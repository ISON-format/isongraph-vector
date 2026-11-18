using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using IsonGraph;

namespace IsonGraph.Vector;

// Optional binary snapshot of an EmbeddingStore, written next to (never
// instead of) the JSON embeddings sidecar. DESIGN.md section 11 is the
// normative byte layout; everything here is little endian explicitly and
// IEEE-754 binary32, whatever the host.

/// <summary>Where a load's embeddings came from.</summary>
public enum LoadSource
{
    /// <summary>The binary snapshot.</summary>
    Snapshot,
    /// <summary>The JSON sidecar.</summary>
    Json,
    /// <summary>Nothing was restored (embeddings off, or no sidecar).</summary>
    NoEmbeddings,
}

/// <summary>What <see cref="SemanticGraph.LoadWithReport"/> did.</summary>
/// <param name="Source">Where the embeddings came from.</param>
/// <param name="SnapshotSkipped">
/// Why a snapshot that exists was not used (and the JSON sidecar was read
/// instead); null when it was used or there is none.
/// </param>
/// <param name="Embeddings">Embeddings restored.</param>
/// <param name="Seconds">Wall time of the whole load.</param>
public sealed record LoadReport(LoadSource Source, string? SnapshotSkipped, int Embeddings, double Seconds);

/// <summary>
/// A cheap identity for a sidecar file: its length and a SHA-256 over its
/// length, its first and last 64 KiB and 64 probes of 4 KiB spread over the
/// rest. Any rewrite of the sidecar by a serializer (every float re-rendered)
/// changes it; a hand edit that keeps the length and avoids every sampled byte
/// does not, which is why a save that does not write a snapshot deletes the
/// old one.
/// </summary>
public sealed class SidecarId : IEquatable<SidecarId>
{
    private const int Edge = 64 * 1024;
    private const int Probe = 4 * 1024;
    private const int Probes = 64;

    public long Length { get; }

    /// <summary>32 bytes.</summary>
    public byte[] Fingerprint { get; }

    public SidecarId(long length, byte[] fingerprint)
    {
        if (fingerprint.Length != 32) throw new ArgumentException("a fingerprint is 32 bytes", nameof(fingerprint));
        Length = length;
        Fingerprint = fingerprint;
    }

    /// <summary>Read the identity of the file at <paramref name="path"/>.</summary>
    public static SidecarId OfFile(string path)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        long len = file.Length;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> eight = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(eight, len);
        hash.AppendData(eight);
        var buf = new byte[Edge];
        void Take(long at, int n)
        {
            n = (int)Math.Min(n, Math.Max(0, len - at));
            file.Seek(at, SeekOrigin.Begin);
            file.ReadExactly(buf, 0, n);
            hash.AppendData(buf, 0, n);
        }
        Take(0, Edge);
        Take(Math.Max(0, len - Edge), Edge);
        for (int i = 1; i <= Probes; i++)
            Take((long)((UInt128)(ulong)len * (ulong)i / (Probes + 1)), Probe);
        return new SidecarId(len, hash.GetHashAndReset());
    }

    public bool Equals(SidecarId? other) =>
        other is not null && Length == other.Length && Fingerprint.AsSpan().SequenceEqual(other.Fingerprint);

    public override bool Equals(object? obj) => Equals(obj as SidecarId);
    public override int GetHashCode() => HashCode.Combine(Length, BitConverter.ToInt32(Fingerprint, 0));
}

/// <summary>What a snapshot is tied to.</summary>
/// <param name="GraphDigest">Digest of the graph file (the JSON sidecar records the same).</param>
/// <param name="Sidecar">The JSON sidecar written for the same save.</param>
public sealed record SnapshotIdentity(string? GraphDigest, SidecarId? Sidecar);

/// <summary>The header of a snapshot, and its size.</summary>
public sealed record SnapshotInfo(
    int Version, int Dimension, int Count, string? GraphDigest, string? GraphDigestBasis,
    string? EncoderModel, string? EncoderVersion, SidecarId? Sidecar, long Bytes);

/// <summary>A snapshot that cannot be used, with the reason.</summary>
internal sealed class SnapshotRejected : Exception
{
    public SnapshotRejected(string message) : base(message) { }
}

/// <summary>Buffers writes, hashing every byte that goes through it.</summary>
internal sealed class SnapshotWriter
{
    private readonly Stream _inner;
    private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    private readonly byte[] _buf = new byte[1 << 22];
    private int _pos;
    public long Written { get; private set; }

    public SnapshotWriter(Stream inner) => _inner = inner;

    private void Drain()
    {
        if (_pos == 0) return;
        _hash.AppendData(_buf, 0, _pos);
        _inner.Write(_buf, 0, _pos);
        Written += _pos;
        _pos = 0;
    }

    public void Bytes(ReadOnlySpan<byte> data)
    {
        if (data.Length > _buf.Length - _pos) Drain();
        if (data.Length > _buf.Length)
        {
            _hash.AppendData(data);
            _inner.Write(data);
            Written += data.Length;
            return;
        }
        data.CopyTo(_buf.AsSpan(_pos));
        _pos += data.Length;
    }

    public void U32(uint v)
    {
        Span<byte> b = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(b, v);
        Bytes(b);
    }

    public void U64(ulong v)
    {
        Span<byte> b = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(b, v);
        Bytes(b);
    }

    public void Str(string s)
    {
        int n = Encoding.UTF8.GetByteCount(s);
        U32((uint)n);
        if (n <= _buf.Length - _pos) { Encoding.UTF8.GetBytes(s, _buf.AsSpan(_pos)); _pos += n; }
        else Bytes(Encoding.UTF8.GetBytes(s));
    }

    public void OptStr(string? s)
    {
        if (s is null) U32(uint.MaxValue); else Str(s);
    }

    public void Floats(ReadOnlySpan<float> v)
    {
        if (BitConverter.IsLittleEndian) { Bytes(MemoryMarshal.AsBytes(v)); return; }
        Span<byte> b = stackalloc byte[4];
        foreach (float f in v)
        {
            BinaryPrimitives.WriteSingleLittleEndian(b, f);
            Bytes(b);
        }
    }

    /// <summary>Flush the body and append its SHA-256 (not itself hashed).</summary>
    public long Finish()
    {
        Drain();
        var digest = _hash.GetCurrentHash();
        _inner.Write(digest);
        Written += digest.Length;
        return Written;
    }
}

/// <summary>Reads a snapshot body, hashing everything it consumes and never reading past the trailer.</summary>
internal sealed class SnapshotReader : IDisposable
{
    private static readonly UTF8Encoding Strict = new(false, true);
    private readonly FileStream _file;
    private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    private long _consumed;
    private readonly long _body;

    public long FileLength { get; }

    public SnapshotReader(string path)
    {
        _file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 22, FileOptions.SequentialScan);
        FileLength = _file.Length;
        if (FileLength < 32 + 8)
        {
            _file.Dispose();
            _hash.Dispose();
            throw new SnapshotRejected("file too short");
        }
        _body = FileLength - 32;
    }

    public long Remaining => Math.Max(0, _body - _consumed);

    public void Exact(Span<byte> dest)
    {
        if (dest.Length > Remaining) throw new SnapshotRejected("truncated");
        _file.ReadExactly(dest);
        _hash.AppendData(dest);
        _consumed += dest.Length;
    }

    public uint U32()
    {
        Span<byte> b = stackalloc byte[4];
        Exact(b);
        return BinaryPrimitives.ReadUInt32LittleEndian(b);
    }

    public ulong U64()
    {
        Span<byte> b = stackalloc byte[8];
        Exact(b);
        return BinaryPrimitives.ReadUInt64LittleEndian(b);
    }

    /// <summary>A length-prefixed string; <paramref name="optional"/> lets the length be 0xFFFFFFFF (absent).</summary>
    public string? Str(bool optional)
    {
        uint len = U32();
        if (len == uint.MaxValue)
            return optional ? null : throw new SnapshotRejected("a required string is marked absent");
        if (len > Remaining) throw new SnapshotRejected("a string is longer than the file");
        Span<byte> b = len <= 1024 ? stackalloc byte[(int)len] : new byte[len];
        Exact(b);
        try { return Strict.GetString(b); }
        catch (DecoderFallbackException) { throw new SnapshotRejected("a string is not UTF-8"); }
    }

    /// <summary>Read <c>dest.Length</c> little-endian IEEE-754 floats.</summary>
    public void Floats(Span<float> dest)
    {
        var bytes = MemoryMarshal.AsBytes(dest);
        Exact(bytes);
        if (!BitConverter.IsLittleEndian)
            for (int i = 0; i < dest.Length; i++)
                dest[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReverseEndianness(BitConverter.SingleToInt32Bits(dest[i])));
    }

    /// <summary>The trailer must equal the hash of the body, and nothing may be left between them.</summary>
    public void VerifyTrailer()
    {
        if (Remaining != 0) throw new SnapshotRejected("bytes left over after the last block");
        var expected = _hash.GetCurrentHash();
        Span<byte> stored = stackalloc byte[32];
        _file.ReadExactly(stored);
        if (!stored.SequenceEqual(expected)) throw new SnapshotRejected("checksum mismatch");
    }

    public void Dispose()
    {
        _file.Dispose();
        _hash.Dispose();
    }
}

public sealed partial class EmbeddingStore
{
    private const int SnapshotBlockRows = 1024;
    private static readonly byte[] SnapshotMagic = "ISONVSNP"u8.ToArray();

    /// <summary>
    /// Write a binary snapshot of every live embedding to <paramref name="path"/>,
    /// atomically (a temp file, flushed, renamed over <paramref name="path"/>: a
    /// failure or crash leaves the previous file untouched). Rows are written in
    /// the order the JSON sidecar uses (node type, then id), so loading a
    /// snapshot gives the same store, slot for slot, as loading the sidecar.
    /// <paramref name="identity"/> ties it to the graph and sidecar it was saved
    /// with; a load that finds a different identity refuses the snapshot.
    /// Holds the shared lock while it writes.
    /// </summary>
    public SnapshotInfo SaveSnapshot(string path, SnapshotIdentity? identity = null)
    {
        SnapshotInfo? info = null;
        AtomicFile.Write(path, stream => info = WriteSnapshot(stream, identity ?? new SnapshotIdentity(null, null)));
        return info!;
    }

    private SnapshotInfo WriteSnapshot(Stream stream, SnapshotIdentity identity)
    {
        _rw.EnterReadLock();
        try
        {
            var t = _table;
            var slots = t is null ? Array.Empty<int>() : SortedLiveSlots(t);
            int dim = t?.Dim ?? 0;

            // models, in order of first appearance among the rows as written
            var models = new List<string>();
            var modelIndex = new Dictionary<string, uint>();
            foreach (int s in slots)
            {
                var m = t!.ModelAt(s);
                if (!modelIndex.ContainsKey(m)) { modelIndex[m] = (uint)models.Count; models.Add(m); }
            }

            string? basis = identity.GraphDigest is null ? null : IsonGraphVectorVersion.GraphDigestBasis;
            string? encoderModel = CurrentModel(), encoderVersion = Version;
            var w = new SnapshotWriter(stream);
            w.Bytes(SnapshotMagic);
            w.U32(1);
            w.U32(identity.Sidecar is null ? 0u : 1u);
            w.U32((uint)dim);
            w.U32(SnapshotBlockRows);
            w.U64((ulong)slots.Length);
            w.OptStr(identity.GraphDigest);
            w.OptStr(basis);
            w.OptStr(encoderModel);
            w.OptStr(encoderVersion);
            if (identity.Sidecar is not null)
            {
                w.U64((ulong)identity.Sidecar.Length);
                w.Bytes(identity.Sidecar.Fingerprint);
            }
            w.U32((uint)models.Count);
            foreach (var m in models) w.Str(m);

            for (int at = 0; at < slots.Length; at += SnapshotBlockRows)
            {
                int end = Math.Min(slots.Length, at + SnapshotBlockRows);
                for (int i = at; i < end; i++)
                {
                    int s = slots[i];
                    var r = t!.RefAt(s);
                    w.Str(r.Type);
                    w.Str(r.Id);
                    w.Str(t.TextAt(s));
                    w.U32(modelIndex[t.ModelAt(s)]);
                }
                for (int i = at; i < end; i++) w.Floats(t!.Row(slots[i]));
            }
            long bytes = w.Finish();
            return new SnapshotInfo(1, dim, slots.Length, identity.GraphDigest, basis, encoderModel, encoderVersion,
                identity.Sidecar, bytes);
        }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>
    /// Replace the store's embeddings with those of a snapshot file. Nothing
    /// changes unless the whole file verifies (structure and checksum); the
    /// approximate-index settings are kept (the index is rebuilt lazily). Does
    /// not check identity: <see cref="SemanticGraph.Load"/> does.
    /// </summary>
    /// <exception cref="EmbeddingException">The file is not a valid snapshot.</exception>
    public SnapshotInfo LoadSnapshot(string path)
    {
        try
        {
            var (stage, info) = ReadSnapshot(path, _ => null);
            Adopt(stage, info);
            return info;
        }
        catch (SnapshotRejected ex)
        {
            throw new EmbeddingException($"invalid snapshot: {ex.Message}");
        }
    }

    /// <summary>
    /// Use the snapshot at <paramref name="snap"/> for the graph whose digest is
    /// <paramref name="graphDigest"/> and whose sidecar is at <paramref name="sidecar"/>,
    /// if it belongs to them. False, with the reason, leaves the store untouched.
    /// </summary>
    internal bool TryAdoptSnapshot(string snap, string sidecar, string graphDigest, out SnapshotInfo? info, out string? skipped)
    {
        info = null;
        skipped = null;
        try
        {
            var current = SidecarId.OfFile(sidecar);
            var (stage, header) = ReadSnapshot(snap, h =>
            {
                if (h.GraphDigest != graphDigest || h.GraphDigestBasis != IsonGraphVectorVersion.GraphDigestBasis)
                    return "it was saved for a different graph (digest differs)";
                if (h.Sidecar is null) return "it records no sidecar identity";
                if (!h.Sidecar.Equals(current))
                    return "the JSON sidecar changed since the snapshot was written (stale snapshot)";
                return null;
            });
            Adopt(stage, header);
            info = header;
            return true;
        }
        catch (SnapshotRejected ex) { skipped = ex.Message; }
        catch (IOException ex) { skipped = ex.Message; }
        catch (UnauthorizedAccessException ex) { skipped = ex.Message; }
        return false;
    }

    private void Adopt(VectorTable? stage, SnapshotInfo info)
    {
        _rw.EnterWriteLock();
        try
        {
            _table = stage;
            _dimension = stage is null ? -1 : info.Dimension;
            _ivf = null;
            var meta = new Dictionary<string, string>();
            if (info.EncoderModel is not null) meta["encoder_model"] = info.EncoderModel;
            if (info.EncoderVersion is not null) meta["encoder_version"] = info.EncoderVersion;
            if (info.GraphDigest is not null) meta["graph_digest"] = info.GraphDigest;
            if (info.GraphDigestBasis is not null) meta["graph_digest_basis"] = info.GraphDigestBasis;
            _lastImportMeta = meta;
        }
        finally { _rw.ExitWriteLock(); }
    }

    /// <summary>
    /// Read and fully verify the snapshot at <paramref name="path"/>, after
    /// <paramref name="accept"/> (a reason to refuse, or null) has approved its
    /// header. Returns the rows in a fresh staging table; the store is not touched.
    /// </summary>
    private static (VectorTable? Stage, SnapshotInfo Info) ReadSnapshot(string path, Func<SnapshotInfo, string?> accept)
    {
        using var r = new SnapshotReader(path);
        Span<byte> magic = stackalloc byte[8];
        r.Exact(magic);
        if (!magic.SequenceEqual(SnapshotMagic)) throw new SnapshotRejected("not a snapshot (bad magic)");
        uint version = r.U32();
        if (version != 1)
            throw new SnapshotRejected($"unsupported snapshot version {version} (this build reads 1)");
        uint flags = r.U32();
        if ((flags & ~1u) != 0) throw new SnapshotRejected($"unknown flags 0x{flags:x}");
        uint dim = r.U32();
        uint blockRows = r.U32();
        ulong count = r.U64();
        if (blockRows == 0 || blockRows > 1u << 20) throw new SnapshotRejected($"implausible block size {blockRows}");
        if (count > int.MaxValue || (count > 0 && dim == 0) || dim > int.MaxValue)
            throw new SnapshotRejected($"implausible shape {count} x {dim}");
        if ((UInt128)count * dim * 4 > (ulong)r.Remaining)
            throw new SnapshotRejected("fewer bytes than the rows need");
        string? digest = r.Str(true), basis = r.Str(true), encoderModel = r.Str(true), encoderVersion = r.Str(true);
        SidecarId? sidecar = null;
        if ((flags & 1) != 0)
        {
            long len = (long)r.U64();
            var fingerprint = new byte[32];
            r.Exact(fingerprint);
            sidecar = new SidecarId(len, fingerprint);
        }
        uint modelCount = r.U32();
        if (modelCount > r.Remaining / 4) throw new SnapshotRejected("more models than the file can hold");
        var models = new string[modelCount];
        for (int i = 0; i < models.Length; i++) models[i] = r.Str(false)!;
        var info = new SnapshotInfo(1, (int)dim, (int)count, digest, basis, encoderModel, encoderVersion, sidecar, r.FileLength);
        var refusal = accept(info);
        if (refusal is not null) throw new SnapshotRejected(refusal);

        VectorTable? stage = count > 0 ? new VectorTable((int)dim) : null;
        var meta = new (string Type, string Id, string Text, string Model)[Math.Min(blockRows, (uint)Math.Max(count, 1UL))];
        var floats = new float[(long)meta.Length * (int)dim];
        ulong left = count;
        while (left > 0)
        {
            int rows = (int)Math.Min(left, blockRows);
            for (int i = 0; i < rows; i++)
            {
                string type = r.Str(false)!, id = r.Str(false)!, text = r.Str(false)!;
                uint model = r.U32();
                if (model >= models.Length) throw new SnapshotRejected("a row names a model that is not listed");
                meta[i] = (type, id, text, models[model]);
            }
            if ((ulong)rows * dim * 4 > (ulong)r.Remaining) throw new SnapshotRejected("truncated");
            r.Floats(floats.AsSpan(0, rows * (int)dim));
            for (int i = 0; i < rows; i++)
            {
                var (type, id, text, model) = meta[i];
                stage!.Put(new NodeRef(type, id), $"{type}:{id}", text, floats.AsSpan(i * (int)dim, (int)dim), model, out bool replaced);
                if (replaced) throw new SnapshotRejected($"the key {type}:{id} appears twice");
            }
            left -= (ulong)rows;
        }
        r.VerifyTrailer();
        return (stage, info);
    }
}
