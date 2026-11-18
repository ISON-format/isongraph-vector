using System.Buffers;
using System.Diagnostics;
using System.Numerics;
using NVec = System.Numerics.Vector;
using System.Runtime.InteropServices;

namespace IsonGraph.Vector;

/// <summary>
/// Inverted-file ("IVF-flat") index over a <see cref="VectorTable"/>.
///
/// Spherical k-means partitions the rows into <c>Nlist</c> cells by direction,
/// which is what cosine similarity compares. A query scores the cells whose
/// centroids are nearest and only the rows inside them, with the same cosine
/// the exact scan uses (the same kernel, so the same bits).
///
/// The index owns no vectors: the table holds the one copy of every row, and
/// the index holds only the centroids and, per slot, which cell it is in:
///   * <c>_assign[slot]</c>: its cell, <see cref="Pending"/>, or <see cref="NoCell"/>;
///   * <c>_pos[slot]</c>: its position in that cell's list (or the pending
///     list), so a row leaves its list in O(1) by swapping the last one in;
///   * <c>_lists[cell]</c>: exactly the live folded rows of a cell, no stale entries.
///
/// It follows the store instead of being rebuilt on every write: a new or
/// updated row is "pending" and every query scores the pending rows exactly,
/// so a write is visible at once; the write that makes <see cref="FoldBatch"/>
/// pending rows folds them into their nearest cells. A delete detaches the row.
/// Compaction renumbers slots and the index follows without retraining.
/// <c>Churn</c> counts writes and deletes since training; past a fraction of
/// the trained size the store retrains.
///
/// Mutating members are called only under the store's write lock; searches
/// read the index under the read lock and touch no shared mutable state
/// (all scratch is per call), so any number run at once.
/// </summary>
internal sealed class IvfIndex
{
    /// <summary>Pending rows that trigger a fold (DESIGN.md section 8).</summary>
    public const int FoldBatch = 256;

    private const int NoCell = -1;
    private const int Pending = -2;

    private readonly int _dim;
    private readonly int _stride;                       // dim rounded up to the SIMD width (centroids only)
    private float[] _centroids = Array.Empty<float>();  // nlist x stride
    private int[] _assign = Array.Empty<int>();
    private int[] _pos = Array.Empty<int>();
    private int[][] _lists = Array.Empty<int[]>();
    private int[] _listLen = Array.Empty<int>();
    private int[] _pend = new int[FoldBatch + 16];
    private int _pendLen;

    public int Nlist { get; private set; }
    public long Churn { get; private set; }
    public int TrainedN { get; private set; }
    public double BuildSeconds { get; private set; }
    public int PendingCount => _pendLen;
    public bool Stale => Churn > 0.5 * TrainedN;

    /// <summary>The index keeps no copy of any vector: the table's rows are the only ones.</summary>
    public long VectorBytes => 0;

    public long IndexBytes
    {
        get
        {
            long bytes = ((long)_centroids.Length + _assign.Length + _pos.Length + _listLen.Length + _pend.Length) * sizeof(int);
            foreach (var l in _lists) bytes += (long)l.Length * sizeof(int);
            return bytes;
        }
    }

    private IvfIndex(int dim)
    {
        _dim = dim;
        int lanes = Vector<float>.Count;
        _stride = (dim + lanes - 1) / lanes * lanes;
    }

    // -- building ---------------------------------------------------------------

    private struct SplitMix64
    {
        private ulong _s;
        public SplitMix64(int seed) => _s = unchecked((ulong)(long)seed * 0x9E3779B97F4A7C15UL + 0x632BE59BD9B4E019UL);
        public ulong Next()
        {
            unchecked
            {
                _s += 0x9E3779B97F4A7C15UL;
                ulong z = _s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                return z ^ (z >> 31);
            }
        }
        public int NextInt(int bound) => (int)(((Next() >> 32) * (ulong)bound) >> 32);
    }

    /// <summary>
    /// Train centroids on a sample of the table's live rows, then assign every
    /// live row. Deterministic for a given data set, <paramref name="nlist"/>,
    /// <paramref name="seed"/> and SIMD width (the assignment passes run in
    /// parallel but every row's answer is independent of the others, and the
    /// centroid sums are accumulated in a fixed order).
    /// </summary>
    public static IvfIndex Build(VectorTable t, int nlist, int seed, int iters = 8)
    {
        var started = Stopwatch.GetTimestamp();
        var ivf = new IvfIndex(t.Dim);
        ivf.Train(t, nlist, seed, iters);
        ivf.BuildSeconds = Stopwatch.GetElapsedTime(started).TotalSeconds;
        return ivf;
    }

    private void Train(VectorTable t, int nlist, int seed, int iters)
    {
        int n = t.Slots;
        var live = new int[t.Live];
        for (int i = 0, j = 0; i < n; i++)
            if (t.IsLive(i)) live[j++] = i;
        int total = live.Length;
        nlist = Math.Max(1, Math.Min(nlist, total));
        var rng = new SplitMix64(seed);

        // Sample up to 40 x nlist live rows (kept in row order).
        int sampleN = (int)Math.Min(total, 40L * nlist);
        int[] sample;
        if (sampleN == total)
            sample = live;
        else
        {
            for (int i = 0; i < sampleN; i++)       // partial Fisher-Yates
            {
                int j = i + rng.NextInt(total - i);
                (live[i], live[j]) = (live[j], live[i]);
            }
            sample = new int[sampleN];
            Array.Copy(live, sample, sampleN);
            Array.Sort(sample);
        }

        // The sample is never copied out of the table: a unit-length, SIMD-padded copy of one row
        // (a zero row stays zero) is produced on demand, so training allocates no sample matrix.
        int stride = _stride;

        // Initial centroids: nlist distinct sample rows.
        var cent = new float[(long)nlist * stride];
        var pick = new int[sampleN];
        for (int i = 0; i < sampleN; i++) pick[i] = i;
        for (int c = 0; c < nlist; c++)
        {
            int j = c + rng.NextInt(sampleN - c);
            (pick[c], pick[j]) = (pick[j], pick[c]);
            UnitRow(t, sample[pick[c]], new Span<float>(cent, c * stride, stride));
        }

        var cell = new int[sampleN];
        var counts = new int[nlist];
        var sums = new double[(long)nlist * stride];
        var unit = new float[stride];
        for (int it = 0; it < iters; it++)
        {
            AssignRows(t, sample, sampleN, cent, nlist, cell, BlockRows, unitRows: true);
            Array.Clear(counts);
            Array.Clear(sums);
            for (int r = 0; r < sampleN; r++)       // sequential: a fixed summation order
            {
                int c = cell[r];
                counts[c]++;
                UnitRow(t, sample[r], unit);
                var acc = new Span<double>(sums, c * stride, stride);
                for (int d = 0; d < stride; d++) acc[d] += unit[d];
            }
            for (int c = 0; c < nlist; c++)
            {
                var dst = new Span<float>(cent, c * stride, stride);
                var acc = new ReadOnlySpan<double>(sums, c * stride, stride);
                for (int d = 0; d < stride; d++) dst[d] = (float)acc[d];
                float length = MathF.Sqrt(Dot(dst, dst));
                if (counts[c] == 0 || length == 0f)  // re-seed a dead centroid on a random point
                {
                    UnitRow(t, sample[rng.NextInt(sampleN)], dst);
                    length = MathF.Sqrt(Dot(dst, dst));
                }
                if (length > 0f)
                    for (int d = 0; d < stride; d++) dst[d] /= length;
            }
        }

        _centroids = cent;
        Nlist = nlist;
        _assign = new int[Math.Max(n, 16)];
        _pos = new int[_assign.Length];
        Array.Fill(_assign, NoCell);
        AssignRows(t, null, n, cent, nlist, _assign, BlockRows, unitRows: false);
        for (int i = 0; i < n; i++)
            if (!t.IsLive(i)) _assign[i] = NoCell;
        _pendLen = 0;
        Churn = 0;
        TrainedN = total;
        Relist(n);
    }

    /// <summary>Rebuild every cell list from <c>_assign</c> (slot order within a cell) and the positions.</summary>
    private void Relist(int n)
    {
        var counts = new int[Nlist];
        for (int i = 0; i < n; i++)
        {
            int c = _assign[i];
            if (c >= 0) counts[c]++;
        }
        _lists = new int[Nlist][];
        _listLen = new int[Nlist];
        for (int c = 0; c < Nlist; c++) _lists[c] = new int[counts[c] + counts[c] / 8 + 4];
        for (int i = 0; i < n; i++)
        {
            int c = _assign[i];
            if (c < 0) continue;
            _pos[i] = _listLen[c];
            _lists[c][_listLen[c]++] = i;
        }
    }

    // -- following the store ----------------------------------------------------

    private void Detach(int slot)
    {
        if ((uint)slot >= (uint)_assign.Length) return;
        int cell = _assign[slot];
        if (cell == NoCell) return;
        int at = _pos[slot];
        int[] list;
        int last;
        if (cell == Pending) { list = _pend; last = --_pendLen; }
        else { list = _lists[cell]; last = --_listLen[cell]; }
        if (at != last)
        {
            int moved = list[last];
            list[at] = moved;
            _pos[moved] = at;
        }
        _assign[slot] = NoCell;
    }

    /// <summary>
    /// A row was written (new, or its vector replaced): it waits in the pending
    /// list. Returns true when the list is long enough that the caller should
    /// <see cref="Fold"/> now.
    /// </summary>
    public bool NotePut(int slot)
    {
        if (slot >= _assign.Length)
        {
            int cap = Math.Max(_assign.Length * 2, slot + 1);
            int old = _assign.Length;
            Array.Resize(ref _assign, cap);
            Array.Resize(ref _pos, cap);
            Array.Fill(_assign, NoCell, old, cap - old);
        }
        Detach(slot);
        if (_pendLen == _pend.Length) Array.Resize(ref _pend, _pend.Length * 2);
        _assign[slot] = Pending;
        _pos[slot] = _pendLen;
        _pend[_pendLen++] = slot;
        Churn++;
        return _pendLen >= FoldBatch;
    }

    public void NoteDrop(int slot)
    {
        Detach(slot);
        Churn++;
    }

    /// <summary>Give every pending row its nearest cell.</summary>
    public void Fold(VectorTable t)
    {
        if (_pendLen == 0) return;
        var slots = new int[_pendLen];
        Array.Copy(_pend, slots, _pendLen);
        var cells = new int[slots.Length];
        AssignRows(t, slots, slots.Length, _centroids, Nlist, cells, FoldBlockRows, unitRows: false);
        _pendLen = 0;
        for (int i = 0; i < slots.Length; i++)
        {
            int slot = slots[i], c = cells[i];
            if (_listLen[c] == _lists[c].Length) Array.Resize(ref _lists[c], _lists[c].Length * 2 + 4);
            _assign[slot] = c;
            _pos[slot] = _listLen[c];
            _lists[c][_listLen[c]++] = slot;
        }
    }

    /// <summary>Follow the table's compaction: <paramref name="remap"/>[old] is the new slot, or -1 for a dead one.</summary>
    public void Compact(int[] remap, int newSlots)
    {
        var assign = new int[Math.Max(newSlots, 16)];
        Array.Fill(assign, NoCell);
        for (int old = 0; old < remap.Length; old++)
        {
            int s = remap[old];
            if (s >= 0 && old < _assign.Length) assign[s] = _assign[old];
        }
        var pending = new int[Math.Max(_pend.Length, FoldBatch + 16)];
        int p = 0;
        for (int i = 0; i < _pendLen; i++)
        {
            int s = remap[_pend[i]];
            if (s >= 0) pending[p++] = s;
        }
        _assign = assign;
        _pos = new int[assign.Length];
        Relist(newSlots);
        _pend = pending;
        _pendLen = p;
        for (int i = 0; i < p; i++) _pos[pending[i]] = i;
    }

    /// <summary>Every invariant of DESIGN.md section 7 that concerns the index; null when they hold.</summary>
    public string? Verify(VectorTable t)
    {
        if (_assign.Length < t.Slots) return $"index covers {_assign.Length} rows, table has {t.Slots}";
        if (_pendLen >= FoldBatch) return $"{_pendLen} rows pending, the cap is {FoldBatch}";
        int seen = 0;
        for (int c = 0; c < Nlist; c++)
            for (int at = 0; at < _listLen[c]; at++)
            {
                int s = _lists[c][at];
                if (_assign[s] != c || _pos[s] != at) return $"slot {s} listed in cell {c} at {at} but assigned {_assign[s]} at {_pos[s]}";
                if (!t.IsLive(s)) return $"dead slot {s} still in cell {c}";
                seen++;
            }
        for (int at = 0; at < _pendLen; at++)
        {
            int s = _pend[at];
            if (_assign[s] != Pending || _pos[s] != at || !t.IsLive(s)) return $"pending slot {s} is inconsistent";
            seen++;
        }
        if (seen != t.Live) return $"{seen} rows indexed, {t.Live} live";
        for (int s = 0; s < t.Slots; s++)
            if (!t.IsLive(s) && _assign[s] != NoCell) return $"dead slot {s} is assigned";
        return null;
    }

    // -- nearest-centroid kernel ------------------------------------------------

    private const int BlockRows = 256;
    private const int FoldBlockRows = 32;
    private const int CentroidTile = 64;

    [ThreadStatic] private static float[]? t_block;

    /// <summary>
    /// A row of the table as the k-means kernel wants it: zero-padded to the
    /// SIMD width and, when <paramref name="unit"/>, scaled to unit length (a
    /// zero row stays zero).
    /// </summary>
    private void UnitRow(VectorTable t, int slot, Span<float> dst, bool unit = true)
    {
        t.Row(slot).CopyTo(dst);
        dst.Slice(_dim).Clear();
        if (!unit) return;
        float norm = MathF.Sqrt(Dot(dst, dst));
        float scale = norm > 0f ? 1f / norm : 0f;
        for (int d = 0; d < _dim; d++) dst[d] *= scale;
    }

    /// <summary>
    /// For <paramref name="count"/> table rows (slots 0.. or the named
    /// <paramref name="slots"/>), the nearest centroid. <paramref name="output"/>
    /// is indexed like the rows (by slot when <paramref name="slots"/> is null).
    /// Each block of rows is copied into a zero-padded buffer first, so the
    /// kernel and its results are the ones the padded layout always had.
    /// </summary>
    private void AssignRows(VectorTable t, int[]? slots, int count, float[] cent, int nlist, int[] output, int blockRows, bool unitRows)
    {
        if (count == 0) return;
        int stride = _stride;
        int blocks = (count + blockRows - 1) / blockRows;
        void One(int b)
        {
            int start = b * blockRows;
            int len = Math.Min(blockRows, count - start);
            var buf = t_block;
            if (buf is null || buf.Length < len * stride) t_block = buf = new float[Math.Max(len, BlockRows) * stride];
            for (int i = 0; i < len; i++)
            {
                int slot = slots is null ? start + i : slots[start + i];
                UnitRow(t, slot, new Span<float>(buf, i * stride, stride), unitRows);
            }
            NearestBlock(buf, null, 0, len, cent, nlist, stride, output, start);
        }
        if (blocks == 1) One(0);
        else Parallel.For(0, blocks, One);
    }

    private static void NearestBlock(
        float[] data, int[]? ids, int start, int count, float[] cent, int nlist, int stride, int[] output,
        int outputStart = -1)
    {
        Span<float> best = stackalloc float[BlockRows];
        Span<int> bestIdx = stackalloc int[BlockRows];
        Span<int> offs = stackalloc int[BlockRows];
        best.Slice(0, count).Fill(float.NegativeInfinity);
        bestIdx.Slice(0, count).Clear();
        for (int i = 0; i < count; i++)
            offs[i] = (ids is null ? start + i : ids[start + i]) * stride;

        for (int tile = 0; tile < nlist; tile += CentroidTile)
        {
            int tileEnd = Math.Min(nlist, tile + CentroidTile);
            int g = 0;
            for (; g + 4 <= count; g += 4)
                Update4(data, offs[g], offs[g + 1], offs[g + 2], offs[g + 3], cent, tile, tileEnd,
                        stride, best, bestIdx, g);
            for (; g < count; g++)
            {
                var row = new ReadOnlySpan<float>(data, offs[g], stride);
                for (int c = tile; c < tileEnd; c++)
                {
                    float s = Dot(row, new ReadOnlySpan<float>(cent, c * stride, stride));
                    if (s > best[g]) { best[g] = s; bestIdx[g] = c; }
                }
            }
        }
        int o = outputStart >= 0 ? outputStart : start;
        for (int i = 0; i < count; i++)
            output[o + i] = bestIdx[i];
    }

    /// <summary>Four rows against centroids [from, to): two centroids per step, eight independent accumulators.</summary>
    private static void Update4(
        float[] data, int o0, int o1, int o2, int o3, float[] cent, int from, int to, int stride,
        Span<float> best, Span<int> bestIdx, int g)
    {
        var r0 = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(data, o0, stride));
        var r1 = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(data, o1, stride));
        var r2 = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(data, o2, stride));
        var r3 = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(data, o3, stride));
        int n = r0.Length;
        int c = from;
        for (; c + 1 < to; c += 2)
        {
            var ca = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(cent, c * stride, stride));
            var cb = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(cent, (c + 1) * stride, stride));
            Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
            Vector<float> b0 = default, b1 = default, b2 = default, b3 = default;
            for (int i = 0; i < n; i++)
            {
                var x = ca[i];
                var y = cb[i];
                var v0 = r0[i]; a0 += v0 * x; b0 += v0 * y;
                var v1 = r1[i]; a1 += v1 * x; b1 += v1 * y;
                var v2 = r2[i]; a2 += v2 * x; b2 += v2 * y;
                var v3 = r3[i]; a3 += v3 * x; b3 += v3 * y;
            }
            Offer(best, bestIdx, g, NVec.Sum(a0), c);
            Offer(best, bestIdx, g, NVec.Sum(b0), c + 1);
            Offer(best, bestIdx, g + 1, NVec.Sum(a1), c);
            Offer(best, bestIdx, g + 1, NVec.Sum(b1), c + 1);
            Offer(best, bestIdx, g + 2, NVec.Sum(a2), c);
            Offer(best, bestIdx, g + 2, NVec.Sum(b2), c + 1);
            Offer(best, bestIdx, g + 3, NVec.Sum(a3), c);
            Offer(best, bestIdx, g + 3, NVec.Sum(b3), c + 1);
        }
        if (c < to)
        {
            var ca = MemoryMarshal.Cast<float, Vector<float>>(new ReadOnlySpan<float>(cent, c * stride, stride));
            Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
            for (int i = 0; i < n; i++)
            {
                var x = ca[i];
                a0 += r0[i] * x; a1 += r1[i] * x; a2 += r2[i] * x; a3 += r3[i] * x;
            }
            Offer(best, bestIdx, g, NVec.Sum(a0), c);
            Offer(best, bestIdx, g + 1, NVec.Sum(a1), c);
            Offer(best, bestIdx, g + 2, NVec.Sum(a2), c);
            Offer(best, bestIdx, g + 3, NVec.Sum(a3), c);
        }
    }

    private static void Offer(Span<float> best, Span<int> bestIdx, int g, float score, int c)
    {
        if (score > best[g]) { best[g] = score; bestIdx[g] = c; }
    }

    /// <summary>Dot product of two equal-length, lane-padded spans (centroid ranking and k-means only).</summary>
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var va = MemoryMarshal.Cast<float, Vector<float>>(a);
        var vb = MemoryMarshal.Cast<float, Vector<float>>(b);
        int n = Math.Min(va.Length, vb.Length);
        Vector<float> s0 = default, s1 = default, s2 = default, s3 = default;
        int i = 0;
        for (; i + 4 <= n; i += 4)
        {
            s0 += va[i] * vb[i];
            s1 += va[i + 1] * vb[i + 1];
            s2 += va[i + 2] * vb[i + 2];
            s3 += va[i + 3] * vb[i + 3];
        }
        for (; i < n; i++) s0 += va[i] * vb[i];
        return NVec.Sum((s0 + s1) + (s2 + s3));
    }

    // -- query ------------------------------------------------------------------

    /// <summary>
    /// Feed the rows of the <paramref name="lists"/> cells nearest the query,
    /// plus every pending row, to <paramref name="visitor"/> (rows of the type
    /// <paramref name="code"/>, or all when it is negative). Allocates only
    /// its own scratch, so concurrent calls are safe. The caller has checked
    /// the query has a direction (a non-zero norm).
    /// </summary>
    public void Scan<TV>(VectorTable t, float[] query, int code, int lists, ref TV visitor)
        where TV : struct, IRowVisitor
    {
        int nlist = Nlist, stride = _stride;
        // the query, padded the way the centroids are
        var qpad = ArrayPool<float>.Shared.Rent(stride);
        var keys = ArrayPool<ulong>.Shared.Rent(nlist);
        try
        {
            Array.Clear(qpad, 0, stride);
            Array.Copy(query, qpad, _dim);
            var q = new ReadOnlySpan<float>(qpad, 0, stride);
            for (int c = 0; c < nlist; c++)
            {
                float s = Dot(q, new ReadOnlySpan<float>(_centroids, c * stride, stride));
                int bits = BitConverter.SingleToInt32Bits(s);
                uint ordered = bits < 0 ? ~(uint)bits : (uint)bits | 0x80000000u;   // ascending = ascending score
                keys[c] = ((ulong)~ordered << 32) | (uint)c;                         // best score first, then lowest cell
            }
            if (lists < nlist) Array.Sort(keys, 0, nlist);
            else lists = nlist;

            for (int l = 0; l < lists; l++)
            {
                int cell = (int)(keys[l] & 0xFFFFFFFFu);
                RowScan.Slots(t, query, code, _lists[cell], 0, _listLen[cell], ref visitor);
            }
            RowScan.Slots(t, query, code, _pend, 0, _pendLen, ref visitor);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(qpad);
            ArrayPool<ulong>.Shared.Return(keys);
        }
    }
}
