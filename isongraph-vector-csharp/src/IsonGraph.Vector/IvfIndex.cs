using System.Diagnostics;
using System.Numerics;
using NVec = System.Numerics.Vector;
using System.Runtime.InteropServices;

namespace IsonGraph.Vector;

/// <summary>
/// Inverted-file ("IVF-flat") index over a contiguous copy of a store's vectors.
///
/// Spherical k-means partitions the rows into <c>Nlist</c> cells by direction,
/// which is what cosine similarity compares. A query scores the <c>lists</c>
/// cells whose centroids are nearest and only the rows inside them, with the
/// same cosine formula the exact scan uses.
///
/// Storage is struct-of-arrays: one <c>float[]</c> matrix (row stride padded
/// to a whole number of SIMD lanes, so the dot product has no scalar tail),
/// the row norms, a type code per row and the owning record. The cells are a
/// per-row cell number (<c>_assign</c>, -1 = none) plus the rows grouped by
/// cell (<c>_order</c>/<c>_offsets</c>).
///
/// It follows the store it belongs to instead of being rebuilt on every write:
///   * a new or updated row is "pending": it has no cell, and every query
///     scores the pending rows exactly, so a write is visible immediately;
///   * pending rows are folded into their nearest cell in one pass once there
///     are enough of them (the store decides when);
///   * a delete clears the row's cell; stale entries in <c>_order</c> are
///     skipped at query time (a row is only taken from the cell it is assigned to);
///   * compaction (when a quarter of the slots are dead) remaps row numbers and
///     carries the cells across without retraining;
///   * <c>Churn</c> counts every write and delete since the centroids were
///     trained; the store retrains when it passes a fraction of the trained size.
///
/// Not thread-safe: the owning store calls it only under its lock. Queries
/// reuse scratch buffers owned by the index, so a query allocates nothing
/// per scanned row (only its result objects).
/// </summary>
internal sealed class IvfIndex
{
    // -- rows -------------------------------------------------------------------
    private readonly int _dim;
    private readonly int _stride;                       // dim rounded up to the SIMD width
    private float[] _matrix;
    private float[] _norms;
    private int[] _codes;
    private bool[] _alive;
    private EmbeddingRecord?[] _recs;
    private int[] _assign;
    private int _n;                                     // slots in use, tombstones included
    private int _dead;
    private readonly Dictionary<string, int> _pos = new();      // record id -> slot
    private readonly Dictionary<string, int> _codeOf = new();   // node type -> code

    // -- cells ------------------------------------------------------------------
    private float[] _centroids = Array.Empty<float>();
    private int[] _order = Array.Empty<int>();
    private int[] _offsets = Array.Empty<int>();
    private readonly HashSet<int> _pending = new();

    public int Nlist { get; private set; }
    public long Churn { get; private set; }
    public int TrainedN { get; private set; }
    public double BuildSeconds { get; private set; }
    public int PendingCount => _pending.Count;
    public int LiveCount => _n - _dead;
    public long VectorBytes => ((long)_matrix.Length + _norms.Length) * sizeof(float);
    public long IndexBytes =>
        ((long)_centroids.Length + _assign.Length + _order.Length + _offsets.Length) * sizeof(int);

    // -- scratch (single-threaded, under the store lock) ------------------------
    private float[] _q;
    private float[] _cKey = Array.Empty<float>();
    private int[] _cId = Array.Empty<int>();
    private int[] _candRows = new int[256];
    private float[] _candScores = new float[256];
    private int _candCount;
    private int[] _topRows = new int[16];
    private float[] _topScores = new float[16];

    private IvfIndex(int dim, int capacity)
    {
        _dim = dim;
        int lanes = Vector<float>.Count;
        _stride = (dim + lanes - 1) / lanes * lanes;
        long cells = (long)capacity * _stride;
        if (cells > Array.MaxLength)
            throw new EmbeddingException(
                $"approximate index would need {cells} floats ({capacity} vectors x {_stride}); " +
                "the contiguous copy is limited to about 2 billion values");
        _matrix = new float[cells];
        _norms = new float[capacity];
        _codes = new int[capacity];
        _alive = new bool[capacity];
        _recs = new EmbeddingRecord?[capacity];
        _assign = new int[capacity];
        Array.Fill(_assign, -1);
        _q = new float[_stride];
    }

    /// <summary>Copy <paramref name="records"/> into a fresh, untrained index. Call <see cref="Train"/> next.</summary>
    public static IvfIndex FromRecords(IEnumerable<EmbeddingRecord> records, int dim, int count)
    {
        var index = new IvfIndex(dim, count + count / 8 + 16);
        foreach (var record in records) index.Append(record);
        return index;
    }

    // -- writes -----------------------------------------------------------------

    private int CodeFor(string nodeType)
    {
        if (!_codeOf.TryGetValue(nodeType, out int code))
            _codeOf[nodeType] = code = _codeOf.Count;
        return code;
    }

    private void Grow()
    {
        int capacity = Math.Max(_alive.Length + _alive.Length / 2, _alive.Length + 64);
        if ((long)capacity * _stride > Array.MaxLength)
            capacity = Array.MaxLength / _stride;
        if (capacity <= _alive.Length)
            throw new EmbeddingException("approximate index is full (contiguous copy limited to about 2 billion values)");
        int old = _alive.Length;
        Array.Resize(ref _matrix, capacity * _stride);
        Array.Resize(ref _norms, capacity);
        Array.Resize(ref _codes, capacity);
        Array.Resize(ref _alive, capacity);
        Array.Resize(ref _recs, capacity);
        Array.Resize(ref _assign, capacity);
        Array.Fill(_assign, -1, old, capacity - old);
    }

    private int Append(EmbeddingRecord record)
    {
        if (_n >= _alive.Length) Grow();
        int i = _n++;
        _pos[record.Id] = i;
        _alive[i] = true;
        Store(i, record);
        return i;
    }

    private void Store(int i, EmbeddingRecord record)
    {
        var dst = new Span<float>(_matrix, i * _stride, _stride);
        record.Vector.AsSpan(0, _dim).CopyTo(dst);
        dst.Slice(_dim).Clear();
        _norms[i] = MathF.Sqrt(Dot(dst, dst));
        _codes[i] = CodeFor(record.NodeRef.Type);
        _recs[i] = record;
    }

    /// <summary>A new or replaced vector: visible to the next query, folded into a cell later.</summary>
    public void Put(EmbeddingRecord record)
    {
        if (_pos.TryGetValue(record.Id, out int i))
            Store(i, record);                       // overwrite in place: keeps its position
        else
            i = Append(record);
        _assign[i] = -1;
        _pending.Add(i);
        Churn++;
    }

    /// <summary>A deleted vector: the slot becomes a tombstone no query will read.</summary>
    public void Drop(string id)
    {
        if (!_pos.Remove(id, out int i)) return;
        _norms[i] = 0f;
        _alive[i] = false;
        _recs[i] = null;
        _assign[i] = -1;
        _pending.Remove(i);
        _dead++;
        Churn++;
        while (_n > 0 && !_alive[_n - 1])           // trailing tombstones just shorten the table
        {
            _n--;
            _dead--;
        }
        if (_dead > 32 && _dead * 4 > _n) Compact();
    }

    /// <summary>Squeeze tombstones out, keeping the live rows in order, and carry the cells across.</summary>
    private void Compact()
    {
        var remap = new int[_n];
        int kept = 0;
        for (int i = 0; i < _n; i++)
        {
            if (!_alive[i]) { remap[i] = -1; continue; }
            remap[i] = kept;
            if (kept != i)                          // destination never overtakes the source
            {
                Array.Copy(_matrix, i * _stride, _matrix, kept * _stride, _stride);
                _norms[kept] = _norms[i];
                _codes[kept] = _codes[i];
                _recs[kept] = _recs[i];
                _assign[kept] = _assign[i];
            }
            kept++;
        }
        if (_pending.Count > 0)
        {
            var moved = new List<int>(_pending.Count);
            foreach (int old in _pending) moved.Add(remap[old]);
            _pending.Clear();
            foreach (int row in moved) _pending.Add(row);
        }
        for (int i = kept; i < _n; i++)
        {
            _recs[i] = null;
            _alive[i] = false;
            _assign[i] = -1;
            _norms[i] = 0f;
        }
        for (int i = 0; i < kept; i++)
        {
            _alive[i] = true;
            _pos[_recs[i]!.Id] = i;
        }
        _n = kept;
        _dead = 0;
        Relist();
    }

    // -- training ---------------------------------------------------------------

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
    /// Train centroids on a sample, then assign every live row. Deterministic
    /// for a given data set, <paramref name="nlist"/>, <paramref name="seed"/>
    /// and SIMD width (the assignment passes run in parallel but every row's
    /// answer is independent of the others, and the centroid sums are
    /// accumulated in a fixed order).
    /// </summary>
    public void Train(int nlist, int seed, int iters = 8)
    {
        var started = Stopwatch.GetTimestamp();
        var live = new int[LiveCount];
        for (int i = 0, j = 0; i < _n; i++)
            if (_alive[i]) live[j++] = i;
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

        // Unit-length sample rows (a zero row stays zero).
        var x = new float[(long)sampleN * _stride];
        for (int r = 0; r < sampleN; r++)
        {
            int row = sample[r];
            float norm = _norms[row];
            float scale = norm > 0f ? 1f / norm : 0f;
            var src = new ReadOnlySpan<float>(_matrix, row * _stride, _stride);
            var dst = new Span<float>(x, r * _stride, _stride);
            for (int d = 0; d < _stride; d++) dst[d] = src[d] * scale;
        }

        // Initial centroids: nlist distinct sample rows.
        var cent = new float[(long)nlist * _stride];
        var pick = new int[sampleN];
        for (int i = 0; i < sampleN; i++) pick[i] = i;
        for (int c = 0; c < nlist; c++)
        {
            int j = c + rng.NextInt(sampleN - c);
            (pick[c], pick[j]) = (pick[j], pick[c]);
            Array.Copy(x, pick[c] * _stride, cent, c * _stride, _stride);
        }

        var cell = new int[sampleN];
        var counts = new int[nlist];
        var sums = new double[(long)nlist * _stride];
        for (int it = 0; it < iters; it++)
        {
            AssignNearest(x, null, sampleN, cent, nlist, cell);
            Array.Clear(counts);
            Array.Clear(sums);
            for (int r = 0; r < sampleN; r++)       // sequential: a fixed summation order
            {
                int c = cell[r];
                counts[c]++;
                var src = new ReadOnlySpan<float>(x, r * _stride, _stride);
                var acc = new Span<double>(sums, c * _stride, _stride);
                for (int d = 0; d < _stride; d++) acc[d] += src[d];
            }
            for (int c = 0; c < nlist; c++)
            {
                var dst = new Span<float>(cent, c * _stride, _stride);
                var acc = new ReadOnlySpan<double>(sums, c * _stride, _stride);
                for (int d = 0; d < _stride; d++) dst[d] = (float)acc[d];
                float length = MathF.Sqrt(Dot(dst, dst));
                if (counts[c] == 0 || length == 0f)  // re-seed a dead centroid on a random point
                {
                    Array.Copy(x, rng.NextInt(sampleN) * _stride, cent, c * _stride, _stride);
                    length = MathF.Sqrt(Dot(dst, dst));
                }
                if (length > 0f)
                    for (int d = 0; d < _stride; d++) dst[d] /= length;
            }
        }

        _centroids = cent;
        Nlist = nlist;
        _cKey = new float[nlist];
        _cId = new int[nlist];

        // Every row, in the order of the table. (Raw rows rank centroids the same
        // as unit rows: scaling by a positive norm does not change the argmax.)
        AssignNearest(_matrix, null, _n, cent, nlist, _assign);
        for (int i = 0; i < _n; i++)
            if (!_alive[i]) _assign[i] = -1;
        _pending.Clear();
        Churn = 0;
        TrainedN = total;
        Relist();
        BuildSeconds = Stopwatch.GetElapsedTime(started).TotalSeconds;
    }

    /// <summary>Regroup the rows by cell (a stable counting sort of <c>_assign</c>).</summary>
    private void Relist()
    {
        if (_offsets.Length != Nlist + 1) _offsets = new int[Nlist + 1];
        else Array.Clear(_offsets);
        int assigned = 0;
        for (int i = 0; i < _n; i++)
        {
            int c = _assign[i];
            if (c >= 0) { _offsets[c + 1]++; assigned++; }
        }
        for (int c = 0; c < Nlist; c++) _offsets[c + 1] += _offsets[c];
        if (_order.Length < assigned) _order = new int[assigned + assigned / 8 + 16];
        var cursor = new int[Nlist];
        Array.Copy(_offsets, cursor, Nlist);
        for (int i = 0; i < _n; i++)
        {
            int c = _assign[i];
            if (c >= 0) _order[cursor[c]++] = i;
        }
    }

    /// <summary>Give every pending row its nearest cell and regroup.</summary>
    public void Fold()
    {
        if (_pending.Count == 0) return;
        var ids = new int[_pending.Count];
        _pending.CopyTo(ids);
        Array.Sort(ids);
        var cells = new int[ids.Length];
        AssignNearest(_matrix, ids, ids.Length, _centroids, Nlist, cells);
        for (int i = 0; i < ids.Length; i++) _assign[ids[i]] = cells[i];
        _pending.Clear();
        Relist();
    }

    // -- nearest-centroid kernel ------------------------------------------------

    private const int BlockRows = 256;
    private const int CentroidTile = 64;

    /// <summary>
    /// For <paramref name="count"/> rows (consecutive from 0, or the rows named
    /// by <paramref name="ids"/>) of <paramref name="data"/>, the index of the
    /// centroid with the largest dot product (the lowest index wins a tie).
    /// <paramref name="output"/> is indexed like the rows, or by the row itself
    /// when <paramref name="ids"/> is null.
    /// </summary>
    private void AssignNearest(float[] data, int[]? ids, int count, float[] cent, int nlist, int[] output)
    {
        if (count == 0) return;
        int stride = _stride;
        int blocks = (count + BlockRows - 1) / BlockRows;
        if (blocks == 1)
        {
            NearestBlock(data, ids, 0, count, cent, nlist, stride, output);
            return;
        }
        Parallel.For(0, blocks, b =>
        {
            int start = b * BlockRows;
            NearestBlock(data, ids, start, Math.Min(BlockRows, count - start), cent, nlist, stride, output);
        });
    }

    private static void NearestBlock(
        float[] data, int[]? ids, int start, int count, float[] cent, int nlist, int stride, int[] output)
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
        for (int i = 0; i < count; i++)
            output[start + i] = bestIdx[i];
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

    /// <summary>Dot product of two equal-length, lane-padded spans.</summary>
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
    /// Score the rows in the <paramref name="lists"/> cells nearest the query,
    /// plus every pending row, into the candidate buffers (rows that pass the
    /// type filter and <paramref name="threshold"/>). Returns false for a zero
    /// query, which has no direction to rank cells by. Allocates nothing per row.
    /// </summary>
    public bool Scan(float[] query, string? nodeType, float threshold, int lists, out int passing)
    {
        passing = 0;
        _candCount = 0;
        query.AsSpan(0, _dim).CopyTo(_q);
        float qnorm = MathF.Sqrt(Dot(_q, _q));
        if (qnorm == 0f) return false;
        int code = -1;
        if (nodeType is not null && !_codeOf.TryGetValue(nodeType, out code))
            return true;                                // no row has this type

        int stride = _stride;
        int nlist = Nlist;
        var centroids = _centroids;
        for (int c = 0; c < nlist; c++)
        {
            _cKey[c] = -Dot(_q, new ReadOnlySpan<float>(centroids, c * stride, stride));
            _cId[c] = c;
        }
        if (lists < nlist)
            MemoryExtensions.Sort(_cKey.AsSpan(0, nlist), _cId.AsSpan(0, nlist));
        else
            lists = nlist;

        var matrix = _matrix;
        var norms = _norms;
        var codes = _codes;
        var assign = _assign;
        var order = _order;
        var offsets = _offsets;
        for (int l = 0; l < lists; l++)
        {
            int cell = _cId[l];
            for (int p = offsets[cell], end = offsets[cell + 1]; p < end; p++)
            {
                int row = order[p];
                if (assign[row] != cell) continue;      // deleted or updated since the last regroup
                if (code >= 0 && codes[row] != code) continue;
                Consider(row, qnorm, threshold);
            }
        }
        foreach (int row in _pending)
        {
            if (code >= 0 && codes[row] != code) continue;
            Consider(row, qnorm, threshold);
        }
        passing = _candCount;
        return true;
    }

    private void Consider(int row, float qnorm, float threshold)
    {
        float dot = Dot(_q, new ReadOnlySpan<float>(_matrix, row * _stride, _stride));
        float denom = _norms[row] * qnorm;
        float score = denom > 0f ? dot / denom : 0f;
        if (!(score >= threshold)) return;
        if (_candCount == _candRows.Length)
        {
            Array.Resize(ref _candRows, _candCount * 2);
            Array.Resize(ref _candScores, _candCount * 2);
        }
        _candRows[_candCount] = row;
        _candScores[_candCount++] = score;
    }

    private static bool Better(float s1, int r1, float s2, int r2) => s1 > s2 || (s1 == s2 && r1 < r2);

    /// <summary>
    /// The best <paramref name="k"/> of the last <see cref="Scan"/>, best first;
    /// equal scores keep row (insertion) order, as in the exact scan.
    /// </summary>
    public List<SimilarityResult> TopK(int k)
    {
        if (_topRows.Length < k)
        {
            _topRows = new int[k];
            _topScores = new float[k];
        }
        var topRows = _topRows;
        var topScores = _topScores;
        int m = 0;
        for (int i = 0; i < _candCount; i++)
        {
            float s = _candScores[i];
            int r = _candRows[i];
            if (m == k && !Better(s, r, topScores[k - 1], topRows[k - 1])) continue;
            int pos = m < k ? m++ : k - 1;
            while (pos > 0 && Better(s, r, topScores[pos - 1], topRows[pos - 1]))
            {
                topScores[pos] = topScores[pos - 1];
                topRows[pos] = topRows[pos - 1];
                pos--;
            }
            topScores[pos] = s;
            topRows[pos] = r;
        }
        var results = new List<SimilarityResult>(m);
        for (int i = 0; i < m; i++)
        {
            var rec = _recs[topRows[i]]!;
            results.Add(new SimilarityResult(rec.NodeRef, topScores[i], rec.Text));
        }
        return results;
    }

    /// <summary>
    /// Every candidate of the last <see cref="Scan"/>, best first, as a
    /// snapshot that stays valid however the index changes afterwards.
    /// </summary>
    public IEnumerable<SimilarityResult> RankedSnapshot()
    {
        int m = _candCount;
        var keys = new ulong[m];
        for (int i = 0; i < m; i++)
        {
            int bits = BitConverter.SingleToInt32Bits(_candScores[i]);
            uint ordered = bits < 0 ? ~(uint)bits : (uint)bits | 0x80000000u;   // ascending = ascending score
            keys[i] = ((ulong)~ordered << 32) | (uint)_candRows[i];               // best score, then lowest row, first
        }
        Array.Sort(keys);
        var recs = new EmbeddingRecord[m];
        var scores = new float[m];
        for (int i = 0; i < m; i++)
        {
            uint ordered = ~(uint)(keys[i] >> 32);
            int bits = (ordered & 0x80000000u) != 0 ? (int)(ordered & 0x7FFFFFFFu) : (int)~ordered;
            scores[i] = BitConverter.Int32BitsToSingle(bits);
            recs[i] = _recs[(int)(keys[i] & 0xFFFFFFFFu)]!;
        }
        return Enumerate(recs, scores);
    }

    private static IEnumerable<SimilarityResult> Enumerate(EmbeddingRecord[] recs, float[] scores)
    {
        for (int i = 0; i < recs.Length; i++)
            yield return new SimilarityResult(recs[i].NodeRef, scores[i], recs[i].Text);
    }
}
