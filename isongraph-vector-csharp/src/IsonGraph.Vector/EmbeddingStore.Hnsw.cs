namespace IsonGraph.Vector;

// Opt-in HNSW search for the in-memory store (AnnMode.Hnsw; DESIGN.md section
// 12), beside the IVF index in EmbeddingStore.Ann.cs. Like IVF it is built
// lazily by the first search that wants it (or BuildAnnIndex) under the
// exclusive lock, read under the shared one, kept current by writes, and never
// persisted.
public sealed partial class EmbeddingStore
{
    private readonly HnswOptions _annHnsw;
    private int _annEfSearch;                 // 0 = automatic
    private HnswIndex? _hnsw;

    /// <summary>Candidates per HNSW query; null means automatic (by store size). Ignored by the exact scan and by IVF.</summary>
    public int? AnnEfSearch
    {
        get { int v = Volatile.Read(ref _annEfSearch); return v == 0 ? null : v; }
        set
        {
            if (value is < 1)
                throw new ArgumentOutOfRangeException(nameof(value), value, "AnnEfSearch must be >= 1");
            Volatile.Write(ref _annEfSearch, value ?? 0);
        }
    }

    /// <summary><c>ef_search</c> by store size when none is configured (DESIGN.md section 12.1).</summary>
    internal static int AutoEf(int live) => live switch
    {
        < 10_000 => 64,
        < 200_000 => 96,
        < 600_000 => 128,
        _ => 192,
    };

    /// <summary>The ef a query uses: the per-call override, else the configured value, else the size-based default; never below k.</summary>
    private int EffectiveEf(int over, int k, int live)
    {
        int ef = over > 0 ? over : Volatile.Read(ref _annEfSearch);
        if (ef <= 0) ef = AutoEf(live);
        return Math.Max(Math.Max(ef, k), 1);
    }

    /// <summary>Scoring more than this many nodes costs more than the exact scan.</summary>
    private static int HnswBudget(int live, int ef) => (int)Math.Min(int.MaxValue, Math.Max(live / 8L, ef * 16L));

    /// <summary>Whether <paramref name="ef"/> at this widening step would reach so much of the store that scanning is as cheap.</summary>
    private static bool HnswExhausted(long ef, long scale, int live) => scale == 1 ? ef >= live : ef * 2 >= live;

    private static bool HasDirection(float qnorm) => qnorm > 0f && float.IsFinite(qnorm);

    /// <summary>
    /// Top-k from the graph, or null to fall back to the exact scan. Caller
    /// holds the shared lock with the graph ready. Widens ef 4x when a filter
    /// leaves fewer than k, and gives up (scan) once ef would cover half the
    /// store or one search scores more nodes than the scan would cost.
    /// </summary>
    private IReadOnlyList<SimilarityResult>? HnswSearch(
        float[] queryVector, int topK, string? nodeType, float threshold, int efOver)
    {
        var h = _hnsw;
        var table = _table;
        if (h is null || table is null) return null;
        float qnorm = QueryNorm(queryVector);
        if (!HasDirection(qnorm)) return null;
        int code = -1;
        if (nodeType is not null)
        {
            code = table.CodeOf(nodeType);
            if (code < 0) return null;                 // no row has this type: the exact scan says so
        }
        int live = table.Live;
        if (topK >= live) return null;                 // everything is wanted: the scan is the answer
        long baseEf = EffectiveEf(efOver, topK, live);
        for (long scale = 1; ; scale *= 4)
        {
            long ef = baseEf * scale;
            if (HnswExhausted(ef, scale, live)) return null;
            var found = h.Search(table, queryVector, qnorm, code, threshold, (int)ef, HnswBudget(live, (int)ef));
            if (found.BudgetHit) return null;
            if (found.Hits.Length >= topK)
            {
                Interlocked.Increment(ref _annApproximate);
                if (scale > 1) Interlocked.Increment(ref _annWidened);
                var hits = found.Hits;
                if (hits.Length > topK) Array.Resize(ref hits, topK);
                return ToResults(table, hits);
            }
        }
    }

    /// <summary>
    /// Every row the widened graph search keeps, best first, or null to fall
    /// back to the exact scan (for the graph's nodeFilter; see
    /// <see cref="RankedForGate"/>). Caller holds the shared lock with the graph ready.
    /// </summary>
    private IEnumerable<SimilarityResult>? HnswRanked(float[] queryVector, string? nodeType, float threshold, long scale)
    {
        var h = _hnsw;
        var table = _table;
        if (h is null || table is null) return null;
        float qnorm = QueryNorm(queryVector);
        if (!HasDirection(qnorm)) return null;
        int code = -1;
        if (nodeType is not null)
        {
            code = table.CodeOf(nodeType);
            if (code < 0) return null;
        }
        int live = table.Live;
        long ef = EffectiveEf(0, 1, live) * scale;
        if (HnswExhausted(ef, scale, live)) return null;
        var found = h.Search(table, queryVector, qnorm, code, threshold, (int)ef, HnswBudget(live, (int)ef));
        if (found.BudgetHit) return null;
        Interlocked.Increment(ref _annApproximate);
        if (scale > 1) Interlocked.Increment(ref _annWidened);
        return Snapshot(table, found.Hits);
    }

    /// <summary>
    /// <see cref="SimilaritySearchVector"/> with <paramref name="efSearch"/> candidates for this call only
    /// (HNSW mode; ignored by the exact scan and by IVF).
    /// </summary>
    public IReadOnlyList<SimilarityResult> SimilaritySearchVectorEf(
        float[] queryVector, int? topK, string? nodeType, float threshold, int efSearch)
    {
        if (efSearch < 1) throw new ArgumentOutOfRangeException(nameof(efSearch), efSearch, "efSearch must be >= 1");
        EnterReadReady(WantsIndex(topK, exact: false));
        try { return SearchCore(queryVector, topK, nodeType, threshold, exact: false, ScanThreads, efSearch); }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>Threads a batch spreads its queries over: the explicit degree, else (HNSW) <see cref="ScanThreads"/> when set, else one per logical CPU.</summary>
    private int BatchDegree(int maxDegreeOfParallelism)
    {
        if (maxDegreeOfParallelism != 0) return maxDegreeOfParallelism;
        int scan = ScanThreads;
        return _ann == AnnMode.Hnsw && scan > 0 ? scan : Environment.ProcessorCount;
    }

    private AnnStats HnswStatsLocked()
    {
        var h = _hnsw;
        int live = _table?.Live ?? 0;
        int configuredEf = Volatile.Read(ref _annEfSearch);
        return new AnnStats(
            Enabled: true, Mode: AnnMode.Hnsw, Active: h is not null,
            Nlist: null, Nprobe: null, TrainedN: h?.Members,
            Pending: 0, Churn: h?.Churn ?? 0, BuildSeconds: h?.BuildSeconds,
            IndexBytes: h?.IndexBytes ?? 0, VectorBytes: 0,
            MinSize: _annMinSize,
            Searches: new AnnSearchCounts(
                Interlocked.Read(ref _annApproximate), Interlocked.Read(ref _annExact),
                Interlocked.Read(ref _annWidened), Interlocked.Read(ref _annBuilds)))
        {
            EfSearch = configuredEf > 0 ? configuredEf : AutoEf(live),
            MaxLevel = h?.MaxLevelInGraph,
            GraphNodes = h?.Members ?? 0,
            GraphDead = h is not null && _table is not null ? h.Dead(_table) : 0,
            Inserts = h?.Inserts ?? 0,
            InsertSeconds = h?.InsertSeconds ?? 0,
        };
    }

    /// <summary>Test hook: the graph's structure digest under the shared lock (null while none is built).</summary>
    internal ulong? HnswStructureHash()
    {
        _rw.EnterReadLock();
        try { return _hnsw?.StructureHash(); }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>Test hook: run <paramref name="inspect"/> on the graph under the shared lock.</summary>
    internal T? InspectHnsw<T>(Func<HnswIndex, VectorTable, T> inspect)
    {
        _rw.EnterReadLock();
        try { return _hnsw is not null && _table is not null ? inspect(_hnsw, _table) : default; }
        finally { _rw.ExitReadLock(); }
    }
}
