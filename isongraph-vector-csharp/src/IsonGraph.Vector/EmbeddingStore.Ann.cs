namespace IsonGraph.Vector;

// Opt-in approximate (IVF) search for the in-memory store. The exact scan in
// EmbeddingStore.cs is untouched; the index is built and retrained under the
// store's exclusive lock and read under the shared one.
public sealed partial class EmbeddingStore
{
    private const double AnnRetrainFraction = 0.5;   // retrain once writes+deletes exceed this x trained size

    private static int AnnAutoNlist(int n) =>
        (int)Math.Min(Math.Max(Math.Round(4 * Math.Sqrt(n)), 16), 65536);

    // Caller holds a lock.
    private bool AnnBelowMin() => (_table?.Live ?? 0) < Math.Max(2, _annMinSize);

    /// <summary>
    /// Whether a search wanting the index can proceed under the shared lock: the
    /// index exists and is current, or the store is under the minimum size and
    /// there is none (so the answer is exact). Caller holds a lock.
    /// </summary>
    private bool IndexReady()
    {
        if (_ann == AnnMode.None) return true;
        if (AnnBelowMin()) return _ivf is null;
        return _ivf is not null && !_ivf.Stale;
    }

    /// <summary>
    /// Build, retrain or drop the index as the policy says. Caller holds the
    /// exclusive lock.
    ///
    /// Policy: built lazily by the first qualifying search (never persisted),
    /// k-means on a sample of up to 40 x nlist vectors, then every row
    /// assigned. New and updated rows are scored exactly until 256 are
    /// pending, when the write that completes them folds them into their
    /// nearest cell. Deletes detach the row. Once writes + deletes since
    /// training exceed 50% of the trained size the index is retrained from the
    /// current data (with an automatic nlist, re-chosen for the new size). A
    /// store that shrinks below the minimum size, or is cleared, drops it.
    /// </summary>
    private void EnsureIndex()
    {
        if (_ann == AnnMode.None) return;
        if (AnnBelowMin())
        {
            _ivf = null;
            return;
        }
        if (_ivf is null || _ivf.Stale)
        {
            var table = _table!;
            int nlist = _annNlist ?? AnnAutoNlist(table.Live);
            _ivf = IvfIndex.Build(table, nlist, _annSeed);
            Interlocked.Increment(ref _annBuilds);
        }
    }

    /// <summary>
    /// Take the shared lock with the index ready if <paramref name="wantIndex"/>:
    /// when it has to be built or retrained, the exclusive lock is taken for
    /// that (searches wait) and the shared lock is then re-acquired. The caller
    /// releases with <c>_rw.ExitReadLock()</c>.
    /// </summary>
    private void EnterReadReady(bool wantIndex)
    {
        for (;;)
        {
            _rw.EnterReadLock();
            if (!wantIndex || IndexReady()) return;
            _rw.ExitReadLock();
            _rw.EnterWriteLock();
            try { EnsureIndex(); }
            finally { _rw.ExitWriteLock(); }
        }
    }

    private int AnnProbe(IvfIndex ivf)
    {
        int configured = Volatile.Read(ref _annNprobe);
        int wanted = configured != 0 ? configured : Math.Max(1, (int)Math.Round(0.05 * ivf.Nlist));
        return Math.Min(wanted, ivf.Nlist);
    }

    /// <summary>
    /// Cells to probe at widening <paramref name="scale"/>; null = widened
    /// enough to just scan. Scale 1 honours nprobe as configured (nprobe &gt;=
    /// nlist is the exact scan). Later steps multiply it by 4 and give up on
    /// the index once they would touch half the cells, where scanning
    /// everything costs about the same.
    /// </summary>
    private int? AnnLists(IvfIndex ivf, long scale)
    {
        long lists = AnnProbe(ivf) * scale;
        if (scale == 1) return lists >= ivf.Nlist ? null : (int)lists;
        return lists * 2 >= ivf.Nlist ? null : (int)lists;
    }

    /// <summary>
    /// Top-k from the index, or null to fall back to the exact scan. Caller
    /// holds the shared lock with the index ready. Filters apply to the probed
    /// rows; if fewer than topK survive, the probe is widened 4x and tried
    /// again, ending in the exact scan, so a rare type or a high threshold
    /// costs time rather than silently returning a short list.
    /// </summary>
    private IReadOnlyList<SimilarityResult>? AnnSearch(
        float[] queryVector, int topK, string? nodeType, float threshold)
    {
        var ivf = _ivf;
        var table = _table;
        if (ivf is null || table is null) return null;
        float qnorm = QueryNorm(queryVector);
        if (qnorm == 0f) return null;                  // a zero query has no direction to rank cells by
        int code = -1;
        if (nodeType is not null)
        {
            code = table.CodeOf(nodeType);
            if (code < 0) return null;                 // no row has this type: the exact scan says so
        }
        int k = Math.Min(topK, table.Live);
        for (long scale = 1; ; scale *= 4)
        {
            var lists = AnnLists(ivf, scale);
            if (lists is null) return null;
            var heap = new TopKHeap(k);
            var visitor = new TopKVisitor { Heap = heap, Norms = table.Norms, QueryNorm = qnorm, Threshold = threshold };
            ivf.Scan(table, queryVector, code, lists.Value, ref visitor);
            if (heap.Count >= topK)
            {
                Interlocked.Increment(ref _annApproximate);
                if (scale > 1) Interlocked.Increment(ref _annWidened);
                return ToResults(table, heap.ToSortedArray());
            }
        }
    }

    /// <summary>
    /// Every match in the probed cells, best first, for callers that apply a
    /// predicate before taking k (the graph's nodeFilter). <c>Approximate</c>
    /// is false when the answer is exact (the full scan) and there is nothing
    /// left to widen to. <paramref name="scale"/> is the widening step
    /// (1, 4, 16, ...). The sequence is a snapshot: it may be enumerated after
    /// the lock is released, with the store changing underneath.
    /// </summary>
    internal (IEnumerable<SimilarityResult> Ranked, bool Approximate) RankedForGate(
        float[] queryVector, string? nodeType, float threshold, long scale)
    {
        EnterReadReady(_ann != AnnMode.None);
        try
        {
            CheckQuery(queryVector);
            var ivf = _ivf;
            var table = _table;
            if (ivf is not null && table is not null)
            {
                var lists = AnnLists(ivf, scale);
                float qnorm = QueryNorm(queryVector);
                int code = nodeType is null ? -1 : table.CodeOf(nodeType);
                if (lists is not null && qnorm != 0f && !(nodeType is not null && code < 0))
                {
                    var found = new HitList();
                    var visitor = new CollectVisitor { List = found, Norms = table.Norms, QueryNorm = qnorm, Threshold = threshold };
                    ivf.Scan(table, queryVector, code, lists.Value, ref visitor);
                    Interlocked.Increment(ref _annApproximate);
                    if (scale > 1) Interlocked.Increment(ref _annWidened);
                    return (Snapshot(table, found.ToSortedArray()), true);
                }
            }
            Interlocked.Increment(ref _annExact);
            return (ExactSearch(queryVector, null, nodeType, threshold, ScanThreads), false);
        }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>The hits as a lazily built sequence that no longer touches the table.</summary>
    private static IEnumerable<SimilarityResult> Snapshot(VectorTable t, Hit[] hits)
    {
        var refs = new IsonGraph.NodeRef[hits.Length];
        var texts = new string[hits.Length];
        var scores = new float[hits.Length];
        for (int i = 0; i < hits.Length; i++)
        {
            refs[i] = t.RefAt(hits[i].Slot);
            texts[i] = t.TextAt(hits[i].Slot);
            scores[i] = hits[i].Score;
        }
        return Enumerate(refs, texts, scores);
    }

    private static IEnumerable<SimilarityResult> Enumerate(IsonGraph.NodeRef[] refs, string[] texts, float[] scores)
    {
        for (int i = 0; i < refs.Length; i++)
            yield return new SimilarityResult(refs[i], scores[i], texts[i]);
    }

    /// <summary>Resolve a text query to its vector (encoder, model check and dimension check).</summary>
    internal float[] EncodeQuery(string query) => ResolveQuery(query);

    /// <summary>
    /// Build the approximate index now instead of on the first search. Returns
    /// <see cref="GetAnnStats"/>. <paramref name="force"/> retrains even if an
    /// index exists. Does nothing (and says so in the stats) while the store is
    /// smaller than the minimum size.
    /// </summary>
    /// <exception cref="EmbeddingException">The store was created without an index mode.</exception>
    public AnnStats BuildAnnIndex(bool force = false)
    {
        if (_ann == AnnMode.None)
            throw new EmbeddingException(
                "this store has no approximate index; create it with ann: AnnMode.Ivf");
        _rw.EnterWriteLock();
        try
        {
            if (force) _ivf = null;
            EnsureIndex();
            if (_ivf is not null && _table is not null) _ivf.Fold(_table);
            return StatsLocked();
        }
        finally { _rw.ExitWriteLock(); }
    }

    /// <summary>What the approximate index looks like right now. Never builds anything.</summary>
    public AnnStats GetAnnStats()
    {
        _rw.EnterReadLock();
        try { return StatsLocked(); }
        finally { _rw.ExitReadLock(); }
    }

    private AnnStats StatsLocked()
    {
        var ivf = _ivf;
        int configured = Volatile.Read(ref _annNprobe);
        return new AnnStats(
            Enabled: _ann != AnnMode.None,
            Mode: _ann,
            Active: ivf is not null,
            Nlist: ivf?.Nlist,
            Nprobe: ivf is not null ? AnnProbe(ivf) : (configured == 0 ? null : configured),
            TrainedN: ivf?.TrainedN,
            Pending: ivf?.PendingCount ?? 0,
            Churn: ivf?.Churn ?? 0,
            BuildSeconds: ivf?.BuildSeconds,
            IndexBytes: ivf?.IndexBytes ?? 0,
            VectorBytes: ivf?.VectorBytes ?? 0,
            MinSize: _annMinSize,
            Searches: new AnnSearchCounts(
                Interlocked.Read(ref _annApproximate), Interlocked.Read(ref _annExact),
                Interlocked.Read(ref _annWidened), Interlocked.Read(ref _annBuilds)));
    }

    /// <summary>
    /// Check the store's invariants (DESIGN.md section 7: table, model counts,
    /// index lists and positions). Returns null when they all hold, else a
    /// description of the first one that does not. For tests and debugging;
    /// O(n).
    /// </summary>
    internal string? VerifyInvariants()
    {
        _rw.EnterReadLock();
        try
        {
            var t = _table;
            if (t is null) return _ivf is null ? null : "index without a table";
            return t.Verify() ?? _ivf?.Verify(t);
        }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>Bytes held by the vector matrix chunks (0 while empty); what the vectors cost, one copy.</summary>
    internal long MatrixBytes
    {
        get
        {
            _rw.EnterReadLock();
            try { return _table?.MatrixBytes ?? 0; }
            finally { _rw.ExitReadLock(); }
        }
    }

    /// <summary>Recall@k for text queries; see the vector overload.</summary>
    public RecallReport RecallCheck(
        IEnumerable<string> queries, int k = 10, string? nodeType = null, float threshold = 0.0f) =>
        RecallCheck(queries.Select(ResolveQuery).ToList(), k, nodeType, threshold);

    /// <summary>Recall@k for a single text query.</summary>
    public RecallReport RecallCheck(string query, int k = 10, string? nodeType = null, float threshold = 0.0f) =>
        RecallCheck(new[] { ResolveQuery(query) }, k, nodeType, threshold);

    /// <summary>Recall@k for a single query vector.</summary>
    public RecallReport RecallCheck(float[] query, int k = 10, string? nodeType = null, float threshold = 0.0f) =>
        RecallCheck(new[] { query }, k, nodeType, threshold);

    /// <summary>
    /// Recall@k of the approximate index against the exact scan.
    ///
    /// Runs every query both ways and compares the sets of top-k nodes. Use
    /// real queries drawn from your own workload: recall depends on how the data
    /// clusters, and a score measured on unrelated queries or data does not
    /// transfer. Raise <see cref="AnnNprobe"/> until the recall you need holds.
    /// Latencies are per-query medians of the search itself (a text query's
    /// encoding is not timed).
    ///
    /// Without an index in use (none requested, a store under the minimum size,
    /// or nprobe &gt;= nlist) both answers are the exact scan, recall is 1.0 and
    /// <see cref="RecallReport.Approximate"/> is false.
    /// </summary>
    public RecallReport RecallCheck(
        IEnumerable<float[]> queries, int k = 10, string? nodeType = null, float threshold = 0.0f)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), k, "k must be positive");
        var batch = queries.ToList();
        var recalls = new List<double>();
        var annTimes = new List<double>();
        var exactTimes = new List<double>();
        bool used = false;
        foreach (var query in batch)
        {
            long before = GetAnnStats().Searches.Approximate;
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var found = SimilaritySearchVector(query, k, nodeType, threshold);
            annTimes.Add(System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            used |= GetAnnStats().Searches.Approximate > before;

            t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var truth = SimilaritySearchVector(query, k, nodeType, threshold, exact: true);
            exactTimes.Add(System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
            if (truth.Count > 0)
            {
                var wanted = new HashSet<IsonGraph.NodeRef>(truth.Select(r => r.NodeRef));
                int hit = found.Count(r => wanted.Contains(r.NodeRef));
                recalls.Add((double)hit / wanted.Count);
            }
        }
        var stats = GetAnnStats();
        return new RecallReport(
            Recall: recalls.Count > 0 ? recalls.Average() : 1.0,
            MinRecall: recalls.Count > 0 ? recalls.Min() : 1.0,
            Queries: batch.Count, K: k,
            Nlist: stats.Nlist ?? 0, Nprobe: stats.Nprobe ?? 0,
            Approximate: used, AnnMs: Median(annTimes), ExactMs: Median(exactTimes));
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0.0;
        var ordered = values.OrderBy(v => v).ToList();
        int mid = ordered.Count / 2;
        return ordered.Count % 2 == 1 ? ordered[mid] : 0.5 * (ordered[mid - 1] + ordered[mid]);
    }
}
