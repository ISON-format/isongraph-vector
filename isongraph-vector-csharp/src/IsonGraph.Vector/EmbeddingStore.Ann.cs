namespace IsonGraph.Vector;

// Opt-in approximate (IVF) search for the in-memory store. The exact scan in
// IsonGraphVector.cs is untouched; everything here runs under the same
// coarse lock as every other store operation.
public sealed partial class EmbeddingStore
{
    private const int AnnFoldMin = 512;              // fold pending rows into cells at >= this many ...
    private const double AnnFoldFraction = 0.02;     // ... and at least this fraction of the indexed rows
    private const double AnnRetrainFraction = 0.5;   // retrain once writes+deletes exceed this x trained size

    private static int AnnAutoNlist(int n) =>
        (int)Math.Min(Math.Max(Math.Round(4 * Math.Sqrt(n)), 16), 65536);

    /// <summary>
    /// The index, built or maintained as the policy says; null means "answer
    /// exactly". Caller holds the lock.
    ///
    /// Policy: built lazily by the first qualifying search (never persisted),
    /// k-means on a sample of up to 40 x nlist vectors, then every row
    /// assigned. New and updated rows are scored exactly until at least
    /// max(512, 2%) of the indexed rows are pending, then folded into their
    /// nearest cell. Deletes just drop the row. Once writes + deletes since
    /// training exceed 50% of the trained size the index is retrained from the
    /// current data (with an automatic nlist, re-chosen for the new size). A
    /// store that shrinks below the minimum size, or is cleared, drops it.
    /// </summary>
    private IvfIndex? AnnIndexForSearch()
    {
        if (_ann == AnnMode.None) return null;
        int live = _embeddings.Count;
        if (live < Math.Max(2, _annMinSize))
        {
            _ivf = null;
            return null;
        }
        if (_ivf is null)
        {
            _ivf = IvfIndex.FromRecords(_embeddings.Values, _dimension!.Value, live);
            TrainIndex(_ivf);
        }
        else if (_ivf.Churn > AnnRetrainFraction * _ivf.TrainedN)
        {
            TrainIndex(_ivf);
        }
        else if (_ivf.PendingCount >= Math.Max(AnnFoldMin, AnnFoldFraction * _ivf.TrainedN))
        {
            _ivf.Fold();
        }
        return _ivf;
    }

    private void TrainIndex(IvfIndex ivf)
    {
        int nlist = _annNlist ?? AnnAutoNlist(ivf.LiveCount);
        ivf.Train(nlist, _annSeed);
        _annBuilds++;
    }

    private int AnnProbe(IvfIndex ivf)
    {
        int wanted = _annNprobe ?? Math.Max(1, (int)Math.Round(0.05 * ivf.Nlist));
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
    /// holds the lock. Filters apply to the probed rows; if fewer than topK
    /// survive, the probe is widened 4x and tried again, ending in the exact
    /// scan, so a rare type or a high threshold costs time rather than
    /// silently returning a short list.
    /// </summary>
    private IReadOnlyList<SimilarityResult>? AnnSearch(
        float[] queryVector, int topK, string? nodeType, float threshold)
    {
        var ivf = AnnIndexForSearch();
        if (ivf is null) return null;
        for (long scale = 1; ; scale *= 4)
        {
            var lists = AnnLists(ivf, scale);
            if (lists is null) return null;
            if (!ivf.Scan(queryVector, nodeType, threshold, lists.Value, out int passing))
                return null;                       // a zero query has no direction to rank cells by
            if (passing >= topK)
            {
                _annApproximate++;
                if (scale > 1) _annWidened++;
                return ivf.TopK(topK);
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
        lock (_lock)
        {
            CheckQuery(queryVector);
            var ivf = AnnIndexForSearch();
            var lists = ivf is null ? null : AnnLists(ivf, scale);
            if (ivf is not null && lists is not null &&
                ivf.Scan(queryVector, nodeType, threshold, lists.Value, out _))
            {
                _annApproximate++;
                if (scale > 1) _annWidened++;
                return (ivf.RankedSnapshot(), true);
            }
            return (SimilaritySearchVector(queryVector, null, nodeType, threshold, exact: true), false);
        }
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
        lock (_lock)
        {
            if (force) _ivf = null;
            AnnIndexForSearch()?.Fold();
            return GetAnnStats();
        }
    }

    /// <summary>What the approximate index looks like right now. Never builds anything.</summary>
    public AnnStats GetAnnStats()
    {
        lock (_lock)
        {
            var ivf = _ivf;
            return new AnnStats(
                Enabled: _ann != AnnMode.None,
                Mode: _ann,
                Active: ivf is not null,
                Nlist: ivf?.Nlist,
                Nprobe: ivf is not null ? AnnProbe(ivf) : _annNprobe,
                TrainedN: ivf?.TrainedN,
                Pending: ivf?.PendingCount ?? 0,
                Churn: ivf?.Churn ?? 0,
                BuildSeconds: ivf?.BuildSeconds,
                IndexBytes: ivf?.IndexBytes ?? 0,
                VectorBytes: ivf?.VectorBytes ?? 0,
                MinSize: _annMinSize,
                Searches: new AnnSearchCounts(_annApproximate, _annExact, _annWidened, _annBuilds));
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
                var wanted = new HashSet<NodeRef>(truth.Select(r => r.NodeRef));
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
