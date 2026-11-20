namespace IsonGraph.Vector;

/// <summary>Approximate nearest-neighbour index a store may keep (see <see cref="EmbeddingStore"/>).</summary>
public enum AnnMode
{
    /// <summary>No index: every search is the exact scan (default).</summary>
    None,
    /// <summary>
    /// In-memory inverted-file ("IVF-flat") index: vectors are clustered by
    /// direction and only the cells nearest the query are scored.
    /// </summary>
    Ivf,
    /// <summary>
    /// In-memory HNSW graph (hierarchical navigable small world) over the stored
    /// vectors, which it does not copy: it holds links only. Answers a bounded
    /// top-k about 4-10x faster than IVF at the same recall on text embeddings,
    /// at the price of a slower build; see <see cref="HnswOptions"/>.
    /// </summary>
    Hnsw,
}

/// <summary>
/// Settings for <see cref="AnnMode.Hnsw"/> (DESIGN.md section 12.1); ignored in
/// any other mode.
/// </summary>
/// <param name="M">
/// Links per node on layers above 0 (layer 0 holds up to 2 x M). More is better
/// recall and more memory. 2 to 64; default 16.
/// </param>
/// <param name="EfConstruction">
/// Candidates kept while linking a new row. More is a better graph and a slower
/// build and write. Default 200 (never less than <paramref name="M"/>).
/// </param>
/// <param name="EfSearch">
/// Candidates kept per query (never less than the topK asked for). More is higher
/// recall and slower. Null picks by store size, aiming at about 0.95 recall@10 on
/// 384-d text embeddings: 64 under 10,000 rows, 96 under 200,000, 128 under
/// 600,000, 192 above. Changeable later via <see cref="EmbeddingStore.AnnEfSearch"/>.
/// </param>
public sealed record HnswOptions(int M = 16, int EfConstruction = 200, int? EfSearch = null)
{
    internal void Validate()
    {
        if (M is < 2 or > 64)
            throw new ArgumentOutOfRangeException(nameof(M), M, "HNSW M must be between 2 and 64");
        if (EfConstruction < 1)
            throw new ArgumentOutOfRangeException(nameof(EfConstruction), EfConstruction, "HNSW EfConstruction must be at least 1");
        if (EfSearch is < 1)
            throw new ArgumentOutOfRangeException(nameof(EfSearch), EfSearch, "HNSW EfSearch must be at least 1");
    }
}

/// <summary>How many searches each route answered, since the store was created.</summary>
/// <param name="Approximate">Answered from the index (including widened probes).</param>
/// <param name="Exact">Answered by the full scan.</param>
/// <param name="Widened">Answered from the index only after widening the probe.</param>
/// <param name="Builds">Times the index was trained (first build and retrains).</param>
public sealed record AnnSearchCounts(long Approximate, long Exact, long Widened, long Builds);

/// <summary>
/// What the approximate index looks like right now; see
/// <see cref="EmbeddingStore.GetAnnStats"/>.
/// </summary>
/// <param name="Enabled">The store was created with an index mode.</param>
/// <param name="Mode">The mode.</param>
/// <param name="Active">An index exists and will answer searches.</param>
/// <param name="Nlist">Cells in the index (null while none, and in HNSW mode).</param>
/// <param name="Nprobe">Cells scanned per query at present (the configured value, or the automatic one; null in HNSW mode).</param>
/// <param name="TrainedN">Vectors the centroids were trained on, or the nodes of the HNSW graph (null while none).</param>
/// <param name="Pending">Rows scored exactly until folded into a cell (always 0 in HNSW mode).</param>
/// <param name="Churn">Writes plus deletes since training (HNSW: updates plus deletes since the build).</param>
/// <param name="BuildSeconds">Time the last training or graph build took (null while none).</param>
/// <param name="IndexBytes">Memory held by centroids and cell lists (HNSW: the graph's links), on top of <paramref name="VectorBytes"/>.</param>
/// <param name="VectorBytes">Memory held by the index's own contiguous copy of the vectors.</param>
/// <param name="MinSize">Below this many vectors search stays exact.</param>
/// <param name="Searches">Per-route search counts.</param>
public sealed record AnnStats(
    bool Enabled, AnnMode Mode, bool Active, int? Nlist, int? Nprobe, int? TrainedN,
    int Pending, long Churn, double? BuildSeconds, long IndexBytes, long VectorBytes,
    int MinSize, AnnSearchCounts Searches)
{
    /// <summary>HNSW mode: candidates per query at present (the configured value, or the automatic one for this store size).</summary>
    public int? EfSearch { get; init; }
    /// <summary>HNSW mode: the highest layer of the graph (null while none).</summary>
    public int? MaxLevel { get; init; }
    /// <summary>HNSW mode: nodes in the graph, tombstones included.</summary>
    public int GraphNodes { get; init; }
    /// <summary>HNSW mode: nodes whose row is deleted and which stay until compaction.</summary>
    public int GraphDead { get; init; }
    /// <summary>HNSW mode: rows linked by writes since the build.</summary>
    public long Inserts { get; init; }
    /// <summary>HNSW mode: total time those writes spent linking.</summary>
    public double InsertSeconds { get; init; }
}

/// <summary>
/// Recall of the approximate index against the exact scan, over some queries.
/// </summary>
/// <param name="Recall">
/// Mean over queries of |approximate top-k ∩ exact top-k| / |exact top-k|
/// (1.0 = identical sets).
/// </param>
/// <param name="MinRecall">The worst single query.</param>
/// <param name="Queries">Queries run.</param>
/// <param name="K">Results compared per query.</param>
/// <param name="Nlist">Cells in the index (0 when none).</param>
/// <param name="Nprobe">Cells scanned per query.</param>
/// <param name="Approximate">
/// False when the index was not actually consulted (it is off, the store is
/// smaller than the minimum, or nprobe &gt;= nlist); recall is then 1.0 by
/// construction.
/// </param>
/// <param name="AnnMs">Median per-query latency of the indexed search, milliseconds.</param>
/// <param name="ExactMs">Median per-query latency of the exact scan, milliseconds.</param>
public sealed record RecallReport(
    double Recall, double MinRecall, int Queries, int K, int Nlist, int Nprobe,
    bool Approximate, double AnnMs, double ExactMs);
