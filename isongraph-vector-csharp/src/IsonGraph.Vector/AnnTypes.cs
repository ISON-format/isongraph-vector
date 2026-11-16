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
/// <param name="Nlist">Cells in the index (null while none).</param>
/// <param name="Nprobe">Cells scanned per query at present (the configured value, or the automatic one).</param>
/// <param name="TrainedN">Vectors the centroids were trained on (null while none).</param>
/// <param name="Pending">Rows scored exactly until folded into a cell.</param>
/// <param name="Churn">Writes plus deletes since training.</param>
/// <param name="BuildSeconds">Time the last training took (null while none).</param>
/// <param name="IndexBytes">Memory held by centroids and cell lists, on top of <paramref name="VectorBytes"/>.</param>
/// <param name="VectorBytes">Memory held by the index's own contiguous copy of the vectors.</param>
/// <param name="MinSize">Below this many vectors search stays exact.</param>
/// <param name="Searches">Per-route search counts.</param>
public sealed record AnnStats(
    bool Enabled, AnnMode Mode, bool Active, int? Nlist, int? Nprobe, int? TrainedN,
    int Pending, long Churn, double? BuildSeconds, long IndexBytes, long VectorBytes,
    int MinSize, AnnSearchCounts Searches);

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
