using System.Globalization;
using System.Text.Json;
using IsonGraph;

namespace IsonGraph.Vector;

// =============================================================================
// Embedding Store
// =============================================================================

/// <summary>
/// In-memory embedding storage with cosine similarity search.
///
/// <b>Storage.</b> Every vector is kept once, in a chunked contiguous matrix
/// (1024 rows per chunk); a record is a row number plus its key, text and model
/// (see DESIGN.md section 1). Nothing else owns the floats: the exact scan, the
/// approximate index and the save paths all read the same rows.
///
/// <b>Concurrency contract.</b> Searches (<c>SimilaritySearch*</c>,
/// <see cref="ScoreMap"/>, <see cref="Get"/>, <see cref="Count"/>, exports and
/// saves) take a shared lock and touch no shared mutable state, so any number
/// run at the same time and a batch sees one state for all its queries.
/// Writes (<c>Add*</c>, <see cref="Remove"/>, <see cref="Clear"/>, imports,
/// <see cref="BuildAnnIndex"/>) take the exclusive lock: a search sees the store
/// before a write or after it, never in between. The one thing a search can do
/// besides read is build or retrain the approximate index, which it does by
/// taking the exclusive lock first (so concurrent searches wait for the
/// build). A write that makes 256 rows pending folds them into the index while
/// it holds the exclusive lock (about 1-3 ms at 1M x 384).
/// </summary>
public sealed partial class EmbeddingStore
{
    private readonly ReaderWriterLockSlim _rw = new(LockRecursionPolicy.NoRecursion);
    private VectorTable? _table;
    private IEmbeddingEncoder? _encoder;
    private int _dimension = -1;              // pinned vector length, -1 while none
    private string? _version;
    private ModelMismatchMode _onModelMismatch;
    private readonly HashSet<string> _warnedMismatch = new();
    private volatile IReadOnlyDictionary<string, string> _lastImportMeta = new Dictionary<string, string>();
    private int _scanThreads;

    // Opt-in approximate index (see AnnMode). _ivf is built lazily by the first
    // search that qualifies, kept current by Add/Remove/Clear, and never saved.
    private readonly AnnMode _ann;
    private readonly int? _annNlist;
    private int _annNprobe;                   // 0 = automatic
    private readonly int _annMinSize;
    private readonly int _annSeed;
    private IvfIndex? _ivf;
    private long _annApproximate, _annExact, _annWidened, _annBuilds;

    /// <summary>Below this many vectors the index is not used by default.</summary>
    public const int DefaultAnnMinSize = 50_000;

    /// <param name="encoder">Encoder used when a text is added or queried without a vector.</param>
    /// <param name="version">
    /// Free-form version of the encoder/model that made these vectors,
    /// recorded in the sidecar. Falls back to the encoder's own Version.
    /// </param>
    /// <param name="onModelMismatch">
    /// What a text query does when the encoder's ModelName differs from the
    /// model that produced the stored vectors. Vectors whose model is
    /// "unknown"/"provided" never trigger it, and vector queries are never
    /// checked.
    /// </param>
    /// <param name="ann">
    /// <see cref="AnnMode.None"/> (default) keeps search exact, exactly as it
    /// always was. <see cref="AnnMode.Ivf"/> answers bounded top-k searches
    /// from an in-memory IVF-flat index: vectors are clustered into
    /// <paramref name="annNlist"/> cells and only the <paramref name="annNprobe"/>
    /// cells nearest the query are scored. It is approximate - a true neighbour
    /// in an unprobed cell is missed - so measure it with
    /// <see cref="RecallCheck(IEnumerable{float[]}, int, string?, float)"/>. The
    /// index is built lazily by the first search, never persisted, and kept
    /// current through writes and deletes. It holds no copy of the vectors
    /// (only centroids and a cell number per row).
    /// </param>
    /// <param name="annNlist">Cells in the index. Null picks about 4 * sqrt(N) when it is built.</param>
    /// <param name="annNprobe">
    /// Cells scanned per query. Null picks about 5% of nlist. More cells = higher
    /// recall, slower; nprobe &gt;= nlist is the exact scan. Adjustable later via
    /// <see cref="AnnNprobe"/>.
    /// </param>
    /// <param name="annMinSize">Below this many vectors search stays exact. Default 50,000.</param>
    /// <param name="annSeed">Seed for the k-means initialisation, so the same data gives the same index.</param>
    public EmbeddingStore(
        IEmbeddingEncoder? encoder = null,
        string? version = null,
        ModelMismatchMode onModelMismatch = ModelMismatchMode.Warn,
        AnnMode ann = AnnMode.None,
        int? annNlist = null,
        int? annNprobe = null,
        int annMinSize = DefaultAnnMinSize,
        int annSeed = 0)
    {
        CheckMode(onModelMismatch);
        if (!Enum.IsDefined(ann))
            throw new ArgumentOutOfRangeException(nameof(ann), ann, "unknown AnnMode");
        if (annNlist is < 1)
            throw new ArgumentOutOfRangeException(nameof(annNlist), annNlist, "annNlist must be >= 1");
        if (annNprobe is < 1)
            throw new ArgumentOutOfRangeException(nameof(annNprobe), annNprobe, "annNprobe must be >= 1");
        if (annMinSize < 0)
            throw new ArgumentOutOfRangeException(nameof(annMinSize), annMinSize, "annMinSize must be >= 0");
        _encoder = encoder;
        _version = version;
        _onModelMismatch = onModelMismatch;
        _ann = ann;
        _annNlist = annNlist;
        _annNprobe = annNprobe ?? 0;
        _annMinSize = annMinSize;
        _annSeed = annSeed;
    }

    /// <summary>The approximate index in use, or <see cref="AnnMode.None"/> for exact search.</summary>
    public AnnMode Ann => _ann;

    /// <summary>Cells scanned per query; null means automatic (about 5% of nlist).</summary>
    public int? AnnNprobe
    {
        get { int v = Volatile.Read(ref _annNprobe); return v == 0 ? null : v; }
        set
        {
            if (value is < 1)
                throw new ArgumentOutOfRangeException(nameof(value), value, "AnnNprobe must be >= 1");
            Volatile.Write(ref _annNprobe, value ?? 0);
        }
    }

    /// <summary>
    /// Threads one exact scan may use: 0 (default) picks automatically (up to 8,
    /// fewer for small stores), 1 never runs a scan in parallel. The answer is
    /// the same for any value; only the time changes.
    /// </summary>
    public int ScanThreads
    {
        get => Volatile.Read(ref _scanThreads);
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), value, "ScanThreads must be >= 0");
            Volatile.Write(ref _scanThreads, value);
        }
    }

    private static void CheckMode(ModelMismatchMode mode)
    {
        if (!Enum.IsDefined(mode))
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "unknown ModelMismatchMode");
    }

    /// <summary>
    /// Raised with a human-readable message when the store has something to
    /// warn about (a model mismatch, a version mismatch on load). With no
    /// subscriber the message goes to standard error.
    /// </summary>
    public event Action<string>? Warning;

    internal void RaiseWarning(string message)
    {
        var handler = Warning;
        if (handler is not null) handler(message);
        else Console.Error.WriteLine("isongraph-vector: warning: " + message);
    }

    public IEmbeddingEncoder? Encoder
    {
        get => _encoder;
        set => _encoder = value;
    }

    /// <summary>
    /// Encoder/model version recorded in the sidecar: the one set explicitly,
    /// else the encoder's own, else null.
    /// </summary>
    public string? Version
    {
        get => _version ?? _encoder?.Version;
        set => _version = value;
    }

    /// <summary>What a text query does when the encoder's model differs from the stored vectors'.</summary>
    public ModelMismatchMode OnModelMismatch
    {
        get => _onModelMismatch;
        set { CheckMode(value); _onModelMismatch = value; }
    }

    /// <summary>
    /// Metadata (encoder_model, encoder_version, graph_digest,
    /// graph_digest_basis) of the most recently imported payload; only the
    /// keys it carried are present, and it is empty before any import.
    /// </summary>
    public IReadOnlyDictionary<string, string> LastImportMeta => _lastImportMeta;

    /// <summary>Vector length this store holds, or null while it is empty.</summary>
    public int? Dimension
    {
        get
        {
            _rw.EnterReadLock();
            try { return _dimension < 0 ? null : _dimension; }
            finally { _rw.ExitReadLock(); }
        }
    }

    private static string Key(NodeRef nodeRef) => $"{nodeRef.Type}:{nodeRef.Id}";

    /// <summary>
    /// Keep the best <paramref name="k"/> results without ordering all of them.
    ///
    /// Sorting every candidate to return ten is O(n log n) where this is
    /// O(n·k) with early rejection — for the small k a search actually asks
    /// for, that is a large saving over 20,000 candidates. Insertion goes
    /// after equal scores, so the order matches a stable descending sort.
    /// </summary>
    internal static List<SimilarityResult> SelectTopK(List<SimilarityResult> results, int k)
    {
        if (k <= 0) return new List<SimilarityResult>();
        if (results.Count <= k)
            return results.OrderByDescending(r => r.Score).ToList();

        var top = new List<SimilarityResult>(k);
        foreach (var candidate in results)
        {
            if (top.Count == k && candidate.Score <= top[k - 1].Score) continue;

            int i = top.Count < k ? top.Count : k - 1;
            if (top.Count < k) top.Add(candidate);
            while (i > 0 && top[i - 1].Score < candidate.Score)
            {
                top[i] = top[i - 1];
                i--;
            }
            top[i] = candidate;
        }
        return top;
    }

    // Caller holds a lock.
    private void CheckDimension(ReadOnlySpan<float> vector, string what)
    {
        if (vector.Length == 0)
            throw new DimensionMismatchException($"{what} is empty");
        if (_dimension < 0)
            _dimension = vector.Length;
        else if (vector.Length != _dimension)
            throw new DimensionMismatchException(
                $"{what} has {vector.Length} dimensions, store holds {_dimension}");
    }

    // Caller holds a lock.
    private void CheckQuery(ReadOnlySpan<float> vector)
    {
        if (vector.Length == 0)
            throw new DimensionMismatchException("query vector is empty");
        if (_dimension >= 0 && vector.Length != _dimension)
            throw new DimensionMismatchException(
                $"query vector has {vector.Length} dimensions, store holds {_dimension}");
    }

    private float[] ResolveQuery(string query)
    {
        var encoder = _encoder ?? throw new NoEncoderException("No encoder provided for text query");
        CheckModelCompat();
        var vector = encoder.Encode(query);
        _rw.EnterReadLock();
        try { CheckQuery(vector); }
        finally { _rw.ExitReadLock(); }
        return vector;
    }

    // -- model lifecycle -------------------------------------------------------

    /// <summary>Model name stamped on a vector written now.</summary>
    private string StampModel() => ModelCompat.Stamp(_encoder);

    /// <summary>The encoder's model name, or null when it has none worth comparing.</summary>
    private string? CurrentModel() => ModelCompat.Current(_encoder);

    /// <summary>
    /// Which models produced the stored vectors: model name to count.
    /// "unknown" and "provided" appear as-is (vectors imported from a port
    /// that records no model, or handed in with no encoder attached).
    /// </summary>
    public IReadOnlyDictionary<string, int> Models()
    {
        _rw.EnterReadLock();
        try { return new Dictionary<string, int>(_table?.ModelCounts ?? new Dictionary<string, int>()); }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>
    /// Complain when a text query's encoder is not the stored vectors' model.
    /// Called before a text query is encoded; vector queries skip it.
    /// </summary>
    internal void CheckModelCompat()
    {
        if (_onModelMismatch == ModelMismatchMode.Ignore) return;
        var current = CurrentModel();
        if (current is null) return;

        var foreign = ModelCompat.Foreign(current, Models().Keys);
        if (foreign.Count == 0) return;

        var message = ModelCompat.Message(current, foreign);
        if (_onModelMismatch == ModelMismatchMode.Error)
            throw new ModelMismatchException(message);

        bool first;
        lock (_warnedMismatch) first = _warnedMismatch.Add(ModelCompat.WarnKey(current, foreign));
        if (first) RaiseWarning(message);
    }

    /// <summary>
    /// Node refs whose stored vector does not belong to the current encoder.
    ///
    /// A vector is stale when its recorded model differs from
    /// <paramref name="model"/> (default: the encoder's ModelName) or its
    /// length differs from <paramref name="dimension"/> (default: the
    /// encoder's Dimension). Vectors whose model is "unknown"/"provided"
    /// cannot be judged by model and are skipped unless
    /// <paramref name="includeUnknown"/> is true (a dimension mismatch still
    /// counts). With no encoder and no arguments nothing can be judged and the
    /// result is empty.
    /// </summary>
    public IReadOnlyList<NodeRef> Stale(string? model = null, int? dimension = null, bool includeUnknown = false)
    {
        _rw.EnterReadLock();
        try
        {
            model ??= CurrentModel();
            dimension ??= _encoder?.Dimension;

            var stale = new List<NodeRef>();
            var t = _table;
            if (t is null) return stale;
            for (int s = 0; s < t.Slots; s++)
            {
                if (!t.IsLive(s)) continue;
                if (ModelCompat.IsStale(t.ModelAt(s), t.Dim, model, dimension, includeUnknown))
                    stale.Add(t.RefAt(s));
            }
            return stale;
        }
        finally { _rw.ExitReadLock(); }
    }

    // -- writes -----------------------------------------------------------------

    /// <summary>
    /// Store a vector for a node, copying it into the store (later changes to
    /// <paramref name="vector"/> do not reach the store). Encodes
    /// <paramref name="text"/> first, outside the lock, when no vector is given.
    /// </summary>
    public string Add(NodeRef nodeRef, string text, float[]? vector = null)
    {
        if (vector is null)
        {
            var encoder = _encoder ??
                throw new NoEncoderException("No encoder provided and no vector given");
            vector = encoder.Encode(text);
        }
        _rw.EnterWriteLock();
        try
        {
            CheckDimension(vector, "vector");
            return PutLocked(nodeRef, text, vector, StampModel());
        }
        finally { _rw.ExitWriteLock(); }
    }

    /// <summary>Store a vector under an explicit model name. Caller holds the write lock and has checked the dimension.</summary>
    private string PutLocked(NodeRef nodeRef, string text, ReadOnlySpan<float> vector, string model)
    {
        var key = Key(nodeRef);
        var table = _table ??= new VectorTable(vector.Length);
        int slot = table.Put(nodeRef, key, text, vector, model, out _);
        if (_ivf is not null && _ivf.NotePut(slot)) _ivf.Fold(table);
        return key;
    }

    /// <summary>Copy a record in, keeping the model it was made by.</summary>
    internal string AddRecord(EmbeddingRecord record)
    {
        _rw.EnterWriteLock();
        try
        {
            CheckDimension(record.Vector, "vector");
            return PutLocked(record.NodeRef, record.Text, record.Vector, record.Model);
        }
        finally { _rw.ExitWriteLock(); }
    }

    /// <summary>Add many embeddings, encoding their texts in one batched call.</summary>
    public int AddBatch(IReadOnlyList<(NodeRef NodeRef, string Text)> entries)
    {
        if (entries.Count == 0) return 0;
        if (_encoder is null)
            throw new NoEncoderException("No encoder provided and no vectors given");

        var vectors = _encoder.EncodeBatch(entries.Select(e => e.Text).ToList());
        if (vectors.Count != entries.Count)
            throw new EmbeddingException(
                $"encoder returned {vectors.Count} vectors for {entries.Count} texts");

        for (int i = 0; i < entries.Count; i++)
            Add(entries[i].NodeRef, entries[i].Text, vectors[i]);
        return entries.Count;
    }

    /// <summary>
    /// The record of a node. <b>The vector is a copy</b> (the store keeps its
    /// one copy in a shared matrix), so changing it does not change the store.
    /// </summary>
    public EmbeddingRecord? Get(NodeRef nodeRef)
    {
        _rw.EnterReadLock();
        try
        {
            var t = _table;
            if (t is null || !t.TryGetSlot(Key(nodeRef), out int slot)) return null;
            return new EmbeddingRecord(t.KeyAt(slot), t.RefAt(slot), t.Row(slot).ToArray(), t.TextAt(slot), t.ModelAt(slot));
        }
        finally { _rw.ExitReadLock(); }
    }

    public bool Remove(NodeRef nodeRef)
    {
        _rw.EnterWriteLock();
        try
        {
            var t = _table;
            if (t is null || !t.Remove(Key(nodeRef), out int slot)) return false;
            _ivf?.NoteDrop(slot);
            if (t.ShouldCompact) CompactLocked();
            if (_ivf is not null && t.Live < Math.Max(2, _annMinSize)) _ivf = null;
            return true;
        }
        finally { _rw.ExitWriteLock(); }
    }

    // Caller holds the write lock.
    private void CompactLocked()
    {
        var t = _table!;
        var remap = t.Compact();
        _ivf?.Compact(remap, t.Slots);
    }

    /// <summary>
    /// Squeeze the tombstones out of the matrix now. It happens by itself once
    /// a quarter of the rows are dead; order is preserved, so results do not
    /// change. A pause proportional to the store's size.
    /// </summary>
    public void Compact()
    {
        _rw.EnterWriteLock();
        try { if (_table is { Dead: > 0 }) CompactLocked(); }
        finally { _rw.ExitWriteLock(); }
    }

    public int Count
    {
        get
        {
            _rw.EnterReadLock();
            try { return _table?.Live ?? 0; }
            finally { _rw.ExitReadLock(); }
        }
    }

    /// <summary>
    /// Clear all embeddings.
    /// </summary>
    /// <param name="resetDimension">
    /// Also forget the pinned vector length, so the next vector may have a
    /// different one (switching to an encoder of a different dimension). True
    /// by default, which is how this method has always behaved; pass false to
    /// keep the store refusing vectors of the wrong length even when empty (the
    /// Python port's default).
    /// </param>
    public void Clear(bool resetDimension = true)
    {
        _rw.EnterWriteLock();
        try { ClearLocked(resetDimension); }
        finally { _rw.ExitWriteLock(); }
    }

    private void ClearLocked(bool resetDimension)
    {
        _table = null;
        _ivf = null;
        if (resetDimension) _dimension = -1;
    }

    // -- search -----------------------------------------------------------------

    /// <summary>
    /// Find the most similar embeddings to a text query.
    ///
    /// With an approximate index (<see cref="Ann"/>) a bounded <paramref name="topK"/>
    /// on a store of at least the minimum size is answered from the index
    /// (scores are the same cosine values, but a true neighbour in an unprobed
    /// cell can be missing). <paramref name="exact"/> forces the full scan.
    /// Equal scores are ordered by insertion order of the stored node.
    /// </summary>
    /// <param name="query">Query text.</param>
    /// <param name="topK">Number of results (null returns every match).</param>
    /// <param name="nodeType">Keep only embeddings of this node type.</param>
    /// <param name="threshold">Minimum similarity.</param>
    /// <param name="exact">Skip the approximate index and scan everything (no effect without one).</param>
    public IReadOnlyList<SimilarityResult> SimilaritySearch(
        string query, int? topK = 10, string? nodeType = null, float threshold = 0.0f, bool exact = false)
        => SimilaritySearchVector(ResolveQuery(query), topK, nodeType, threshold, exact);

    private bool WantsIndex(int? topK, bool exact) => _ann != AnnMode.None && !exact && topK is > 0;

    /// <summary>Find the most similar embeddings to a query vector; see <see cref="SimilaritySearch"/>.</summary>
    public IReadOnlyList<SimilarityResult> SimilaritySearchVector(
        float[] queryVector, int? topK = 10, string? nodeType = null, float threshold = 0.0f,
        bool exact = false)
    {
        EnterReadReady(WantsIndex(topK, exact));
        try { return SearchCore(queryVector, topK, nodeType, threshold, exact, ScanThreads); }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>
    /// Run many text queries at once; see the vector overload. The texts are
    /// encoded in one batched encoder call.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<SimilarityResult>> SimilaritySearchBatch(
        IReadOnlyList<string> queries, int? topK = 10, string? nodeType = null, float threshold = 0.0f,
        bool exact = false, int maxDegreeOfParallelism = 0)
    {
        if (queries.Count == 0) return Array.Empty<IReadOnlyList<SimilarityResult>>();
        var encoder = _encoder ?? throw new NoEncoderException("No encoder provided for text query");
        CheckModelCompat();
        var vectors = encoder.EncodeBatch(queries);
        if (vectors.Count != queries.Count)
            throw new EmbeddingException($"encoder returned {vectors.Count} vectors for {queries.Count} texts");
        return SimilaritySearchBatch(vectors, topK, nodeType, threshold, exact, maxDegreeOfParallelism);
    }

    /// <summary>
    /// Run many queries in parallel: result <c>i</c> is <b>bit-identical</b> (same
    /// nodes, same scores, same order) to <c>SimilaritySearchVector(queries[i], ...)</c>
    /// at the same moment. The whole batch observes one state of the store
    /// (it holds the shared lock throughout, so a concurrent write waits for it).
    /// Each query runs on one thread (no nested parallelism); the batch is
    /// spread over at most <paramref name="maxDegreeOfParallelism"/> threads
    /// (0 = one per logical CPU).
    /// </summary>
    public IReadOnlyList<IReadOnlyList<SimilarityResult>> SimilaritySearchBatch(
        IReadOnlyList<float[]> queries, int? topK = 10, string? nodeType = null, float threshold = 0.0f,
        bool exact = false, int maxDegreeOfParallelism = 0)
    {
        if (maxDegreeOfParallelism < 0)
            throw new ArgumentOutOfRangeException(nameof(maxDegreeOfParallelism), maxDegreeOfParallelism, "must be >= 0");
        int n = queries.Count;
        if (n == 0) return Array.Empty<IReadOnlyList<SimilarityResult>>();

        EnterReadReady(WantsIndex(topK, exact));
        try
        {
            foreach (var q in queries) CheckQuery(q);       // fail before any work, deterministically
            var results = new IReadOnlyList<SimilarityResult>[n];
            int degree = Math.Min(maxDegreeOfParallelism == 0 ? Environment.ProcessorCount : maxDegreeOfParallelism, n);
            if (degree <= 1)
            {
                for (int i = 0; i < n; i++)
                    results[i] = SearchCore(queries[i], topK, nodeType, threshold, exact, 1);
            }
            else
            {
                Parallel.For(0, n, new ParallelOptions { MaxDegreeOfParallelism = degree },
                    i => results[i] = SearchCore(queries[i], topK, nodeType, threshold, exact, 1));
            }
            return results;
        }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>One search, with a shared lock held and (if wanted) the index ready.</summary>
    private IReadOnlyList<SimilarityResult> SearchCore(
        float[] queryVector, int? topK, string? nodeType, float threshold, bool exact, int maxThreads)
    {
        CheckQuery(queryVector);
        if (_ann != AnnMode.None)
        {
            if (exact || topK is null || topK <= 0)
                Interlocked.Increment(ref _annExact);
            else
            {
                var answer = AnnSearch(queryVector, topK.Value, nodeType, threshold);
                if (answer is not null) return answer;
                Interlocked.Increment(ref _annExact);
            }
        }
        return ExactSearch(queryVector, topK, nodeType, threshold, maxThreads);
    }

    private static float QueryNorm(float[] q) => MathF.Sqrt(RowScan.SumSq(q));

    private List<SimilarityResult> ToResults(VectorTable t, Hit[] hits)
    {
        var list = new List<SimilarityResult>(hits.Length);
        foreach (var h in hits) list.Add(new SimilarityResult(t.RefAt(h.Slot), h.Score, t.TextAt(h.Slot)));
        return list;
    }

    private int ScanThreadsFor(VectorTable t, int maxThreads)
    {
        if (maxThreads == 1) return 1;
        long work = (long)t.Slots * t.Dim;
        int auto = (int)Math.Min(Math.Min(Environment.ProcessorCount, 8), work >> 21);
        return Math.Max(1, maxThreads > 0 ? Math.Min(maxThreads, t.ChunkCount) : auto);
    }

    /// <summary>The full scan. Caller holds a lock.</summary>
    private List<SimilarityResult> ExactSearch(
        float[] queryVector, int? topK, string? nodeType, float threshold, int maxThreads)
    {
        var t = _table;
        if (t is null || t.Live == 0 || topK is <= 0) return new List<SimilarityResult>();
        int code = -1;
        if (nodeType is not null)
        {
            code = t.CodeOf(nodeType);
            if (code < 0) return new List<SimilarityResult>();       // no row has this type
        }
        float qnorm = QueryNorm(queryVector);
        int chunks = t.ChunkCount;

        if (topK is null || topK >= t.Live)
        {
            var list = new HitList();
            var visitor = new CollectVisitor { List = list, Norms = t.Norms, QueryNorm = qnorm, Threshold = threshold };
            RowScan.Chunks(t, queryVector, code, 0, chunks, ref visitor);
            return ToResults(t, list.ToSortedArray());
        }

        int k = topK.Value;
        int threads = ScanThreadsFor(t, maxThreads);
        if (threads <= 1)
        {
            var heap = new TopKHeap(k);
            var visitor = new TopKVisitor { Heap = heap, Norms = t.Norms, QueryNorm = qnorm, Threshold = threshold };
            RowScan.Chunks(t, queryVector, code, 0, chunks, ref visitor);
            return ToResults(t, heap.ToSortedArray());
        }

        var merged = new TopKHeap(k);
        var norms = t.Norms;
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = threads },
            () => new TopKHeap(k),
            (c, _, heap) =>
            {
                var visitor = new TopKVisitor { Heap = heap, Norms = norms, QueryNorm = qnorm, Threshold = threshold };
                RowScan.Chunks(t, queryVector, code, c, c + 1, ref visitor);
                return heap;
            },
            heap => { lock (merged) merged.MergeFrom(heap); });
        return ToResults(t, merged.ToSortedArray());
    }

    /// <summary>
    /// Cosine score for every embedded node, keyed by "type:id".
    ///
    /// One scan serving both seed selection and the per-node relevance that
    /// SemanticMultiHop blends in, instead of two.
    /// </summary>
    public IReadOnlyDictionary<string, float> ScoreMap(float[] queryVector, string? nodeType = null)
    {
        _rw.EnterReadLock();
        try
        {
            CheckQuery(queryVector);
            var scores = new Dictionary<string, float>();
            var t = _table;
            if (t is null) return scores;
            int code = -1;
            if (nodeType is not null)
            {
                code = t.CodeOf(nodeType);
                if (code < 0) return scores;
            }
            var all = new float[t.Slots];
            var visitor = new ScoreAllVisitor { Scores = all, Norms = t.Norms, QueryNorm = QueryNorm(queryVector) };
            RowScan.Chunks(t, queryVector, code, 0, t.ChunkCount, ref visitor);
            for (int s = 0; s < t.Slots; s++)
            {
                if (!t.IsLive(s)) continue;
                if (code >= 0 && t.Codes[s] != code) continue;
                scores[t.KeyAt(s)] = all[s];
            }
            return scores;
        }
        finally { _rw.ExitReadLock(); }
    }

    // -- persistence (JSON sidecar) ------------------------------------------------

    /// <summary>Live slots in the sidecar's order: node type (default string comparer), then id (ordinal).</summary>
    private int[] SortedLiveSlots(VectorTable t)
    {
        var slots = new int[t.Live];
        for (int s = 0, j = 0; s < t.Slots; s++)
            if (t.IsLive(s)) slots[j++] = s;

        // rank the (few) distinct types once instead of comparing strings per row
        var names = t.TypeNames.ToArray();
        var order = Enumerable.Range(0, names.Length).ToArray();
        Array.Sort(order, (a, b) => Comparer<string>.Default.Compare(names[a], names[b]));
        var rank = new int[names.Length];
        for (int i = 0, r = 0; i < order.Length; i++)
        {
            if (i > 0 && Comparer<string>.Default.Compare(names[order[i - 1]], names[order[i]]) != 0) r++;
            rank[order[i]] = r;
        }
        var codes = t.Codes;
        Array.Sort(slots, (a, b) =>
        {
            int c = rank[codes[a]].CompareTo(rank[codes[b]]);
            return c != 0 ? c : string.CompareOrdinal(t.RefAt(a).Id, t.RefAt(b).Id);
        });
        return slots;
    }

    /// <summary>
    /// Serialize every embedding into the portable payload. <b>This copies every
    /// vector</b> (the store keeps its one copy in a shared matrix); for a large
    /// store prefer <see cref="SaveEmbeddings"/>, which streams to disk.
    ///
    /// Besides the embeddings it carries <c>encoder_model</c> (the encoder's
    /// ModelName) and <c>encoder_version</c> when known and, when
    /// <paramref name="graphDigest"/> is given, <c>graph_digest</c> and its
    /// basis. Absent values are omitted, not written as null.
    /// </summary>
    public ExportedEmbeddings ExportEmbeddings(string? graphDigest = null)
    {
        _rw.EnterReadLock();
        try
        {
            var payload = new ExportedEmbeddings { Dimension = _dimension < 0 ? null : _dimension };
            var t = _table;
            if (t is not null)
            {
                foreach (int slot in SortedLiveSlots(t))
                {
                    var r = t.RefAt(slot);
                    payload.Embeddings.Add(new ExportedEmbedding
                    {
                        Type = r.Type,
                        // C# node ids are always strings; the other ports use this
                        // to tell a numeric id from a numeric-looking string one.
                        Id = JsonSerializer.SerializeToElement(r.Id),
                        IdType = "str",
                        Text = t.TextAt(slot),
                        Model = t.ModelAt(slot),
                        Vector = t.Row(slot).ToArray(),
                    });
                }
            }
            payload.Count = payload.Embeddings.Count;
            payload.EncoderModel = CurrentModel();
            payload.EncoderVersion = Version;
            if (graphDigest is not null)
            {
                payload.GraphDigest = graphDigest;
                payload.GraphDigestBasis = IsonGraphVectorVersion.GraphDigestBasis;
            }
            return payload;
        }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>
    /// Load a payload produced by ExportEmbeddings (in any port). Fields this
    /// build does not know are ignored; the metadata it does know is exposed as
    /// <see cref="LastImportMeta"/>.
    /// </summary>
    public int ImportEmbeddings(ExportedEmbeddings payload, bool replace = false)
    {
        if (payload.Format != IsonGraphVectorVersion.EmbeddingFormat)
            throw new EmbeddingException(
                $"not an {IsonGraphVectorVersion.EmbeddingFormat} payload: format={payload.Format}");
        if (payload.Version != IsonGraphVectorVersion.EmbeddingFormatVersion)
            throw new EmbeddingException(
                $"unsupported {IsonGraphVectorVersion.EmbeddingFormat} version {payload.Version} " +
                $"(this build reads version {IsonGraphVectorVersion.EmbeddingFormatVersion})");

        _rw.EnterWriteLock();
        try
        {
            var meta = new Dictionary<string, string>();
            if (payload.EncoderModel is not null) meta["encoder_model"] = payload.EncoderModel;
            if (payload.EncoderVersion is not null) meta["encoder_version"] = payload.EncoderVersion;
            if (payload.GraphDigest is not null) meta["graph_digest"] = payload.GraphDigest;
            if (payload.GraphDigestBasis is not null) meta["graph_digest_basis"] = payload.GraphDigestBasis;
            _lastImportMeta = meta;

            if (replace)
            {
                ClearLocked(resetDimension: true);
                _dimension = payload.Dimension ?? -1;
            }

            int written = 0;
            foreach (var item in payload.Embeddings)
            {
                var nodeRef = new NodeRef(item.Type, item.IdAsString());
                CheckDimension(item.Vector, $"embedding for :{item.Type}:{nodeRef.Id}");
                PutLocked(nodeRef, item.Text, item.Vector, item.Model ?? "unknown");
                written++;
            }
            return written;
        }
        finally { _rw.ExitWriteLock(); }
    }

    /// <summary>Serialize the portable payload to a JSON string.</summary>
    public string ToJson(string? graphDigest = null) => JsonSerializer.Serialize(ExportEmbeddings(graphDigest));

    /// <summary>Load embeddings from a JSON payload written by any port.</summary>
    public int FromJson(string json, bool replace = false)
    {
        ExportedEmbeddings? payload;
        try { payload = JsonSerializer.Deserialize<ExportedEmbeddings>(json); }
        catch (JsonException ex) { throw new EmbeddingException($"unreadable embeddings payload: {ex.Message}"); }
        if (payload is null) throw new EmbeddingException("payload did not deserialize");
        return ImportEmbeddings(payload, replace);
    }

    /// <summary>
    /// Write the JSON payload (exactly the bytes <c>JsonSerializer.Serialize(ExportEmbeddings(...))</c>
    /// gives) straight from the stored rows to <paramref name="stream"/>: no
    /// payload, no vector copies and no giant string. Holds the shared lock
    /// while it writes.
    /// </summary>
    internal int WriteSidecar(Stream stream, string? graphDigest)
    {
        _rw.EnterReadLock();
        try
        {
            var t = _table;
            var slots = t is null ? Array.Empty<int>() : SortedLiveSlots(t);
            using var w = new Utf8JsonWriter(stream);
            w.WriteStartObject();
            w.WriteString("format", IsonGraphVectorVersion.EmbeddingFormat);
            w.WriteNumber("version", IsonGraphVectorVersion.EmbeddingFormatVersion);
            if (_dimension < 0) w.WriteNull("dimension"); else w.WriteNumber("dimension", _dimension);
            w.WriteNumber("count", slots.Length);
            w.WriteStartArray("embeddings");
            for (int i = 0; i < slots.Length; i++)
            {
                int slot = slots[i];
                var r = t!.RefAt(slot);
                w.WriteStartObject();
                w.WriteString("type", r.Type);
                w.WriteString("id", r.Id);
                w.WriteString("id_type", "str");
                w.WriteString("text", t.TextAt(slot));
                w.WriteString("model", t.ModelAt(slot));
                w.WriteStartArray("vector");
                foreach (float f in t.Row(slot)) w.WriteNumberValue(f);
                w.WriteEndArray();
                w.WriteEndObject();
                if ((i & 255) == 255) w.Flush();
            }
            w.WriteEndArray();
            var model = CurrentModel();
            if (model is not null) w.WriteString("encoder_model", model);
            var version = Version;
            if (version is not null) w.WriteString("encoder_version", version);
            if (graphDigest is not null)
            {
                w.WriteString("graph_digest", graphDigest);
                w.WriteString("graph_digest_basis", IsonGraphVectorVersion.GraphDigestBasis);
            }
            w.WriteEndObject();
            w.Flush();
            return slots.Length;
        }
        finally { _rw.ExitReadLock(); }
    }

    /// <summary>
    /// Write every embedding to a JSON sidecar, atomically. Returns the count.
    ///
    /// The file is written to a temp file and renamed into place, so a crash
    /// never leaves a half-written sidecar. <paramref name="graphDigest"/> ties
    /// the sidecar to one particular graph (see <see cref="GraphDigest"/>). The
    /// rows are streamed from the store to the file (no intermediate payload),
    /// holding the shared lock: searches continue, writes wait.
    /// </summary>
    public int SaveEmbeddings(string path, string? graphDigest = null)
    {
        int count = 0;
        AtomicFile.Write(path, stream => count = WriteSidecar(stream, graphDigest));
        return count;
    }

    /// <summary>Read a JSON sidecar written by <see cref="SaveEmbeddings"/> (or by any port).</summary>
    public int LoadEmbeddings(string path, bool replace = false)
    {
        var payload = ReadPayload(path);
        return ImportEmbeddings(payload, replace);
    }

    /// <summary>Parse a JSON sidecar from disk without building a string of the whole file (a JsonException propagates).</summary>
    internal static ExportedEmbeddings? DeserializePayload(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
        return JsonSerializer.Deserialize<ExportedEmbeddings>(stream);
    }

    private static ExportedEmbeddings ReadPayload(string path)
    {
        try
        {
            return DeserializePayload(path) ?? throw new EmbeddingException("payload did not deserialize");
        }
        catch (JsonException ex)
        {
            throw new EmbeddingException($"unreadable embeddings payload: {ex.Message}");
        }
    }
}
