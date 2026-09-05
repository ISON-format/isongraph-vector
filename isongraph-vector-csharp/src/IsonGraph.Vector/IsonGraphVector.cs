/*
 * ISONGraph Vector - Semantic Graph Extension for C#
 *
 * Extends ISONGraph with embedding-based similarity search and semantic traversal.
 *
 * Example:
 *   var graph = new SemanticGraph("social", new MockEncoder(384));
 *   graph.AddNode("person", "1", new() { ["name"] = "Alice" }, "software engineer");
 *   var results = graph.SimilaritySearch("engineer", 5, null, -1.0);
 *
 * Author: Mahesh Vaikri
 * Version: 1.0.0
 */

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using IsonGraph;

namespace IsonGraph.Vector;

/// <summary>Version of this package.</summary>
public static class IsonGraphVectorVersion
{
    public const string Version = "1.0.0";

    /// <summary>
    /// Portable embedding payload shared by every language port, so a store
    /// built in one can be searched by another.
    /// </summary>
    public const string EmbeddingFormat = "ison-embeddings";

    public const int EmbeddingFormatVersion = 1;
}

// =============================================================================
// Errors
// =============================================================================

/// <summary>Base class for embedding-specific failures.</summary>
public class EmbeddingException : Exception
{
    public EmbeddingException(string message) : base(message) { }
}

/// <summary>
/// A vector was stored or queried with the wrong number of dimensions.
///
/// Cosine similarity over a mismatched pair returns a plausible-looking number
/// rather than an error, so a store that quietly accepted mixed widths would
/// degrade silently. A store pins its dimension to the first vector it sees.
/// </summary>
public sealed class DimensionMismatchException : EmbeddingException
{
    public DimensionMismatchException(string message) : base(message) { }
}

/// <summary>Text was handed to a store with no encoder to embed it with.</summary>
public sealed class NoEncoderException : EmbeddingException
{
    public NoEncoderException(string message) : base(message) { }
}

// =============================================================================
// Embedding Encoder
// =============================================================================

public interface IEmbeddingEncoder
{
    int Dimension { get; }
    float[] Encode(string text);
    IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts);
}

/// <summary>
/// Deterministic encoder with no dependencies, for tests and offline work.
///
/// The same text produces the same vector in every language port
/// (Python/TypeScript/JavaScript/Rust/C++/C#), so a store built by one can be
/// searched by another: a 32-bit FNV-1a hash over the UTF-8 bytes seeds an
/// xorshift32 generator, and the top 24 bits of each state become a value in
/// [-1, 1).
///
/// These are pseudo-embeddings with no semantic structure. Use them to test
/// plumbing, never to judge relevance quality.
/// </summary>
public sealed class MockEncoder : IEmbeddingEncoder
{
    private const uint FnvOffsetBasis = 0x811c9dc5u;
    private const uint FnvPrime = 0x01000193u;

    public int Dimension { get; }

    public MockEncoder(int dimension = 384)
    {
        if (dimension <= 0)
            throw new EmbeddingException($"dimension must be positive, got {dimension}");
        Dimension = dimension;
    }

    public float[] Encode(string text)
    {
        uint state = FnvOffsetBasis;
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(text))
        {
            state = unchecked((state ^ b) * FnvPrime);
        }
        // xorshift32 has no way out of a zero state; steer off it.
        if (state == 0) state = 0x9e3779b9u;

        var values = new float[Dimension];
        for (int i = 0; i < Dimension; i++)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            values[i] = (state >> 8) / 8388608.0f - 1.0f;  // top 24 bits
        }
        return values;
    }

    public IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts)
    {
        var results = new List<float[]>(texts.Count);
        foreach (var text in texts) results.Add(Encode(text));
        return results;
    }
}

// =============================================================================
// Records
// =============================================================================

public sealed record EmbeddingRecord(string Id, NodeRef NodeRef, float[] Vector, string Text);

public sealed record SimilarityResult(NodeRef NodeRef, float Score, string Text);

public sealed record SemanticSearchResult(
    NodeRef NodeRef,
    Node? Node,
    float Score,
    int HopCount,
    IReadOnlyList<NodeRef> Path);

// =============================================================================
// Portable payload
// =============================================================================

/// <summary>One entry of the portable embedding payload.</summary>
public sealed class ExportedEmbedding
{
    [JsonPropertyName("type")] public string Type { get; set; } = "";

    /// <summary>
    /// Node id. Ports whose ids are always strings write "007"; the ports that
    /// keep an integer id write 1 unquoted. Both land on the same string id
    /// here, so this is read as a raw JSON element.
    /// </summary>
    [JsonPropertyName("id")] public JsonElement Id { get; set; }

    [JsonPropertyName("id_type")] public string IdType { get; set; } = "str";
    [JsonPropertyName("text")] public string Text { get; set; } = "";
    [JsonPropertyName("vector")] public float[] Vector { get; set; } = Array.Empty<float>();

    public string IdAsString() =>
        Id.ValueKind == JsonValueKind.String ? Id.GetString() ?? "" : Id.ToString();
}

/// <summary>Payload written by ExportEmbeddings, readable by every port.</summary>
public sealed class ExportedEmbeddings
{
    [JsonPropertyName("format")] public string Format { get; set; } = IsonGraphVectorVersion.EmbeddingFormat;
    [JsonPropertyName("version")] public int Version { get; set; } = IsonGraphVectorVersion.EmbeddingFormatVersion;
    [JsonPropertyName("dimension")] public int? Dimension { get; set; }
    [JsonPropertyName("count")] public int Count { get; set; }
    [JsonPropertyName("embeddings")] public List<ExportedEmbedding> Embeddings { get; set; } = new();
}

// =============================================================================
// Embedding Store
// =============================================================================

/// <summary>
/// In-memory embedding storage with brute-force cosine similarity search.
/// </summary>
public sealed class EmbeddingStore
{
    private readonly Dictionary<string, EmbeddingRecord> _embeddings = new();
    private readonly object _lock = new();
    private IEmbeddingEncoder? _encoder;
    private int? _dimension;

    public EmbeddingStore(IEmbeddingEncoder? encoder = null) => _encoder = encoder;

    public IEmbeddingEncoder? Encoder
    {
        get => _encoder;
        set => _encoder = value;
    }

    /// <summary>Vector length this store holds, or null while it is empty.</summary>
    public int? Dimension
    {
        get { lock (_lock) return _dimension; }
    }

    private static string Key(NodeRef nodeRef) => $"{nodeRef.Type}:{nodeRef.Id}";

    private void CheckDimension(IReadOnlyList<float> vector, string what)
    {
        if (vector.Count == 0)
            throw new DimensionMismatchException($"{what} is empty");
        if (_dimension is null)
            _dimension = vector.Count;
        else if (vector.Count != _dimension)
            throw new DimensionMismatchException(
                $"{what} has {vector.Count} dimensions, store holds {_dimension}");
    }

    private void CheckQuery(IReadOnlyList<float> vector)
    {
        if (vector.Count == 0)
            throw new DimensionMismatchException("query vector is empty");
        if (_dimension is not null && vector.Count != _dimension)
            throw new DimensionMismatchException(
                $"query vector has {vector.Count} dimensions, store holds {_dimension}");
    }

    private static float CosineSimilarity(IReadOnlyList<float> v1, IReadOnlyList<float> v2)
    {
        float dot = 0f, norm1 = 0f, norm2 = 0f;
        int n = Math.Min(v1.Count, v2.Count);
        for (int i = 0; i < n; i++)
        {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        norm1 = MathF.Sqrt(norm1);
        norm2 = MathF.Sqrt(norm2);
        if (norm1 == 0f || norm2 == 0f) return 0f;
        return dot / (norm1 * norm2);
    }

    private float[] ResolveQuery(string query)
    {
        if (_encoder is null)
            throw new NoEncoderException("No encoder provided for text query");
        var vector = _encoder.Encode(query);
        CheckQuery(vector);
        return vector;
    }

    public string Add(NodeRef nodeRef, string text, float[]? vector = null)
    {
        lock (_lock)
        {
            if (vector is null)
            {
                if (_encoder is null)
                    throw new NoEncoderException("No encoder provided and no vector given");
                vector = _encoder.Encode(text);
            }
            CheckDimension(vector, "vector");

            var key = Key(nodeRef);
            _embeddings[key] = new EmbeddingRecord(key, nodeRef, vector, text);
            return key;
        }
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

    public EmbeddingRecord? Get(NodeRef nodeRef)
    {
        lock (_lock)
            return _embeddings.TryGetValue(Key(nodeRef), out var record) ? record : null;
    }

    public bool Remove(NodeRef nodeRef)
    {
        lock (_lock) return _embeddings.Remove(Key(nodeRef));
    }

    public int Count
    {
        get { lock (_lock) return _embeddings.Count; }
    }

    public void Clear()
    {
        lock (_lock)
        {
            _embeddings.Clear();
            _dimension = null;
        }
    }

    public IReadOnlyList<SimilarityResult> SimilaritySearch(
        string query, int? topK = 10, string? nodeType = null, float threshold = 0.0f)
        => SimilaritySearchVector(ResolveQuery(query), topK, nodeType, threshold);

    public IReadOnlyList<SimilarityResult> SimilaritySearchVector(
        float[] queryVector, int? topK = 10, string? nodeType = null, float threshold = 0.0f)
    {
        lock (_lock)
        {
            CheckQuery(queryVector);

            var results = new List<SimilarityResult>();
            foreach (var record in _embeddings.Values)
            {
                if (nodeType is not null && record.NodeRef.Type != nodeType) continue;
                float score = CosineSimilarity(queryVector, record.Vector);
                if (score >= threshold)
                    results.Add(new SimilarityResult(record.NodeRef, score, record.Text));
            }

            results.Sort((a, b) => b.Score.CompareTo(a.Score));
            return topK is null || results.Count <= topK
                ? results
                : results.GetRange(0, topK.Value);
        }
    }

    /// <summary>
    /// Cosine score for every embedded node, keyed by "type:id".
    ///
    /// One scan serving both seed selection and the per-node relevance that
    /// SemanticMultiHop blends in, instead of two.
    /// </summary>
    public IReadOnlyDictionary<string, float> ScoreMap(float[] queryVector, string? nodeType = null)
    {
        lock (_lock)
        {
            CheckQuery(queryVector);
            var scores = new Dictionary<string, float>();
            foreach (var (key, record) in _embeddings)
            {
                if (nodeType is not null && record.NodeRef.Type != nodeType) continue;
                scores[key] = CosineSimilarity(queryVector, record.Vector);
            }
            return scores;
        }
    }

    /// <summary>Serialize every embedding into the portable payload.</summary>
    public ExportedEmbeddings ExportEmbeddings()
    {
        lock (_lock)
        {
            var payload = new ExportedEmbeddings { Dimension = _dimension };
            foreach (var record in _embeddings.Values.OrderBy(r => r.NodeRef.Type)
                                                     .ThenBy(r => r.NodeRef.Id, StringComparer.Ordinal))
            {
                payload.Embeddings.Add(new ExportedEmbedding
                {
                    Type = record.NodeRef.Type,
                    // C# node ids are always strings; the other ports use this
                    // to tell a numeric id from a numeric-looking string one.
                    Id = JsonSerializer.SerializeToElement(record.NodeRef.Id),
                    IdType = "str",
                    Text = record.Text,
                    Vector = record.Vector,
                });
            }
            payload.Count = payload.Embeddings.Count;
            return payload;
        }
    }

    /// <summary>Load a payload produced by ExportEmbeddings (in any port).</summary>
    public int ImportEmbeddings(ExportedEmbeddings payload, bool replace = false)
    {
        if (payload.Format != IsonGraphVectorVersion.EmbeddingFormat)
            throw new EmbeddingException(
                $"not an {IsonGraphVectorVersion.EmbeddingFormat} payload: format={payload.Format}");
        if (payload.Version != IsonGraphVectorVersion.EmbeddingFormatVersion)
            throw new EmbeddingException(
                $"unsupported {IsonGraphVectorVersion.EmbeddingFormat} version {payload.Version} " +
                $"(this build reads version {IsonGraphVectorVersion.EmbeddingFormatVersion})");

        if (replace) Clear();

        int written = 0;
        foreach (var item in payload.Embeddings)
        {
            Add(new NodeRef(item.Type, item.IdAsString()), item.Text, item.Vector);
            written++;
        }
        return written;
    }

    /// <summary>Serialize the portable payload to a JSON string.</summary>
    public string ToJson() => JsonSerializer.Serialize(ExportEmbeddings());

    /// <summary>Load embeddings from a JSON payload written by any port.</summary>
    public int FromJson(string json, bool replace = false)
    {
        var payload = JsonSerializer.Deserialize<ExportedEmbeddings>(json)
            ?? throw new EmbeddingException("payload did not deserialize");
        return ImportEmbeddings(payload, replace);
    }
}

// =============================================================================
// Semantic Graph
// =============================================================================

/// <summary>Options for SemanticMultiHop.</summary>
public sealed class MultiHopOptions
{
    public string? RelType { get; set; }
    public int MaxHops { get; set; } = 2;
    public int TopKSeeds { get; set; } = 5;
    public int TopKResults { get; set; } = 10;
    public Direction Direction { get; set; } = Direction.Out;
    public float Decay { get; set; } = 0.8f;
    public float Threshold { get; set; } = 0.3f;

    /// <summary>
    /// How much of the relevance of a node itself to mix into its traversal
    /// score, in [0, 1]. 0 (the default) keeps pure seed-and-decay scoring.
    /// Raise it so a node found two hops out but highly relevant in its own
    /// right is not buried under a closer, unrelated neighbour. Hop-0 seeds
    /// score the same either way.
    /// </summary>
    public float Blend { get; set; } = 0.0f;
}

/// <summary>
/// ISONGraph extended with embedding-based semantic search.
///
/// ISONGraph is sealed, so this wraps it by composition; the underlying graph
/// stays reachable through <see cref="Graph"/> and every ISONGraph method keeps
/// working unchanged.
/// </summary>
public sealed class SemanticGraph
{
    private static readonly string[] DefaultEmbedFields = { "name", "description", "content", "text" };

    private readonly ISONGraph _graph;
    private readonly EmbeddingStore _store;
    private readonly bool _autoEmbed;
    private readonly List<string> _embedFields;

    public SemanticGraph(
        string name = "semantic_graph",
        IEmbeddingEncoder? encoder = null,
        bool autoEmbed = true,
        IEnumerable<string>? embedFields = null,
        bool directed = true)
    {
        _graph = new ISONGraph(name, directed);
        _store = new EmbeddingStore(encoder);
        _autoEmbed = autoEmbed;
        _embedFields = (embedFields ?? DefaultEmbedFields).ToList();
    }

    public ISONGraph Graph => _graph;
    public EmbeddingStore EmbeddingStore => _store;
    public IEmbeddingEncoder? Encoder => _store.Encoder;
    public string Name => _graph.Name;

    public void SetEncoder(IEmbeddingEncoder encoder) => _store.Encoder = encoder;

    private static string Key(NodeRef nodeRef) => $"{nodeRef.Type}:{nodeRef.Id}";

    /// <summary>
    /// Concatenate the configured embed fields present in a property bag.
    ///
    /// ISONGraph property values are typed (string, number, bool, null), so
    /// they are rendered with the invariant culture: a float property must
    /// embed as "1.5" on every machine, not "1,5" on some.
    /// </summary>
    public string EmbedTextFor(IReadOnlyDictionary<string, object?> properties)
    {
        var parts = new List<string>();
        foreach (var field in _embedFields)
        {
            if (!properties.TryGetValue(field, out var value)) continue;
            var text = PropertyText(value);
            if (text.Length > 0) parts.Add(text);
        }
        return string.Join(" ", parts);
    }

    /// <summary>
    /// Render one property value as embedding text.
    ///
    /// Booleans render as ISON spells them - "true"/"false", not .NET's
    /// "True"/"False" - and null contributes nothing, so the same graph
    /// produces the same text (and so the same vector) in every port.
    /// Everything else uses the invariant culture: a float property must embed
    /// as "1.5" on every machine, not "1,5" on some.
    /// </summary>
    public static string PropertyText(object? value) => value switch
    {
        null => "",
        bool flag => flag ? "true" : "false",
        _ => Convert.ToString(value, CultureInfo.InvariantCulture) ?? "",
    };

    /// <summary>
    /// Add a node, embedding it from <paramref name="embedText"/> or, when
    /// autoEmbed is on, from its properties.
    /// </summary>
    public Node AddNode(
        string nodeType,
        string nodeId,
        IDictionary<string, object?>? properties = null,
        string? embedText = null)
    {
        var node = _graph.AddNode(nodeType, nodeId, properties);

        var text = embedText;
        if (text is null && _autoEmbed && _store.Encoder is not null)
        {
            var derived = EmbedTextFor(node.Properties);
            if (derived.Length > 0) text = derived;
        }
        if (!string.IsNullOrEmpty(text))
            EmbedNode(new NodeRef(nodeType, nodeId), text);

        return node;
    }

    public string EmbedNode(NodeRef nodeRef, string text) => _store.Add(nodeRef, text);

    /// <summary>Embed many nodes in a single batched encoder call.</summary>
    public int EmbedNodes(IReadOnlyList<(NodeRef NodeRef, string Text)> items)
    {
        var pairs = items.Where(i => !string.IsNullOrEmpty(i.Text)).ToList();
        return pairs.Count == 0 ? 0 : _store.AddBatch(pairs);
    }

    public EmbeddingRecord? GetEmbedding(NodeRef nodeRef) => _store.Get(nodeRef);

    public void RemoveNode(string nodeType, string nodeId)
    {
        _store.Remove(new NodeRef(nodeType, nodeId));
        _graph.RemoveNode(nodeType, nodeId);
    }

    public IReadOnlyList<SimilarityResult> SimilaritySearch(
        string query, int? topK = 10, string? nodeType = null, float threshold = 0.0f)
        => _store.SimilaritySearch(query, topK, nodeType, threshold);

    /// <summary>
    /// Find nodes semantically closest to one already in the graph.
    ///
    /// The "more like this" query - no text and no encoder call: it searches
    /// with the stored vector of the node and drops the node itself from the
    /// results.
    /// </summary>
    public IReadOnlyList<SimilarityResult> SimilarToNode(
        NodeRef nodeRef, int topK = 10, string? nodeType = null, float threshold = 0.0f)
    {
        var record = _store.Get(nodeRef);
        if (record is null) return Array.Empty<SimilarityResult>();

        var results = _store.SimilaritySearchVector(record.Vector, topK + 1, nodeType, threshold)
                            .Where(r => r.NodeRef != nodeRef)
                            .ToList();
        return results.Count <= topK ? results : results.GetRange(0, topK);
    }

    private Node? ResolveNode(NodeRef nodeRef)
    {
        try { return _graph.GetNode(nodeRef); }
        catch (GraphError) { return null; }
    }

    /// <summary>
    /// Semantic multi-hop search.
    ///
    /// 1. Find top-k similar nodes as seeds (hop 0)
    /// 2. Traverse the graph following relationships
    /// 3. Score by max(seedSimilarity, 0) * decay^hopCount, optionally blended
    ///    with the relevance of each node itself
    /// </summary>
    public IReadOnlyList<SemanticSearchResult> SemanticMultiHop(string query, MultiHopOptions? options = null)
    {
        options ??= new MultiHopOptions();

        if (options.Blend < 0f || options.Blend > 1f)
            throw new EmbeddingException($"blend must be in [0, 1], got {options.Blend}");
        if (_store.Encoder is null)
            throw new NoEncoderException("No encoder provided for text query");

        var queryVector = _store.Encoder.Encode(query);

        // One scan serves both seed selection and per-node relevance.
        var scores = _store.ScoreMap(queryVector);
        if (scores.Count == 0) return Array.Empty<SemanticSearchResult>();

        var seeds = _store.SimilaritySearchVector(
            queryVector, options.TopKSeeds, null, options.Threshold);
        if (seeds.Count == 0) return Array.Empty<SemanticSearchResult>();

        float Combined(float baseScore, string key) =>
            options.Blend <= 0f
                ? baseScore
                : (1f - options.Blend) * baseScore +
                  options.Blend * (scores.TryGetValue(key, out var own) ? own : 0f);

        var results = new Dictionary<string, SemanticSearchResult>();

        foreach (var seed in seeds)
        {
            var seedKey = Key(seed.NodeRef);

            // Cosine similarity is signed, and `base * decay ** hops` only
            // decays a non-negative base: multiplying a negative seed score by
            // 0.8 moves it toward zero, so everything it reached outranked the
            // seed itself and relevance grew with distance.
            float traversalBase = MathF.Max(seed.Score, 0f);

            // A seed is a hop-0 match on its own merit. An earlier traversal
            // may already have recorded it with a decayed score; that must not
            // win over the higher, direct similarity of the node.
            if (!results.TryGetValue(seedKey, out var seeded) || seeded.Score < seed.Score)
            {
                results[seedKey] = new SemanticSearchResult(
                    seed.NodeRef, ResolveNode(seed.NodeRef), seed.Score, 0,
                    new List<NodeRef> { seed.NodeRef });
            }

            // BFS traversal
            var queue = new Queue<(NodeRef NodeRef, List<NodeRef> Path, int Hop)>();
            queue.Enqueue((seed.NodeRef, new List<NodeRef> { seed.NodeRef }, 0));
            var visited = new HashSet<string> { seedKey };

            while (queue.Count > 0)
            {
                var (nodeRef, path, hop) = queue.Dequeue();
                if (hop >= options.MaxHops) continue;

                foreach (var neighbor in _graph.Neighbors(nodeRef, options.RelType, options.Direction))
                {
                    var neighborKey = Key(neighbor);
                    if (!visited.Add(neighborKey)) continue;

                    var newPath = new List<NodeRef>(path) { neighbor };
                    int newHop = hop + 1;
                    float newScore = Combined(
                        traversalBase * MathF.Pow(options.Decay, newHop), neighborKey);

                    if (!results.TryGetValue(neighborKey, out var existing) || existing.Score < newScore)
                    {
                        results[neighborKey] = new SemanticSearchResult(
                            neighbor, ResolveNode(neighbor), newScore, newHop, newPath);
                    }

                    queue.Enqueue((neighbor, newPath, newHop));
                }
            }
        }

        var sorted = results.Values.OrderByDescending(r => r.Score).ToList();
        return sorted.Count <= options.TopKResults
            ? sorted
            : sorted.GetRange(0, options.TopKResults);
    }

    /// <summary>
    /// Find a semantically-rooted path to a target node.
    ///
    /// Starts from the most similar nodes and returns the first path found to
    /// the target, scored by the similarity of the seed decayed per hop.
    /// </summary>
    public SemanticSearchResult? SemanticPath(
        string query,
        NodeRef targetRef,
        string? relType = null,
        int maxHops = 5,
        int topKSeeds = 3,
        Direction direction = Direction.Out,
        float decay = 0.9f,
        float threshold = 0.0f)
    {
        foreach (var seed in _store.SimilaritySearch(query, topKSeeds, null, threshold))
        {
            var path = _graph.ShortestPath(seed.NodeRef, targetRef, relType, maxHops, direction);
            if (path is null) continue;

            int hops = path.Length;
            return new SemanticSearchResult(
                targetRef, ResolveNode(targetRef),
                seed.Score * MathF.Pow(decay, hops), hops, path.Nodes);
        }
        return null;
    }

    /// <summary>
    /// Extract the slice of the graph a query is actually about.
    ///
    /// Runs SemanticMultiHop, then returns a new SemanticGraph holding exactly
    /// those nodes, every edge induced between them, and their embeddings - so
    /// the result can be traversed, searched or serialized on its own. This is
    /// what makes an ISON graph useful as LLM context: a small, on-topic graph
    /// to inject rather than the whole store.
    /// </summary>
    public SemanticGraph SemanticSubgraph(
        string query, MultiHopOptions? options = null, string? name = null)
    {
        options ??= new MultiHopOptions();
        var results = SemanticMultiHop(query, options);
        var keep = results.Select(r => Key(r.NodeRef)).ToHashSet();

        var sub = new SemanticGraph(
            name ?? $"{_graph.Name}_subgraph",
            _store.Encoder,
            autoEmbed: false,
            embedFields: _embedFields,
            directed: _graph.Directed);

        foreach (var result in results)
        {
            var node = result.Node ?? ResolveNode(result.NodeRef);
            if (node is null) continue;

            sub._graph.AddNode(node.Type, node.Id, node.Properties);
            var record = _store.Get(result.NodeRef);
            if (record is not null)
                sub._store.Add(result.NodeRef, record.Text, record.Vector);
        }

        foreach (var edge in _graph.Edges(options.RelType))
        {
            if (keep.Contains(Key(edge.Source)) && keep.Contains(Key(edge.Target)))
                sub._graph.AddEdge(edge.RelType, edge.Source, edge.Target, edge.Properties);
        }

        return sub;
    }

    /// <summary>Embed all nodes in the graph, in one batched encoder call.</summary>
    public int EmbedAllNodes(Func<Node, string>? textFn = null, bool skipExisting = false)
    {
        textFn ??= node =>
        {
            var text = EmbedTextFor(node.Properties);
            return text.Length > 0 ? text : $"{node.Type}:{node.Id}";
        };

        var batch = new List<(NodeRef, string)>();
        foreach (var node in _graph.Nodes())
        {
            var nodeRef = new NodeRef(node.Type, node.Id);
            if (skipExisting && _store.Get(nodeRef) is not null) continue;
            var text = textFn(node);
            if (!string.IsNullOrEmpty(text)) batch.Add((nodeRef, text));
        }

        return EmbedNodes(batch);
    }

    public override string ToString() =>
        $"SemanticGraph(name={_graph.Name}, nodes={_graph.NodeCount()}, " +
        $"edges={_graph.EdgeCount()}, embeddings={_store.Count})";
}
