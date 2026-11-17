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

    /// <summary>
    /// Suffix appended to a graph path to locate its embeddings sidecar
    /// (shared with the other ports).
    /// </summary>
    public const string EmbeddingSidecarSuffix = ".embeddings.json";

    /// <summary>
    /// Suffix appended to a graph path to locate its optional binary
    /// embeddings snapshot (see <see cref="EmbeddingStore.SaveSnapshot"/>).
    /// </summary>
    public const string EmbeddingSnapshotSuffix = ".embeddings.snap";

    /// <summary>
    /// Name of the serialization <see cref="GraphDigest"/> hashes. Written next
    /// to the digest so a reader that computes a different digest (or none) can
    /// tell it is not comparable and skip the check instead of reporting a
    /// false mismatch.
    /// </summary>
    public const string GraphDigestBasis = "nodes-edges-json-v1";

    /// <summary>
    /// Model names that carry no information about which encoder made a
    /// vector: "provided" (vector handed in, store has no encoder) and
    /// "unknown" (sidecars from ports that do not record a model).
    /// </summary>
    public static readonly IReadOnlySet<string> UnknownModels =
        new HashSet<string> { "unknown", "provided" };
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

/// <summary>
/// A text query was encoded by a different model than the stored vectors.
///
/// Two models of the same dimension give a plausible-looking cosine score
/// between vectors that live in unrelated spaces, so the ranking is noise
/// rather than an error. Thrown only with <see cref="ModelMismatchMode.Error"/>;
/// the default is a warning.
/// </summary>
public sealed class ModelMismatchException : EmbeddingException
{
    public ModelMismatchException(string message) : base(message) { }
}

/// <summary>
/// A graph file and its embeddings sidecar are not from the same save.
///
/// Thrown by <see cref="SemanticGraph.Load"/> when the sidecar's recorded
/// graph digest does not match the graph that was read: the sidecar came from
/// a different save, or the graph file was edited afterwards.
/// </summary>
public sealed class GraphEmbeddingMismatchException : EmbeddingException
{
    public GraphEmbeddingMismatchException(string message) : base(message) { }
}

/// <summary>What a text query does when the encoder's model is not the stored vectors' model.</summary>
public enum ModelMismatchMode
{
    /// <summary>Raise the store's Warning event, once per distinct pair of models (default).</summary>
    Warn,
    /// <summary>Throw <see cref="ModelMismatchException"/>.</summary>
    Error,
    /// <summary>Say nothing.</summary>
    Ignore,
}

/// <summary>What <see cref="SemanticGraph.Load"/> does when the sidecar belongs to a different graph.</summary>
public enum GraphMismatchMode
{
    /// <summary>Throw <see cref="GraphEmbeddingMismatchException"/> (default).</summary>
    Error,
    /// <summary>Warn and load the sidecar's vectors anyway.</summary>
    Warn,
    /// <summary>Discard the sidecar's vectors and re-embed every node with the encoder.</summary>
    Reembed,
}

// =============================================================================
// Embedding Encoder
// =============================================================================

public interface IEmbeddingEncoder
{
    int Dimension { get; }
    float[] Encode(string text);
    IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts);

    /// <summary>
    /// Name of the model behind this encoder, stamped on every vector it
    /// writes; it decides whether stored vectors are stale for this encoder.
    /// Null (the default) records the vectors as "unknown", which never
    /// counts as stale.
    /// </summary>
    string? ModelName => null;

    /// <summary>
    /// Free-form encoder/model version (a revision or weights hash), written
    /// to the sidecar. Null when unknown.
    /// </summary>
    string? Version => null;
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

    /// <summary>"mock-&lt;dimension&gt;" unless set; settable so tests can stand in for another model.</summary>
    public string? ModelName { get; set; }

    public string? Version { get; set; }

    public MockEncoder(int dimension = 384, string? version = null)
    {
        if (dimension <= 0)
            throw new EmbeddingException($"dimension must be positive, got {dimension}");
        Dimension = dimension;
        ModelName = $"mock-{dimension}";
        Version = version;
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

/// <param name="Id">Surrogate key, "type:id".</param>
/// <param name="NodeRef">The node the vector belongs to.</param>
/// <param name="Vector">The embedding. The store keeps its one copy of every vector in a shared matrix, so a record handed out by <see cref="EmbeddingStore.Get"/> carries a copy of it: changing it does not change the store.</param>
/// <param name="Text">The text it was made from.</param>
/// <param name="Model">
/// Name of the model that made the vector: the encoder's ModelName, "unknown"
/// (an encoder without one, or a sidecar from a port that records none) or
/// "provided" (a vector handed in with no encoder attached).
/// </param>
public sealed record EmbeddingRecord(
    string Id, NodeRef NodeRef, float[] Vector, string Text, string Model = "unknown");

public sealed record SimilarityResult(NodeRef NodeRef, float Score, string Text);

public sealed record SemanticSearchResult(
    NodeRef NodeRef,
    Node? Node,
    float Score,
    int HopCount,
    IReadOnlyList<NodeRef> Path);

// =============================================================================
// Model lifecycle helpers (shared with the SQLite store)
// =============================================================================

internal static class ModelCompat
{
    /// <summary>Model name stamped on a vector written with this encoder.</summary>
    public static string Stamp(IEmbeddingEncoder? encoder) =>
        encoder is null ? "provided" : encoder.ModelName ?? "unknown";

    /// <summary>The encoder's model name, or null when it has none worth comparing.</summary>
    public static string? Current(IEmbeddingEncoder? encoder)
    {
        var name = encoder?.ModelName;
        return string.IsNullOrEmpty(name) || IsonGraphVectorVersion.UnknownModels.Contains(name) ? null : name;
    }

    /// <summary>The stored models that are neither unknown nor the current one, sorted.</summary>
    public static List<string> Foreign(string current, IEnumerable<string> stored) =>
        stored.Where(m => !IsonGraphVectorVersion.UnknownModels.Contains(m) && m != current)
              .OrderBy(m => m, StringComparer.Ordinal)
              .ToList();

    public static string Message(string current, IReadOnlyList<string> foreign) =>
        $"encoder model '{current}' differs from the model that produced the stored " +
        $"vectors ({string.Join(", ", foreign.Select(m => $"'{m}'"))}); scores compare " +
        "vectors from different spaces. Re-embed with ReembedStale() or search with " +
        "the original encoder.";

    public static string WarnKey(string current, IReadOnlyList<string> foreign) =>
        current + "\u0000" + string.Join("\u0000", foreign);

    public static bool IsStale(
        string recordModel, int recordDimension, string? model, int? dimension, bool includeUnknown)
    {
        bool byModel = model is not null && recordModel != model &&
            (includeUnknown || !IsonGraphVectorVersion.UnknownModels.Contains(recordModel));
        bool byDimension = dimension is not null && recordDimension != dimension;
        return byModel || byDimension;
    }
}

// =============================================================================
// Portable payload
// =============================================================================

/// <summary>
/// Reads a JSON string, but also accepts a number or boolean and keeps its
/// text, so a sidecar from a port that wrote a version as <c>2</c> rather than
/// <c>"2"</c> still imports.
/// </summary>
internal sealed class LenientStringConverter : JsonConverter<string?>
{
    public override string? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        reader.TokenType switch
        {
            JsonTokenType.String => reader.GetString(),
            JsonTokenType.Null => null,
            _ => JsonDocument.ParseValue(ref reader).RootElement.GetRawText(),
        };

    public override void Write(Utf8JsonWriter writer, string? value, JsonSerializerOptions options)
    {
        if (value is null) writer.WriteNullValue(); else writer.WriteStringValue(value);
    }
}

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

    /// <summary>Model that made the vector; absent from sidecars of ports that record none.</summary>
    [JsonPropertyName("model")]
    [JsonConverter(typeof(LenientStringConverter))]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Model { get; set; }

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

    // Optional metadata. Version 1 of the payload is unchanged: these trail the
    // array as flat top-level fields, so readers that do not know them ignore
    // them, and absent values are omitted rather than written as null.

    /// <summary>Model name of the encoder in use when the payload was written.</summary>
    [JsonPropertyName("encoder_model")]
    [JsonConverter(typeof(LenientStringConverter))]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? EncoderModel { get; set; }

    /// <summary>Version of the encoder/model that made the vectors.</summary>
    [JsonPropertyName("encoder_version")]
    [JsonConverter(typeof(LenientStringConverter))]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? EncoderVersion { get; set; }

    /// <summary>Digest of the graph these embeddings belong to (see <see cref="IsonGraph.Vector.GraphDigest"/>).</summary>
    [JsonPropertyName("graph_digest")]
    [JsonConverter(typeof(LenientStringConverter))]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? GraphDigest { get; set; }

    /// <summary>Serialization the digest was taken over; a reader that does not know it skips the check.</summary>
    [JsonPropertyName("graph_digest_basis")]
    [JsonConverter(typeof(LenientStringConverter))]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? GraphDigestBasis { get; set; }
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

    /// <summary>
    /// Name of an edge property whose numeric value multiplies the decay of
    /// every hop that crosses that edge (clamped to [0, 1]; a missing,
    /// non-numeric, boolean or NaN value counts as 1.0, so partly-weighted
    /// graphs work, and a weight of 0 zeroes the score without making the node
    /// unreachable). A node is scored by its best route, which with weights can
    /// be a longer one than the fewest-hops route. Null (the default) ignores
    /// edge properties entirely and scores exactly as before.
    /// </summary>
    public string? WeightField { get; set; }

    /// <summary>
    /// Predicate on a node. Combines with <see cref="NodeTypes"/> by AND. See
    /// the filtering rules on <see cref="SemanticGraph.SemanticMultiHop"/>.
    /// A reference with no node in the graph fails whenever this is set.
    /// </summary>
    public Func<Node, bool>? NodeFilter { get; set; }

    /// <summary>Node types allowed. An empty collection passes nothing; null means no type filter.</summary>
    public IEnumerable<string>? NodeTypes { get; set; }

    /// <summary>
    /// Make the filter a hard wall for traversal too: a failing node is never
    /// entered, so every node on every returned path passes. By default a
    /// failing node can still be walked through (it is not returned, but it
    /// costs a hop and its decay, and a returned node's path may list it).
    /// </summary>
    public bool StrictFilter { get; set; }
}

/// <summary>
/// ISONGraph extended with embedding-based semantic search.
///
/// ISONGraph is sealed, so this wraps it by composition. The wrapped graph is
/// private: <see cref="Graph"/> is a read-only view (queries, traversal,
/// serialization), and every mutation - AddNode, UpdateNode, RemoveNode,
/// AddEdge, RemoveEdge - goes through this class, so a node's embedding can
/// never be left stale by a change made behind its back. Nodes and edges that
/// come back are detached copies, for the same reason: changing a returned
/// <c>Node.Properties</c> does not change the graph.
/// </summary>
public sealed class SemanticGraph
{
    private static readonly string[] DefaultEmbedFields = { "name", "description", "content", "text" };

    private readonly ISONGraph _graph;
    private readonly ReadOnlyGraphView _view;
    private readonly EmbeddingStore _store;
    private readonly bool _autoEmbed;
    private readonly List<string> _embedFields;

    /// <param name="name">Graph name.</param>
    /// <param name="encoder">Encoder used to embed nodes and text queries.</param>
    /// <param name="autoEmbed">Embed nodes on add, from the embed fields.</param>
    /// <param name="embedFields">Fields concatenated for auto-embedding.</param>
    /// <param name="directed">Whether edges are directed.</param>
    /// <param name="version">Encoder/model version recorded in the sidecar (default: the encoder's own Version).</param>
    /// <param name="onModelMismatch">What a text search does when the encoder's model differs from the stored vectors' model.</param>
    /// <param name="ann">Opt-in approximate index for the graph's embedding store (see <see cref="EmbeddingStore"/>); default is exact search.</param>
    /// <param name="annNlist">Cells in the index; null picks about 4 * sqrt(N).</param>
    /// <param name="annNprobe">Cells scanned per query; null picks about 5% of nlist.</param>
    /// <param name="annMinSize">Below this many vectors search stays exact.</param>
    /// <param name="annSeed">Seed for the k-means initialisation.</param>
    public SemanticGraph(
        string name = "semantic_graph",
        IEmbeddingEncoder? encoder = null,
        bool autoEmbed = true,
        IEnumerable<string>? embedFields = null,
        bool directed = true,
        string? version = null,
        ModelMismatchMode onModelMismatch = ModelMismatchMode.Warn,
        AnnMode ann = AnnMode.None,
        int? annNlist = null,
        int? annNprobe = null,
        int annMinSize = EmbeddingStore.DefaultAnnMinSize,
        int annSeed = 0)
    {
        _graph = new ISONGraph(name, directed);
        _view = new ReadOnlyGraphView(_graph);
        _store = new EmbeddingStore(encoder, version, onModelMismatch, ann, annNlist, annNprobe, annMinSize, annSeed);
        _autoEmbed = autoEmbed;
        _embedFields = (embedFields ?? DefaultEmbedFields).ToList();
    }

    /// <summary>
    /// Read-only view of the graph: lookups, counts, neighbors, traversals,
    /// paths and ISON serialization. It has no mutating members; change the
    /// graph through this SemanticGraph so embeddings stay in step.
    /// </summary>
    public IReadOnlyGraph Graph => _view;
    public EmbeddingStore EmbeddingStore => _store;
    public IEmbeddingEncoder? Encoder => _store.Encoder;
    public string Name => _graph.Name;

    /// <summary>Warnings raised by the embedding store (see <see cref="EmbeddingStore.Warning"/>).</summary>
    public event Action<string> Warning
    {
        add => _store.Warning += value;
        remove => _store.Warning -= value;
    }

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
        Dictionary<string, object?>? properties = null,
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

        return ReadOnlyGraphView.Copy(node);
    }

    /// <summary>
    /// Add an edge. Edges carry no embedding, so this is a plain pass-through
    /// to the graph; it exists so every mutation goes through the
    /// SemanticGraph. Returns a copy of the new edge. Throws
    /// <see cref="NodeNotFoundError"/> for a missing endpoint and
    /// <see cref="DuplicateEdgeError"/> for an existing edge.
    /// </summary>
    public Edge AddEdge(
        string relType, NodeRef source, NodeRef target,
        IDictionary<string, object?>? properties = null) =>
        ReadOnlyGraphView.Copy(_graph.AddEdge(relType, source, target, properties));

    /// <summary>
    /// Remove an edge. Throws <see cref="EdgeNotFoundError"/> if it does not
    /// exist. (An undirected graph's mirror edge goes with it.)
    /// </summary>
    public void RemoveEdge(string relType, NodeRef source, NodeRef target) =>
        _graph.RemoveEdge(relType, source, target);

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

    /// <summary>
    /// Merge properties into a node, keeping its embedding in step with them.
    ///
    /// With autoEmbed on, a change to any of the embed fields re-embeds the
    /// node from its new properties, so searches never rank it by text it no
    /// longer carries. If the new properties leave nothing to embed, the stale
    /// vector is dropped. Updates that touch no embed field, and graphs with
    /// autoEmbed off (whose vectors the caller manages), leave the embedding
    /// alone.
    ///
    /// <see cref="Graph"/> is read-only, so this is the only way to change a
    /// node's properties and the embedding cannot go stale behind your back.
    /// Returns a copy of the updated node.
    /// </summary>
    public Node UpdateNode(
        string nodeType, string nodeId, Dictionary<string, object?> properties)
    {
        var node = _graph.UpdateNode(nodeType, nodeId, properties);

        if (_autoEmbed && _embedFields.Any(properties.ContainsKey))
        {
            var nodeRef = new NodeRef(nodeType, nodeId);
            var text = EmbedTextFor(node.Properties);
            if (text.Length > 0)
                EmbedNode(nodeRef, text);
            else
                _store.Remove(nodeRef);
        }

        return ReadOnlyGraphView.Copy(node);
    }

    // -- node filtering ----------------------------------------------------------

    private Node? ResolveNode(NodeRef nodeRef)
    {
        try { return ReadOnlyGraphView.Copy(_graph.GetNode(nodeRef)); }
        catch (GraphError) { return null; }
    }

    /// <summary>
    /// Build the ref-to-bool predicate behind nodeFilter/nodeTypes.
    ///
    /// Returns null when neither is set, so unfiltered calls take exactly the
    /// pre-existing code path. A node passes when its type is in nodeTypes (an
    /// empty collection passes nothing) and nodeFilter(node) is true. A ref
    /// with no node in the graph cannot be shown to nodeFilter, so it fails
    /// whenever a nodeFilter is set. Answers are memoized for the one call.
    /// </summary>
    private Func<NodeRef, bool>? NodeGate(Func<Node, bool>? nodeFilter, IEnumerable<string>? nodeTypes)
    {
        if (nodeFilter is null && nodeTypes is null) return null;
        var types = nodeTypes is null ? null : new HashSet<string>(nodeTypes);
        var memo = new Dictionary<NodeRef, bool>();

        return nodeRef =>
        {
            if (memo.TryGetValue(nodeRef, out var cached)) return cached;
            bool ok = types is null || types.Contains(nodeRef.Type);
            if (ok && nodeFilter is not null)
            {
                var node = ResolveNode(nodeRef);
                ok = node is not null && nodeFilter(node);
            }
            memo[nodeRef] = ok;
            return ok;
        };
    }

    /// <summary>
    /// Top-k similarity search among the nodes that pass the gate.
    ///
    /// With a gate the store cannot truncate first: the best k unfiltered
    /// matches may all be filtered out, so every match is scored, filtered, and
    /// only then cut to k.
    ///
    /// With the store's approximate index (<paramref name="ann"/> given) the
    /// candidates are the probed cells only. If fewer than k of them pass the
    /// gate, the probe is widened (4x each step) and the search retried,
    /// ending in the exact scan, so a selective filter costs time rather than
    /// results.
    /// </summary>
    private IReadOnlyList<SimilarityResult> SeedSearch(
        Func<int?, IReadOnlyList<SimilarityResult>> search, int? topK, Func<NodeRef, bool>? gate,
        Func<long, (IEnumerable<SimilarityResult> Ranked, bool Approximate)>? ann = null)
    {
        if (gate is null) return search(topK);
        if (ann is null || topK is not > 0)
        {
            var passing = search(null).Where(r => gate(r.NodeRef)).ToList();
            return topK is null ? passing : passing.Take(Math.Max(topK.Value, 0)).ToList();
        }

        int limit = topK.Value;
        for (long scale = 1; ; scale *= 4)
        {
            var (ranked, approximate) = ann(scale);
            // Best first, and stop as soon as k have passed: the gate (a user
            // callback that may touch the graph) runs on the matches that are
            // needed, not on every stored vector.
            var results = new List<SimilarityResult>();
            foreach (var result in ranked)
            {
                if (!gate(result.NodeRef)) continue;
                results.Add(result);
                if (results.Count >= limit) break;
            }
            if (!approximate || results.Count >= limit) return results;
        }
    }

    /// <summary>The widening search for <see cref="SeedSearch"/>, or null when the store has no index in play.</summary>
    private Func<long, (IEnumerable<SimilarityResult>, bool)>? AnnRanked(
        float[] queryVector, string? nodeType, float threshold, bool exact = false) =>
        _store.Ann == AnnMode.None || exact
            ? null
            : scale => _store.RankedForGate(queryVector, nodeType, threshold, scale);

    /// <summary>
    /// Search by embedding similarity.
    /// </summary>
    /// <param name="query">Query text.</param>
    /// <param name="topK">Number of results (null returns every match).</param>
    /// <param name="nodeType">Keep only nodes of this type.</param>
    /// <param name="threshold">Minimum similarity.</param>
    /// <param name="nodeFilter">
    /// Keep only nodes for which this is true. Applied before topK is taken, so
    /// you still get topK matching nodes when that many exist. An embedding
    /// whose node is not in the graph fails the filter.
    /// </param>
    /// <param name="nodeTypes">
    /// Keep only nodes whose type is in this collection (combines with
    /// nodeType and nodeFilter by AND; empty matches nothing).
    /// </param>
    /// <param name="exact">
    /// Force the full scan even when the graph was created with an approximate
    /// index (<c>ann</c>); no effect otherwise.
    /// </param>
    public IReadOnlyList<SimilarityResult> SimilaritySearch(
        string query, int? topK = 10, string? nodeType = null, float threshold = 0.0f,
        Func<Node, bool>? nodeFilter = null, IEnumerable<string>? nodeTypes = null, bool exact = false)
    {
        var gate = NodeGate(nodeFilter, nodeTypes);
        if (gate is null)
            return _store.SimilaritySearch(query, topK, nodeType, threshold, exact);
        var queryVector = _store.EncodeQuery(query);
        return SeedSearch(
            k => _store.SimilaritySearchVector(queryVector, k, nodeType, threshold, exact), topK, gate,
            AnnRanked(queryVector, nodeType, threshold, exact));
    }

    /// <summary>
    /// Recall@k of the approximate index against the exact scan, over text
    /// queries (see <see cref="EmbeddingStore.RecallCheck(IEnumerable{float[]}, int, string?, float)"/>).
    /// </summary>
    public RecallReport RecallCheck(
        IEnumerable<string> queries, int k = 10, string? nodeType = null, float threshold = 0.0f) =>
        _store.RecallCheck(queries, k, nodeType, threshold);

    /// <summary>Recall@k over query vectors.</summary>
    public RecallReport RecallCheck(
        IEnumerable<float[]> queries, int k = 10, string? nodeType = null, float threshold = 0.0f) =>
        _store.RecallCheck(queries, k, nodeType, threshold);

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

    // -- traversal ---------------------------------------------------------------

    /// <summary>
    /// An edge's numeric weight property clamped to [0, 1].
    ///
    /// A missing, non-numeric (including boolean) or NaN value counts as 1.0,
    /// so an unweighted edge never penalises a hop.
    /// </summary>
    private static float EdgeWeight(Edge edge, string weightField)
    {
        if (!edge.Properties.TryGetValue(weightField, out var raw)) return 1f;
        double value;
        switch (raw)
        {
            case long l: value = l; break;
            case int i: value = i; break;
            case double d: value = d; break;
            case float f: value = f; break;
            case short s: value = s; break;
            case byte b: value = b; break;
            case decimal m: value = (double)m; break;
            default: return 1f;
        }
        if (double.IsNaN(value)) return 1f;
        return (float)Math.Min(1.0, Math.Max(0.0, value));
    }

    /// <summary>(neighbor, connecting edge) pairs, in the order <c>Neighbors</c> yields them.</summary>
    private List<(NodeRef Neighbor, Edge Edge)> Adjacent(NodeRef nodeRef, string? relType, Direction direction)
    {
        var pairs = new List<(NodeRef, Edge)>();
        if (direction is Direction.Out or Direction.Both)
            foreach (var edge in _graph.Edges(relType, source: nodeRef))
                pairs.Add((edge.Target, edge));
        if (direction is Direction.In or Direction.Both)
            foreach (var edge in _graph.Edges(relType, target: nodeRef))
                pairs.Add((edge.Source, edge));
        return pairs;
    }

    /// <summary>
    /// Semantic multi-hop search.
    ///
    /// 1. Find top-k similar nodes as seeds (hop 0)
    /// 2. Traverse the graph following relationships
    /// 3. Score by max(seedSimilarity, 0) * product(decay * weight) over the
    ///    hops (weight is 1 unless <see cref="MultiHopOptions.WeightField"/>
    ///    is set), optionally blended with the relevance of each node itself
    ///
    /// Filtering (<see cref="MultiHopOptions.NodeFilter"/> and
    /// <see cref="MultiHopOptions.NodeTypes"/> combine with AND):
    /// seeds must pass - the top seeds are chosen among passing nodes, so a
    /// filtered-out best match never occupies a seed slot; only passing nodes
    /// are returned; and by default a node that fails can still be walked
    /// through (it costs a hop and its decay, and a returned node's path may
    /// list it) unless <see cref="MultiHopOptions.StrictFilter"/> is set.
    /// </summary>
    public IReadOnlyList<SemanticSearchResult> SemanticMultiHop(string query, MultiHopOptions? options = null)
    {
        options ??= new MultiHopOptions();

        if (options.Blend < 0f || options.Blend > 1f)
            throw new EmbeddingException($"blend must be in [0, 1], got {options.Blend}");
        if (_store.Encoder is null)
            throw new NoEncoderException("No encoder provided for text query");

        var gate = NodeGate(options.NodeFilter, options.NodeTypes);
        var weightField = options.WeightField;

        _store.CheckModelCompat();
        var queryVector = _store.Encoder.Encode(query);

        // Blending needs a score for every node; one scan serves both the
        // seed selection and the per-node relevance.
        IReadOnlyDictionary<string, float> scores = new Dictionary<string, float>();
        if (options.Blend > 0f)
        {
            scores = _store.ScoreMap(queryVector);
            if (scores.Count == 0) return Array.Empty<SemanticSearchResult>();
        }

        // With blend the scan above already scored every node, so the seeds come
        // from the exact scan, never the approximate index (as in the Python port).
        bool exactSeeds = options.Blend > 0f;
        var seeds = SeedSearch(
            k => _store.SimilaritySearchVector(queryVector, k, null, options.Threshold, exactSeeds),
            options.TopKSeeds, gate, AnnRanked(queryVector, null, options.Threshold, exactSeeds));
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

            // Breadth-first traversal. Each node keeps its best decay factor
            // and its fewest hops separately: with edge weights the route with
            // the best factor is not always the shortest one, and a shorter
            // route must still be expanded because it has hops left. Without
            // weights the first (shortest) arrival is both, so this reduces to
            // the plain visited-set BFS.
            var bestFactor = new Dictionary<NodeRef, float> { [seed.NodeRef] = 1f };
            var bestHops = new Dictionary<NodeRef, int> { [seed.NodeRef] = 0 };
            var queue = new Queue<(NodeRef NodeRef, List<NodeRef> Path, float Factor, int Hop)>();
            queue.Enqueue((seed.NodeRef, new List<NodeRef> { seed.NodeRef }, 1f, 0));

            while (queue.Count > 0)
            {
                var (nodeRef, path, factor, hop) = queue.Dequeue();
                if (hop >= options.MaxHops) continue;

                foreach (var (neighbor, edge) in Adjacent(nodeRef, options.RelType, options.Direction))
                {
                    if (path.Contains(neighbor)) continue;
                    if (options.StrictFilter && gate is not null && !gate(neighbor)) continue;

                    int newHop = hop + 1;
                    // Unweighted: decay ^ hops exactly, not a running product,
                    // so scores are bit-identical to the pre-weight releases.
                    float newFactor = weightField is null
                        ? MathF.Pow(options.Decay, newHop)
                        : factor * options.Decay * EdgeWeight(edge, weightField);

                    float knownFactor = bestFactor.TryGetValue(neighbor, out var bf) ? bf : -1f;
                    bool knownHops = bestHops.TryGetValue(neighbor, out var bh);
                    if (newFactor <= knownFactor && newHop >= (knownHops ? bh : newHop + 1))
                        continue;
                    bestFactor[neighbor] = MathF.Max(newFactor, knownFactor);
                    bestHops[neighbor] = knownHops ? Math.Min(newHop, bh) : newHop;

                    var newPath = new List<NodeRef>(path) { neighbor };
                    if (gate is null || gate(neighbor))
                    {
                        var neighborKey = Key(neighbor);
                        float newScore = Combined(traversalBase * newFactor, neighborKey);
                        if (!results.TryGetValue(neighborKey, out var existing) || existing.Score < newScore)
                        {
                            results[neighborKey] = new SemanticSearchResult(
                                neighbor, ResolveNode(neighbor), newScore, newHop, newPath);
                        }
                    }

                    queue.Enqueue((neighbor, newPath, newFactor, newHop));
                }
            }
        }

        var sorted = results.Values.OrderByDescending(r => r.Score).ToList();
        return sorted.Count <= options.TopKResults
            ? sorted
            : sorted.GetRange(0, options.TopKResults);
    }

    /// <summary>Fewest-hops path from start to end whose every node passes the gate.</summary>
    private List<NodeRef>? GatedShortestPath(
        NodeRef start, NodeRef end, string? relType, int maxHops, Direction direction,
        Func<NodeRef, bool> gate)
    {
        if (!gate(start) || !gate(end)) return null;
        if (start == end) return new List<NodeRef> { start };

        var visited = new HashSet<NodeRef> { start };
        var queue = new Queue<List<NodeRef>>();
        queue.Enqueue(new List<NodeRef> { start });
        while (queue.Count > 0)
        {
            var path = queue.Dequeue();
            if (path.Count - 1 >= maxHops) continue;
            foreach (var (neighbor, _) in Adjacent(path[^1], relType, direction))
            {
                if (visited.Contains(neighbor) || !gate(neighbor)) continue;
                var next = new List<NodeRef>(path) { neighbor };
                if (neighbor == end) return next;
                visited.Add(neighbor);
                queue.Enqueue(next);
            }
        }
        return null;
    }

    /// <summary>
    /// Find a semantically-rooted path to a target node.
    ///
    /// Starts from the most similar nodes and scores every seed that reaches
    /// the target as max(seedSimilarity, 0) * decay^hops, returning the best:
    /// a slightly less similar seed one hop from the target beats a better
    /// seed five hops away. Seeds arrive best-first, so a tie goes to the more
    /// similar seed. As in <see cref="SemanticMultiHop"/>, a negative seed
    /// passes no relevance along rather than "decaying" toward zero and
    /// outscoring itself.
    ///
    /// Filtering: seeds are chosen among passing nodes, and the target must
    /// pass or the result is null. By default the path may run through failing
    /// nodes (it is the fewest-hops path, as without a filter); with
    /// <paramref name="strictFilter"/> the path is the fewest-hops path that
    /// uses passing nodes only, which can be longer or not exist.
    /// </summary>
    public SemanticSearchResult? SemanticPath(
        string query,
        NodeRef targetRef,
        string? relType = null,
        int maxHops = 5,
        int topKSeeds = 3,
        Direction direction = Direction.Out,
        float decay = 0.9f,
        float threshold = 0.0f,
        Func<Node, bool>? nodeFilter = null,
        IEnumerable<string>? nodeTypes = null,
        bool strictFilter = false)
    {
        var gate = NodeGate(nodeFilter, nodeTypes);
        if (gate is not null && !gate(targetRef)) return null;

        float[]? pathQuery = null;
        if (gate is not null && _store.Ann != AnnMode.None) pathQuery = _store.EncodeQuery(query);
        var seeds = SeedSearch(
            k => pathQuery is null
                ? _store.SimilaritySearch(query, k, null, threshold)
                : _store.SimilaritySearchVector(pathQuery, k, null, threshold),
            topKSeeds, gate,
            pathQuery is null ? null : AnnRanked(pathQuery, null, threshold));

        SemanticSearchResult? best = null;
        foreach (var seed in seeds)
        {
            List<NodeRef> nodes;
            if (strictFilter && gate is not null)
            {
                var gated = GatedShortestPath(seed.NodeRef, targetRef, relType, maxHops, direction, gate);
                if (gated is null) continue;
                nodes = gated;
            }
            else
            {
                var path = _graph.ShortestPath(seed.NodeRef, targetRef, relType, maxHops, direction);
                if (path is null) continue;
                nodes = path.Nodes;
            }

            int hops = nodes.Count - 1;
            float score = MathF.Max(seed.Score, 0f) * MathF.Pow(decay, hops);
            if (best is null || score > best.Score)
            {
                best = new SemanticSearchResult(
                    targetRef, ResolveNode(targetRef), score, hops, nodes);
            }
        }
        return best;
    }

    // -- subgraph connectivity -----------------------------------------------------

    /// <summary>Order by (type, id) in Unicode code point order, as Python orders the tuple.</summary>
    private static int CompareRefs(NodeRef a, NodeRef b)
    {
        int c = CompareCodePoints(a.Type, b.Type);
        return c != 0 ? c : CompareCodePoints(a.Id, b.Id);
    }

    private static int CompareCodePoints(string a, string b)
    {
        static int Fix(char c) => c >= 0xE000 ? c - 0x800 : c >= 0xD800 ? c + 0x2000 : c;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            if (a[i] == b[i]) continue;
            return Fix(a[i]).CompareTo(Fix(b[i]));
        }
        return a.Length.CompareTo(b.Length);
    }

    /// <summary>
    /// Grow <paramref name="keep"/> with connector nodes until it is as
    /// connected as the hop budget allows.
    ///
    /// Greedy: each round finds, over all pairs of components, the shortest
    /// route (at most maxHops edges, honouring relType/direction) whose interior
    /// nodes lie outside keep, adds its interior, and recomputes components.
    /// Each round merges at least two, so it terminates; it stops when one
    /// component remains or no route fits the budget. The gate (set for
    /// strictFilter) bars failing nodes from serving as connectors.
    /// </summary>
    private void ConnectComponents(
        HashSet<NodeRef> keep, List<NodeRef> ordered, string? relType, Direction direction,
        int maxHops, Func<NodeRef, bool>? gate)
    {
        while (keep.Count > 1)
        {
            // Weakly connected components of the induced subgraph.
            var parent = keep.ToDictionary(r => r, r => r);
            NodeRef Find(NodeRef r)
            {
                while (parent[r] != r)
                {
                    parent[r] = parent[parent[r]];
                    r = parent[r];
                }
                return r;
            }

            foreach (var edge in _graph.Edges(relType))
                if (keep.Contains(edge.Source) && keep.Contains(edge.Target))
                    parent[Find(edge.Source)] = Find(edge.Target);

            var groups = new Dictionary<NodeRef, List<NodeRef>>();
            foreach (var r in keep)
            {
                var root = Find(r);
                if (!groups.TryGetValue(root, out var members)) groups[root] = members = new List<NodeRef>();
                members.Add(r);
            }
            if (groups.Count <= 1) break;

            var ordering = groups.Values.ToList();
            foreach (var g in ordering) g.Sort(CompareRefs);
            ordering.Sort((x, y) => CompareRefs(x[0], y[0]));

            List<NodeRef>? bestRoute = null;
            foreach (var members in ordering)
            {
                var route = ShortestLink(members, keep, relType, direction, maxHops, gate);
                if (route is not null && (bestRoute is null || route.Count < bestRoute.Count))
                    bestRoute = route;
            }
            if (bestRoute is null) break;

            for (int i = 1; i < bestRoute.Count - 1; i++)
            {
                if (keep.Add(bestRoute[i])) ordered.Add(bestRoute[i]);
            }
        }
    }

    /// <summary>Shortest route from any of <paramref name="members"/> to a kept node outside them.</summary>
    private List<NodeRef>? ShortestLink(
        List<NodeRef> members, HashSet<NodeRef> keep, string? relType, Direction direction,
        int maxHops, Func<NodeRef, bool>? gate)
    {
        var visited = new HashSet<NodeRef>(members);
        var queue = new Queue<List<NodeRef>>();
        foreach (var m in members) queue.Enqueue(new List<NodeRef> { m });
        while (queue.Count > 0)
        {
            var path = queue.Dequeue();
            if (path.Count - 1 >= maxHops) continue;
            foreach (var (neighbor, _) in Adjacent(path[^1], relType, direction))
            {
                if (visited.Contains(neighbor)) continue;
                if (keep.Contains(neighbor)) return new List<NodeRef>(path) { neighbor };
                if (gate is not null && !gate(neighbor)) continue;
                if (ResolveNode(neighbor) is null) continue;
                visited.Add(neighbor);
                queue.Enqueue(new List<NodeRef>(path) { neighbor });
            }
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
    ///
    /// The options carry the same weighting and filtering as SemanticMultiHop.
    /// </summary>
    /// <param name="query">Query text.</param>
    /// <param name="options">Same options as SemanticMultiHop.</param>
    /// <param name="name">Name for the returned graph (default: "&lt;name&gt;_subgraph").</param>
    /// <param name="connect">
    /// Also add connector nodes so the selected nodes fall into as few
    /// connected components as possible. Components are weak (edge direction
    /// ignored when judging the result), but connector routes honour the
    /// options' RelType and Direction. Each round adds the interior of the
    /// shortest route between two components (greedy; not a minimum Steiner
    /// tree), until one component remains or no route fits
    /// <paramref name="connectMaxHops"/>. Nothing but those interior nodes is
    /// added, and they are not marked in any way. With connect the result can
    /// hold more than TopKResults nodes. Without StrictFilter a connector may
    /// be a node that fails the filter (it is only a bridge); with it,
    /// connectors must pass too.
    /// </param>
    /// <param name="connectMaxHops">
    /// Longest route, counted in edges, that may join two components. The
    /// default 2 allows one connector node per join; 3 allows two. A direct
    /// edge needs no connector (it is already an induced edge). Must be at
    /// least 1.
    /// </param>
    public SemanticGraph SemanticSubgraph(
        string query, MultiHopOptions? options = null, string? name = null,
        bool connect = false, int connectMaxHops = 2)
    {
        if (connectMaxHops < 1)
            throw new ArgumentOutOfRangeException(
                nameof(connectMaxHops), connectMaxHops, $"connectMaxHops must be >= 1, got {connectMaxHops}");

        options ??= new MultiHopOptions();
        var results = SemanticMultiHop(query, options);

        var ordered = results.Select(r => r.NodeRef).ToList();
        var keep = ordered.ToHashSet();
        if (connect && keep.Count > 0)
        {
            var gate = options.StrictFilter ? NodeGate(options.NodeFilter, options.NodeTypes) : null;
            ConnectComponents(keep, ordered, options.RelType, options.Direction, connectMaxHops, gate);
        }

        var sub = new SemanticGraph(
            name ?? $"{_graph.Name}_subgraph",
            _store.Encoder,
            autoEmbed: false,
            embedFields: _embedFields,
            directed: _graph.Directed,
            version: null,
            onModelMismatch: _store.OnModelMismatch);

        foreach (var nodeRef in ordered)
        {
            var node = ResolveNode(nodeRef);
            if (node is null) continue;

            sub._graph.AddNode(node.Type, node.Id, node.Properties);
            var record = _store.Get(nodeRef);
            if (record is not null)
                sub._store.AddRecord(record);   // keeps the model that made the vector
        }

        foreach (var edge in _graph.Edges(options.RelType))
        {
            if (keep.Contains(edge.Source) && keep.Contains(edge.Target))
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
            var text = textFn(ReadOnlyGraphView.Copy(node));
            if (!string.IsNullOrEmpty(text)) batch.Add((nodeRef, text));
        }

        return EmbedNodes(batch);
    }

    // -- model lifecycle -----------------------------------------------------------

    /// <summary>
    /// Nodes whose stored vector was not made by the current encoder.
    ///
    /// Stale means the vector's recorded model differs from the encoder's
    /// ModelName, or its length differs from the encoder's Dimension. Vectors
    /// recorded as "unknown"/"provided" (from ports that store no model, or
    /// supplied without an encoder) cannot be judged by model and are left out
    /// unless <paramref name="includeUnknown"/> is true; a dimension mismatch
    /// always counts.
    /// </summary>
    public IReadOnlyList<NodeRef> StaleEmbeddings(bool includeUnknown = false) =>
        _store.Stale(includeUnknown: includeUnknown);

    /// <summary>
    /// Re-encode every stale embedding with the current encoder.
    ///
    /// Each vector is rebuilt from the text it was originally embedded from (so
    /// a custom embedText survives); a vector stored with no text falls back to
    /// the node's embed fields. If the encoder's dimension differs from the
    /// store's, every stored vector is stale, so the store is emptied, its
    /// pinned dimension reset, and everything re-embedded.
    /// </summary>
    /// <param name="includeUnknown">Also re-embed vectors of unknown provenance.</param>
    /// <param name="batchSize">Texts per encoder call.</param>
    /// <returns>Number of embeddings rewritten.</returns>
    public int ReembedStale(bool includeUnknown = false, int batchSize = 256)
    {
        if (batchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "batchSize must be >= 1");

        var stale = StaleEmbeddings(includeUnknown);
        if (stale.Count == 0) return 0;

        var encoder = _store.Encoder ?? throw new NoEncoderException("No encoder provided to re-embed with");

        var pairs = new List<(NodeRef NodeRef, string Text)>();
        foreach (var nodeRef in stale)
        {
            var record = _store.Get(nodeRef);
            if (record is null) continue;
            var text = record.Text;
            if (string.IsNullOrEmpty(text))
            {
                var node = ResolveNode(nodeRef);
                text = node is null ? "" : EmbedTextFor(node.Properties);
                if (text.Length == 0) text = $"{nodeRef.Type}:{nodeRef.Id}";
            }
            pairs.Add((nodeRef, text));
        }

        if (_store.Dimension is not null && _store.Dimension != encoder.Dimension)
            _store.Clear(resetDimension: true);

        int count = 0;
        for (int start = 0; start < pairs.Count; start += batchSize)
            count += EmbedNodes(pairs.GetRange(start, Math.Min(batchSize, pairs.Count - start)));
        return count;
    }

    // -- persistence ---------------------------------------------------------------

    /// <summary>Sidecar path holding the embeddings for a saved graph.</summary>
    public static string EmbeddingsPathFor(string path) =>
        path + IsonGraphVectorVersion.EmbeddingSidecarSuffix;

    private static bool IsIsonl(string path) => path.EndsWith(".isonl", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Save the graph to <paramref name="path"/> and its embeddings alongside.
    ///
    /// The ISON graph format carries nodes and edges only, so embeddings go to
    /// a JSON sidecar at <c>&lt;path&gt;.embeddings.json</c> (the
    /// ison-embeddings payload every port reads and writes) that
    /// <see cref="Load"/> picks up automatically. Without it a save/load
    /// round-trip silently returns a graph whose searches all come back empty.
    ///
    /// Both files are written atomically (temp file, flush, move), so a crash
    /// never leaves a truncated file. The pair cannot be replaced as one unit,
    /// so the sidecar also records a <see cref="GraphDigest"/> of the graph it
    /// belongs to; <see cref="Load"/> checks it and catches a crash between the
    /// two writes, a sidecar copied from another save, and a graph edited
    /// afterwards. The digest is taken of the graph as it reads back from the
    /// text just written, so a round trip that normalises a value does not
    /// cause a false mismatch.
    ///
    /// With <paramref name="embeddings"/> false an existing sidecar is left in
    /// place (and will fail the digest check on load, as it no longer matches).
    ///
    /// With <paramref name="snapshot"/> true a binary snapshot of the
    /// embeddings is also written to <c>&lt;path&gt;.embeddings.snap</c>, after
    /// and in addition to the JSON sidecar, which stays the portable copy.
    /// <see cref="Load"/> uses the snapshot when it provably belongs to the
    /// graph and sidecar on disk (and its checksum verifies), and the JSON
    /// otherwise; see <see cref="LoadWithReport"/> to learn which. A save
    /// without a snapshot deletes any existing one. If the snapshot cannot be
    /// written the graph and sidecar are saved anyway, a stale snapshot is
    /// removed, and the error is thrown.
    /// </summary>
    /// <param name="path">Graph file path (.isonl for ISONL, otherwise ISON).</param>
    /// <param name="embeddings">Write the embeddings sidecar.</param>
    /// <param name="snapshot">Also write the binary snapshot (needs <paramref name="embeddings"/>).</param>
    public void Save(string path, bool embeddings = true, bool snapshot = false)
    {
        bool isonl = IsIsonl(path);
        var text = isonl ? _graph.ToIsonl() : _graph.ToIson();
        AtomicFile.WriteText(path, text);

        if (!embeddings) return;
        var digest = GraphDigest.OfText(text, isonl);
        var sidecar = EmbeddingsPathFor(path);
        var snap = SnapshotPathFor(path);
        if (!snapshot)
            DeleteQuietly(snap);        // the sidecar is about to change: a snapshot of the old one is stale
        _store.SaveEmbeddings(sidecar, digest);
        if (!snapshot) return;
        try
        {
            _store.SaveSnapshot(snap, new SnapshotIdentity(digest, SidecarId.OfFile(sidecar)));
        }
        catch
        {
            DeleteQuietly(snap);        // the pair is saved; a stale snapshot must not survive
            throw;
        }
    }

    private static void DeleteQuietly(string path)
    {
        try { File.Delete(path); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    /// <summary>
    /// Load a graph from <paramref name="path"/>, restoring its embeddings from
    /// the sidecar written by <see cref="Save"/> (or by any other port).
    ///
    /// Nodes are loaded without auto-embedding, so the saved vectors are kept
    /// exactly rather than re-derived. Pass the encoder the graph was saved
    /// with so later text queries land in the same vector space.
    ///
    /// The ISON file format stores nodes and edges only: the graph name comes
    /// back from the file name and the graph comes back directed (an
    /// undirected graph's mirrored edges are stored, so its edge set is kept).
    /// This matches the Python port.
    /// </summary>
    /// <param name="path">Graph file path (.isonl for ISONL, otherwise ISON).</param>
    /// <param name="encoder">Encoder for the loaded graph.</param>
    /// <param name="embeddings">Restore the sidecar written by Save.</param>
    /// <param name="autoEmbed">Auto-embed nodes added to the loaded graph.</param>
    /// <param name="embedFields">Fields concatenated for auto-embedding.</param>
    /// <param name="onMismatch">
    /// What to do when the sidecar carries a graph digest that does not match
    /// the graph file (a sidecar from a different save, or a graph edited
    /// since). A sidecar without a digest, or with a digest of a basis this
    /// build does not know, is loaded as before and never checked.
    /// </param>
    /// <param name="version">
    /// Encoder/model version for the loaded graph's store. If both it (or the
    /// encoder's own Version) and the sidecar's encoder_version are set and
    /// differ, a warning is raised.
    /// </param>
    /// <param name="onModelMismatch">
    /// What a text search does when the encoder's model differs from the stored
    /// vectors' model (see <see cref="EmbeddingStore"/>).
    /// </param>
    /// <param name="onWarning">
    /// Receives warnings raised while loading and afterwards; null sends them to
    /// standard error.
    /// </param>
    /// <param name="ann">
    /// Opt-in approximate index for the loaded graph's store. Nothing about it
    /// is saved or loaded: the index is built lazily from the loaded vectors.
    /// </param>
    /// <param name="annNlist">Cells in the index; null picks about 4 * sqrt(N).</param>
    /// <param name="annNprobe">Cells scanned per query; null picks about 5% of nlist.</param>
    /// <param name="annMinSize">Below this many vectors search stays exact.</param>
    /// <param name="annSeed">Seed for the k-means initialisation.</param>
    /// <exception cref="GraphEmbeddingMismatchException">Digest mismatch with <see cref="GraphMismatchMode.Error"/>.</exception>
    public static SemanticGraph Load(
        string path,
        IEmbeddingEncoder? encoder = null,
        bool embeddings = true,
        bool autoEmbed = true,
        IEnumerable<string>? embedFields = null,
        GraphMismatchMode onMismatch = GraphMismatchMode.Error,
        string? version = null,
        ModelMismatchMode onModelMismatch = ModelMismatchMode.Warn,
        Action<string>? onWarning = null,
        AnnMode ann = AnnMode.None,
        int? annNlist = null,
        int? annNprobe = null,
        int annMinSize = EmbeddingStore.DefaultAnnMinSize,
        int annSeed = 0) =>
        LoadWithReport(path, encoder, embeddings, autoEmbed, embedFields, onMismatch, version,
            onModelMismatch, onWarning, ann, annNlist, annNprobe, annMinSize, annSeed).Graph;

    /// <summary>Binary snapshot path for a saved graph (<c>&lt;path&gt;.embeddings.snap</c>).</summary>
    public static string SnapshotPathFor(string path) =>
        path + IsonGraphVectorVersion.EmbeddingSnapshotSuffix;

    /// <summary>
    /// <see cref="Load"/>, also saying where the embeddings came from.
    ///
    /// If a binary snapshot (<c>&lt;path&gt;.embeddings.snap</c>, written by
    /// <see cref="Save"/> with <c>snapshot: true</c>) sits beside the sidecar,
    /// it is used when its header names this graph's digest and the sidecar
    /// file as it is now and its checksum verifies; the result is identical to
    /// loading the JSON. Otherwise the JSON sidecar is read, exactly as before,
    /// and <see cref="LoadReport.SnapshotSkipped"/> says why the snapshot was
    /// not used (a snapshot that fails any check never changes the store).
    /// Parameters are those of <see cref="Load"/>.
    /// </summary>
    public static (SemanticGraph Graph, LoadReport Report) LoadWithReport(
        string path,
        IEmbeddingEncoder? encoder = null,
        bool embeddings = true,
        bool autoEmbed = true,
        IEnumerable<string>? embedFields = null,
        GraphMismatchMode onMismatch = GraphMismatchMode.Error,
        string? version = null,
        ModelMismatchMode onModelMismatch = ModelMismatchMode.Warn,
        Action<string>? onWarning = null,
        AnnMode ann = AnnMode.None,
        int? annNlist = null,
        int? annNprobe = null,
        int annMinSize = EmbeddingStore.DefaultAnnMinSize,
        int annSeed = 0)
    {
        if (!Enum.IsDefined(onMismatch))
            throw new ArgumentOutOfRangeException(nameof(onMismatch), onMismatch, "unknown GraphMismatchMode");

        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        LoadReport Report(LoadSource source, string? skipped, int count) =>
            new(source, skipped, count, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalSeconds);

        bool isonl = IsIsonl(path);
        var graphText = File.ReadAllText(path, System.Text.Encoding.UTF8);
        var name = System.IO.Path.GetFileNameWithoutExtension(path);
        var loaded = isonl ? ISONGraph.FromIsonl(graphText, name) : ISONGraph.FromIson(graphText, name);

        var graph = new SemanticGraph(
            loaded.Name, encoder, autoEmbed, embedFields, loaded.Directed, version, onModelMismatch,
            ann, annNlist, annNprobe, annMinSize, annSeed);
        if (onWarning is not null) graph._store.Warning += onWarning;

        foreach (var node in loaded.Nodes())
            graph._graph.AddNode(node.Type, node.Id, node.Properties);
        foreach (var edge in loaded.Edges())
            graph._graph.AddEdge(edge.RelType, edge.Source, edge.Target, edge.Properties);

        if (!embeddings) return (graph, Report(LoadSource.NoEmbeddings, null, 0));

        var sidecar = EmbeddingsPathFor(path);
        if (!File.Exists(sidecar)) return (graph, Report(LoadSource.NoEmbeddings, null, 0));

        string? snapshotSkipped = null;
        var snap = SnapshotPathFor(path);
        if (File.Exists(snap))
        {
            if (graph._store.TryAdoptSnapshot(snap, sidecar, GraphDigest.OfText(graphText, isonl), out var info, out snapshotSkipped))
            {
                graph.WarnOnVersionSkew();
                return (graph, Report(LoadSource.Snapshot, null, info!.Count));
            }
        }

        ExportedEmbeddings? payload;
        try
        {
            payload = EmbeddingStore.DeserializePayload(sidecar);
        }
        catch (JsonException ex)
        {
            throw new EmbeddingException($"unreadable embeddings sidecar {sidecar}: {ex.Message}");
        }
        if (payload is null)
            throw new EmbeddingException($"unreadable embeddings sidecar {sidecar}: empty payload");

        string? problem = null;
        var recorded = payload.GraphDigest;
        if (recorded is not null &&
            (payload.GraphDigestBasis ?? IsonGraphVectorVersion.GraphDigestBasis) ==
                IsonGraphVectorVersion.GraphDigestBasis)
        {
            var actual = GraphDigest.OfText(graphText, isonl);
            if (recorded != actual)
            {
                problem =
                    $"embeddings sidecar {sidecar} does not belong to graph {path}: it was " +
                    $"written for graph digest {recorded}, but the graph reads as {actual} " +
                    "(the sidecar is from a different save, or the graph was edited since). " +
                    "Pass onMismatch: GraphMismatchMode.Reembed to rebuild the vectors or " +
                    "GraphMismatchMode.Warn to use them anyway.";
            }
        }

        if (problem is not null && onMismatch == GraphMismatchMode.Error)
            throw new GraphEmbeddingMismatchException(problem);

        if (problem is not null && onMismatch == GraphMismatchMode.Reembed)
        {
            graph.EmbedAllNodes();
            return (graph, Report(LoadSource.Json, snapshotSkipped, graph._store.Count));
        }

        if (problem is not null) graph._store.RaiseWarning(problem);
        graph._store.ImportEmbeddings(payload);
        graph.WarnOnVersionSkew();
        return (graph, Report(LoadSource.Json, snapshotSkipped, graph._store.Count));
    }

    private void WarnOnVersionSkew()
    {
        var storedVersion = _store.LastImportMeta.GetValueOrDefault("encoder_version");
        var currentVersion = _store.Version;
        if (storedVersion is not null && currentVersion is not null && storedVersion != currentVersion)
        {
            _store.RaiseWarning(
                $"embeddings were made by encoder version '{storedVersion}', " +
                $"but the encoder in use is version '{currentVersion}'");
        }
    }

    public override string ToString() =>
        $"SemanticGraph(name={_graph.Name}, nodes={_graph.NodeCount()}, " +
        $"edges={_graph.EdgeCount()}, embeddings={_store.Count})";
}
