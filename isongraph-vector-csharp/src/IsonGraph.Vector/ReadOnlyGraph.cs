// Read-only view over an ISONGraph.
//
// SemanticGraph keeps node embeddings in step with node properties. ISONGraph
// is sealed and hands out its live Node objects (whose Properties dictionary is
// mutable), so exposing it directly lets callers change a node behind the
// embedding store's back. This view closes that hole: it offers every query,
// traversal and serialization member of ISONGraph, none of the mutating ones,
// and returns detached copies of nodes and edges rather than the live objects.

using IsonGraph;

namespace IsonGraph.Vector;

/// <summary>
/// A read-only window onto a <see cref="SemanticGraph"/>'s graph: lookups,
/// counts, neighbors, traversals, paths, analysis and serialization.
///
/// Nodes, edges and paths returned from it are detached copies: changing a
/// returned <see cref="Node.Properties"/> or <see cref="Edge.Properties"/>
/// bag does not change the graph. Mutate through <see cref="SemanticGraph"/>
/// (AddNode, UpdateNode, RemoveNode, AddEdge, RemoveEdge), which keeps the
/// embeddings in step. The view is live: it reflects later changes made
/// through the SemanticGraph.
///
/// Not exposed (use the SemanticGraph member instead): AddNode, UpdateNode,
/// RemoveNode, AddEdge, RemoveEdge, the Name setter, Save/Load (use
/// SemanticGraph.Save/Load, which also writes the embeddings sidecar) and
/// Start (its GraphTraversal hands out live nodes; use Neighbors, MultiHop or
/// Traverse).
/// </summary>
public interface IReadOnlyGraph
{
    string Name { get; }
    bool Directed { get; }

    // -- nodes ---------------------------------------------------------------
    /// <summary>Get a copy of a node. Throws <see cref="NodeNotFoundError"/>.</summary>
    Node GetNode(string nodeType, string nodeId);
    Node GetNode(NodeRef nodeRef);
    bool HasNode(string nodeType, string nodeId);
    bool HasNode(NodeRef nodeRef);
    /// <summary>Copies of the nodes, optionally of one type, in insertion order.</summary>
    IEnumerable<Node> Nodes(string? nodeType = null);
    int NodeCount(string? nodeType = null);
    List<string> NodeTypes();

    // -- edges ---------------------------------------------------------------
    bool HasEdge(string relType, NodeRef source, NodeRef target);
    /// <summary>Get a copy of an edge. Throws <see cref="EdgeNotFoundError"/>.</summary>
    Edge GetEdge(string relType, NodeRef source, NodeRef target);
    /// <summary>Copies of the edges matching the optional filters.</summary>
    IEnumerable<Edge> Edges(string? relType = null, NodeRef? source = null, NodeRef? target = null);
    int EdgeCount(string? relType = null);
    List<string> EdgeTypes();

    // -- traversal -----------------------------------------------------------
    List<NodeRef> Neighbors(NodeRef nodeRef, string? relType = null, Direction direction = Direction.Out);
    List<NodeRef> MultiHop(NodeRef start, string? relType = null, int hops = 1, Direction direction = Direction.Out);
    List<NodeRef> MultiHopRange(NodeRef start, string? relType = null, int minHops = 1, int maxHops = 3,
                                Direction direction = Direction.Out);
    /// <summary>Follow (relType, direction) steps; <paramref name="filter"/> sees copies of nodes.</summary>
    List<NodeRef> Traverse(NodeRef start, IEnumerable<(string RelType, Direction Direction)> pattern,
                           Func<Node, bool>? filter = null);
    IsonGraph.Path? ShortestPath(NodeRef start, NodeRef end, string? relType = null, int maxHops = 10,
                                 Direction direction = Direction.Out);
    List<IsonGraph.Path> AllPaths(NodeRef start, NodeRef end, string? relType = null, int maxHops = 5,
                                  Direction direction = Direction.Out);
    bool PathExists(NodeRef start, NodeRef end, string? relType = null, int maxHops = 10);

    // -- analysis ------------------------------------------------------------
    int InDegree(NodeRef nodeRef);
    int OutDegree(NodeRef nodeRef);
    int Degree(NodeRef nodeRef);
    bool IsConnected();
    bool HasCycle(string? relType = null);
    List<HashSet<NodeRef>> ConnectedComponents();
    List<NodeRef> Query(string pattern);

    // -- serialization -------------------------------------------------------
    /// <summary>The graph as canonical ISON text (no embeddings; see SemanticGraph.Save).</summary>
    string ToIson();
    /// <summary>The graph as canonical ISONL text.</summary>
    string ToIsonl();
}

internal sealed class ReadOnlyGraphView : IReadOnlyGraph
{
    private readonly ISONGraph _g;

    public ReadOnlyGraphView(ISONGraph graph) => _g = graph;

    internal static Node Copy(Node n) => new(n.Type, n.Id, n.Properties);
    internal static Edge Copy(Edge e) => new(e.RelType, e.Source, e.Target, e.Properties);
    private static IsonGraph.Path Copy(IsonGraph.Path p) =>
        new(new List<NodeRef>(p.Nodes), p.Edges.Select(Copy).ToList());

    public string Name => _g.Name;
    public bool Directed => _g.Directed;

    public Node GetNode(string nodeType, string nodeId) => Copy(_g.GetNode(nodeType, nodeId));
    public Node GetNode(NodeRef nodeRef) => Copy(_g.GetNode(nodeRef));
    public bool HasNode(string nodeType, string nodeId) => _g.HasNode(nodeType, nodeId);
    public bool HasNode(NodeRef nodeRef) => _g.HasNode(nodeRef);
    public IEnumerable<Node> Nodes(string? nodeType = null) => _g.Nodes(nodeType).Select(Copy);
    public int NodeCount(string? nodeType = null) => _g.NodeCount(nodeType);
    public List<string> NodeTypes() => _g.NodeTypes();

    public bool HasEdge(string relType, NodeRef source, NodeRef target) => _g.HasEdge(relType, source, target);
    public Edge GetEdge(string relType, NodeRef source, NodeRef target) => Copy(_g.GetEdge(relType, source, target));
    public IEnumerable<Edge> Edges(string? relType = null, NodeRef? source = null, NodeRef? target = null) =>
        _g.Edges(relType, source, target).Select(Copy);
    public int EdgeCount(string? relType = null) => _g.EdgeCount(relType);
    public List<string> EdgeTypes() => _g.EdgeTypes();

    public List<NodeRef> Neighbors(NodeRef nodeRef, string? relType = null, Direction direction = Direction.Out) =>
        _g.Neighbors(nodeRef, relType, direction);
    public List<NodeRef> MultiHop(NodeRef start, string? relType = null, int hops = 1, Direction direction = Direction.Out) =>
        _g.MultiHop(start, relType, hops, direction);
    public List<NodeRef> MultiHopRange(NodeRef start, string? relType = null, int minHops = 1, int maxHops = 3,
                                       Direction direction = Direction.Out) =>
        _g.MultiHopRange(start, relType, minHops, maxHops, direction);
    public List<NodeRef> Traverse(NodeRef start, IEnumerable<(string RelType, Direction Direction)> pattern,
                                  Func<Node, bool>? filter = null) =>
        _g.Traverse(start, pattern, filter is null ? null : n => filter(Copy(n)));
    public IsonGraph.Path? ShortestPath(NodeRef start, NodeRef end, string? relType = null, int maxHops = 10,
                                        Direction direction = Direction.Out) =>
        _g.ShortestPath(start, end, relType, maxHops, direction) is { } p ? Copy(p) : null;
    public List<IsonGraph.Path> AllPaths(NodeRef start, NodeRef end, string? relType = null, int maxHops = 5,
                                         Direction direction = Direction.Out) =>
        _g.AllPaths(start, end, relType, maxHops, direction).Select(Copy).ToList();
    public bool PathExists(NodeRef start, NodeRef end, string? relType = null, int maxHops = 10) =>
        _g.PathExists(start, end, relType, maxHops);

    public int InDegree(NodeRef nodeRef) => _g.InDegree(nodeRef);
    public int OutDegree(NodeRef nodeRef) => _g.OutDegree(nodeRef);
    public int Degree(NodeRef nodeRef) => _g.Degree(nodeRef);
    public bool IsConnected() => _g.IsConnected();
    public bool HasCycle(string? relType = null) => _g.HasCycle(relType);
    public List<HashSet<NodeRef>> ConnectedComponents() => _g.ConnectedComponents();
    public List<NodeRef> Query(string pattern) => _g.Query(pattern);

    public string ToIson() => _g.ToIson();
    public string ToIsonl() => _g.ToIsonl();

    public override string ToString() => _g.ToString();
}
