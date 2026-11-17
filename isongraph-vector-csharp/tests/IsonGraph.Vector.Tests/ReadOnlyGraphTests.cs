using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>
/// SemanticGraph.Graph is a read-only view, so a node cannot be changed
/// without going through the SemanticGraph (and its re-embedding).
/// </summary>
public class ReadOnlyGraphTests
{
    private static NodeRef P(string id) => new("person", id);

    private static SemanticGraph Build()
    {
        var g = new SemanticGraph("g", new MockEncoder(16));
        g.AddNode("person", "1", new() { ["name"] = "Alice" });
        g.AddNode("person", "2", new() { ["name"] = "Bob" });
        g.AddNode("person", "3", new() { ["name"] = "Carol" });
        g.AddEdge("KNOWS", P("1"), P("2"));
        g.AddEdge("KNOWS", P("2"), P("3"));
        return g;
    }

    [Fact]
    public void GraphIsAViewWithNoMutatingMembers()
    {
        var g = Build();
        Assert.IsNotType<ISONGraph>(g.Graph);
        Assert.False(((object)g.Graph) is ISONGraph);

        var names = typeof(IReadOnlyGraph).GetMethods().Select(m => m.Name).ToHashSet();
        foreach (var banned in new[] { "AddNode", "UpdateNode", "RemoveNode", "AddEdge", "RemoveEdge", "Save", "Start" })
            Assert.DoesNotContain(banned, names);
        Assert.Null(typeof(IReadOnlyGraph).GetProperty("Name")!.SetMethod);
    }

    [Fact]
    public void ViewIsLive()
    {
        var g = Build();
        var view = g.Graph;
        Assert.Equal(3, view.NodeCount());
        g.AddNode("person", "4", new() { ["name"] = "Dan" });
        g.RemoveNode("person", "1");
        Assert.Equal(3, view.NodeCount());
        Assert.False(view.HasNode("person", "1"));
        Assert.True(view.HasNode(P("4")));
        Assert.Equal("g", view.Name);
        Assert.True(view.Directed);
    }

    [Fact]
    public void ReturnedNodesAreDetachedCopies()
    {
        var g = Build();
        var before = g.GetEmbedding(P("1"))!;

        g.Graph.GetNode("person", "1").Properties["name"] = "Mallory";
        g.Graph.GetNode(P("1")).Properties["name"] = "Mallory";
        foreach (var n in g.Graph.Nodes()) n.Properties["name"] = "Mallory";
        foreach (var n in g.Graph.Nodes("person")) n.Properties["name"] = "Mallory";

        Assert.Equal("Alice", g.Graph.GetNode("person", "1").Properties["name"]);
        Assert.Equal("Alice", g.GetEmbedding(P("1"))!.Text);
        Assert.Equal(before.Vector, g.GetEmbedding(P("1"))!.Vector);   // the store hands out copies, so equal content, not the same instance
        Assert.Equal(before.Text, g.GetEmbedding(P("1"))!.Text);
    }

    [Fact]
    public void NodesReturnedFromMutatorsAreDetachedCopies()
    {
        var g = new SemanticGraph("g", new MockEncoder(16));
        var added = g.AddNode("person", "1", new() { ["name"] = "Alice" });
        added.Properties["name"] = "Mallory";
        Assert.Equal("Alice", g.Graph.GetNode("person", "1").Properties["name"]);

        var updated = g.UpdateNode("person", "1", new() { ["name"] = "Alicia" });
        updated.Properties["name"] = "Mallory";
        Assert.Equal("Alicia", g.Graph.GetNode("person", "1").Properties["name"]);
        Assert.Equal("Alicia", g.GetEmbedding(P("1"))!.Text);
    }

    [Fact]
    public void CallbacksAndResultsSeeCopiesToo()
    {
        var g = Build();

        // nodeFilter
        g.SimilaritySearch("Alice", nodeFilter: n => { n.Properties["name"] = "Mallory"; return true; });
        // EmbedAllNodes text function
        g.EmbedAllNodes(n => { n.Properties["name"] = "Mallory"; return "x"; });
        // Traverse filter on the view
        g.Graph.Traverse(P("1"), new[] { ("KNOWS", Direction.Out) },
            n => { n.Properties["name"] = "Mallory"; return true; });
        // semantic multi-hop results
        foreach (var r in g.SemanticMultiHop("Alice"))
            r.Node?.Properties.Clear();

        foreach (var n in g.Graph.Nodes())
            Assert.NotEqual("Mallory", n.Properties["name"]);
        Assert.Equal(3, g.Graph.Nodes().Count(n => n.Properties.Count == 1));
    }

    [Fact]
    public void ReturnedEdgesAndPathsAreDetachedCopies()
    {
        var g = Build();
        g.AddEdge("LIKES", P("1"), P("3"), new Dictionary<string, object?> { ["w"] = 1L });

        g.Graph.GetEdge("LIKES", P("1"), P("3")).Properties["w"] = 99L;
        foreach (var e in g.Graph.Edges()) e.Properties["w"] = 99L;
        g.Graph.ShortestPath(P("1"), P("3"), "LIKES")!.Edges[0].Properties["w"] = 99L;
        g.Graph.AllPaths(P("1"), P("3"))[0].Edges[0].Properties["w"] = 99L;

        Assert.Equal(1L, g.Graph.GetEdge("LIKES", P("1"), P("3")).Properties["w"]);
    }

    [Fact]
    public void QueriesAndSerializationWorkThroughTheView()
    {
        var g = Build();
        var v = g.Graph;
        Assert.Equal(new[] { P("2") }, v.Neighbors(P("1"), "KNOWS"));
        Assert.Equal(new[] { P("3") }, v.MultiHop(P("1"), "KNOWS", 2));
        Assert.Equal(2, v.EdgeCount());
        Assert.Equal(2, v.EdgeCount("KNOWS"));
        Assert.Equal(new[] { "person" }, v.NodeTypes());
        Assert.Equal(new[] { "KNOWS" }, v.EdgeTypes());
        Assert.True(v.HasEdge("KNOWS", P("1"), P("2")));
        Assert.True(v.PathExists(P("1"), P("3")));
        Assert.Equal(2, v.ShortestPath(P("1"), P("3"))!.Length);
        Assert.Equal(1, v.OutDegree(P("1")));
        Assert.Equal(2, v.Degree(P("2")));
        Assert.Single(v.ConnectedComponents());
        Assert.False(v.HasCycle());
        Assert.NotEmpty(v.Query(":person:1 -[:KNOWS]-> *"));
        Assert.Contains("Alice", v.ToIson());
        Assert.Contains("Alice", v.ToIsonl());
        Assert.Throws<NodeNotFoundError>(() => v.GetNode("person", "nope"));
    }

    [Fact]
    public void AddEdgeAndRemoveEdgeGoThroughTheSemanticGraph()
    {
        var g = Build();
        var edge = g.AddEdge("LIKES", P("1"), P("3"), new Dictionary<string, object?> { ["w"] = 2L });
        Assert.Equal("LIKES", edge.RelType);
        Assert.Equal(2L, edge.Properties["w"]);
        Assert.Equal(3, g.Graph.EdgeCount());

        Assert.Throws<DuplicateEdgeError>(() => g.AddEdge("LIKES", P("1"), P("3")));
        Assert.Throws<NodeNotFoundError>(() => g.AddEdge("LIKES", P("1"), P("404")));

        g.RemoveEdge("LIKES", P("1"), P("3"));
        Assert.False(g.Graph.HasEdge("LIKES", P("1"), P("3")));
        Assert.Throws<EdgeNotFoundError>(() => g.RemoveEdge("LIKES", P("1"), P("3")));

        // edges feed traversal, so the new edge is visible to semantic multi-hop
        g.AddEdge("LIKES", P("3"), P("1"));
        Assert.Contains(P("1"), g.Graph.Neighbors(P("3"), "LIKES"));
    }

    [Fact]
    public void RemovingANodeDropsItsEmbeddingAndEdges()
    {
        var g = Build();
        g.RemoveNode("person", "2");
        Assert.Null(g.GetEmbedding(P("2")));
        Assert.Equal(0, g.Graph.EdgeCount());
    }

    [Fact]
    public void DigestOfViewMatchesDigestOfText()
    {
        var g = Build();
        Assert.Equal(GraphDigest.OfText(g.Graph.ToIson()), GraphDigest.Compute(g.Graph, _ => false)); // string ids are written quoted
    }

    [Fact]
    public void LoadedGraphIsAlsoReadOnly()
    {
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"ro-{Guid.NewGuid():N}.ison");
        try
        {
            var g = Build();
            g.Save(path);
            var loaded = SemanticGraph.Load(path, new MockEncoder(16));
            loaded.Graph.GetNode(P("1")).Properties["name"] = "Mallory";
            Assert.Equal("Alice", loaded.Graph.GetNode(P("1")).Properties["name"]);
            Assert.Equal(2, loaded.Graph.EdgeCount());
        }
        finally
        {
            foreach (var f in new[] { path, SemanticGraph.EmbeddingsPathFor(path) })
                if (File.Exists(f)) File.Delete(f);
        }
    }
}
