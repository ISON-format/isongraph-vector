using System.Text.Json;
using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>Encoder that maps every text to the same unit vector, so scores are exact.</summary>
internal sealed class UnitEncoder : IEmbeddingEncoder
{
    public UnitEncoder(int dimension = 4) => Dimension = dimension;
    public int Dimension { get; }
    public float[] Encode(string text)
    {
        var v = new float[Dimension];
        v[0] = 1f;
        return v;
    }
    public IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts) =>
        texts.Select(Encode).ToList();
}

internal static class Fixtures
{
    public static NodeRef Ref(string type, string id) => new(type, id);

    /// <summary>Auto-embedding graph whose vectors derive from node names.</summary>
    public static SemanticGraph NamedGraph(
        IEmbeddingEncoder encoder, string[]? names = null,
        ModelMismatchMode onModelMismatch = ModelMismatchMode.Warn)
    {
        var graph = new SemanticGraph("g", encoder, autoEmbed: true, onModelMismatch: onModelMismatch);
        names ??= new[] { "Alice", "Bob", "Carol" };
        for (int i = 0; i < names.Length; i++)
            graph.AddNode("person", (i + 1).ToString(), new() { ["name"] = names[i] });
        if (names.Length >= 2)
            graph.Graph.AddEdge("KNOWS", Ref("person", "1"), Ref("person", "2"));
        return graph;
    }

    /// <summary>
    /// Hand-vectored graph for traversal tests (query [1,0,0,0] is the target).
    ///
    ///   person 1  --KNOWS--&gt; company 100 --KNOWS--&gt; person 3
    ///   person 2  (isolated)                         person 3 has no vector
    ///
    /// company 100 is the single best match, person 1 next.
    /// </summary>
    public static SemanticGraph VectorGraph()
    {
        var graph = new SemanticGraph("v", new UnitEncoder(), autoEmbed: false);
        graph.AddNode("person", "1", new() { ["name"] = "P1", ["active"] = true });
        graph.AddNode("person", "2", new() { ["name"] = "P2", ["active"] = false });
        graph.AddNode("person", "3", new() { ["name"] = "P3", ["active"] = true });
        graph.AddNode("company", "100", new() { ["name"] = "C" });
        graph.Graph.AddEdge("KNOWS", Ref("person", "1"), Ref("company", "100"));
        graph.Graph.AddEdge("KNOWS", Ref("company", "100"), Ref("person", "3"));
        var store = graph.EmbeddingStore;
        store.Add(Ref("company", "100"), "c", new[] { 1f, 0f, 0f, 0f });
        store.Add(Ref("person", "1"), "p1", new[] { 0.9f, 0.1f, 0f, 0f });
        store.Add(Ref("person", "2"), "p2", new[] { 0.8f, 0.2f, 0f, 0f });
        return graph;
    }

    public static bool IsActive(Node n) => n.Properties.TryGetValue("active", out var a) && a is true;
}

// =============================================================================
// Node filtering
// =============================================================================

public class NodeFilteringTests
{
    private static readonly string[] Person = { "person" };

    private static MultiHopOptions Hop(int maxHops = 2, int seeds = 1) =>
        new() { MaxHops = maxHops, TopKSeeds = seeds, Threshold = -1f };

    [Fact]
    public void SimilaritySearchFiltersBeforeTakingTopK()
    {
        var graph = Fixtures.VectorGraph();
        // the single best match is the company; filtering must look past it
        Assert.Equal(Fixtures.Ref("company", "100"), graph.SimilaritySearch("q", 1)[0].NodeRef);
        var top = graph.SimilaritySearch("q", 1, nodeTypes: Person);
        Assert.Equal(new[] { Fixtures.Ref("person", "1") }, top.Select(r => r.NodeRef));
    }

    [Fact]
    public void NodeFilterAndTypesCombineWithAnd()
    {
        var graph = Fixtures.VectorGraph();
        var active = graph.SimilaritySearch("q", null, threshold: -1f, nodeFilter: Fixtures.IsActive);
        Assert.Equal(new[] { Fixtures.Ref("person", "1") }, active.Select(r => r.NodeRef));
        var none = graph.SimilaritySearch("q", null, threshold: -1f,
            nodeFilter: Fixtures.IsActive, nodeTypes: new[] { "company" });
        Assert.Empty(none);
    }

    [Fact]
    public void EmptyTypeListMatchesNothing()
    {
        var graph = Fixtures.VectorGraph();
        Assert.Empty(graph.SimilaritySearch("q", null, nodeTypes: Array.Empty<string>()));
        Assert.Single(graph.SimilaritySearch("q", null, nodeTypes: new[] { "company" }));
    }

    [Fact]
    public void EmbeddingsWithoutANodeFailNodeFilterOnly()
    {
        var graph = Fixtures.VectorGraph();
        graph.EmbeddingStore.Add(Fixtures.Ref("ghost", "1"), "g", new[] { 1f, 0f, 0f, 0f });
        var ghost = Fixtures.Ref("ghost", "1");
        Assert.Contains(ghost, graph.SimilaritySearch("q", null, threshold: -1f,
            nodeTypes: new[] { "ghost" }).Select(r => r.NodeRef));
        Assert.DoesNotContain(ghost, graph.SimilaritySearch("q", null, threshold: -1f,
            nodeFilter: _ => true).Select(r => r.NodeRef));
    }

    [Fact]
    public void MultiHopSeedsAndResultsAreFiltered()
    {
        var graph = Fixtures.VectorGraph();
        var options = Hop();
        options.NodeTypes = Person;
        var refs = graph.SemanticMultiHop("q", options).Select(r => r.NodeRef).ToList();
        // the best match (company) did not take the seed slot, and is not returned
        Assert.Equal(Fixtures.Ref("person", "1"), refs[0]);
        Assert.DoesNotContain(Fixtures.Ref("company", "100"), refs);
    }

    [Fact]
    public void FilteredNodesAreTraversedButNotReturnedByDefault()
    {
        var graph = Fixtures.VectorGraph();
        var options = Hop();
        options.NodeTypes = Person;
        var found = graph.SemanticMultiHop("q", options).ToDictionary(r => r.NodeRef);
        var reach = found[Fixtures.Ref("person", "3")];
        Assert.Equal(new[] { Fixtures.Ref("person", "1"), Fixtures.Ref("company", "100"),
                             Fixtures.Ref("person", "3") }, reach.Path);
        Assert.Equal(2, reach.HopCount);   // the company still cost a hop
        Assert.Equal(found[Fixtures.Ref("person", "1")].Score * 0.8f * 0.8f, reach.Score, 5);
    }

    [Fact]
    public void StrictFilterWallsOffTheTraversal()
    {
        var graph = Fixtures.VectorGraph();
        var options = Hop();
        options.NodeTypes = Person;
        options.StrictFilter = true;
        var found = graph.SemanticMultiHop("q", options).Select(r => r.NodeRef).ToList();
        Assert.DoesNotContain(Fixtures.Ref("person", "3"), found);
        Assert.DoesNotContain(Fixtures.Ref("company", "100"), found);
    }

    [Fact]
    public void BlendPathRespectsTheFilterToo()
    {
        var graph = Fixtures.VectorGraph();
        var options = Hop(1, 2);
        options.Blend = 0.5f;
        options.NodeTypes = Person;
        var found = graph.SemanticMultiHop("q", options);
        Assert.NotEmpty(found);
        Assert.All(found, r => Assert.Equal("person", r.NodeRef.Type));
    }

    [Fact]
    public void SemanticPathRequiresAPassingTarget()
    {
        var graph = Fixtures.VectorGraph();
        var company = Fixtures.Ref("company", "100");
        Assert.Null(graph.SemanticPath("q", company, threshold: -1f, nodeTypes: Person));
        Assert.NotNull(graph.SemanticPath("q", company, threshold: -1f));
    }

    [Fact]
    public void SemanticPathMayCrossFailingNodesUnlessStrict()
    {
        var graph = Fixtures.VectorGraph();
        var target = Fixtures.Ref("person", "3");
        var lax = graph.SemanticPath("q", target, threshold: -1f, topKSeeds: 1, nodeTypes: Person);
        Assert.NotNull(lax);
        Assert.Equal(new[] { Fixtures.Ref("person", "1"), Fixtures.Ref("company", "100"), target }, lax!.Path);
        var strict = graph.SemanticPath("q", target, threshold: -1f, topKSeeds: 1,
            nodeTypes: Person, strictFilter: true);
        Assert.Null(strict);
    }

    [Fact]
    public void SemanticPathSeedsComeFromPassingNodes()
    {
        var graph = Fixtures.VectorGraph();
        var result = graph.SemanticPath("q", Fixtures.Ref("person", "3"), threshold: -1f, topKSeeds: 1,
            nodeFilter: n => n.Type == "person");
        Assert.NotNull(result);
        Assert.Equal(Fixtures.Ref("person", "1"), result!.Path[0]);
    }

    [Fact]
    public void SubgraphHoldsOnlyPassingNodes()
    {
        var graph = Fixtures.VectorGraph();
        var options = Hop(2, 2);
        options.NodeTypes = Person;
        var sub = graph.SemanticSubgraph("q", options);
        Assert.Equal(new[] { "person" }, sub.Graph.Nodes().Select(n => n.Type).Distinct());
        Assert.True(sub.Graph.HasNode("person", "3"));
    }

    [Fact]
    public void DefaultsAreUnchanged()
    {
        var graph = Fixtures.VectorGraph();
        var plain = graph.SemanticMultiHop("q", Hop());
        Assert.Equal(Fixtures.Ref("company", "100"), plain[0].NodeRef);
        Assert.Contains(Fixtures.Ref("person", "3"), plain.Select(r => r.NodeRef));
    }
}

// =============================================================================
// Edge weighting
// =============================================================================

public class EdgeWeightingTests
{
    private static SemanticGraph Fan(object? pWeight, bool qHasWeight = false, object? qWeight = null)
    {
        var graph = new SemanticGraph("fan", new UnitEncoder(), autoEmbed: false);
        foreach (var id in new[] { "S", "P", "Q" }) graph.AddNode("n", id);
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "S"), Fixtures.Ref("n", "P"), new Dictionary<string, object?> { ["w"] = pWeight });
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "S"), Fixtures.Ref("n", "Q"),
            qHasWeight ? new Dictionary<string, object?> { ["w"] = qWeight } : null);
        graph.EmbeddingStore.Add(Fixtures.Ref("n", "S"), "s", new[] { 1f, 0f, 0f, 0f });
        return graph;
    }

    private static Dictionary<string, float> Scores(SemanticGraph graph, string? weightField)
    {
        var found = graph.SemanticMultiHop("q", new MultiHopOptions
        {
            MaxHops = 1, TopKSeeds = 1, Threshold = -1f, WeightField = weightField,
        });
        return found.ToDictionary(r => r.NodeRef.Id, r => r.Score);
    }

    [Fact]
    public void DefaultIgnoresEdgeProperties()
    {
        var scores = Scores(Fan(0.1), null);
        Assert.Equal(0.8f, scores["P"], 6);
        Assert.Equal(0.8f, scores["Q"], 6);
        // bit-identical to the unweighted formula
        Assert.Equal(MathF.Pow(0.8f, 1), scores["P"]);
    }

    [Fact]
    public void WeightMultipliesTheDecay()
    {
        var scores = Scores(Fan(0.5), "w");
        Assert.Equal(1.0f, scores["S"], 6);
        Assert.Equal(0.8f * 0.5f, scores["P"], 6);
        Assert.Equal(0.8f, scores["Q"], 6);   // missing -> 1.0
    }

    [Fact]
    public void WeightsAreClampedAndJunkCountsAsOne()
    {
        Assert.Equal(0.8f, Scores(Fan(5.0), "w")["P"], 6);
        Assert.Equal(0.8f, Scores(Fan(5L), "w")["P"], 6);
        Assert.Equal(0.0f, Scores(Fan(-3.0), "w")["P"], 6);
        Assert.Equal(0.8f, Scores(Fan("heavy"), "w")["P"], 6);
        Assert.Equal(0.8f, Scores(Fan(true), "w")["P"], 6);
        Assert.Equal(0.8f, Scores(Fan(double.NaN), "w")["P"], 6);
        Assert.Equal(0.8f, Scores(Fan(null), "w")["P"], 6);
    }

    [Fact]
    public void ZeroWeightScoresZeroButStaysReachable()
    {
        var found = Scores(Fan(0.0), "w");
        Assert.True(found.ContainsKey("P"));
        Assert.Equal(0f, found["P"]);
    }

    private static SemanticGraph Diamond()
    {
        var graph = new SemanticGraph("d", new UnitEncoder(), autoEmbed: false);
        foreach (var id in new[] { "S", "M", "T" }) graph.AddNode("n", id);
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "S"), Fixtures.Ref("n", "T"), new Dictionary<string, object?> { ["w"] = 0.1 });   // direct but weak
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "S"), Fixtures.Ref("n", "M"), new Dictionary<string, object?> { ["w"] = 1.0 });
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "M"), Fixtures.Ref("n", "T"), new Dictionary<string, object?> { ["w"] = 1.0 }); // 2 hops, strong
        graph.EmbeddingStore.Add(Fixtures.Ref("n", "S"), "s", new[] { 1f, 0f, 0f, 0f });
        return graph;
    }

    [Fact]
    public void BestRouteCanBeLongerThanTheShortest()
    {
        var graph = Diamond();
        var plain = graph.SemanticMultiHop("q", new MultiHopOptions { MaxHops = 2, TopKSeeds = 1, Threshold = -1f })
                         .ToDictionary(r => r.NodeRef.Id);
        Assert.Equal(1, plain["T"].HopCount);
        Assert.Equal(0.8f, plain["T"].Score, 6);

        var weighted = graph.SemanticMultiHop("q", new MultiHopOptions
        {
            MaxHops = 2, TopKSeeds = 1, Threshold = -1f, WeightField = "w",
        }).ToDictionary(r => r.NodeRef.Id);
        Assert.Equal(2, weighted["T"].HopCount);
        Assert.Equal(new[] { Fixtures.Ref("n", "S"), Fixtures.Ref("n", "M"), Fixtures.Ref("n", "T") },
                     weighted["T"].Path);
        Assert.Equal(0.64f, weighted["T"].Score, 5);
    }

    [Fact]
    public void HopBudgetStillAppliesToTheBetterRoute()
    {
        var found = Diamond().SemanticMultiHop("q", new MultiHopOptions
        {
            MaxHops = 1, TopKSeeds = 1, Threshold = -1f, WeightField = "w",
        }).ToDictionary(r => r.NodeRef.Id);
        Assert.Equal(0.8f * 0.1f, found["T"].Score, 6);
    }

    [Fact]
    public void SubgraphSelectionFollowsTheWeights()
    {
        var sub = Fan(0.1).SemanticSubgraph("q", new MultiHopOptions
        {
            MaxHops = 1, TopKSeeds = 1, TopKResults = 2, Threshold = -1f, WeightField = "w",
        });
        Assert.Equal(new[] { "Q", "S" }, sub.Graph.Nodes().Select(n => n.Id).OrderBy(x => x));
    }
}

// =============================================================================
// Subgraph connectivity
// =============================================================================

public class SubgraphConnectivityTests
{
    private static SemanticGraph Bridge(string relAB = "R", string relXB = "R",
                                        bool aToX = true, bool bToX = false)
    {
        var graph = new SemanticGraph("b", new UnitEncoder(), autoEmbed: false);
        foreach (var id in new[] { "A", "B", "X" })
            graph.AddNode("n", id, new() { ["kind"] = id == "X" ? "hub" : "leaf" });
        // a -> x -> b (or reversed, per the flags)
        if (aToX) graph.Graph.AddEdge(relAB, Fixtures.Ref("n", "A"), Fixtures.Ref("n", "X"));
        else graph.Graph.AddEdge(relAB, Fixtures.Ref("n", "X"), Fixtures.Ref("n", "A"));
        if (bToX) graph.Graph.AddEdge(relXB, Fixtures.Ref("n", "B"), Fixtures.Ref("n", "X"));
        else graph.Graph.AddEdge(relXB, Fixtures.Ref("n", "X"), Fixtures.Ref("n", "B"));
        graph.EmbeddingStore.Add(Fixtures.Ref("n", "A"), "a", new[] { 1f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(Fixtures.Ref("n", "B"), "b", new[] { 0.9f, 0.1f, 0f, 0f });
        return graph;
    }

    private static SemanticGraph Sub(SemanticGraph graph, Action<MultiHopOptions>? tweak = null,
                                     bool connect = false, int connectMaxHops = 2)
    {
        var options = new MultiHopOptions { MaxHops = 0, TopKSeeds = 2, TopKResults = 2, Threshold = -1f };
        tweak?.Invoke(options);
        return graph.SemanticSubgraph("q", options, connect: connect, connectMaxHops: connectMaxHops);
    }

    private static HashSet<string> Ids(SemanticGraph g) => g.Graph.Nodes().Select(n => n.Id).ToHashSet();

    [Fact]
    public void DefaultLeavesTheSelectionDisconnected()
    {
        var sub = Sub(Bridge());
        Assert.Equal(new HashSet<string> { "A", "B" }, Ids(sub));
        Assert.Equal(0, sub.Graph.EdgeCount());
    }

    [Fact]
    public void ConnectAddsTheBridgingNodeAndItsEdges()
    {
        var sub = Sub(Bridge(), connect: true);
        Assert.Equal(new HashSet<string> { "A", "B", "X" }, Ids(sub));
        Assert.Equal(2, sub.Graph.EdgeCount());
        // the connector carries its own properties
        Assert.Equal("hub", sub.Graph.GetNode("n", "X").Properties["kind"]);
    }

    [Fact]
    public void ConnectorNodesMayExceedTopKResults()
    {
        Assert.Equal(3, Sub(Bridge(), o => o.TopKResults = 2, connect: true).Graph.NodeCount());
    }

    [Fact]
    public void HopBudgetLimitsTheSearch()
    {
        var graph = Bridge();
        Assert.Equal(new HashSet<string> { "A", "B" }, Ids(Sub(graph, connect: true, connectMaxHops: 1)));
        Assert.Contains("X", Ids(Sub(graph, connect: true, connectMaxHops: 2)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Sub(graph, connect: true, connectMaxHops: 0));
    }

    [Fact]
    public void LongerBudgetAddsAChainOfConnectors()
    {
        var graph = new SemanticGraph("chain", new UnitEncoder(), autoEmbed: false);
        foreach (var id in new[] { "A", "M1", "M2", "B" }) graph.AddNode("n", id);
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "A"), Fixtures.Ref("n", "M1"));
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "M1"), Fixtures.Ref("n", "M2"));
        graph.Graph.AddEdge("R", Fixtures.Ref("n", "M2"), Fixtures.Ref("n", "B"));
        graph.EmbeddingStore.Add(Fixtures.Ref("n", "A"), "a", new[] { 1f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(Fixtures.Ref("n", "B"), "b", new[] { 0.9f, 0.1f, 0f, 0f });
        Assert.Equal(2, Sub(graph, connect: true, connectMaxHops: 2).Graph.NodeCount());
        Assert.Equal(4, Sub(graph, connect: true, connectMaxHops: 3).Graph.NodeCount());
    }

    [Fact]
    public void DirectionIsRespected()
    {
        var graph = Bridge(aToX: true, bToX: true);   // A -> X <- B
        Assert.Equal(new HashSet<string> { "A", "B" }, Ids(Sub(graph, connect: true)));
        var both = Sub(graph, o => o.Direction = Direction.Both, connect: true);
        Assert.Contains("X", Ids(both));
    }

    [Fact]
    public void RelTypeIsRespected()
    {
        var graph = Bridge(relAB: "KNOWS", relXB: "LIKES");
        Assert.Equal(new HashSet<string> { "A", "B" }, Ids(Sub(graph, o => o.RelType = "KNOWS", connect: true)));
        Assert.Contains("X", Ids(Sub(graph, connect: true)));
    }

    [Fact]
    public void OneHubJoinsThreeComponents()
    {
        var graph = new SemanticGraph("hub", new UnitEncoder(), autoEmbed: false);
        graph.AddNode("n", "HUB");
        var vectors = new Dictionary<string, float[]>
        {
            ["A"] = new[] { 1.0f, 0.0f, 0f, 0f },
            ["B"] = new[] { 0.95f, 0.1f, 0f, 0f },
            ["C"] = new[] { 0.9f, 0.2f, 0f, 0f },
        };
        foreach (var (id, vector) in vectors)
        {
            graph.AddNode("n", id);
            graph.Graph.AddEdge("R", Fixtures.Ref("n", id), Fixtures.Ref("n", "HUB"));
            graph.EmbeddingStore.Add(Fixtures.Ref("n", id), id, vector);
        }
        var sub = graph.SemanticSubgraph("q", new MultiHopOptions
        {
            MaxHops = 0, TopKSeeds = 3, TopKResults = 3, Threshold = -1f, Direction = Direction.Both,
        }, connect: true);
        Assert.Equal(new HashSet<string> { "A", "B", "C", "HUB" }, Ids(sub));
    }

    [Fact]
    public void StrictFilterBarsFailingNodesAsConnectors()
    {
        var graph = Bridge();
        Action<MultiHopOptions> filter = o =>
            o.NodeFilter = n => (string?)n.Properties["kind"] == "leaf";
        // lax: X fails the filter but may still bridge
        Assert.Contains("X", Ids(Sub(graph, filter, connect: true)));
        var strict = Sub(graph, o => { filter(o); o.StrictFilter = true; }, connect: true);
        Assert.Equal(new HashSet<string> { "A", "B" }, Ids(strict));
    }
}

// =============================================================================
// Model lifecycle
// =============================================================================

public class ModelLifecycleTests
{
    private static MockEncoder Retagged(int dimension, string name) =>
        new(dimension) { ModelName = name };

    private static (SemanticGraph Graph, List<string> Warnings) WithWarnings(SemanticGraph graph)
    {
        var warnings = new List<string>();
        graph.Warning += warnings.Add;
        return (graph, warnings);
    }

    [Fact]
    public void ModelsCountsVectorsPerModel()
    {
        var encoder = new MockEncoder(8);
        var graph = Fixtures.NamedGraph(encoder);
        Assert.Equal(new Dictionary<string, int> { [encoder.ModelName!] = 3 },
                     graph.EmbeddingStore.Models());
        Assert.Equal("mock-8", encoder.ModelName);
    }

    [Fact]
    public void StaleByModelNameThenReembed()
    {
        var graph = Fixtures.NamedGraph(new MockEncoder(16));
        graph.SetEncoder(Retagged(16, "next-gen-16"));

        var stale = graph.StaleEmbeddings();
        Assert.Equal(new[] { "1", "2", "3" }, stale.Select(r => r.Id).OrderBy(x => x));
        Assert.Equal(new Dictionary<string, int> { ["mock-16"] = 3 }, graph.EmbeddingStore.Models());

        Assert.Equal(3, graph.ReembedStale());
        Assert.Empty(graph.StaleEmbeddings());
        Assert.Equal(new Dictionary<string, int> { ["next-gen-16"] = 3 }, graph.EmbeddingStore.Models());
    }

    [Fact]
    public void NothingIsStaleUnderTheSameEncoder()
    {
        var graph = Fixtures.NamedGraph(new MockEncoder(8));
        Assert.Empty(graph.StaleEmbeddings());
        Assert.Equal(0, graph.ReembedStale());
    }

    [Fact]
    public void StaleByDimensionResetsTheStore()
    {
        var graph = Fixtures.NamedGraph(new MockEncoder(8));
        graph.SetEncoder(new MockEncoder(16));
        Assert.Equal(3, graph.StaleEmbeddings().Count);

        Assert.Equal(3, graph.ReembedStale());
        Assert.Equal(16, graph.EmbeddingStore.Dimension);
        Assert.Equal(16, graph.GetEmbedding(Fixtures.Ref("person", "1"))!.Vector.Length);
        Assert.Empty(graph.StaleEmbeddings());
    }

    [Fact]
    public void ReembedKeepsCustomEmbedText()
    {
        var graph = new SemanticGraph("g", new MockEncoder(8), autoEmbed: false);
        graph.AddNode("person", "1", new() { ["name"] = "A" }, embedText: "a very specific blurb");
        graph.SetEncoder(Retagged(8, "v2"));
        graph.ReembedStale();
        var record = graph.GetEmbedding(Fixtures.Ref("person", "1"))!;
        Assert.Equal("a very specific blurb", record.Text);
        Assert.Equal("v2", record.Model);
    }

    [Fact]
    public void UnknownModelIsNotStaleUnlessAsked()
    {
        var graph = new SemanticGraph("g", new MockEncoder(8), autoEmbed: false);
        graph.AddNode("person", "1");
        var encoder = graph.Encoder!;
        graph.EmbeddingStore.Encoder = null;     // a vector handed in -> "provided"
        graph.EmbeddingStore.Add(Fixtures.Ref("person", "1"), "x", Enumerable.Repeat(0.1f, 8).ToArray());
        graph.EmbeddingStore.Encoder = encoder;

        Assert.Equal(new Dictionary<string, int> { ["provided"] = 1 }, graph.EmbeddingStore.Models());
        Assert.Empty(graph.StaleEmbeddings());
        Assert.Equal(new[] { Fixtures.Ref("person", "1") }, graph.StaleEmbeddings(includeUnknown: true));
    }

    [Fact]
    public void EncoderWithoutAModelNameStampsUnknown()
    {
        var graph = new SemanticGraph("g", new UnitEncoder(), autoEmbed: false);
        graph.AddNode("person", "1", embedText: "x");
        Assert.Equal(new Dictionary<string, int> { ["unknown"] = 1 }, graph.EmbeddingStore.Models());
        Assert.Empty(graph.StaleEmbeddings());
    }

    [Fact]
    public void TextSearchWithADifferentModelWarnsOnce()
    {
        var (graph, warnings) = WithWarnings(Fixtures.NamedGraph(new MockEncoder(16)));
        graph.SetEncoder(Retagged(16, "other-16"));
        graph.SimilaritySearch("hello", 1, threshold: -1f);
        Assert.Single(warnings);
        Assert.Contains("other-16", warnings[0]);
        Assert.Contains("mock-16", warnings[0]);
        graph.SimilaritySearch("hello again", 1, threshold: -1f);
        graph.SemanticMultiHop("hello");
        Assert.Single(warnings);
    }

    [Fact]
    public void TextSearchWithADifferentModelCanThrow()
    {
        var graph = Fixtures.NamedGraph(new MockEncoder(16), new[] { "Alice" }, ModelMismatchMode.Error);
        graph.SetEncoder(Retagged(16, "other-16"));
        var ex = Assert.Throws<ModelMismatchException>(() => graph.SimilaritySearch("hello", 1));
        Assert.Contains("mock-16", ex.Message);
        Assert.Throws<ModelMismatchException>(() => graph.SemanticMultiHop("hello"));
        Assert.Throws<ModelMismatchException>(() =>
            graph.SemanticPath("hello", Fixtures.Ref("person", "1")));
        // re-embedding is the cure
        graph.ReembedStale();
        Assert.NotEmpty(graph.SimilaritySearch("hello", 1, threshold: -1f));
    }

    [Fact]
    public void VectorQueriesAndIgnoreModeAreNeverChecked()
    {
        var (graph, warnings) = WithWarnings(Fixtures.NamedGraph(new MockEncoder(16)));
        graph.SetEncoder(Retagged(16, "other-16"));
        graph.EmbeddingStore.SimilaritySearchVector(new MockEncoder(16).Encode("x"), 1, null, -1f);
        Assert.Empty(warnings);

        var (quiet, quietWarnings) = WithWarnings(
            Fixtures.NamedGraph(new MockEncoder(16), new[] { "Alice" }, ModelMismatchMode.Ignore));
        quiet.SetEncoder(Retagged(16, "other-16"));
        quiet.SimilaritySearch("x", 1, threshold: -1f);
        Assert.Empty(quietWarnings);
    }

    [Fact]
    public void RejectsAnUnknownMismatchMode()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(null, null, (ModelMismatchMode)99));
    }

    [Fact]
    public void VersionIsRecordedInTheSidecar()
    {
        var encoder = new MockEncoder(8, version: "2026.1");
        var graph = Fixtures.NamedGraph(encoder);
        Assert.Equal("2026.1", graph.EmbeddingStore.ExportEmbeddings().EncoderVersion);

        // an explicit version wins over the encoder's
        var store = new EmbeddingStore(encoder, version: "explicit");
        store.Add(Fixtures.Ref("a", "1"), "x");
        Assert.Equal("explicit", store.ExportEmbeddings().EncoderVersion);

        // and nothing is written when no version is known
        var plain = Fixtures.NamedGraph(new MockEncoder(8));
        var json = plain.EmbeddingStore.ToJson();
        Assert.DoesNotContain("encoder_version", json);
    }

    [Fact]
    public void ClearKeepsOrResetsThePinnedDimension()
    {
        var store = new EmbeddingStore();
        store.Add(Fixtures.Ref("a", "1"), "x", new[] { 1f, 2f });
        store.Clear(resetDimension: false);
        Assert.Equal(2, store.Dimension);
        Assert.Throws<DimensionMismatchException>(() => store.Add(Fixtures.Ref("a", "1"), "x", new[] { 1f, 2f, 3f }));
        store.Clear(resetDimension: true);
        Assert.Null(store.Dimension);
        store.Add(Fixtures.Ref("a", "1"), "x", new[] { 1f, 2f, 3f });
        store.Clear();   // the long-standing default resets
        Assert.Null(store.Dimension);
    }

    [Fact]
    public void SubgraphKeepsTheModelThatMadeEachVector()
    {
        var graph = Fixtures.NamedGraph(new MockEncoder(8));
        graph.SetEncoder(Retagged(8, "other"));
        var sub = graph.SemanticSubgraph("Alice", new MultiHopOptions { Threshold = -1f, MaxHops = 0 });
        Assert.All(sub.EmbeddingStore.Models().Keys, m => Assert.Equal("mock-8", m));
    }
}

// =============================================================================
// Graph / vector consistency
// =============================================================================

public class SaveLoadConsistencyTests : IDisposable
{
    private readonly string _dir =
        Directory.CreateDirectory(System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "isongraph-vector-" + Guid.NewGuid().ToString("N"))).FullName;

    public void Dispose() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

    private string P(string name) => System.IO.Path.Combine(_dir, name);

    private static JsonElement ReadSidecar(string graphPath) =>
        JsonDocument.Parse(File.ReadAllText(SemanticGraph.EmbeddingsPathFor(graphPath))).RootElement;

    [Fact]
    public void SidecarRecordsDigestAndStaysVersion1()
    {
        var encoder = new MockEncoder(8);
        var path = P("g.isong");
        Fixtures.NamedGraph(encoder).Save(path);
        var payload = ReadSidecar(path);

        Assert.Equal("ison-embeddings", payload.GetProperty("format").GetString());
        Assert.Equal(1, payload.GetProperty("version").GetInt32());
        Assert.StartsWith("sha256:", payload.GetProperty("graph_digest").GetString());
        Assert.Equal(GraphDigest.Basis, payload.GetProperty("graph_digest_basis").GetString());
        Assert.Equal("mock-8", payload.GetProperty("encoder_model").GetString());
        // Optional metadata trails the array so naive readers (the C++ one
        // scans for keys) meet format/version/embeddings first.
        var keys = payload.EnumerateObject().Select(p => p.Name).ToList();
        Assert.Equal(new[] { "format", "version", "dimension", "count", "embeddings" }, keys.Take(5));
        Assert.Equal(new[] { "encoder_model", "graph_digest", "graph_digest_basis" },
                     keys.Skip(5).OrderBy(k => k));
        // each embedding records its model
        Assert.Equal("mock-8", payload.GetProperty("embeddings")[0].GetProperty("model").GetString());
    }

    [Fact]
    public void SavesLeaveNoTempFiles()
    {
        Fixtures.NamedGraph(new MockEncoder(8)).Save(P("g.isong"));
        Assert.Equal(new[] { "g.isong", "g.isong.embeddings.json" },
                     Directory.GetFileSystemEntries(_dir).Select(System.IO.Path.GetFileName).OrderBy(x => x));
    }

    [Fact]
    public void AFailedWriteLeavesNoTempFileAndKeepsWhatWasThere()
    {
        var graph = Fixtures.NamedGraph(new MockEncoder(8));
        // the sidecar's location is a directory, so moving the temp file over it fails
        Directory.CreateDirectory(P("g.isong.embeddings.json"));
        Assert.ThrowsAny<Exception>(() => graph.Save(P("g.isong")));
        Assert.True(Directory.Exists(P("g.isong.embeddings.json")));
        Assert.Empty(Directory.GetFiles(_dir, "*.tmp"));

        // and likewise when the graph file itself cannot be replaced
        Directory.CreateDirectory(P("h.isong"));
        Assert.ThrowsAny<Exception>(() => graph.Save(P("h.isong")));
        Assert.True(Directory.Exists(P("h.isong")));
        Assert.Empty(Directory.GetFiles(_dir, "*.tmp"));
    }

    [Fact]
    public void MatchingPairLoadsCleanWithTypedValues()
    {
        var encoder = new MockEncoder(8);
        var graph = new SemanticGraph("t", encoder, autoEmbed: false);
        graph.AddNode("person", "1", new()
        {
            ["name"] = "A b", ["age"] = 30L, ["ratio"] = 1.0, ["ok"] = true, ["note"] = null,
            ["tiny"] = 1.5e-7, ["huge"] = 1.0e20, ["third"] = 1.0 / 3.0, ["accent"] = "résumé 日本語 é \"q\" \\ \U0001F600",
        });
        graph.AddNode("account", "007", new() { ["name"] = "Bond" });
        graph.Graph.AddEdge("KNOWS", Fixtures.Ref("person", "1"), Fixtures.Ref("account", "007"),
            new Dictionary<string, object?> { ["w"] = 0.5, ["n"] = 2L });
        graph.EmbedNode(Fixtures.Ref("person", "1"), "a");
        var path = P("t.isong");
        graph.Save(path);

        var warnings = new List<string>();
        var loaded = SemanticGraph.Load(path, encoder, onWarning: warnings.Add);
        Assert.Equal(1, loaded.EmbeddingStore.Count);
        Assert.Empty(warnings);
    }

    [Fact]
    public void IsonlRoundTripVerifies()
    {
        var encoder = new MockEncoder(8);
        var path = P("g.isonl");
        Fixtures.NamedGraph(encoder).Save(path);
        Assert.Equal(3, SemanticGraph.Load(path, encoder).EmbeddingStore.Count);
    }

    [Fact]
    public void UndirectedGraphRoundTripVerifies()
    {
        var encoder = new MockEncoder(8);
        var graph = new SemanticGraph("u", encoder, directed: false);
        graph.AddNode("n", "1", new() { ["name"] = "one" });
        graph.AddNode("n", "2", new() { ["name"] = "two" });
        graph.Graph.AddEdge("E", Fixtures.Ref("n", "1"), Fixtures.Ref("n", "2"), new Dictionary<string, object?> { ["w"] = 0.25 });
        var path = P("u.isong");
        graph.Save(path);
        Assert.Equal(2, SemanticGraph.Load(path, encoder).EmbeddingStore.Count);
    }

    [Fact]
    public void SidecarFromAnotherSaveIsRejected()
    {
        var encoder = new MockEncoder(8);
        Fixtures.NamedGraph(encoder, new[] { "Alice", "Bob", "Carol" }).Save(P("a.isong"));
        Fixtures.NamedGraph(encoder, new[] { "Xavier", "Yolanda", "Zed" }).Save(P("b.isong"));
        // b's vectors dropped next to a's graph
        File.WriteAllText(SemanticGraph.EmbeddingsPathFor(P("a.isong")),
                          File.ReadAllText(SemanticGraph.EmbeddingsPathFor(P("b.isong"))));
        var ex = Assert.Throws<GraphEmbeddingMismatchException>(() => SemanticGraph.Load(P("a.isong"), encoder));
        Assert.Contains("digest", ex.Message);
    }

    private string SaveThenEdit(MockEncoder encoder)
    {
        var path = P("g.isong");
        Fixtures.NamedGraph(encoder).Save(path);
        File.WriteAllText(path, File.ReadAllText(path).Replace("Bob", "Robert"));
        return path;
    }

    [Fact]
    public void EditedGraphIsRejectedByDefault()
    {
        var encoder = new MockEncoder(8);
        var path = SaveThenEdit(encoder);
        Assert.Throws<GraphEmbeddingMismatchException>(() => SemanticGraph.Load(path, encoder));
    }

    [Fact]
    public void WarnModeLoadsTheOldVectorsAnyway()
    {
        var encoder = new MockEncoder(8);
        var path = SaveThenEdit(encoder);
        var warnings = new List<string>();
        var loaded = SemanticGraph.Load(path, encoder, onMismatch: GraphMismatchMode.Warn, onWarning: warnings.Add);
        Assert.Single(warnings);
        Assert.Contains("digest", warnings[0]);
        Assert.Equal("Bob", loaded.GetEmbedding(Fixtures.Ref("person", "2"))!.Text);   // the stale text
    }

    [Fact]
    public void ReembedModeRebuildsFromTheEditedGraph()
    {
        var encoder = new MockEncoder(8);
        var path = SaveThenEdit(encoder);
        var loaded = SemanticGraph.Load(path, encoder, onMismatch: GraphMismatchMode.Reembed);
        var record = loaded.GetEmbedding(Fixtures.Ref("person", "2"))!;
        Assert.Equal("Robert", record.Text);
        Assert.Equal(encoder.Encode("Robert"), record.Vector);
        Assert.Equal(3, loaded.EmbeddingStore.Count);
    }

    private void RewriteSidecar(string graphPath, Action<Dictionary<string, JsonElement>> edit)
    {
        var sidecar = SemanticGraph.EmbeddingsPathFor(graphPath);
        var payload = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(File.ReadAllText(sidecar))!;
        edit(payload);
        File.WriteAllText(sidecar, JsonSerializer.Serialize(payload));
    }

    private static JsonElement Str(string s) => JsonSerializer.SerializeToElement(s);

    [Fact]
    public void SidecarWithoutADigestLoadsAsBefore()
    {
        var encoder = new MockEncoder(8);
        var path = P("g.isong");
        Fixtures.NamedGraph(encoder).Save(path);
        RewriteSidecar(path, p => { p.Remove("graph_digest"); p.Remove("graph_digest_basis"); p.Remove("encoder_model"); });
        // even an edited graph is accepted: there is nothing to compare with
        File.WriteAllText(path, File.ReadAllText(path).Replace("Bob", "Robert"));
        var warnings = new List<string>();
        var loaded = SemanticGraph.Load(path, encoder, onWarning: warnings.Add);
        Assert.Equal(3, loaded.EmbeddingStore.Count);
        Assert.Empty(warnings);
    }

    [Fact]
    public void DigestOfAnUnknownBasisIsNotCompared()
    {
        var encoder = new MockEncoder(8);
        var path = P("g.isong");
        Fixtures.NamedGraph(encoder).Save(path);
        RewriteSidecar(path, p =>
        {
            p["graph_digest"] = Str("sha256:" + new string('0', 64));
            p["graph_digest_basis"] = Str("some-other-port-v9");
        });
        Assert.Equal(3, SemanticGraph.Load(path, encoder).EmbeddingStore.Count);
    }

    [Fact]
    public void LegacyPayloadWithUnknownFieldsImports()
    {
        var encoder = new MockEncoder(8);
        var source = new EmbeddingStore(encoder);
        source.Add(Fixtures.Ref("person", "1"), "Alice");
        var payload = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(source.ToJson("sha256:abc"))!;
        payload["some_future_field"] = JsonSerializer.SerializeToElement(new { x = 1 });
        var target = new EmbeddingStore(encoder);
        Assert.Equal(1, target.FromJson(JsonSerializer.Serialize(payload)));
        Assert.Equal("sha256:abc", target.LastImportMeta["graph_digest"]);
        Assert.Equal(GraphDigest.Basis, target.LastImportMeta["graph_digest_basis"]);
        Assert.Equal("mock-8", target.LastImportMeta["encoder_model"]);
    }

    [Fact]
    public void NumericVersionsAndMissingModelsImport()
    {
        // a port that wrote encoder_version as a number, and no per-vector model
        var json = """
            {"format":"ison-embeddings","version":1,"dimension":2,"count":1,
             "embeddings":[{"type":"p","id":1,"id_type":"int","text":"t","vector":[1,0]}],
             "encoder_version":2}
            """;
        var store = new EmbeddingStore();
        Assert.Equal(1, store.FromJson(json));
        Assert.Equal("2", store.LastImportMeta["encoder_version"]);
        Assert.Equal(new Dictionary<string, int> { ["unknown"] = 1 }, store.Models());
    }

    [Fact]
    public void LoadingWithADifferentVersionWarns()
    {
        var path = P("g.isong");
        Fixtures.NamedGraph(new MockEncoder(8, version: "1")).Save(path);

        var warnings = new List<string>();
        SemanticGraph.Load(path, new MockEncoder(8, version: "2"), onWarning: warnings.Add);
        Assert.Single(warnings);
        Assert.Contains("version '1'", warnings[0]);

        warnings.Clear();
        SemanticGraph.Load(path, new MockEncoder(8, version: "1"), onWarning: warnings.Add);
        SemanticGraph.Load(path, new MockEncoder(8), onWarning: warnings.Add);   // unversioned: no claim
        Assert.Empty(warnings);
    }

    [Fact]
    public void RejectsAnUnknownMismatchMode()
    {
        var path = P("g.isong");
        Fixtures.NamedGraph(new MockEncoder(8)).Save(path);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SemanticGraph.Load(path, new MockEncoder(8), onMismatch: (GraphMismatchMode)99));
    }

    [Fact]
    public void UnreadableSidecarIsAnEmbeddingError()
    {
        var path = P("g.isong");
        Fixtures.NamedGraph(new MockEncoder(8)).Save(path);
        File.WriteAllText(SemanticGraph.EmbeddingsPathFor(path), "{not json");
        Assert.Throws<EmbeddingException>(() => SemanticGraph.Load(path, new MockEncoder(8)));
    }

    [Fact]
    public void DigestIgnoresInsertionOrderButSeesContent()
    {
        var a = new SemanticGraph(autoEmbed: false);
        var b = new SemanticGraph(autoEmbed: false);
        a.AddNode("t", "1", new() { ["x"] = 1L });
        a.AddNode("t", "2", new() { ["x"] = 2L });
        a.Graph.AddEdge("R", Fixtures.Ref("t", "1"), Fixtures.Ref("t", "2"));
        b.AddNode("t", "2", new() { ["x"] = 2L });
        b.AddNode("t", "1", new() { ["x"] = 1L });
        b.Graph.AddEdge("R", Fixtures.Ref("t", "1"), Fixtures.Ref("t", "2"));
        Assert.Equal(GraphDigest.Compute(a.Graph), GraphDigest.Compute(b.Graph));
        b.Graph.UpdateNode("t", "1", new Dictionary<string, object?> { ["x"] = 99L });
        Assert.NotEqual(GraphDigest.Compute(a.Graph), GraphDigest.Compute(b.Graph));

        // an int id and a str id that print alike are different nodes
        var c = new SemanticGraph(autoEmbed: false);
        c.AddNode("t", "7");
        Assert.NotEqual(GraphDigest.Compute(c.Graph, _ => true), GraphDigest.Compute(c.Graph, _ => false));
    }
}

// =============================================================================
// Cross-port digest: the golden fixture shared by every port
// =============================================================================

public class GraphDigestGoldenTests
{
    private static string FindFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = System.IO.Path.Combine(dir.FullName, "tests", "fixtures", "graph_digest_golden.json");
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException("tests/fixtures/graph_digest_golden.json not found above " + AppContext.BaseDirectory);
    }

    private static object? Value(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Null => null,
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.String => e.GetString(),
        JsonValueKind.Number => e.TryGetInt64(out var l) ? (object)l : e.GetDouble(),
        _ => throw new InvalidOperationException("unexpected property value " + e),
    };

    private static Dictionary<string, object?> Props(JsonElement e) =>
        e.EnumerateObject().ToDictionary(p => p.Name, p => Value(p.Value));

    [Fact]
    public void ReproducesTheGoldenDigests()
    {
        var root = JsonDocument.Parse(File.ReadAllText(FindFixture())).RootElement;
        Assert.Equal(GraphDigest.Basis, root.GetProperty("basis").GetString());

        int cases = 0;
        foreach (var c in root.GetProperty("cases").EnumerateArray())
        {
            var name = c.GetProperty("name").GetString();
            var graph = new ISONGraph("golden", c.GetProperty("directed").GetBoolean());
            var intIds = new HashSet<NodeRef>();

            string IdText(JsonElement id) => id.ValueKind == JsonValueKind.Number ? id.GetRawText() : id.GetString()!;

            foreach (var n in c.GetProperty("nodes").EnumerateArray())
            {
                var id = n.GetProperty("id");
                var nodeRef = new NodeRef(n.GetProperty("type").GetString()!, IdText(id));
                if (id.ValueKind == JsonValueKind.Number) intIds.Add(nodeRef);
                graph.AddNode(nodeRef.Type, nodeRef.Id, Props(n.GetProperty("properties")));
            }
            foreach (var e in c.GetProperty("edges").EnumerateArray())
            {
                NodeRef End(JsonElement pair) => new(pair[0].GetString()!, IdText(pair[1]));
                graph.AddEdge(e.GetProperty("rel_type").GetString()!, End(e.GetProperty("source")),
                              End(e.GetProperty("target")), Props(e.GetProperty("properties")));
            }

            Assert.True(c.GetProperty("digest").GetString() == GraphDigest.Compute(graph, intIds.Contains),
                $"digest mismatch for golden case '{name}'");
            cases++;
        }
        Assert.True(cases >= 5, "expected the golden cases to be exercised");
    }

    [Theory]
    [InlineData(0.5, "0.5")]
    [InlineData(1.0, "1.0")]
    [InlineData(100.0, "100.0")]
    [InlineData(-2.5, "-2.5")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(0.00001, "1e-05")]
    [InlineData(1.5e-7, "1.5e-07")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(1e16, "1e+16")]
    [InlineData(123456789012345680.0, "1.2345678901234568e+17")]
    [InlineData(1e22, "1e+22")]
    [InlineData(123456.789, "123456.789")]
    [InlineData(0.1, "0.1")]
    [InlineData(1.0 / 3.0, "0.3333333333333333")]
    [InlineData(5e-324, "5e-324")]
    [InlineData(1.7976931348623157e308, "1.7976931348623157e+308")]
    public void FormatsFloatsLikePythonRepr(double value, string expected)
    {
        Assert.Equal(expected, GraphDigest.PyFloat(value));
    }

    [Fact]
    public void EscapesStringsLikeEnsureAscii()
    {
        var sb = new System.Text.StringBuilder();
        GraphDigest.WriteString(sb, "a\"b\\c\n\t\u0001\u007f é日\U0001F600");
        Assert.Equal("\"a\\\"b\\\\c\\n\\t\\u0001\\u007f \\u00e9\\u65e5\\ud83d\\ude00\"", sb.ToString());
    }
}
