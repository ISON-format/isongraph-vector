using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>Encoder returning a hand-picked vector so scores are exact.</summary>
internal sealed class FixedEncoder : IEmbeddingEncoder
{
    public int Dimension => 4;
    public float[] Encode(string text) => new[] { 1.0f, 0.0f, 0.0f, 0.0f };
    public IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts) =>
        texts.Select(Encode).ToList();
}

public class MockEncoderTests
{
    // Asserted identically in the Python, TypeScript, JavaScript, Rust and C++
    // suites - keeping the ports bit-identical is what lets an embedding store
    // written by one be searched by another.
    private static readonly float[] GoldenHello =
        { 0.8381705284f, -0.8255031109f, 0.5899617672f, 0.4615051746f };
    private static readonly float[] GoldenIson =
        { -0.3658730984f, -0.2765417099f, 0.2944871187f, -0.6502785683f };
    private static readonly float[] GoldenEmpty =
        { -0.4520676136f, -0.3797093630f, 0.3339765072f, 0.9023951292f };

    [Fact]
    public void ProducesTheSameVectorsAsEveryOtherPort()
    {
        var encoder = new MockEncoder(4);
        foreach (var (text, want) in new[]
                 {
                     ("hello", GoldenHello), ("ison", GoldenIson), ("", GoldenEmpty),
                 })
        {
            var got = encoder.Encode(text);
            Assert.Equal(want.Length, got.Length);
            for (int i = 0; i < want.Length; i++)
                Assert.True(Math.Abs(got[i] - want[i]) < 1e-6f,
                    $"{text}[{i}] = {got[i]}, expected {want[i]}");
        }
    }

    [Fact]
    public void IsDeterministic()
    {
        var encoder = new MockEncoder(64);
        Assert.Equal(encoder.Encode("hello"), encoder.Encode("hello"));
        Assert.NotEqual(encoder.Encode("hello"), encoder.Encode("world"));
    }

    [Fact]
    public void DoesNotCollideAcrossManyTexts()
    {
        // The scheme this replaced emitted the low 16 bits of an LCG, so every
        // vector was determined by seed mod 65536: 280 of these 5,000 texts
        // collided outright.
        var encoder = new MockEncoder(8);
        var seen = new HashSet<string>();
        for (int i = 0; i < 5000; i++)
            seen.Add(string.Join(",", encoder.Encode($"text{i}")));
        Assert.Equal(5000, seen.Count);
    }

    [Fact]
    public void StaysWithinRange()
    {
        foreach (var value in new MockEncoder(128).Encode("range check"))
        {
            Assert.True(value >= -1.0f);
            Assert.True(value < 1.0f);
        }
    }

    [Fact]
    public void RejectsANonPositiveDimension() =>
        Assert.Throws<EmbeddingException>(() => new MockEncoder(0));

    [Fact]
    public void BatchMatchesSingle()
    {
        var encoder = new MockEncoder(16);
        var batch = encoder.EncodeBatch(new[] { "a", "b" });
        Assert.Equal(2, batch.Count);
        Assert.Equal(encoder.Encode("a"), batch[0]);
        Assert.Equal(encoder.Encode("b"), batch[1]);
    }
}

public class EmbeddingStoreTests
{
    [Fact]
    public void AddsAndRetrieves()
    {
        var store = new EmbeddingStore(new MockEncoder(16));
        store.Add(new NodeRef("person", "1"), "Alice is an engineer");

        var record = store.Get(new NodeRef("person", "1"));
        Assert.NotNull(record);
        Assert.Equal("Alice is an engineer", record!.Text);
        Assert.Equal(16, record.Vector.Length);
    }

    [Fact]
    public void RemovesAndCounts()
    {
        var store = new EmbeddingStore(new MockEncoder(16));
        store.Add(new NodeRef("person", "1"), "test");
        Assert.Equal(1, store.Count);
        Assert.True(store.Remove(new NodeRef("person", "1")));
        Assert.Null(store.Get(new NodeRef("person", "1")));
        Assert.Equal(0, store.Count);
    }

    [Fact]
    public void SearchesBySimilarity()
    {
        var store = new EmbeddingStore(new MockEncoder(64));
        store.Add(new NodeRef("person", "1"), "software engineer");
        store.Add(new NodeRef("person", "2"), "data scientist");

        // Cosine similarity is signed and MockEncoder vectors are
        // near-orthogonal, so mock scores straddle zero - the default
        // threshold of 0.0 can reject every match.
        var results = store.SimilaritySearch("software engineer", 2, null, -1.0f);
        Assert.Equal(2, results.Count);
        Assert.True(results[0].Score >= results[1].Score);
        // an exact text match scores 1.0 against itself
        Assert.True(Math.Abs(results[0].Score - 1.0f) < 1e-5f);
    }

    [Fact]
    public void FiltersByNodeType()
    {
        var store = new EmbeddingStore(new MockEncoder(64));
        store.Add(new NodeRef("person", "1"), "engineer");
        store.Add(new NodeRef("company", "1"), "tech company");

        var results = store.SimilaritySearch("tech", 10, "company", -1.0f);
        Assert.Single(results);
        Assert.Equal("company", results[0].NodeRef.Type);
    }

    [Fact]
    public void ClearsEverything()
    {
        var store = new EmbeddingStore(new MockEncoder(16));
        store.Add(new NodeRef("a", "1"), "x");
        store.Clear();
        Assert.Equal(0, store.Count);
        Assert.Null(store.Dimension);
    }

    [Fact]
    public void ScoreMapCoversEveryNode()
    {
        var store = new EmbeddingStore(new MockEncoder(16));
        store.Add(new NodeRef("a", "1"), "one");
        store.Add(new NodeRef("a", "2"), "two");

        var scores = store.ScoreMap(new MockEncoder(16).Encode("query"));
        Assert.Equal(2, scores.Count);
    }

    [Fact]
    public void AddBatchEncodesInOneCall()
    {
        var store = new EmbeddingStore(new MockEncoder(16));
        var written = store.AddBatch(new[]
        {
            (new NodeRef("doc", "1"), "first"),
            (new NodeRef("doc", "2"), "second"),
        });
        Assert.Equal(2, written);
        Assert.Equal(2, store.Count);
    }
}

public class DimensionValidationTests
{
    [Fact]
    public void RejectsAVectorOfTheWrongWidth()
    {
        var store = new EmbeddingStore(new MockEncoder(8));
        store.Add(new NodeRef("a", "1"), "full width");
        Assert.Throws<DimensionMismatchException>(
            () => store.Add(new NodeRef("a", "2"), "too short", new[] { 1.0f, 0.0f }));
    }

    [Fact]
    public void RejectsAnEmptyVector()
    {
        var store = new EmbeddingStore(new MockEncoder(8));
        Assert.Throws<DimensionMismatchException>(
            () => store.Add(new NodeRef("a", "1"), "empty", Array.Empty<float>()));
    }

    [Fact]
    public void RejectsAQueryOfTheWrongWidth()
    {
        var store = new EmbeddingStore(new MockEncoder(8));
        store.Add(new NodeRef("a", "1"), "full width");
        Assert.Throws<DimensionMismatchException>(
            () => store.SimilaritySearchVector(new[] { 1.0f, 0.0f }, 1, null, -1.0f));
    }

    [Fact]
    public void PinsItsDimensionToTheFirstVector()
    {
        var store = new EmbeddingStore();
        Assert.Null(store.Dimension);
        store.Add(new NodeRef("a", "1"), "x", new[] { 1.0f, 0.0f, 0.0f });
        Assert.Equal(3, store.Dimension);
    }

    [Fact]
    public void RejectsTextWithNoEncoder()
    {
        var store = new EmbeddingStore();
        Assert.Throws<NoEncoderException>(() => store.Add(new NodeRef("a", "1"), "no encoder"));
    }
}

public class SeedScoringTests
{
    private static SemanticGraph SeededGraph()
    {
        var graph = new SemanticGraph("seeds", new FixedEncoder(), autoEmbed: false);
        graph.AddNode("person", "1");
        graph.AddNode("person", "2");
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));
        // person:1 matches the query perfectly, person:2 slightly less well.
        graph.EmbeddingStore.Add(new NodeRef("person", "1"), "a", new[] { 1.0f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "2"), "b",
            new[] { 0.9f, MathF.Sqrt(0.19f), 0f, 0f });
        return graph;
    }

    private static SemanticSearchResult ById(IReadOnlyList<SemanticSearchResult> results, string id) =>
        results.First(r => r.NodeRef.Id == id);

    [Fact]
    public void SeedKeepsItsOwnScore()
    {
        // Seeds were inserted only when absent from the result map, so an entry
        // written by an earlier traversal won even when it scored lower:
        // person:2 is itself a 0.9 match at hop 0, but came back at
        // seed(1.0) * decay(0.8) = 0.8, at hop 1, down a path through person:1
        // that it never needed.
        var graph = SeededGraph();
        var results = graph.SemanticMultiHop("query", new MultiHopOptions
        {
            RelType = "KNOWS", MaxHops = 2, Decay = 0.8f, Threshold = 0.3f,
        });

        var second = ById(results, "2");
        Assert.True(Math.Abs(second.Score - 0.9f) < 1e-5f, $"score was {second.Score}");
        Assert.Equal(0, second.HopCount);
        Assert.Single(second.Path);
        Assert.Equal("2", second.Path[0].Id);
    }

    [Fact]
    public void TraversalStillReachesNonSeeds()
    {
        var graph = SeededGraph();
        graph.AddNode("person", "3");
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "2"), new NodeRef("person", "3"));
        graph.EmbeddingStore.Add(new NodeRef("person", "3"), "c", new[] { 0f, 0f, 1.0f, 0f });

        var results = graph.SemanticMultiHop("query", new MultiHopOptions
        {
            RelType = "KNOWS", MaxHops = 2, Threshold = 0.3f,
        });

        var third = ById(results, "3");
        Assert.True(third.HopCount > 0);
        Assert.True(third.Score > 0f);
    }

    [Fact]
    public void NegativeSeedDoesNotGrowWithDistance()
    {
        var graph = new SemanticGraph("neg", new FixedEncoder(), autoEmbed: false);
        graph.AddNode("person", "1");
        graph.AddNode("person", "2");
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));
        graph.EmbeddingStore.Add(new NodeRef("person", "1"), "a", new[] { -1.0f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "2"), "b", new[] { 0f, 0f, 1.0f, 0f });

        var results = graph.SemanticMultiHop("query", new MultiHopOptions
        {
            RelType = "KNOWS", MaxHops = 1, Threshold = -1.0f,
        });

        Assert.True(Math.Abs(ById(results, "1").Score + 1.0f) < 1e-5f);
        // 0.0, not -1.0 * 0.8 = -0.8, which would have outranked the seed.
        Assert.True(Math.Abs(ById(results, "2").Score) < 1e-5f);
    }

    [Fact]
    public void BlendLiftsARelevantDistantNode()
    {
        var graph = new SemanticGraph("blend", new FixedEncoder(), autoEmbed: false);
        foreach (var id in new[] { "1", "2", "3" }) graph.AddNode("person", id);
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "3"));
        graph.EmbeddingStore.Add(new NodeRef("person", "1"), "seed", new[] { 1.0f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "2"), "off", new[] { 0f, 1.0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "3"), "on", new[] { 0.8f, 0.6f, 0f, 0f });

        MultiHopOptions Options(float blend) => new()
        {
            RelType = "KNOWS", MaxHops = 1, TopKSeeds = 1, Threshold = 0.5f, Blend = blend,
        };

        var plain = graph.SemanticMultiHop("query", Options(0f));
        Assert.True(Math.Abs(ById(plain, "2").Score - ById(plain, "3").Score) < 1e-5f);

        var blended = graph.SemanticMultiHop("query", Options(0.5f));
        Assert.True(ById(blended, "3").Score > ById(blended, "2").Score);
        // A seed is unaffected by blending: its own score is its base.
        Assert.True(Math.Abs(ById(blended, "1").Score - ById(plain, "1").Score) < 1e-5f);
    }

    [Fact]
    public void RejectsABlendOutsideRange()
    {
        var graph = SeededGraph();
        Assert.Throws<EmbeddingException>(
            () => graph.SemanticMultiHop("query", new MultiHopOptions { Blend = 1.5f }));
    }

    [Fact]
    public void ResultsCarryTheResolvedNode()
    {
        var graph = SeededGraph();
        var results = graph.SemanticMultiHop("query", new MultiHopOptions
        {
            RelType = "KNOWS", Threshold = 0.3f,
        });
        Assert.All(results, r => Assert.NotNull(r.Node));
    }
}

public class SemanticGraphTests
{
    [Fact]
    public void AddNodeAutoEmbedsFromProperties()
    {
        var graph = new SemanticGraph("auto", new MockEncoder(16));
        graph.AddNode("person", "1", new Dictionary<string, object?>
        {
            ["name"] = "Alice", ["description"] = "engineer",
        });

        Assert.Equal(1, graph.EmbeddingStore.Count);
        Assert.Equal("Alice engineer", graph.GetEmbedding(new NodeRef("person", "1"))!.Text);
    }

    [Fact]
    public void LeavesNodesUnembeddedWhenAutoEmbedIsOff()
    {
        var graph = new SemanticGraph("manual", new MockEncoder(16), autoEmbed: false);
        graph.AddNode("person", "1", new Dictionary<string, object?> { ["name"] = "Alice" });
        Assert.Equal(0, graph.EmbeddingStore.Count);
    }

    [Fact]
    public void ExplicitEmbedTextWins()
    {
        var graph = new SemanticGraph("explicit", new MockEncoder(16));
        graph.AddNode("person", "1", new Dictionary<string, object?> { ["name"] = "Alice" },
            "a completely different text");
        Assert.Equal("a completely different text",
            graph.GetEmbedding(new NodeRef("person", "1"))!.Text);
    }

    [Fact]
    public void RemoveNodeRemovesItsEmbedding()
    {
        var graph = new SemanticGraph("remove", new MockEncoder(16));
        graph.AddNode("person", "1", new Dictionary<string, object?> { ["name"] = "Alice" });
        Assert.Equal(1, graph.EmbeddingStore.Count);
        graph.RemoveNode("person", "1");
        Assert.Equal(0, graph.EmbeddingStore.Count);
        Assert.Equal(0, graph.Graph.NodeCount());
    }

    [Fact]
    public void GraphTraversalStillWorks()
    {
        var graph = new SemanticGraph("traverse", new MockEncoder(16));
        graph.AddNode("person", "1", new Dictionary<string, object?> { ["name"] = "Alice" });
        graph.AddNode("person", "2", new Dictionary<string, object?> { ["name"] = "Bob" });
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));

        var neighbors = graph.Graph.Neighbors(new NodeRef("person", "1"), "KNOWS");
        Assert.Single(neighbors);
        Assert.Equal("2", neighbors[0].Id);
    }

    [Fact]
    public void SimilarToNodeExcludesItself()
    {
        var graph = new SemanticGraph("similar", new MockEncoder(16));
        foreach (var (id, name) in new[] { ("1", "Alice"), ("2", "Bob"), ("3", "Carol") })
            graph.AddNode("person", id, new Dictionary<string, object?> { ["name"] = name });

        var results = graph.SimilarToNode(new NodeRef("person", "1"), 2, null, -1.0f);
        Assert.Equal(2, results.Count);
        Assert.All(results, r => Assert.NotEqual("1", r.NodeRef.Id));
    }

    [Fact]
    public void SimilarToNodeWithoutAnEmbedding()
    {
        var graph = new SemanticGraph("similar", new MockEncoder(16), autoEmbed: false);
        graph.AddNode("person", "9");
        Assert.Empty(graph.SimilarToNode(new NodeRef("person", "9"), 5, null, -1.0f));
    }

    [Fact]
    public void SemanticPathFindsARoute()
    {
        var graph = new SemanticGraph("paths", new FixedEncoder(), autoEmbed: false);
        foreach (var id in new[] { "1", "2", "3" }) graph.AddNode("person", id);
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "2"), new NodeRef("person", "3"));
        // Distinct vectors and a threshold that admits only person:1, so the
        // seed is deterministic; otherwise a run seeded on the target would
        // find the zero-hop path from the target to itself.
        graph.EmbeddingStore.Add(new NodeRef("person", "1"), "a", new[] { 1.0f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "2"), "b", new[] { 0f, 1.0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "3"), "c", new[] { 0f, 0f, 1.0f, 0f });

        var result = graph.SemanticPath("query", new NodeRef("person", "3"), "KNOWS",
            maxHops: 5, topKSeeds: 1, threshold: 0.5f);

        Assert.NotNull(result);
        Assert.Equal("3", result!.NodeRef.Id);
        Assert.Equal(2, result.HopCount);
        Assert.True(Math.Abs(result.Score - 0.81f) < 1e-5f, $"score was {result.Score}");
    }

    [Fact]
    public void SemanticPathReturnsNullWithoutARoute()
    {
        var graph = new SemanticGraph("paths", new FixedEncoder(), autoEmbed: false);
        graph.AddNode("person", "1");
        graph.AddNode("person", "2");
        graph.EmbeddingStore.Add(new NodeRef("person", "1"), "a", new[] { 1.0f, 0f, 0f, 0f });
        graph.EmbeddingStore.Add(new NodeRef("person", "2"), "b", new[] { 0f, 1.0f, 0f, 0f });

        Assert.Null(graph.SemanticPath("query", new NodeRef("person", "2"), "KNOWS",
            maxHops: 5, topKSeeds: 1, threshold: 0.5f));
    }

    [Fact]
    public void SemanticSubgraphIsAWorkingGraph()
    {
        var graph = new SemanticGraph("full", new MockEncoder(16));
        foreach (var (id, name) in new[] { ("1", "Alice"), ("2", "Bob"), ("3", "Carol") })
            graph.AddNode("person", id, new Dictionary<string, object?> { ["name"] = name });
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "1"), new NodeRef("person", "2"));
        graph.Graph.AddEdge("KNOWS", new NodeRef("person", "2"), new NodeRef("person", "3"));

        var sub = graph.SemanticSubgraph("Alice", new MultiHopOptions
        {
            RelType = "KNOWS", MaxHops = 1, TopKSeeds = 1, TopKResults = 2, Threshold = -1.0f,
        }, "slice");

        Assert.Equal("slice", sub.Name);
        Assert.True(sub.Graph.NodeCount() > 0);
        Assert.True(sub.Graph.NodeCount() <= 2);
        // every kept node brought its embedding along
        Assert.Equal(sub.Graph.NodeCount(), sub.EmbeddingStore.Count);
    }

    [Fact]
    public void EmbedAllNodesBatches()
    {
        var graph = new SemanticGraph("batch", new MockEncoder(16), autoEmbed: false);
        for (int i = 0; i < 10; i++)
            graph.AddNode("doc", i.ToString(),
                new Dictionary<string, object?> { ["name"] = $"Document {i}" });

        Assert.Equal(10, graph.EmbedAllNodes());
        Assert.Equal(10, graph.EmbeddingStore.Count);
    }

    [Fact]
    public void EmbedAllNodesCanSkipExisting()
    {
        var graph = new SemanticGraph("skip", new MockEncoder(16), autoEmbed: false);
        graph.AddNode("doc", "1", new Dictionary<string, object?> { ["name"] = "One" });
        graph.AddNode("doc", "2", new Dictionary<string, object?> { ["name"] = "Two" });
        graph.EmbedNode(new NodeRef("doc", "1"), "already embedded");

        Assert.Equal(1, graph.EmbedAllNodes(skipExisting: true));
        Assert.Equal("already embedded", graph.GetEmbedding(new NodeRef("doc", "1"))!.Text);
    }
}

public class PropertyRenderingTests
{
    // The same golden string is asserted in all six suites. ISONGraph 1.4.0
    // made property values typed, which is where the ports could drift: .NET
    // renders a bool as "True", Python as "True" too, JavaScript as "true".
    // Different text means a different vector, which would break the
    // cross-port guarantee.
    [Fact]
    public void ProducesEmbedTextIdenticalToEveryOtherPort()
    {
        var graph = new SemanticGraph("props", new MockEncoder(8));
        graph.AddNode("doc", "1", new Dictionary<string, object?>
        {
            ["name"] = "Report",
            ["description"] = null,
            ["content"] = 1.5,
            ["text"] = true,
        });

        Assert.Equal("Report 1.5 true", graph.GetEmbedding(new NodeRef("doc", "1"))!.Text);
    }

    [Fact]
    public void NullContributesNothing()
    {
        var graph = new SemanticGraph("props", new MockEncoder(8));
        Assert.Equal("kept", graph.EmbedTextFor(new Dictionary<string, object?>
        {
            ["name"] = null,
            ["description"] = "kept",
        }));
    }

    [Fact]
    public void BooleansUseTheIsonSpelling()
    {
        Assert.Equal("true", SemanticGraph.PropertyText(true));
        Assert.Equal("false", SemanticGraph.PropertyText(false));
    }

    [Fact]
    public void FloatsUseTheInvariantCulture()
    {
        // "1.5" on every machine, never "1,5".
        Assert.Equal("1.5", SemanticGraph.PropertyText(1.5));
    }
}

public class PortabilityTests
{
    [Fact]
    public void RoundTripsThroughJson()
    {
        var store = new EmbeddingStore(new MockEncoder(8));
        store.Add(new NodeRef("person", "1"), "Alice");
        store.Add(new NodeRef("account", "007"), "Bond");

        var json = store.ToJson();
        Assert.Contains(IsonGraphVectorVersion.EmbeddingFormat, json, StringComparison.Ordinal);

        var other = new EmbeddingStore(new MockEncoder(8));
        Assert.Equal(2, other.FromJson(json));
        Assert.Equal("Alice", other.Get(new NodeRef("person", "1"))!.Text);
        // a numeric-looking string id keeps its exact form
        Assert.Equal("Bond", other.Get(new NodeRef("account", "007"))!.Text);
    }

    [Fact]
    public void RejectsAForeignPayload()
    {
        var store = new EmbeddingStore(new MockEncoder(8));
        Assert.Throws<EmbeddingException>(() => store.FromJson(
            """{"format":"something-else","version":1,"embeddings":[]}"""));
    }

    [Fact]
    public void ImportsAPayloadWrittenByThePythonPort()
    {
        // Captured verbatim from the Python port. Its node ids are typed, so an
        // int id arrives unquoted ("id":1) where this port always writes a
        // string - both have to land on the same NodeRef here.
        const string pythonPayload = """
            {"format":"ison-embeddings","version":1,"dimension":4,"count":2,"embeddings":[{"type":"person","id":1,"id_type":"int","text":"Alice","model":"mock-4","vector":[-0.8610728979110718,-0.33433854579925537,0.8408418893814087,-0.2771846055984497]},{"type":"account","id":"007","id_type":"str","text":"Bond","model":"mock-4","vector":[-0.9531009197235107,0.9167747497558594,0.5073114633560181,-0.0420149564743042]}]}
            """;

        var encoder = new MockEncoder(4);
        var store = new EmbeddingStore(encoder);
        Assert.Equal(2, store.FromJson(pythonPayload));
        Assert.Equal("Alice", store.Get(new NodeRef("person", "1"))!.Text);
        Assert.Equal("Bond", store.Get(new NodeRef("account", "007"))!.Text);

        // and the vectors match what this port encodes for the same text
        var expected = encoder.Encode("Alice");
        var stored = store.Get(new NodeRef("person", "1"))!.Vector;
        Assert.Equal(expected.Length, stored.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(stored[i] - expected[i]) < 1e-6f);
    }
}

public class IntegrationTests
{
    [Fact]
    public void FullWorkflow()
    {
        var graph = new SemanticGraph("social", new MockEncoder(64));

        graph.AddNode("user", "1", new Dictionary<string, object?> { ["name"] = "Alice" },
            "Alice is a software engineer");
        graph.AddNode("user", "2", new Dictionary<string, object?> { ["name"] = "Bob" },
            "Bob is a data scientist");
        graph.AddNode("project", "1", new Dictionary<string, object?> { ["name"] = "ML Platform" },
            "Machine learning infrastructure");

        graph.Graph.AddEdge("KNOWS", new NodeRef("user", "1"), new NodeRef("user", "2"));
        graph.Graph.AddEdge("WORKS_ON", new NodeRef("user", "2"), new NodeRef("project", "1"));

        Assert.Equal(3, graph.Graph.NodeCount());
        Assert.Equal(2, graph.Graph.EdgeCount());
        Assert.Equal(3, graph.EmbeddingStore.Count);

        var similar = graph.SimilaritySearch("Machine learning infrastructure", 5, null, -1.0f);
        Assert.NotEmpty(similar);
        Assert.Equal("project", similar[0].NodeRef.Type);

        var related = graph.SemanticMultiHop("Alice is a software engineer", new MultiHopOptions
        {
            MaxHops = 2, Threshold = -1.0f,
        });
        Assert.NotEmpty(related);
        // results come back sorted by score
        for (int i = 1; i < related.Count; i++)
            Assert.True(related[i - 1].Score >= related[i].Score);

        Assert.Contains("embeddings=3", graph.ToString(), StringComparison.Ordinal);
    }
}
