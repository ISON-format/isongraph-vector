using IsonGraph;
using IsonGraph.Vector;
using IsonGraph.Vector.Sqlite;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>The model lifecycle on the SQLite-backed store (same rules as the in-memory one).</summary>
public class SqliteLifecycleTests
{
    private static readonly NodeRef A = new("person", "1");
    private static readonly NodeRef B = new("person", "2");

    private static MockEncoder Retagged(int dimension, string name) => new(dimension) { ModelName = name };

    [Fact]
    public void RecordsTheModelAndFindsStaleVectors()
    {
        using var store = new SqliteEmbeddingStore(":memory:", new MockEncoder(8));
        store.Add(A, "alice");
        Assert.Equal(new Dictionary<string, int> { ["mock-8"] = 1 }, store.Models());
        Assert.Equal("mock-8", store.Get(A)!.Model);
        Assert.Empty(store.Stale());

        store.Encoder = Retagged(8, "next-8");
        Assert.Equal(new[] { A }, store.Stale());
    }

    [Fact]
    public void ProvidedVectorsAreStaleOnlyWhenAsked()
    {
        using var store = new SqliteEmbeddingStore(":memory:", null);
        store.Add(A, "alice", new float[8]);
        Assert.Equal(new Dictionary<string, int> { ["provided"] = 1 }, store.Models());
        store.Encoder = new MockEncoder(8);
        Assert.Empty(store.Stale());
        Assert.Equal(new[] { A }, store.Stale(includeUnknown: true));
        // a dimension mismatch always counts
        store.Encoder = new MockEncoder(16);
        Assert.Equal(new[] { A }, store.Stale());
    }

    [Fact]
    public void TextSearchWithADifferentModelWarnsOnceOrThrows()
    {
        using var warnStore = new SqliteEmbeddingStore(":memory:", new MockEncoder(8));
        var warnings = new List<string>();
        warnStore.Warning += warnings.Add;
        warnStore.Add(A, "alice");
        warnStore.Encoder = Retagged(8, "other-8");
        warnStore.SimilaritySearch("x", 1, threshold: -1f);
        warnStore.SimilaritySearch("y", 1, threshold: -1f);
        Assert.Single(warnings);
        Assert.Contains("other-8", warnings[0]);

        using var strict = new SqliteEmbeddingStore(":memory:", new MockEncoder(8),
            onModelMismatch: ModelMismatchMode.Error);
        strict.Add(A, "alice");
        strict.Encoder = Retagged(8, "other-8");
        Assert.Throws<ModelMismatchException>(() => strict.SimilaritySearch("x", 1));
        // a vector query is never checked
        strict.SimilaritySearchVector(new MockEncoder(8).Encode("x"), 1, null, -1f);
    }

    [Fact]
    public void SidecarCarriesModelAndMetadata()
    {
        using var source = new SqliteEmbeddingStore(":memory:", new MockEncoder(8, version: "7"));
        source.Add(A, "alice");
        var payload = source.ExportEmbeddings("sha256:abc");
        Assert.Equal("mock-8", payload.EncoderModel);
        Assert.Equal("7", payload.EncoderVersion);
        Assert.Equal("sha256:abc", payload.GraphDigest);
        Assert.Equal(GraphDigest.Basis, payload.GraphDigestBasis);
        Assert.Equal("mock-8", payload.Embeddings[0].Model);

        using var target = new SqliteEmbeddingStore(":memory:", new MockEncoder(8));
        Assert.Equal(1, target.ImportEmbeddings(payload));
        Assert.Equal("mock-8", target.Get(A)!.Model);
        Assert.Equal("sha256:abc", target.LastImportMeta["graph_digest"]);
        Assert.Equal("7", target.LastImportMeta["encoder_version"]);
    }

    [Fact]
    public void ClearCanKeepOrResetTheDimension()
    {
        using var store = new SqliteEmbeddingStore(":memory:", null);
        store.Add(A, "a", new float[4]);
        store.Clear(resetDimension: false);
        Assert.Equal(4, store.Dimension);
        Assert.Throws<DimensionMismatchException>(() => store.Add(A, "a", new float[8]));
        store.Clear();
        Assert.Null(store.Dimension);
        store.Add(A, "a", new float[8]);
        Assert.Equal(8, store.Dimension);
    }

    [Fact]
    public void ResettingTheDimensionRebuildsTheVecIndex()
    {
        using var store = new SqliteEmbeddingStore(":memory:", new MockEncoder(8), sqliteVec: true);
        store.Add(A, "alice");
        store.Add(B, "bob");
        store.Clear();                       // drops the fixed-width index
        store.Encoder = new MockEncoder(16);
        store.Add(A, "alice");
        store.Add(B, "bob");
        var hits = store.SimilaritySearch("alice", 2, threshold: -1f);
        Assert.Equal(2, hits.Count);
        Assert.Equal(A, hits[0].NodeRef);
    }
}
