using IsonGraph;
using IsonGraph.Vector;
using IsonGraph.Vector.Sqlite;
using Xunit;

namespace IsonGraph.Vector.Tests;

public class SqliteEmbeddingStoreTests
{
    private static SqliteEmbeddingStore Store(bool sqliteVec = false, int dim = 32) =>
        new(":memory:", new MockEncoder(dim), sqliteVec);

    [Fact]
    public void AddsAndReadsBack()
    {
        using var store = Store();
        store.Add(new NodeRef("person", "1"), "Alice is an engineer");

        var record = store.Get(new NodeRef("person", "1"));
        Assert.NotNull(record);
        Assert.Equal("Alice is an engineer", record!.Text);
        Assert.Equal(32, record.Vector.Length);
        Assert.Equal(1, store.Count);
    }

    [Fact]
    public void RemovesAndClears()
    {
        using var store = Store();
        store.Add(new NodeRef("a", "1"), "one");
        store.Add(new NodeRef("a", "2"), "two");

        Assert.True(store.Remove(new NodeRef("a", "1")));
        Assert.False(store.Remove(new NodeRef("a", "1")));
        Assert.Equal(1, store.Count);

        store.Clear();
        Assert.Equal(0, store.Count);
    }

    [Fact]
    public void OverwritesRatherThanDuplicating()
    {
        using var store = Store();
        store.Add(new NodeRef("a", "1"), "first");
        store.Add(new NodeRef("a", "1"), "second");
        Assert.Equal(1, store.Count);
        Assert.Equal("second", store.Get(new NodeRef("a", "1"))!.Text);
    }

    [Fact]
    public void ValidatesDimensions()
    {
        using var store = Store();
        store.Add(new NodeRef("a", "1"), "full width");

        Assert.Throws<DimensionMismatchException>(
            () => store.Add(new NodeRef("a", "2"), "short", new[] { 1f, 0f }));
        Assert.Throws<DimensionMismatchException>(
            () => store.Add(new NodeRef("a", "3"), "empty", Array.Empty<float>()));
        Assert.Throws<DimensionMismatchException>(
            () => store.SimilaritySearchVector(new[] { 1f, 0f }, 1, null, -1.0f));
    }

    [Fact]
    public void BatchesInOneTransaction()
    {
        using var store = Store();
        var entries = Enumerable.Range(0, 10)
            .Select(i => (new NodeRef("doc", i.ToString()), $"document {i}"))
            .ToList();
        Assert.Equal(10, store.AddBatch(entries));
        Assert.Equal(10, store.Count);
    }

    [Fact]
    public void SurvivesCloseAndReopen()
    {
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"igv-{Guid.NewGuid():N}.db");
        try
        {
            float[] probe;
            using (var store = new SqliteEmbeddingStore(path, new MockEncoder(32)))
            {
                store.Add(new NodeRef("person", "1"), "Alice");
                probe = store.Get(new NodeRef("person", "1"))!.Vector;
            }

            using var reopened = new SqliteEmbeddingStore(path, new MockEncoder(32));
            Assert.Equal(1, reopened.Count);
            Assert.Equal(probe, reopened.Get(new NodeRef("person", "1"))!.Vector);
            // dimension is recovered from the stored rows
            Assert.Equal(32, reopened.Dimension);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // -- sqlite-vec index --------------------------------------------------

    [Fact]
    public void ReportsTheBackend()
    {
        using var plain = Store();
        using var indexed = Store(sqliteVec: true);
        Assert.False(plain.SqliteVec);
        Assert.True(indexed.SqliteVec);
    }

    [Fact]
    public void IndexMatchesTheScanExactly()
    {
        var encoder = new MockEncoder(32);
        var entries = Enumerable.Range(0, 200)
            .Select(i => (new NodeRef("n", i.ToString()), $"text {i}"))
            .ToList();

        using var scan = Store();
        using var indexed = Store(sqliteVec: true);
        scan.AddBatch(entries);
        indexed.AddBatch(entries);

        foreach (var probe in new[] { "text 0", "text 57", "text 199" })
        {
            var query = encoder.Encode(probe);
            var want = scan.SimilaritySearchVector(query, 10, null, -1.0f);
            var got = indexed.SimilaritySearchVector(query, 10, null, -1.0f);

            Assert.Equal(want.Count, got.Count);
            for (int i = 0; i < want.Count; i++)
            {
                Assert.Equal(want[i].NodeRef, got[i].NodeRef);
                Assert.True(Math.Abs(want[i].Score - got[i].Score) < 1e-5f);
            }
        }
    }

    [Fact]
    public void TypeFilterStaysExact()
    {
        // Filtering after a top-k fetch would drop the rare node entirely; the
        // type is a vec0 metadata column so the search itself is narrowed.
        using var store = Store(sqliteVec: true);
        for (int i = 0; i < 50; i++)
            store.Add(new NodeRef("crowd", i.ToString()), $"crowd {i}");
        store.Add(new NodeRef("rare", "1"), "the only rare node");

        var results = store.SimilaritySearch("anything", 5, "rare", -1.0f);
        Assert.Single(results);
        Assert.Equal(new NodeRef("rare", "1"), results[0].NodeRef);
    }

    [Fact]
    public void UpdatesAndDeletesReachTheIndex()
    {
        using var store = Store(sqliteVec: true);
        store.Add(new NodeRef("a", "1"), "first");
        store.Add(new NodeRef("a", "2"), "second");

        var second = store.Get(new NodeRef("a", "2"))!.Vector;
        store.Add(new NodeRef("a", "1"), "first", second);

        var top = store.SimilaritySearchVector(second, 2, null, -1.0f);
        Assert.Equal(2, top.Count);
        Assert.All(top, r => Assert.True(Math.Abs(r.Score - 1.0f) < 1e-5f));

        store.Remove(new NodeRef("a", "1"));
        var after = store.SimilaritySearchVector(second, 10, null, -1.0f);
        Assert.Single(after);
        Assert.Equal(new NodeRef("a", "2"), after[0].NodeRef);
    }

    [Fact]
    public void IndexesADatabaseWrittenWithoutIt()
    {
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"igv-{Guid.NewGuid():N}.db");
        try
        {
            float[] probe;
            using (var plain = new SqliteEmbeddingStore(path, new MockEncoder(32)))
            {
                for (int i = 0; i < 20; i++) plain.Add(new NodeRef("n", i.ToString()), $"text {i}");
                probe = plain.Get(new NodeRef("n", "7"))!.Vector;
            }

            using var indexed = new SqliteEmbeddingStore(path, new MockEncoder(32), sqliteVec: true);
            Assert.Equal(20, indexed.Count);

            var results = indexed.SimilaritySearchVector(probe, 3, null, -1.0f);
            Assert.Equal(new NodeRef("n", "7"), results[0].NodeRef);
            Assert.True(Math.Abs(results[0].Score - 1.0f) < 1e-5f);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ScoreMapStillCoversEverything()
    {
        using var store = Store(sqliteVec: true);
        for (int i = 0; i < 30; i++) store.Add(new NodeRef("n", i.ToString()), $"text {i}");
        Assert.Equal(30, store.ScoreMap(new MockEncoder(32).Encode("anything")).Count);
    }

    [Fact]
    public void ZeroQueryVectorFallsBackToTheScan()
    {
        using var store = new SqliteEmbeddingStore(":memory:", new MockEncoder(4), sqliteVec: true);
        store.Add(new NodeRef("a", "1"), "x", new[] { 1f, 0f, 0f, 0f });

        var results = store.SimilaritySearchVector(new[] { 0f, 0f, 0f, 0f }, 1, null, -1.0f);
        Assert.Single(results);
        Assert.Equal(0f, results[0].Score);
    }

    [Fact]
    public void RoundTripsThePortablePayload()
    {
        using var source = Store();
        source.Add(new NodeRef("person", "1"), "Alice");
        source.Add(new NodeRef("account", "007"), "Bond");

        using var target = Store(sqliteVec: true);
        Assert.Equal(2, target.FromJson(source.ToJson()));
        Assert.Equal("Bond", target.Get(new NodeRef("account", "007"))!.Text);
    }

    [Fact]
    public void WritesTheSchemaEveryPortShares()
    {
        // A database written here must be readable by the Python, Rust,
        // TypeScript, JavaScript and C++ ports, which means the same table,
        // the same columns, and little-endian float32 vectors.
        var path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"igv-{Guid.NewGuid():N}.db");
        try
        {
            using (var store = new SqliteEmbeddingStore(path, new MockEncoder(4)))
                store.Add(new NodeRef("person", "1"), "Alice");

            // Pooling=False so this inspection connection releases the file
            // handle on dispose and the temp database can be deleted.
            using var conn = new Microsoft.Data.Sqlite.SqliteConnection(
                $"Data Source={path};Pooling=False");
            conn.Open();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT id, node_type, node_id, node_id_type, vector, text " +
                              "FROM embeddings";
            using var reader = cmd.ExecuteReader();
            Assert.True(reader.Read());

            Assert.Equal(16, reader.GetString(0).Length);        // md5 prefix
            Assert.Equal("person", reader.GetString(1));
            Assert.Equal("1", reader.GetString(2));
            Assert.Equal("str", reader.GetString(3));
            Assert.Equal(16, ((byte[])reader["vector"]).Length);  // 4 dims x 4 bytes
            Assert.Equal("Alice", reader.GetString(5));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
