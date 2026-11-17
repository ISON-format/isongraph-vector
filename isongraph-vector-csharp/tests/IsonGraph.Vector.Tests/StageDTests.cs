using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

internal static class StageD
{
    public static float[] Vec(Random r, int dim)
    {
        var v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float)(r.NextDouble() * 2 - 1);
        return v;
    }

    public static (string Key, int Score, string Text)[] Shape(IReadOnlyList<SimilarityResult> r) =>
        r.Select(x => ($"{x.NodeRef.Type}:{x.NodeRef.Id}", BitConverter.SingleToInt32Bits(x.Score), x.Text)).ToArray();

    public static EmbeddingStore Filled(int n, int dim, int seed = 1, bool ties = true, AnnMode ann = AnnMode.None, int annMin = 0, int? nprobe = null)
    {
        var rng = new Random(seed);
        var store = new EmbeddingStore(ann: ann, annMinSize: annMin, annNprobe: nprobe);
        var keep = new List<float[]>();
        for (int i = 0; i < n; i++)
        {
            float[] v;
            if (ties && i % 11 == 3 && keep.Count > 0) v = (float[])keep[rng.Next(keep.Count)].Clone();   // exact duplicate
            else if (ties && i % 29 == 7) v = new float[dim];                                             // zero vector
            else v = Vec(rng, dim);
            keep.Add(v);
            store.Add(new NodeRef(i % 3 == 0 ? "a" : i % 3 == 1 ? "b" : "c", i.ToString()), "text " + i, v);
        }
        return store;
    }

    public static string TempDir()
    {
        var dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ison-stage-d-" + Guid.NewGuid().ToString("N")[..10]);
        Directory.CreateDirectory(dir);
        return dir;
    }

    public static string FixtureDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            var candidate = System.IO.Path.Combine(dir.FullName, "tests", "fixtures", "snapshot");
            if (File.Exists(System.IO.Path.Combine(candidate, "golden.ison"))) return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("tests/fixtures/snapshot (the golden snapshot fixture) not found above " + AppContext.BaseDirectory);
    }
}

public class OneCopyTests
{
    [Fact]
    public void Invariants_hold_through_random_writes_updates_deletes_and_compaction()
    {
        foreach (var ann in new[] { AnnMode.None, AnnMode.Ivf })
        {
            var store = new EmbeddingStore(ann: ann, annMinSize: 300, annSeed: 2);
            var rng = new Random(5);
            var live = new Dictionary<string, float[]>();
            for (int step = 0; step < 6000; step++)
            {
                int id = rng.Next(1500);
                var r = new NodeRef(id % 2 == 0 ? "x" : "y", id.ToString());
                if (rng.NextDouble() < 0.35)
                {
                    bool removed = store.Remove(r);
                    Assert.Equal(live.Remove($"{r.Type}:{r.Id}"), removed);
                }
                else
                {
                    var v = StageD.Vec(rng, 24);
                    store.Add(r, "t" + step, v);
                    live[$"{r.Type}:{r.Id}"] = v;
                }
                if (step % 250 == 0)
                {
                    store.SimilaritySearchVector(StageD.Vec(rng, 24), 5);   // builds / retrains the index when it qualifies
                    Assert.Null(store.VerifyInvariants());
                }
            }
            Assert.Null(store.VerifyInvariants());
            Assert.Equal(live.Count, store.Count);
            if (ann == AnnMode.Ivf) Assert.True(store.GetAnnStats().Searches.Builds >= 1);   // the index really was live through the churn
            foreach (var (key, v) in live.Take(200))
            {
                var parts = key.Split(':');
                Assert.Equal(v, store.Get(new NodeRef(parts[0], parts[1]))!.Vector);
            }
            store.Compact();
            Assert.Null(store.VerifyInvariants());
            Assert.Equal(live.Count, store.Count);
        }
    }

    [Fact]
    public void A_store_emptied_by_deletes_compacts_and_takes_new_rows()
    {
        foreach (int n in new[] { 1, 5, 1500 })
        {
            var store = StageD.Filled(n, 12);
            for (int i = 0; i < n; i++)
                Assert.True(store.Remove(new NodeRef(i % 3 == 0 ? "a" : i % 3 == 1 ? "b" : "c", i.ToString())));
            Assert.Equal(0, store.Count);
            Assert.Empty(store.SimilaritySearchVector(new float[12], 5));
            Assert.Null(store.VerifyInvariants());
            store.Add(new NodeRef("a", "again"), "x", new float[12]);
            Assert.Equal(1, store.Count);
            Assert.Null(store.VerifyInvariants());
            Assert.Single(store.SimilaritySearchVector(new float[12], 5));
        }
    }

    [Fact]
    public void Get_returns_a_copy_the_store_does_not_alias()
    {
        var store = new EmbeddingStore();
        var original = new float[] { 1f, 2f, 3f };
        store.Add(new NodeRef("t", "1"), "x", original);

        original[0] = 99f;                                   // the caller's array is not the store's
        var record = store.Get(new NodeRef("t", "1"))!;
        Assert.Equal(1f, record.Vector[0]);

        record.Vector[1] = 77f;                              // and neither is what Get hands back
        Assert.Equal(2f, store.Get(new NodeRef("t", "1"))!.Vector[1]);
        Assert.Equal(1.0, store.SimilaritySearchVector(new float[] { 1f, 2f, 3f }, 1)[0].Score, 5);
    }

    [Fact]
    public void The_store_holds_one_copy_of_every_vector_even_with_the_index_built()
    {
        const int n = 6000, dim = 64;
        var store = StageD.Filled(n, dim, ann: AnnMode.Ivf, annMin: 0);
        store.BuildAnnIndex();
        long raw = (long)n * dim * sizeof(float);
        Assert.InRange(store.MatrixBytes, raw, raw + (long)VectorTable.ChunkRows * dim * sizeof(float));   // at most one partly empty chunk over
        Assert.Equal(0, store.GetAnnStats().VectorBytes);
    }

    [Fact]
    public void Concurrent_readers_get_the_same_answers_as_one_reader()
    {
        var store = StageD.Filled(8000, 48, ann: AnnMode.Ivf, annMin: 0, nprobe: 4);
        var rng = new Random(9);
        var queries = Enumerable.Range(0, 64).Select(_ => StageD.Vec(rng, 48)).ToArray();
        var expected = queries.Select(q => StageD.Shape(store.SimilaritySearchVector(q, 10))).ToArray();
        var expectedExact = queries.Select(q => StageD.Shape(store.SimilaritySearchVector(q, 10, exact: true))).ToArray();

        var errors = new List<Exception>();
        var threads = Enumerable.Range(0, 8).Select(t => new Thread(() =>
        {
            try
            {
                for (int round = 0; round < 5; round++)
                    for (int i = 0; i < queries.Length; i++)
                    {
                        Assert.Equal(expected[i], StageD.Shape(store.SimilaritySearchVector(queries[i], 10)));
                        Assert.Equal(expectedExact[i], StageD.Shape(store.SimilaritySearchVector(queries[i], 10, exact: true)));
                    }
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
        })).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        Assert.Empty(errors);
    }

    [Fact]
    public void Readers_and_a_writer_never_see_a_torn_store()
    {
        var store = StageD.Filled(3000, 16, ties: false);
        // the writer replaces one row with a vector whose direction is exactly +x or exactly +y; a reader
        // must see one or the other, never a mix, and never a result twice
        var target = new NodeRef("w", "flip");
        store.Add(target, "flip", Enumerable.Range(0, 16).Select(i => i == 0 ? 1f : 0f).ToArray());
        var stop = false;
        var errors = new List<Exception>();
        var writer = new Thread(() =>
        {
            try
            {
                for (int i = 0; !Volatile.Read(ref stop); i++)
                {
                    var v = new float[16];
                    v[i % 2] = 1f;
                    store.Add(target, "flip", v);
                    store.Add(new NodeRef("w", "churn" + (i % 50)), "c", StageD.Vec(new Random(i), 16));
                    store.Remove(new NodeRef("w", "churn" + ((i + 25) % 50)));
                }
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
        });
        var readers = Enumerable.Range(0, 4).Select(_ => new Thread(() =>
        {
            try
            {
                var qx = Enumerable.Range(0, 16).Select(i => i == 0 ? 1f : 0f).ToArray();
                for (int i = 0; i < 400; i++)
                {
                    var res = store.SimilaritySearchVector(qx, null, "w", -1f);
                    Assert.Equal(res.Count, res.Select(r => r.NodeRef).Distinct().Count());
                    var flip = res.First(r => r.NodeRef == target);
                    Assert.True(flip.Score == 1f || flip.Score == 0f, $"torn score {flip.Score}");
                }
            }
            catch (Exception e) { lock (errors) errors.Add(e); }
        })).ToList();
        writer.Start();
        readers.ForEach(t => t.Start());
        readers.ForEach(t => t.Join());
        stop = true;
        writer.Join();
        Assert.Empty(errors);
        Assert.Null(store.VerifyInvariants());
    }
}

public class BatchSearchTests
{
    [Theory]
    [InlineData(AnnMode.None, 0)]
    [InlineData(AnnMode.None, 1)]
    [InlineData(AnnMode.None, 3)]
    [InlineData(AnnMode.Ivf, 0)]
    [InlineData(AnnMode.Ivf, 2)]
    public void A_batch_equals_the_single_query_answers_bit_for_bit(AnnMode ann, int degree)
    {
        var store = StageD.Filled(9000, 40, ann: ann, annMin: 0, nprobe: 5);
        var rng = new Random(77);
        var queries = Enumerable.Range(0, 50).Select(_ => StageD.Vec(rng, 40)).ToList();
        queries.Add(new float[40]);                                  // a zero query
        queries.Add(store.Get(new NodeRef("a", "3"))!.Vector);        // a stored vector (tied with its duplicates)

        foreach (var (topK, type, threshold) in new (int?, string?, float)[] { (10, null, 0f), (1, "b", -1f), (200, null, 0.1f), (null, "c", 0.2f) })
        {
            var batch = store.SimilaritySearchBatch(queries, topK, type, threshold, maxDegreeOfParallelism: degree);
            Assert.Equal(queries.Count, batch.Count);
            for (int i = 0; i < queries.Count; i++)
                Assert.Equal(StageD.Shape(store.SimilaritySearchVector(queries[i], topK, type, threshold)), StageD.Shape(batch[i]));
        }
    }

    [Fact]
    public void Scan_thread_count_does_not_change_an_answer()
    {
        var store = StageD.Filled(30000, 64);
        var q = StageD.Vec(new Random(3), 64);
        store.ScanThreads = 1;
        var one = StageD.Shape(store.SimilaritySearchVector(q, 25));
        foreach (int threads in new[] { 0, 2, 3, 8 })
        {
            store.ScanThreads = threads;
            Assert.Equal(one, StageD.Shape(store.SimilaritySearchVector(q, 25)));
        }
    }

    [Fact]
    public void The_text_batch_encodes_once_and_matches_single_text_searches()
    {
        var graph = new SemanticGraph("g", new MockEncoder(32));
        for (int i = 0; i < 400; i++) graph.AddNode("n", i.ToString(), new() { ["name"] = "item number " + i });
        var texts = new[] { "item number 7", "number 12", "zebra", "item" };
        var batch = graph.EmbeddingStore.SimilaritySearchBatch(texts, 5);
        for (int i = 0; i < texts.Length; i++)
            Assert.Equal(StageD.Shape(graph.EmbeddingStore.SimilaritySearch(texts[i], 5)), StageD.Shape(batch[i]));
    }

    [Fact]
    public void A_batch_validates_every_query_before_running_any()
    {
        var store = StageD.Filled(100, 8);
        var bad = new List<float[]> { new float[8], new float[3] };
        Assert.Throws<DimensionMismatchException>(() => store.SimilaritySearchBatch(bad));
        Assert.Empty(store.SimilaritySearchBatch(new List<float[]>()));
    }
}

public class SidecarStreamingTests
{
    [Fact]
    public void Streamed_sidecar_bytes_equal_the_serializer_bytes()
    {
        var store = new EmbeddingStore(new MockEncoder(10) { Version = "v-2" });
        var rng = new Random(4);
        string[] ids = { "1", "00007", "a:b", "é", "😀", "quote\"back\\slash", "ctrl\u0001\n\t", "<&>+" };
        foreach (var (id, i) in ids.Select((x, i) => (x, i)))
            store.Add(new NodeRef(i % 2 == 0 ? "t" : "Ünï", id), id + " \u2028 text", StageD.Vec(rng, 10));
        store.Add(new NodeRef("t", "zero"), "", new float[10]);
        store.Add(new NodeRef("t", "tiny"), "", Enumerable.Repeat(1e-7f, 10).ToArray());

        foreach (string? digest in new string?[] { null, "sha256:abc" })
        {
            var expected = JsonSerializer.Serialize(store.ExportEmbeddings(digest));
            using var ms = new MemoryStream();
            int count = store.WriteSidecar(ms, digest);
            Assert.Equal(store.Count, count);
            Assert.Equal(expected, Encoding.UTF8.GetString(ms.ToArray()));
        }

        var empty = new EmbeddingStore();
        using var ms2 = new MemoryStream();
        empty.WriteSidecar(ms2, null);
        Assert.Equal(JsonSerializer.Serialize(empty.ExportEmbeddings()), Encoding.UTF8.GetString(ms2.ToArray()));
    }
}

public class SnapshotTests
{
    private static SemanticGraph MakeGraph(int n = 300, int dim = 16, int seed = 3)
    {
        var graph = new SemanticGraph("snap", new MockEncoder(dim) { Version = "enc-1" });
        var rng = new Random(seed);
        for (int i = 0; i < n; i++)
        {
            var id = i % 7 == 0 ? $"id_{i}" : i.ToString("D5");
            graph.AddNode(i % 2 == 0 ? "doc" : "note", id, new() { ["name"] = "n" + i }, embedText: i % 5 == 0 ? "" : $"unicode \U0001F600 é中 \"q\" {i}\nline");
            if (i % 5 != 0 && i % 13 == 0)
                graph.EmbeddingStore.Add(new NodeRef(i % 2 == 0 ? "doc" : "note", id), "dup", i == 13 ? new float[dim] : StageD.Vec(rng, dim));
        }
        // keys that are not graph nodes, with colons and unicode in type and id (the key is "type:id")
        graph.EmbeddingStore.Add(new NodeRef("a:b", "id:with:colons"), "colons", StageD.Vec(rng, dim));
        graph.EmbeddingStore.Add(new NodeRef("ünï", "ключ"), "unicode", StageD.Vec(rng, dim));
        graph.SetEncoder(new MockEncoder(dim) { ModelName = "other-model" });
        graph.AddNode("late", "1", new() { ["name"] = "late" });
        return graph;
    }

    private static void AssertSameStore(EmbeddingStore a, EmbeddingStore b, bool compareMeta = true)
    {
        Assert.Equal(a.Count, b.Count);
        Assert.Equal(a.Dimension, b.Dimension);
        Assert.Equal(a.Models().OrderBy(x => x.Key).ToList(), b.Models().OrderBy(x => x.Key).ToList());
        var pa = a.ExportEmbeddings("d");
        var pb = b.ExportEmbeddings("d");
        Assert.Equal(JsonSerializer.Serialize(pa.Embeddings), JsonSerializer.Serialize(pb.Embeddings));              // same keys, texts, models, vector bits
        foreach (var e in pa.Embeddings)
        {
            var r = new NodeRef(e.Type, e.IdAsString());
            var x = a.Get(r)!; var y = b.Get(r)!;
            Assert.Equal(x.Text, y.Text);
            Assert.Equal(x.Model, y.Model);
            Assert.Equal(x.Vector.Select(BitConverter.SingleToInt32Bits), y.Vector.Select(BitConverter.SingleToInt32Bits));
        }
        var rng = new Random(21);
        int dim = a.Dimension ?? 0;
        for (int i = 0; i < 20 && dim > 0; i++)
        {
            var q = StageD.Vec(rng, dim);
            foreach (var (k, type) in new (int?, string?)[] { (5, null), (null, null), (3, "doc") })
                Assert.Equal(StageD.Shape(a.SimilaritySearchVector(q, k, type, -1f)), StageD.Shape(b.SimilaritySearchVector(q, k, type, -1f)));
        }
        if (compareMeta)
            Assert.Equal(a.LastImportMeta.OrderBy(x => x.Key).ToList(), b.LastImportMeta.OrderBy(x => x.Key).ToList());
    }

    private static (string Path, SemanticGraph Graph) SaveWithSnapshot(string dir, SemanticGraph? graph = null)
    {
        graph ??= MakeGraph();
        var path = System.IO.Path.Combine(dir, "g.ison");
        graph.Save(path, snapshot: true);
        return (path, graph);
    }

    private static string Snap(string path) => SemanticGraph.SnapshotPathFor(path);

    private static void Seal(string snapPath, byte[] body) =>
        File.WriteAllBytes(snapPath, body.Concat(SHA256.HashData(body)).ToArray());

    // -- round trip ---------------------------------------------------------------------

    [Fact]
    public void Own_save_then_load_uses_the_snapshot_and_equals_a_json_load()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, graph) = SaveWithSnapshot(dir);
            Assert.True(File.Exists(Snap(path)));

            var (viaSnap, report) = SemanticGraph.LoadWithReport(path, new MockEncoder(16) { Version = "enc-1" });
            Assert.Equal(LoadSource.Snapshot, report.Source);
            Assert.Null(report.SnapshotSkipped);
            Assert.Equal(graph.EmbeddingStore.Count, report.Embeddings);

            File.Delete(Snap(path));
            var (viaJson, jsonReport) = SemanticGraph.LoadWithReport(path, new MockEncoder(16) { Version = "enc-1" });
            Assert.Equal(LoadSource.Json, jsonReport.Source);
            AssertSameStore(viaSnap.EmbeddingStore, viaJson.EmbeddingStore);
            AssertSameStore(graph.EmbeddingStore, viaSnap.EmbeddingStore, compareMeta: false);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Snapshot_save_does_not_change_the_graph_or_the_json_sidecar_bytes()
    {
        var dir = StageD.TempDir();
        try
        {
            var graph = MakeGraph();
            var a = System.IO.Path.Combine(dir, "plain.ison");
            var b = System.IO.Path.Combine(dir, "snap.ison");
            graph.Save(a);
            graph.Save(b, snapshot: true);
            Assert.Equal(File.ReadAllBytes(a), File.ReadAllBytes(b));
            Assert.Equal(File.ReadAllBytes(SemanticGraph.EmbeddingsPathFor(a)), File.ReadAllBytes(SemanticGraph.EmbeddingsPathFor(b)));
            Assert.False(File.Exists(Snap(a)));
            Assert.True(File.Exists(Snap(b)));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_save_without_snapshot_removes_an_old_one_and_an_empty_store_round_trips()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, graph) = SaveWithSnapshot(dir);
            graph.Save(path);                                                // JSON only now
            Assert.False(File.Exists(Snap(path)));

            var empty = new SemanticGraph("e", new MockEncoder(8));
            var p2 = System.IO.Path.Combine(dir, "e.ison");
            empty.Save(p2, snapshot: true);
            var (loaded, report) = SemanticGraph.LoadWithReport(p2, new MockEncoder(8));
            Assert.Equal(LoadSource.Snapshot, report.Source);
            Assert.Equal(0, loaded.EmbeddingStore.Count);
            Assert.Null(loaded.EmbeddingStore.Dimension);
            loaded.EmbeddingStore.Add(new NodeRef("a", "1"), "x", new float[5]);     // the dimension is not pinned by an empty snapshot
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Round_trip_reloads_through_ann_index_and_exact_search_the_same()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, graph) = SaveWithSnapshot(dir);
            var (loaded, report) = SemanticGraph.LoadWithReport(path, new MockEncoder(16) { Version = "enc-1" }, ann: AnnMode.Ivf, annMinSize: 0, annNprobe: 3);
            Assert.Equal(LoadSource.Snapshot, report.Source);
            var q = StageD.Vec(new Random(1), 16);
            var hits = loaded.EmbeddingStore.SimilaritySearchVector(q, 5);
            Assert.Equal(5, hits.Count);
            Assert.Equal(StageD.Shape(graph.EmbeddingStore.SimilaritySearchVector(q, 5)),
                         StageD.Shape(loaded.EmbeddingStore.SimilaritySearchVector(q, 5, exact: true)));
        }
        finally { Directory.Delete(dir, true); }
    }

    // -- the byte layout (DESIGN.md 11.2) ------------------------------------------------

    [Fact]
    public void The_smallest_example_of_the_spec_is_byte_for_byte_what_we_write()
    {
        var store = new EmbeddingStore();                                    // no encoder: model "provided"
        store.Add(new NodeRef("t", "a"), "x", new[] { 1.0f, -2.0f });
        var dir = StageD.TempDir();
        try
        {
            var snap = System.IO.Path.Combine(dir, "x.snap");
            var info = store.SaveSnapshot(snap);
            var bytes = File.ReadAllBytes(snap);

            var body = new List<byte>();
            body.AddRange("ISONVSNP"u8.ToArray());
            body.AddRange(new byte[] { 1, 0, 0, 0, 0, 0, 0, 0 });            // version 1, flags 0
            body.AddRange(new byte[] { 2, 0, 0, 0, 0, 4, 0, 0 });            // dim 2, block_rows 1024
            body.AddRange(new byte[] { 1, 0, 0, 0, 0, 0, 0, 0 });            // count 1
            for (int i = 0; i < 16; i++) body.Add(0xff);                     // four absent strings
            body.AddRange(new byte[] { 1, 0, 0, 0, 8, 0, 0, 0 });
            body.AddRange("provided"u8.ToArray());
            body.AddRange(new byte[] { 1, 0, 0, 0, (byte)'t', 1, 0, 0, 0, (byte)'a', 1, 0, 0, 0, (byte)'x', 0, 0, 0, 0 });
            body.AddRange(new byte[] { 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0 });
            Assert.Equal(91, body.Count);
            Assert.Equal(123, bytes.Length);
            Assert.Equal(123, info.Bytes);
            Assert.Equal(body.ToArray(), bytes[..91]);
            Assert.Equal(SHA256.HashData(body.ToArray()), bytes[91..]);

            var loaded = new EmbeddingStore();
            var read = loaded.LoadSnapshot(snap);
            Assert.Equal(1, read.Count);
            Assert.Equal(new[] { 1.0f, -2.0f }, loaded.Get(new NodeRef("t", "a"))!.Vector);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void The_sidecar_fingerprint_follows_the_spec_for_short_and_long_files()
    {
        var dir = StageD.TempDir();
        try
        {
            foreach (int len in new[] { 0, 1, 5000, 70_000, 400_000 })
            {
                var path = System.IO.Path.Combine(dir, $"f{len}");
                var data = new byte[len];
                new Random(len).NextBytes(data);
                File.WriteAllBytes(path, data);

                var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                h.AppendData(BitConverter.GetBytes((long)len));               // little endian on every host we run on
                void Take(long at, int n)
                {
                    n = (int)Math.Min(n, Math.Max(0, len - at));
                    h.AppendData(data, (int)at, n);
                }
                Take(0, 65536);
                Take(Math.Max(0, len - 65536), 65536);
                for (int i = 1; i <= 64; i++) Take((long)len * i / 65, 4096);
                var id = SidecarId.OfFile(path);
                Assert.Equal(len, id.Length);
                Assert.Equal(h.GetHashAndReset(), id.Fingerprint);
            }
        }
        finally { Directory.Delete(dir, true); }
    }

    // -- golden fixture written by the Rust crate -------------------------------------------

    private static string CopyGolden(string dir, bool withSnap)
    {
        var src = StageD.FixtureDir();
        foreach (var name in new[] { "golden.ison", "golden.ison.embeddings.json", "golden.ison.embeddings.snap" })
            if (withSnap || !name.EndsWith(".snap")) File.Copy(System.IO.Path.Combine(src, name), System.IO.Path.Combine(dir, name));
        return System.IO.Path.Combine(dir, "golden.ison");
    }

    [Fact]
    public void The_rust_golden_snapshot_is_used_and_equals_the_json_load()
    {
        var dir = StageD.TempDir();
        try
        {
            var golden = CopyGolden(dir, withSnap: true);
            var (viaSnap, report) = SemanticGraph.LoadWithReport(golden, new MockEncoder(12));
            Assert.True(report.Source == LoadSource.Snapshot, $"snapshot was not used: {report.Source}, skipped: {report.SnapshotSkipped}");
            Assert.Equal(36, report.Embeddings);

            var dir2 = StageD.TempDir();
            try
            {
                var plain = CopyGolden(dir2, withSnap: false);
                var (viaJson, jsonReport) = SemanticGraph.LoadWithReport(plain, new MockEncoder(12));
                Assert.Equal(LoadSource.Json, jsonReport.Source);
                Assert.Equal(36, viaJson.EmbeddingStore.Count);
                AssertSameStore(viaSnap.EmbeddingStore, viaJson.EmbeddingStore);
                Assert.Contains("second-model", viaSnap.EmbeddingStore.Models().Keys);
                Assert.Equal(2, viaSnap.EmbeddingStore.Models().Count);
            }
            finally { Directory.Delete(dir2, true); }
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Our_writer_reproduces_the_rust_golden_snapshot_byte_for_byte()
    {
        var dir = StageD.TempDir();
        try
        {
            var golden = CopyGolden(dir, withSnap: false);
            var goldenSnap = File.ReadAllBytes(System.IO.Path.Combine(StageD.FixtureDir(), "golden.ison.embeddings.snap"));
            var encoder = new MockEncoder(12) { ModelName = "second-model" };
            var (graph, report) = SemanticGraph.LoadWithReport(golden, encoder);
            Assert.Equal(LoadSource.Json, report.Source);
            var digest = graph.EmbeddingStore.LastImportMeta["graph_digest"];

            var mine = System.IO.Path.Combine(dir, "mine.snap");
            graph.EmbeddingStore.SaveSnapshot(mine, new SnapshotIdentity(digest, SidecarId.OfFile(SemanticGraph.EmbeddingsPathFor(golden))));
            Assert.Equal(goldenSnap, File.ReadAllBytes(mine));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void The_rust_golden_snapshot_rejections_are_the_same_as_for_our_own()
    {
        var dir = StageD.TempDir();
        try
        {
            var golden = CopyGolden(dir, withSnap: true);
            var snap = Snap(golden);
            var good = File.ReadAllBytes(snap);
            var bad = (byte[])good.Clone();
            bad[bad.Length / 2] ^= 0x40;
            File.WriteAllBytes(snap, bad);
            var (_, report) = SemanticGraph.LoadWithReport(golden, new MockEncoder(12));
            Assert.Equal(LoadSource.Json, report.Source);
            Assert.NotNull(report.SnapshotSkipped);
        }
        finally { Directory.Delete(dir, true); }
    }

    // -- every rejection falls back to JSON without touching the store ---------------------------

    private static void AssertFallsBack(string path, string reasonPart)
    {
        var (graph, report) = SemanticGraph.LoadWithReport(path, new MockEncoder(16) { Version = "enc-1" });
        Assert.Equal(LoadSource.Json, report.Source);
        Assert.NotNull(report.SnapshotSkipped);
        Assert.Contains(reasonPart, report.SnapshotSkipped!, StringComparison.OrdinalIgnoreCase);
        Assert.True(graph.EmbeddingStore.Count > 0);                          // the JSON sidecar was read
    }

    [Fact]
    public void Corrupt_truncated_and_stale_snapshots_fall_back_to_json()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, graph) = SaveWithSnapshot(dir);
            var snap = Snap(path);
            var good = File.ReadAllBytes(snap);

            // a flipped bit in the body: checksum
            var flipped = (byte[])good.Clone();
            flipped[good.Length - 100] ^= 0x01;
            File.WriteAllBytes(snap, flipped);
            AssertFallsBack(path, "checksum");

            // a flipped bit in the trailer itself
            flipped = (byte[])good.Clone();
            flipped[^1] ^= 0x80;
            File.WriteAllBytes(snap, flipped);
            AssertFallsBack(path, "checksum");

            // truncation, everywhere that matters
            foreach (int len in new[] { 0, 7, 39, 40, 100, good.Length / 2, good.Length - 33, good.Length - 32, good.Length - 1 })
            {
                File.WriteAllBytes(snap, good[..len]);
                var (g, r) = SemanticGraph.LoadWithReport(path, new MockEncoder(16) { Version = "enc-1" });
                Assert.Equal(LoadSource.Json, r.Source);
                Assert.NotNull(r.SnapshotSkipped);
                Assert.Equal(graph.EmbeddingStore.Count, g.EmbeddingStore.Count);
            }

            // bytes appended after the trailer move it: the checksum is read from the wrong place
            File.WriteAllBytes(snap, good.Concat(new byte[] { 0 }).ToArray());
            AssertFallsBack(path, "");

            // not a snapshot at all
            File.WriteAllText(snap, new string('x', 200));
            AssertFallsBack(path, "magic");

            // stale: the sidecar was rewritten (same graph digest, different bytes)
            File.WriteAllBytes(snap, good);
            var (_, ok) = SemanticGraph.LoadWithReport(path, new MockEncoder(16) { Version = "enc-1" });
            Assert.Equal(LoadSource.Snapshot, ok.Source);
            var sidecar = SemanticGraph.EmbeddingsPathFor(path);
            var text = File.ReadAllText(sidecar);
            File.WriteAllText(sidecar, text.Replace("\"text\":\"dup\"", "\"text\":\"dux\""));      // same length, sampled bytes differ
            AssertFallsBack(path, "stale");
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_snapshot_for_another_graph_is_refused_by_digest_and_by_a_missing_sidecar()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, graph) = SaveWithSnapshot(dir);
            var sidecar = SemanticGraph.EmbeddingsPathFor(path);

            // the graph file is rewritten after the save (a node added): the digest in the header no longer matches
            graph.AddNode("extra", "1", new() { ["name"] = "x" }, embedText: "");
            graph.Save(path, embeddings: false);
            Assert.True(File.Exists(Snap(path)));
            Assert.Throws<GraphEmbeddingMismatchException>(() => SemanticGraph.LoadWithReport(path, new MockEncoder(16)));
            var (g, report) = SemanticGraph.LoadWithReport(path, new MockEncoder(16), onMismatch: GraphMismatchMode.Warn, onWarning: _ => { });
            Assert.Equal(LoadSource.Json, report.Source);
            Assert.Contains("different graph", report.SnapshotSkipped);
            Assert.True(g.EmbeddingStore.Count > 0);

            // no sidecar: never use a snapshot alone
            File.Delete(sidecar);
            var (loaded, none) = SemanticGraph.LoadWithReport(path, new MockEncoder(16));
            Assert.Equal(LoadSource.NoEmbeddings, none.Source);
            Assert.Equal(0, loaded.EmbeddingStore.Count);
        }
        finally { Directory.Delete(dir, true); }
    }

    private static byte[] Body(string snapPath) => File.ReadAllBytes(snapPath)[..^32];

    private static void ExpectRejected(string path, byte[] body, string reason)
    {
        Seal(Snap(path), body);
        AssertFallsBack(path, reason);
        var store = StoreWithRow();
        var before = JsonSerializer.Serialize(store.ExportEmbeddings());
        var ex = Assert.Throws<EmbeddingException>(() => store.LoadSnapshot(Snap(path)));    // a failed load leaves the store as it was
        Assert.Contains(reason, ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(before, JsonSerializer.Serialize(store.ExportEmbeddings()));
        Assert.Equal(1, store.Count);
    }

    private static EmbeddingStore StoreWithRow()
    {
        var s = new EmbeddingStore();
        s.Add(new NodeRef("keep", "me"), "kept", new float[16]);
        return s;
    }

    [Fact]
    public void Every_structural_rejection_rule_of_the_spec_is_enforced()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, _) = SaveWithSnapshot(dir);
            var good = Body(Snap(path));

            byte[] With(Action<byte[]> edit) { var c = (byte[])good.Clone(); edit(c); return c; }

            ExpectRejected(path, With(b => b[0] = (byte)'X'), "magic");
            ExpectRejected(path, With(b => b[8] = 2), "version");
            ExpectRejected(path, With(b => b[12] = 2), "flags");                                 // an unknown flag bit
            ExpectRejected(path, With(b => { b[20] = 0; b[21] = 0; b[22] = 0; b[23] = 0; }), "block size");   // block_rows 0
            ExpectRejected(path, With(b => { b[22] = 0x20; }), "block size");                   // block_rows > 2^20
            ExpectRejected(path, With(b => { b[16] = 0; b[17] = 0; b[18] = 0; b[19] = 0; }), "shape");        // dim 0 with rows
            ExpectRejected(path, With(b => b[27] = 0x80), "shape");                              // count past 2^32-1... and rows
            ExpectRejected(path, With(b => b[24] += 1), "");                                     // count one too many: runs out of rows
            ExpectRejected(path, With(b => b[24] -= 1), "left over");                            // count one too few: bytes left over
            ExpectRejected(path, good.Concat(new byte[] { 7, 7 }).ToArray(), "left over");      // bytes between the last block and the trailer
            ExpectRejected(path, good[..^1], "");                                                // short by one byte (sealed, so only structure can object)
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_repeated_key_a_bad_model_index_a_bad_utf8_string_and_an_absent_required_string_are_rejected()
    {
        var dir = StageD.TempDir();
        try
        {
            // hand-built single-block snapshots, sealed with a correct checksum, so only the rule under test can object
            byte[] Build(Action<List<byte>> rows, int count = 2, int modelCount = 1)
            {
                var b = new List<byte>();
                b.AddRange("ISONVSNP"u8.ToArray());
                b.AddRange(new byte[] { 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0 });
                b.AddRange(BitConverter.GetBytes((long)count));
                for (int i = 0; i < 16; i++) b.Add(0xff);
                b.AddRange(BitConverter.GetBytes((uint)modelCount));
                for (int m = 0; m < modelCount; m++) { b.AddRange(BitConverter.GetBytes(1u)); b.Add((byte)'m'); }
                rows(b);
                return b.ToArray();
            }
            void Str(List<byte> b, string s) { var raw = Encoding.UTF8.GetBytes(s); b.AddRange(BitConverter.GetBytes((uint)raw.Length)); b.AddRange(raw); }
            void Row(List<byte> b, string type, string id, uint model) { Str(b, type); Str(b, id); Str(b, ""); b.AddRange(BitConverter.GetBytes(model)); }
            void Floats(List<byte> b, int n) { for (int i = 0; i < n; i++) b.AddRange(BitConverter.GetBytes(1.5f)); }

            var path = System.IO.Path.Combine(dir, "h.ison");
            var graph = MakeGraph(20, 2);
            graph.Save(path, snapshot: true);

            ExpectRejectedHand(path, Build(b => { Row(b, "t", "1", 0); Row(b, "t", "1", 0); Floats(b, 4); }), "twice");
            ExpectRejectedHand(path, Build(b => { Row(b, "t", "1", 0); Row(b, "t", "2", 1); Floats(b, 4); }), "model");
            ExpectRejectedHand(path, Build(b => { Row(b, "t", "1", 0); b.AddRange(BitConverter.GetBytes(1u)); b.Add(0xC3); Str(b, ""); Str(b, ""); b.AddRange(BitConverter.GetBytes(0u)); Floats(b, 4); }), "utf-8");
            ExpectRejectedHand(path, Build(b => { Row(b, "t", "1", 0); b.AddRange(BitConverter.GetBytes(uint.MaxValue)); }), "absent");
            ExpectRejectedHand(path, Build(b => { Row(b, "t", "1", 0); Row(b, "t", "2", 0); Floats(b, 3); }), "");        // one float short
            ExpectRejectedHand(path, Build(b => { }, modelCount: 0, count: 0).Take(40).ToArray(), "");                       // cut inside the header

            // and the same two rows with distinct keys are fine
            Seal(Snap(path), Build(b => { Row(b, "t", "1", 0); Row(b, "t", "2", 0); Floats(b, 4); }));
            var store = new EmbeddingStore();
            Assert.Equal(2, store.LoadSnapshot(Snap(path)).Count);
        }
        finally { Directory.Delete(dir, true); }
    }

    private static void ExpectRejectedHand(string path, byte[] body, string reason)
    {
        Seal(Snap(path), body);
        var store = StoreWithRow();
        var ex = Assert.Throws<EmbeddingException>(() => store.LoadSnapshot(Snap(path)));
        Assert.Contains(reason, ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(1, store.Count);                                                          // untouched
        Assert.Equal(new float[16], store.Get(new NodeRef("keep", "me"))!.Vector);
    }

    [Fact]
    public void A_failed_snapshot_write_keeps_the_previous_file_and_leaves_no_temp()
    {
        var dir = StageD.TempDir();
        try
        {
            var (path, graph) = SaveWithSnapshot(dir);
            var snap = Snap(path);
            var before = File.ReadAllBytes(snap);

            // a read-only target directory entry cannot be forced portably; make the write itself fail instead
            Assert.Throws<ArgumentException>(() => AtomicFile.Write(snap, stream =>
            {
                stream.Write(new byte[] { 1, 2, 3 });
                throw new ArgumentException("injected");
            }));
            Assert.Equal(before, File.ReadAllBytes(snap));
            Assert.Empty(Directory.GetFiles(dir, "*.tmp"));
            Assert.Empty(Directory.GetFiles(dir, ".*.tmp"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void A_snapshot_made_of_several_blocks_round_trips()
    {
        var dir = StageD.TempDir();
        try
        {
            var graph = MakeGraph(n: 2600, dim: 8);                                  // three blocks of 1024 rows
            var (path, _) = SaveWithSnapshot(dir, graph);
            var (viaSnap, report) = SemanticGraph.LoadWithReport(path, new MockEncoder(8) { Version = "enc-1" });
            Assert.Equal(LoadSource.Snapshot, report.Source);
            AssertSameStore(graph.EmbeddingStore, viaSnap.EmbeddingStore, compareMeta: false);
        }
        finally { Directory.Delete(dir, true); }
    }
}
