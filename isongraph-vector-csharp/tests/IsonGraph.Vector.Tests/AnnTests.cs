using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>Seeded clustered unit-ish vectors, the shape of real embeddings: tight groups around a few centres.</summary>
internal sealed class Clustered
{
    private readonly Random _rng;
    private readonly float[][] _centers;
    private readonly int _dim;
    private readonly float _noise;

    public Clustered(int dim, int clusters = 40, int seed = 1, float noise = 0.35f)
    {
        _rng = new Random(seed);
        _dim = dim;
        _noise = noise;
        _centers = new float[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            var v = new float[dim];
            double norm = 0;
            for (int d = 0; d < dim; d++) { v[d] = (float)Gauss(); norm += v[d] * v[d]; }
            float inv = (float)(1 / Math.Sqrt(norm));
            for (int d = 0; d < dim; d++) v[d] *= inv;
            _centers[c] = v;
        }
    }

    private double Gauss()
    {
        double u1 = 1.0 - _rng.NextDouble(), u2 = _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    public float[] Draw()
    {
        var center = _centers[_rng.Next(_centers.Length)];
        var v = new float[_dim];
        float scale = _noise / MathF.Sqrt(_dim);
        for (int d = 0; d < _dim; d++) v[d] = center[d] + scale * (float)Gauss();
        return v;
    }

    /// <summary>A vector from the same clusters, drawn from its own stream (so it does not disturb <see cref="Draw()"/>).</summary>
    public float[] Query(int salt)
    {
        var r = new Random(salt);
        var center = _centers[r.Next(_centers.Length)];
        var v = new float[_dim];
        float scale = _noise / MathF.Sqrt(_dim);
        for (int d = 0; d < _dim; d++)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            v[d] = center[d] + scale * (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }

    public float[][] Draw(int count)
    {
        var all = new float[count][];
        for (int i = 0; i < count; i++) all[i] = Draw();
        return all;
    }
}

/// <summary>Encodes "item N" as the N-th clustered vector and any other text as a clustered query vector.</summary>
internal sealed class ClusteredEncoder : IEmbeddingEncoder
{
    private readonly Clustered _gen;
    private readonly float[][] _items;
    public ClusteredEncoder(int dim, int items)
    {
        Dimension = dim;
        _gen = new Clustered(dim, seed: 5);
        _items = _gen.Draw(items);
    }
    public int Dimension { get; }
    public float[] Encode(string text) =>
        text.StartsWith("item ") ? _items[int.Parse(text.AsSpan(5))] : _gen.Query(text.Aggregate(17, (h, c) => unchecked(h * 31 + c)));
    public IReadOnlyList<float[]> EncodeBatch(IReadOnlyList<string> texts) => texts.Select(Encode).ToList();
}

public class AnnTests
{
    private const double RecallFloor = 0.95;

    private static void Close(double expected, double actual, double tol = 1e-5) =>
        Assert.True(Math.Abs(expected - actual) <= tol, $"expected {expected} but was {actual}");

    private static NodeRef Ref(int i, string type = "doc") => new(type, i.ToString());

    private static (EmbeddingStore Store, float[][] Data, Clustered Gen) Build(
        int n = 4000, int dim = 32, int seed = 1, Func<int, string>? types = null,
        AnnMode ann = AnnMode.Ivf, int? nlist = 64, int? nprobe = 8, int minSize = 0, int annSeed = 0)
    {
        var gen = new Clustered(dim, seed: seed);
        var data = gen.Draw(n);
        var store = new EmbeddingStore(
            ann: ann, annNlist: nlist, annNprobe: nprobe, annMinSize: minSize, annSeed: annSeed);
        for (int i = 0; i < n; i++)
            store.Add(Ref(i, types?.Invoke(i) ?? "doc"), "text " + i, data[i]);
        return (store, data, gen);
    }

    /// <summary>The definition of exact: cosine over every vector, best first, ties in insertion order.</summary>
    private static List<(NodeRef Ref, float Score)> BruteForce(
        IReadOnlyList<float[]> data, IReadOnlyList<NodeRef> refs, float[] q, int k, float threshold = 0f)
    {
        double qn = Math.Sqrt(q.Sum(x => (double)x * x));
        var scored = new List<(NodeRef, float, int)>();
        for (int i = 0; i < data.Count; i++)
        {
            double dot = 0, vn = 0;
            for (int d = 0; d < q.Length; d++) { dot += (double)q[d] * data[i][d]; vn += (double)data[i][d] * data[i][d]; }
            float s = vn == 0 || qn == 0 ? 0f : (float)(dot / (Math.Sqrt(vn) * qn));
            if (s >= threshold) scored.Add((refs[i], s, i));
        }
        return scored.OrderByDescending(t => t.Item2).ThenBy(t => t.Item3).Take(k)
                     .Select(t => (t.Item1, t.Item2)).ToList();
    }

    private static List<NodeRef> Refs(int n, Func<int, string>? types = null) =>
        Enumerable.Range(0, n).Select(i => Ref(i, types?.Invoke(i) ?? "doc")).ToList();

    // -- off by default -----------------------------------------------------------

    [Fact]
    public void Off_by_default_is_exact_and_has_no_index()
    {
        var (store, data, gen) = Build(ann: AnnMode.None);
        Assert.Equal(AnnMode.None, store.Ann);
        var stats = store.GetAnnStats();
        Assert.False(stats.Enabled);
        Assert.False(stats.Active);
        Assert.Throws<EmbeddingException>(() => store.BuildAnnIndex());

        var refs = Refs(data.Length);
        foreach (var q in gen.Draw(5))
        {
            var got = store.SimilaritySearchVector(q, 10);
            var want = BruteForce(data, refs, q, 10);
            Assert.Equal(want.Select(w => w.Ref), got.Select(g => g.NodeRef));
        }
        Assert.Equal(0, store.GetAnnStats().Searches.Approximate);
        Assert.Equal(0, store.GetAnnStats().Searches.Builds);
    }

    [Fact]
    public void Exact_results_are_identical_with_or_without_the_flag_on()
    {
        var (off, _, gen) = Build(ann: AnnMode.None);
        var (on, _, _) = Build(ann: AnnMode.Ivf);
        foreach (var q in gen.Draw(5))
        {
            var a = off.SimilaritySearchVector(q, 10);
            var b = on.SimilaritySearchVector(q, 10, exact: true);
            Assert.Equal(a.Select(r => (r.NodeRef, r.Score)), b.Select(r => (r.NodeRef, r.Score)));
        }
    }

    [Fact]
    public void Argument_validation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: (AnnMode)99));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Ivf, annNlist: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Ivf, annNprobe: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: -1));
        var store = new EmbeddingStore(ann: AnnMode.Ivf);
        Assert.Throws<ArgumentOutOfRangeException>(() => store.AnnNprobe = 0);
        store.AnnNprobe = 5;
        Assert.Equal(5, store.AnnNprobe);
        Assert.Equal(EmbeddingStore.DefaultAnnMinSize, store.GetAnnStats().MinSize);
        Assert.Equal(50_000, EmbeddingStore.DefaultAnnMinSize);
    }

    // -- recall, scores, exact override -------------------------------------------

    [Fact]
    public void Recall_floor_on_clustered_data()
    {
        var (store, _, gen) = Build();
        var report = store.RecallCheck(gen.Draw(60), k: 10);
        Assert.True(report.Approximate);
        Assert.Equal(64, report.Nlist);
        Assert.Equal(8, report.Nprobe);
        Assert.Equal(60, report.Queries);
        Assert.True(report.Recall >= RecallFloor, $"recall {report.Recall:F3}");
        Assert.InRange(report.MinRecall, 0.0, report.Recall);
        Assert.True(report.Recall <= 1.0);
        var stats = store.GetAnnStats();
        Assert.True(stats.Active);
        Assert.Equal(64, stats.Nlist);
        Assert.Equal(4000, stats.TrainedN);
        Assert.True(stats.IndexBytes > 0);
        Assert.Equal(0, stats.VectorBytes);                              // the index keeps no copy of any vector
        Assert.True(stats.Searches.Approximate >= 60);
        Assert.Equal(1, stats.Searches.Builds);
    }

    [Fact]
    public void Index_actually_narrows_the_search()
    {
        // With 8 of 64 cells probed the index must not be reading every row:
        // an index that silently scanned everything would score recall 1.0.
        var (store, _, gen) = Build(nprobe: 1);
        var report = store.RecallCheck(gen.Draw(80), k: 10);
        Assert.True(report.Approximate);
        Assert.True(report.Recall < 1.0, "nprobe=1 of 64 cells should miss something");
        Assert.True(report.Recall > 0.3);
    }

    [Fact]
    public void Exact_override_equals_brute_force_and_skips_the_index()
    {
        var (store, data, gen) = Build();
        var refs = Refs(data.Length);
        store.SimilaritySearchVector(data[0], 5);                    // warm the index
        long before = store.GetAnnStats().Searches.Approximate;
        foreach (var q in gen.Draw(10))
        {
            var got = store.SimilaritySearchVector(q, 10, exact: true);
            var want = BruteForce(data, refs, q, 10);
            Assert.Equal(want.Select(w => w.Ref), got.Select(g => g.NodeRef));
            for (int i = 0; i < want.Count; i++) Close(want[i].Score, got[i].Score);
        }
        Assert.Equal(before, store.GetAnnStats().Searches.Approximate);
    }

    [Fact]
    public void Results_carry_exact_scores_sorted_descending()
    {
        var (store, data, gen) = Build();
        var refs = Refs(data.Length);
        foreach (var q in gen.Draw(10))
        {
            var truth = BruteForce(data, refs, q, data.Length, -1f).ToDictionary(t => t.Ref, t => t.Score);
            var got = store.SimilaritySearchVector(q, 10);
            Assert.Equal(10, got.Count);
            foreach (var r in got) Close(truth[r.NodeRef], r.Score);
            Assert.Equal(got.Select(r => r.Score).OrderByDescending(s => s), got.Select(r => r.Score));
            Assert.Equal(10, got.Select(r => r.NodeRef).Distinct().Count());
            Assert.All(got, r => Assert.StartsWith("text ", r.Text));
        }
    }

    [Fact]
    public void Probing_every_cell_is_the_exact_scan()
    {
        var (store, data, gen) = Build(nprobe: 64);                    // nprobe == nlist
        var refs = Refs(data.Length);
        foreach (var q in gen.Draw(8))
        {
            var got = store.SimilaritySearchVector(q, 10);
            var want = BruteForce(data, refs, q, 10);
            Assert.Equal(want.Select(w => w.Ref), got.Select(g => g.NodeRef));
        }
        var report = store.RecallCheck(gen.Draw(5));
        Assert.False(report.Approximate);
        Assert.Equal(1.0, report.Recall);
        Assert.Equal(0, store.GetAnnStats().Searches.Approximate);
    }

    [Fact]
    public void Small_store_falls_back_to_exact()
    {
        var (store, data, gen) = Build(n: 500, minSize: 1000);
        var refs = Refs(data.Length);
        var q = gen.Draw();
        Assert.Equal(BruteForce(data, refs, q, 10).Select(w => w.Ref),
                     store.SimilaritySearchVector(q, 10).Select(r => r.NodeRef));
        Assert.False(store.GetAnnStats().Active);
        Assert.Equal(0, store.GetAnnStats().Searches.Builds);
        Assert.False(store.RecallCheck(q).Approximate);
        // BuildAnnIndex on a small store says so in the stats and builds nothing.
        Assert.False(store.BuildAnnIndex().Active);
    }

    [Fact]
    public void Index_appears_when_a_growing_store_reaches_the_minimum_size()
    {
        var (store, data, gen) = Build(n: 90, minSize: 100, nlist: 8, nprobe: 2);
        store.SimilaritySearchVector(gen.Draw(), 5);
        Assert.False(store.GetAnnStats().Active);
        foreach (var v in gen.Draw(20)) store.Add(Ref(1000 + store.Count), "x", v);
        store.SimilaritySearchVector(gen.Draw(), 5);
        Assert.True(store.GetAnnStats().Active);
        // ... and goes away again when it shrinks below it.
        for (int i = 0; i < 30; i++) store.Remove(Ref(i));
        store.SimilaritySearchVector(gen.Draw(), 5);
        Assert.False(store.GetAnnStats().Active);
    }

    [Fact]
    public void TopK_null_and_zero_are_not_approximate_and_zero_query_is_exact()
    {
        var (store, data, gen) = Build();
        var q = gen.Draw();
        long before = store.GetAnnStats().Searches.Approximate;
        Assert.Equal(4000, store.SimilaritySearchVector(q, null, threshold: -1f).Count);
        Assert.Empty(store.SimilaritySearchVector(q, 0));
        Assert.Equal(before, store.GetAnnStats().Searches.Approximate);

        var zero = store.SimilaritySearchVector(new float[32], 5, threshold: 0f);
        Assert.Equal(5, zero.Count);
        Assert.All(zero, r => Assert.Equal(0f, r.Score));
    }

    // -- writes: add / update / remove ---------------------------------------------

    [Fact]
    public void Added_vectors_are_visible_immediately()
    {
        var (store, data, gen) = Build();
        store.BuildAnnIndex();
        var fresh = gen.Draw();
        store.Add(new NodeRef("new", "1"), "fresh", fresh);
        var top = store.SimilaritySearchVector(fresh, 3);
        Assert.Equal(new NodeRef("new", "1"), top[0].NodeRef);
        Assert.Equal(1.0, top[0].Score, 4);
        Assert.Equal(1, store.GetAnnStats().Pending);
    }

    [Fact]
    public void Updated_vectors_move_and_are_listed_once()
    {
        var (store, data, gen) = Build();
        store.BuildAnnIndex();
        var target = gen.Draw();
        store.Add(Ref(7), "moved", target);
        var hit = store.SimilaritySearchVector(target, 1)[0];
        Assert.Equal(Ref(7), hit.NodeRef);
        Assert.Equal(1.0, hit.Score, 4);
        Assert.Equal("moved", hit.Text);

        // Scored from the new vector everywhere, and no stale second entry.
        foreach (var q in new[] { data[7], target })
        {
            var res = store.SimilaritySearchVector(q, 400, threshold: -1f);
            Assert.Equal(res.Count, res.Select(r => r.NodeRef).Distinct().Count());
            var truth = store.SimilaritySearchVector(q, null, threshold: -1f).ToDictionary(r => r.NodeRef, r => r.Score);
            Assert.All(res, r => Close(truth[r.NodeRef], r.Score));
        }
        Assert.Equal(4000, store.Count);
        Assert.True(store.GetAnnStats().Searches.Approximate > 0);       // answered by the index
    }

    [Fact]
    public void Removed_vectors_never_come_back_and_compaction_keeps_the_index()
    {
        var (store, data, gen) = Build();
        store.BuildAnnIndex();
        var gone = new HashSet<NodeRef>();
        for (int i = 0; i < 4000; i += 3)                               // a third: forces compaction
        {
            Assert.True(store.Remove(Ref(i)));
            gone.Add(Ref(i));
        }
        long builds = store.GetAnnStats().Searches.Builds;
        foreach (var q in gen.Draw(15))
        {
            Assert.DoesNotContain(store.SimilaritySearchVector(q, 10).Select(r => r.NodeRef), gone.Contains);
            // A negative threshold with many results would surface a tombstone scoring 0.
            Assert.DoesNotContain(store.SimilaritySearchVector(q, 250, threshold: -1f).Select(r => r.NodeRef), gone.Contains);
        }
        var report = store.RecallCheck(gen.Draw(40));
        Assert.True(report.Recall >= RecallFloor, $"recall {report.Recall:F3}");
        Assert.Equal(builds, store.GetAnnStats().Searches.Builds);       // followed, not retrained
        Assert.True(store.GetAnnStats().Active);
        Assert.True(report.Approximate);
        Assert.Equal(2666, store.Count);
    }

    [Fact]
    public void Pending_rows_fold_and_heavy_churn_retrains()
    {
        var (store, data, gen) = Build(n: 3000);
        store.BuildAnnIndex();
        long builds = store.GetAnnStats().Searches.Builds;
        var extra = gen.Draw(700);
        for (int i = 0; i < 700; i++) store.Add(Ref(i, "late"), "", extra[i]);
        // every 256th write folds the pending rows into their cells (at write time, not at search time)
        Assert.Equal(700 % IvfIndex.FoldBatch, store.GetAnnStats().Pending);
        store.SimilaritySearchVector(extra[0], 5);
        var stats = store.GetAnnStats();
        Assert.True(stats.Pending < IvfIndex.FoldBatch);
        Assert.Equal(builds, stats.Searches.Builds);

        var more = gen.Draw(1200);
        for (int i = 0; i < 1200; i++) store.Add(Ref(i, "later"), "", more[i]);
        store.SimilaritySearchVector(more[0], 5);                        // churn 1900 > 50% of 3000
        Assert.Equal(builds + 1, store.GetAnnStats().Searches.Builds);
        Assert.Equal(4900, store.GetAnnStats().TrainedN);
        Assert.True(store.RecallCheck(gen.Draw(30)).Recall >= RecallFloor);
    }

    [Fact]
    public void Clear_drops_the_index_and_a_new_dimension_gets_a_new_one()
    {
        var (store, data, gen) = Build(n: 1500, nlist: 16, nprobe: 4);
        store.BuildAnnIndex();
        Assert.True(store.GetAnnStats().Active);
        store.Clear();
        Assert.Empty(store.SimilaritySearchVector(data[0], 5));          // dimension reset: no mismatch either
        Assert.False(store.GetAnnStats().Active);

        var small = new Clustered(8, clusters: 12, seed: 3).Draw(1200);
        for (int i = 0; i < small.Length; i++) store.Add(Ref(i), "", small[i]);
        var res = store.SimilaritySearchVector(small[5], 5);
        Assert.Equal(Ref(5), res[0].NodeRef);
        Assert.True(store.GetAnnStats().Active);
        Assert.Equal(8, store.Dimension);
    }

    [Fact]
    public void Importing_with_replace_swaps_the_indexed_data()
    {
        var (store, data, gen) = Build(n: 1500, nlist: 16, nprobe: 4);
        store.BuildAnnIndex();
        var other = new EmbeddingStore();
        var fresh = gen.Draw(1200);
        for (int i = 0; i < fresh.Length; i++) other.Add(Ref(i, "fresh"), "f" + i, fresh[i]);
        store.ImportEmbeddings(other.ExportEmbeddings(), replace: true);
        Assert.Equal(1200, store.Count);
        var res = store.SimilaritySearchVector(fresh[3], 5);
        Assert.Equal(Ref(3, "fresh"), res[0].NodeRef);
        Assert.All(res, r => Assert.Equal("fresh", r.NodeRef.Type));
    }

    // -- filters --------------------------------------------------------------------

    [Fact]
    public void NodeType_filter_widens_for_a_rare_type()
    {
        var (store, data, gen) = Build(types: i => i % 200 == 0 ? "rare" : "common", nprobe: 2);
        var q = gen.Draw();
        var common = store.SimilaritySearchVector(q, 10, nodeType: "common");
        Assert.Equal(10, common.Count);
        Assert.All(common, r => Assert.Equal("common", r.NodeRef.Type));
        Assert.True(store.GetAnnStats().Searches.Approximate > 0);

        // 20 rare rows in the store, top-15: the two probed cells cannot hold 15 of them.
        var rare = store.SimilaritySearchVector(q, 15, nodeType: "rare", threshold: -1f);
        Assert.Equal(15, rare.Count);
        Assert.All(rare, r => Assert.Equal("rare", r.NodeRef.Type));
        var counts = store.GetAnnStats().Searches;
        Assert.True(counts.Widened > 0 || counts.Exact > 0);

        Assert.Empty(store.SimilaritySearchVector(q, 5, nodeType: "nope"));
    }

    [Fact]
    public void Threshold_is_respected()
    {
        var (store, data, gen) = Build();
        var q = gen.Draw();
        float best = store.SimilaritySearchVector(q, 1, exact: true)[0].Score;
        var res = store.SimilaritySearchVector(q, 10, threshold: best - 1e-3f);
        Assert.All(res, r => Assert.True(r.Score >= best - 1e-3f));
        Assert.Empty(store.SimilaritySearchVector(q, 10, threshold: 2f));

        float cut = store.SimilaritySearchVector(q, 10, exact: true)[4].Score;
        var got = store.SimilaritySearchVector(q, 10, threshold: cut);
        var want = store.SimilaritySearchVector(q, 10, threshold: cut, exact: true);
        Assert.Equal(want.Select(r => r.NodeRef).ToHashSet(), got.Select(r => r.NodeRef).ToHashSet());
        // A threshold that fewer than k rows clear widens the probe until the scan answers.
        store.SimilaritySearchVector(q, 10, threshold: 0f);
        Assert.True(store.GetAnnStats().Searches.Approximate > 0);
    }

    // -- determinism ---------------------------------------------------------------

    [Fact]
    public void Same_data_and_seed_give_the_same_index_and_answers()
    {
        var (a, _, gen) = Build(annSeed: 7, nprobe: 3);
        var (b, _, _) = Build(annSeed: 7, nprobe: 3);
        var queries = gen.Draw(30);
        foreach (var q in queries)
        {
            var ra = a.SimilaritySearchVector(q, 10);
            var rb = b.SimilaritySearchVector(q, 10);
            Assert.Equal(ra.Select(r => (r.NodeRef, r.Score)), rb.Select(r => (r.NodeRef, r.Score)));
        }
        // ... and again from the same store, and after a forced retrain on unchanged data.
        var first = queries.Select(q => a.SimilaritySearchVector(q, 10).Select(r => r.NodeRef).ToList()).ToList();
        a.BuildAnnIndex(force: true);
        var second = queries.Select(q => a.SimilaritySearchVector(q, 10).Select(r => r.NodeRef).ToList()).ToList();
        Assert.Equal(first, second);
        Assert.Equal(a.GetAnnStats().TrainedN, b.GetAnnStats().TrainedN);
        Assert.True(a.GetAnnStats().Active && b.GetAnnStats().Active);
        Assert.Equal(2, a.GetAnnStats().Searches.Builds);                // first build + forced retrain
    }

    // -- persistence: nothing changes ------------------------------------------------

    private static SemanticGraph Graph(AnnMode ann, int n = 1500, int dim = 32)
    {
        var encoder = new ClusteredEncoder(dim, n);
        var graph = new SemanticGraph(
            "ann", encoder, autoEmbed: false,
            ann: ann, annMinSize: 0, annNlist: 32, annNprobe: 4);
        for (int i = 0; i < n; i++)
            graph.AddNode("item", i.ToString(), new() { ["group"] = (long)(i % 50) }, embedText: "item " + i);
        return graph;
    }

    [Fact]
    public void Graph_file_sidecar_and_digest_are_unchanged_by_the_index()
    {
        string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ivf-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var plain = Graph(AnnMode.None);
            var indexed = Graph(AnnMode.Ivf);
            indexed.EmbeddingStore.BuildAnnIndex();                        // an index definitely exists
            Assert.True(indexed.EmbeddingStore.GetAnnStats().Active);
            indexed.EmbeddingStore.SimilaritySearchVector(new float[32].Select((_, i) => (float)(i + 1)).ToArray(), 5);

            string pa = System.IO.Path.Combine(dir, "plain.ison"), pb = System.IO.Path.Combine(dir, "indexed.ison");
            plain.Save(pa);
            indexed.Save(pb);

            Assert.Equal(File.ReadAllBytes(pa), File.ReadAllBytes(pb));
            var sa = File.ReadAllText(SemanticGraph.EmbeddingsPathFor(pa));
            var sb = File.ReadAllText(SemanticGraph.EmbeddingsPathFor(pb));
            Assert.Equal(sa, sb);
            Assert.DoesNotContain("ivf", sb, StringComparison.OrdinalIgnoreCase);
            Assert.DoesNotContain("centroid", sb, StringComparison.OrdinalIgnoreCase);
            Assert.Equal(
                plain.EmbeddingStore.ExportEmbeddings("d").GraphDigest,
                indexed.EmbeddingStore.ExportEmbeddings("d").GraphDigest);

            // Loading builds nothing; the index appears lazily on the first search.
            var loaded = SemanticGraph.Load(
                pb, new MockEncoder(32), ann: AnnMode.Ivf, annMinSize: 0, annNlist: 32, annNprobe: 4);
            Assert.Equal(AnnMode.Ivf, loaded.EmbeddingStore.Ann);
            Assert.False(loaded.EmbeddingStore.GetAnnStats().Active);
            Assert.Equal(5, loaded.EmbeddingStore.SimilaritySearchVector(
                new float[32].Select((_, i) => (float)(i + 1)).ToArray(), 5).Count);
            Assert.True(loaded.EmbeddingStore.GetAnnStats().Active);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // -- through the graph --------------------------------------------------------------

    [Fact]
    public void Graph_passes_the_options_through_and_exact_forces_the_scan()
    {
        var graph = Graph(AnnMode.Ivf);
        var store = graph.EmbeddingStore;
        Assert.Equal(AnnMode.Ivf, store.Ann);
        Assert.Equal(AnnMode.None, new SemanticGraph().EmbeddingStore.Ann);
        Assert.Equal(4, store.AnnNprobe);

        var res = graph.SimilaritySearch("anything", 5);
        Assert.Equal(5, res.Count);
        Assert.True(store.GetAnnStats().Active);

        long before = store.GetAnnStats().Searches.Approximate;
        graph.SimilaritySearch("anything", 5, exact: true);
        graph.SimilaritySearch("anything", 5, exact: true, nodeFilter: _ => true);
        Assert.Equal(before, store.GetAnnStats().Searches.Approximate);

        var near = graph.SimilarToNode(new NodeRef("item", "3"), topK: 5);
        Assert.Equal(5, near.Count);
        Assert.DoesNotContain(near, r => r.NodeRef == new NodeRef("item", "3"));
        Assert.True(graph.RecallCheck(new[] { "a", "b", "c" }).Queries == 3);
    }

    [Fact]
    public void NodeFilter_still_returns_k_results_when_selective()
    {
        var graph = Graph(AnnMode.Ivf);
        long Group(Node n) => (long)n.Properties["group"]!;

        // 1 node in 50 passes: the probed cells hold few of them, so the search widens.
        var res = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) == 7);
        Assert.Equal(10, res.Count);
        Assert.All(res, r => Assert.Equal(7, int.Parse(r.NodeRef.Id) % 50));
        Assert.Equal(res.Select(r => r.Score).OrderByDescending(s => s), res.Select(r => r.Score));
        var want = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) == 7, exact: true);
        Assert.True(res.Select(r => r.NodeRef).Intersect(want.Select(r => r.NodeRef)).Count() >= 5);

        Assert.Empty(graph.SimilaritySearch("q", 10, nodeFilter: _ => false));
        var even = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) % 2 == 0);
        var evenExact = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) % 2 == 0, exact: true);
        Assert.True(even.Select(r => r.NodeRef).Intersect(evenExact.Select(r => r.NodeRef)).Count() >= 6);
    }

    [Fact]
    public void MultiHop_uses_the_index_for_seeds_but_blend_does_not()
    {
        var graph = Graph(AnnMode.Ivf, n: 1200);
        for (int i = 0; i < 1199; i++)
            graph.AddEdge("NEXT", new NodeRef("item", i.ToString()), new NodeRef("item", (i + 1).ToString()));
        var store = graph.EmbeddingStore;
        var found = graph.SemanticMultiHop("q", new MultiHopOptions
        { RelType = "NEXT", MaxHops = 2, TopKSeeds = 3, Threshold = -1f });
        Assert.NotEmpty(found);
        Assert.True(store.GetAnnStats().Searches.Approximate >= 1);

        long before = store.GetAnnStats().Searches.Approximate;
        graph.SemanticMultiHop("q", new MultiHopOptions
        { RelType = "NEXT", MaxHops = 1, Blend = 0.3f, Threshold = -1f });
        Assert.Equal(before, store.GetAnnStats().Searches.Approximate);
    }

    // -- concurrency --------------------------------------------------------------------

    [Fact]
    public void Concurrent_searches_and_writes_stay_consistent()
    {
        var (store, data, gen) = Build(n: 3000, nlist: 32, nprobe: 4);
        var extra = gen.Draw(600);
        var queries = gen.Draw(40);
        var errors = new System.Collections.Concurrent.ConcurrentBag<string>();

        void Reader(int seed)
        {
            try
            {
                for (int i = 0; i < 120; i++)
                {
                    var res = store.SimilaritySearchVector(queries[(i + seed) % 40], 8, threshold: -1f);
                    if (res.Count != 8) errors.Add($"short result {res.Count}");
                    if (!res.Select(r => r.Score).SequenceEqual(res.Select(r => r.Score).OrderByDescending(s => s)))
                        errors.Add("unsorted");
                    if (res.Select(r => r.NodeRef).Distinct().Count() != res.Count) errors.Add("duplicate");
                }
            }
            catch (Exception ex) { errors.Add(ex.ToString()); }
        }

        void Writer(int seed)
        {
            try
            {
                for (int i = 0; i < 100; i++)
                {
                    store.Add(new NodeRef("w", (seed * 1000 + i).ToString()), "", extra[(seed * 100 + i) % 600]);
                    if (i % 4 == 0) store.Remove(Ref(seed * 300 + i));
                    if (i % 20 == 0) store.BuildAnnIndex();
                }
            }
            catch (Exception ex) { errors.Add(ex.ToString()); }
        }

        var threads = Enumerable.Range(0, 3).Select(t => new Thread(() => Reader(t)))
            .Concat(Enumerable.Range(0, 2).Select(t => new Thread(() => Writer(t)))).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        Assert.Empty(errors);
        Assert.True(store.RecallCheck(queries.Take(20)).Recall >= RecallFloor);
    }

    // -- the vector kernel against awkward dimensions ------------------------------------

    [Theory]
    [InlineData(3)]
    [InlineData(10)]
    [InlineData(33)]
    public void Odd_dimensions_work(int dim)
    {
        var (store, data, gen) = Build(n: 1200, dim: dim, nlist: 16, nprobe: 16);   // all cells: must equal brute force
        var refs = Refs(data.Length);
        foreach (var q in gen.Draw(5))
        {
            var got = store.SimilaritySearchVector(q, 10);
            var want = BruteForce(data, refs, q, 10);
            Assert.Equal(want.Select(w => w.Ref), got.Select(g => g.NodeRef));
        }
        store.AnnNprobe = 4;
        var report = store.RecallCheck(gen.Draw(30));
        Assert.True(report.Approximate);
        Assert.True(report.Recall >= RecallFloor);
    }
}
