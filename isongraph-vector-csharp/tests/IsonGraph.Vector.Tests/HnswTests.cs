using IsonGraph;
using IsonGraph.Vector;
using Xunit;

namespace IsonGraph.Vector.Tests;

/// <summary>The opt-in HNSW index (DESIGN.md section 12).</summary>
public class HnswTests
{
    private static NodeRef Ref(int i, string type = "doc") => new(type, i.ToString());

    private static int Bits(float f) => BitConverter.SingleToInt32Bits(f);

    private static float[] Random(Random r, int dim) => StageD.Vec(r, dim);

    /// <summary>Vectors for <paramref name="n"/> docs: tight clusters (like real embeddings) or uniform noise.</summary>
    private static float[][] Data(int n, int dim, bool clustered, int seed = 1)
    {
        if (clustered) return new Clustered(dim, seed: seed).Draw(n);
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => Random(r, dim)).ToArray();
    }

    private static float[][] Queries(int count, int dim, bool clustered, int seed = 1)
    {
        if (clustered)
        {
            var gen = new Clustered(dim, seed: seed);
            return Enumerable.Range(0, count).Select(i => gen.Query(1000 + i)).ToArray();
        }
        var r = new Random(seed + 77);
        return Enumerable.Range(0, count).Select(_ => Random(r, dim)).ToArray();
    }

    private static EmbeddingStore Store(
        float[][] data, Func<int, string>? types = null, int minSize = 0, int seed = 0, HnswOptions? opts = null,
        AnnMode mode = AnnMode.Hnsw, int scanThreads = 0)
    {
        var store = new EmbeddingStore(ann: mode, annMinSize: minSize, annSeed: seed, annHnsw: opts) { ScanThreads = scanThreads };
        for (int i = 0; i < data.Length; i++)
            store.Add(Ref(i, types?.Invoke(i) ?? "doc"), "text " + i, data[i]);
        return store;
    }

    /// <summary>A returned hit's score is the exact scan's score for that id, bit for bit.</summary>
    private static void AssertScoresAreExact(EmbeddingStore store, float[] q, IReadOnlyList<SimilarityResult> hits)
    {
        var map = store.ScoreMap(q);
        foreach (var h in hits)
            Assert.Equal(Bits(map[$"{h.NodeRef.Type}:{h.NodeRef.Id}"]), Bits(h.Score));
    }

    /// <summary>Best first, equal scores in slot (here: insertion = numeric id) order.</summary>
    private static void AssertOrdered(IReadOnlyList<SimilarityResult> hits)
    {
        for (int i = 1; i < hits.Count; i++)
        {
            Assert.True(hits[i - 1].Score >= hits[i].Score, "scores must not increase");
            if (hits[i - 1].Score == hits[i].Score)
                Assert.True(int.Parse(hits[i - 1].NodeRef.Id) < int.Parse(hits[i].NodeRef.Id), "ties order by slot");
        }
    }

    private static double Recall(EmbeddingStore store, IEnumerable<float[]> queries, int k = 10) =>
        store.RecallCheck(queries, k).Recall;

    // -- level generator (portable, integers only) -----------------------------------------

    [Fact]
    public void Level_draw_matches_an_independent_splitmix64_reference()
    {
        // Expected values come from a separate implementation (Python big integers):
        // state = seed ^ 0x4857535748575357; per draw state += 0x9E3779B97F4A7C15, mix, high 64 bits of next * m.
        int[] m16 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
        Assert.Equal(m16, HnswIndex.LevelSequence(0, 16, 64));
        int[] m4 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0 };
        Assert.Equal(m4, HnswIndex.LevelSequence(0, 4, 64));
        int[] m8 = { 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
        Assert.Equal(m8, HnswIndex.LevelSequence(12345, 8, 64));
    }

    [Fact]
    public void Levels_follow_a_geometric_law_and_the_seed_matters()
    {
        var levels = HnswIndex.LevelSequence(0, 16, 160_000);
        double p1 = levels.Count(l => l >= 1) / 160_000.0, p2 = levels.Count(l => l >= 2) / 160_000.0;
        Assert.InRange(Math.Abs(p1 - 1.0 / 16), 0, 0.004);
        Assert.InRange(Math.Abs(p2 - 1.0 / 256), 0, 0.001);
        Assert.Equal(HnswIndex.LevelSequence(3, 16, 500), HnswIndex.LevelSequence(3, 16, 500));
        Assert.NotEqual(HnswIndex.LevelSequence(3, 16, 500), HnswIndex.LevelSequence(4, 16, 500));
    }

    // -- configuration -----------------------------------------------------------------------

    [Fact]
    public void Options_are_validated_and_ef_is_adjustable()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Hnsw, annHnsw: new HnswOptions(M: 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Hnsw, annHnsw: new HnswOptions(M: 65)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Hnsw, annHnsw: new HnswOptions(EfConstruction: 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingStore(ann: AnnMode.Hnsw, annHnsw: new HnswOptions(EfSearch: 0)));

        var store = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 0, annHnsw: new HnswOptions(M: 8, EfSearch: 33));
        Assert.Equal(AnnMode.Hnsw, store.Ann);
        Assert.Equal(33, store.AnnEfSearch);
        Assert.Equal(33, store.GetAnnStats().EfSearch);
        store.AnnEfSearch = null;
        Assert.Null(store.AnnEfSearch);
        Assert.Equal(64, store.GetAnnStats().EfSearch);                  // automatic, by store size
        Assert.Throws<ArgumentOutOfRangeException>(() => store.AnnEfSearch = 0);
        store.AnnEfSearch = 70;
        Assert.Equal(70, store.GetAnnStats().EfSearch);

        Assert.Equal(64, EmbeddingStore.AutoEf(9_999));
        Assert.Equal(96, EmbeddingStore.AutoEf(10_000));
        Assert.Equal(96, EmbeddingStore.AutoEf(199_999));
        Assert.Equal(128, EmbeddingStore.AutoEf(200_000));
        Assert.Equal(128, EmbeddingStore.AutoEf(599_999));
        Assert.Equal(192, EmbeddingStore.AutoEf(600_000));

        // an options object given to another mode is ignored; IVF and None are untouched
        Assert.Equal(AnnMode.Ivf, new EmbeddingStore(ann: AnnMode.Ivf, annHnsw: new HnswOptions(M: 8)).Ann);
        var data = Data(50, 8, true);
        Assert.Throws<ArgumentOutOfRangeException>(() => Store(data).SimilaritySearchVectorEf(data[0], 5, null, 0f, 0));
        Assert.Throws<EmbeddingException>(() => new EmbeddingStore().BuildAnnIndex());
    }

    // -- recall and exactness -------------------------------------------------------------------

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Recall_against_exact_is_high_and_scores_are_bit_identical(bool clustered)
    {
        const int n = 4000, dim = 32;
        var data = Data(n, dim, clustered);
        var store = Store(data);
        var queries = Queries(100, dim, clustered);
        var report = store.RecallCheck(queries, 10);
        Assert.True(report.Approximate);
        Assert.True(report.Recall >= (clustered ? 0.97 : 0.90), $"recall {report.Recall} (min {report.MinRecall})");

        foreach (var q in queries.Take(40))
        {
            var hits = store.SimilaritySearchVector(q, 10);
            Assert.Equal(10, hits.Count);
            AssertScoresAreExact(store, q, hits);
            AssertOrdered(hits);
        }
        var stats = store.GetAnnStats();
        Assert.Equal(AnnMode.Hnsw, stats.Mode);
        Assert.True(stats.Active);
        Assert.Equal(n, stats.GraphNodes);
        Assert.Equal(n, stats.TrainedN);
        Assert.Null(stats.Nlist);
        Assert.Equal(1, stats.Searches.Builds);
        Assert.Null(store.VerifyInvariants());
    }

    [Fact]
    public void A_higher_ef_does_not_lower_recall_and_the_per_call_ef_matches_the_configured_one()
    {
        var data = Data(4000, 32, clustered: false);
        var store = Store(data, opts: new HnswOptions(M: 8, EfConstruction: 60));
        var queries = Queries(60, 32, clustered: false);
        store.AnnEfSearch = 12;
        double low = Recall(store, queries);
        store.AnnEfSearch = 400;
        double high = Recall(store, queries);
        Assert.True(high >= low, $"ef 400: {high}, ef 12: {low}");
        Assert.True(high > 0.97, $"ef 400 recall {high}");
        Assert.True(low < high, "a tiny ef should be visibly worse than a large one on noise");

        store.AnnEfSearch = 12;
        foreach (var q in queries.Take(20))
        {
            var viaCall = store.SimilaritySearchVectorEf(q, 10, null, 0f, 400);
            store.AnnEfSearch = 400;
            var viaConfig = store.SimilaritySearchVector(q, 10);
            store.AnnEfSearch = 12;
            Assert.Equal(StageD.Shape(viaConfig), StageD.Shape(viaCall));
        }
    }

    [Fact]
    public void Results_are_the_exact_scan_below_min_size_and_when_top_k_covers_the_store()
    {
        var data = Data(500, 16, clustered: true);
        var queries = Queries(10, 16, clustered: true);

        var belowMin = Store(data, minSize: 1000);                       // 500 live < 1000: no graph
        foreach (var q in queries)
            Assert.Equal(StageD.Shape(belowMin.SimilaritySearchVector(q, 10, exact: true)),
                         StageD.Shape(belowMin.SimilaritySearchVector(q, 10)));
        Assert.False(belowMin.GetAnnStats().Active);

        var store = Store(data, minSize: 0);
        foreach (var q in queries)
            foreach (int k in new[] { 500, 501, 100000 })
                Assert.Equal(StageD.Shape(store.SimilaritySearchVector(q, k, exact: true)),
                             StageD.Shape(store.SimilaritySearchVector(q, k)));
        Assert.Equal(0, store.GetAnnStats().Searches.Approximate);
        Assert.Equal(StageD.Shape(store.SimilaritySearchVector(queries[0], null, exact: true)),
                     StageD.Shape(store.SimilaritySearchVector(queries[0], null)));

        long before = store.GetAnnStats().Searches.Approximate;
        store.SimilaritySearchVector(queries[0], 10);
        Assert.Equal(before + 1, store.GetAnnStats().Searches.Approximate);
        store.SimilaritySearchVector(queries[0], 10, exact: true);
        Assert.Equal(before + 1, store.GetAnnStats().Searches.Approximate);
    }

    [Fact]
    public void Duplicates_and_zero_vectors_keep_the_scan_order_and_exact_scores()
    {
        // StageD.Filled has exact duplicates (equal scores) and zero rows (norm 0, live and finite)
        var rng = new Random(9);
        const int dim = 16;
        var store = StageD.Filled(2500, dim, seed: 3, ties: true, ann: AnnMode.Hnsw, annMin: 0);
        for (int t = 0; t < 40; t++)
        {
            var q = StageD.Vec(rng, dim);
            var hits = store.SimilaritySearchVector(q, 25);
            AssertScoresAreExact(store, q, hits);
            AssertOrdered(hits);
        }
        // a duplicate of a stored row: both copies come back with the same score, lower slot first
        var dupQuery = store.Get(new NodeRef("a", "0"))!.Vector;
        var top = store.SimilaritySearchVector(dupQuery, 5);
        Assert.Equal(1f, top[0].Score, 4);
        Assert.Null(store.VerifyInvariants());
        // a zero query has no direction: it is answered by the scan
        long approx = store.GetAnnStats().Searches.Approximate;
        var zero = store.SimilaritySearchVector(new float[dim], 5);
        Assert.Equal(StageD.Shape(store.SimilaritySearchVector(new float[dim], 5, exact: true)), StageD.Shape(zero));
        Assert.Equal(approx, store.GetAnnStats().Searches.Approximate);
    }

    // -- filters -------------------------------------------------------------------------------------

    [Fact]
    public void Type_filter_and_threshold_including_selective_ones_that_fall_back()
    {
        const int n = 3000, dim = 24;
        var data = Data(n, dim, clustered: true, seed: 4);
        // 1 in 100 is "rare" (30 rows), 5 rows are "tiny" (fewer than k), the rest "doc"
        string TypeOf(int i) => i % 600 == 5 ? "tiny" : i % 100 == 0 ? "rare" : "doc";
        var store = Store(data, TypeOf);
        var queries = Queries(30, dim, clustered: true, seed: 4);

        foreach (var q in queries)
        {
            // a type that holds a third of... most rows: plain graph answer
            var docs = store.SimilaritySearchVector(q, 10, "doc", -1f);
            Assert.Equal(10, docs.Count);
            Assert.All(docs, r => Assert.Equal("doc", r.NodeRef.Type));
            AssertScoresAreExact(store, q, docs);

            // selective: the search widens ef, possibly all the way to the scan; always k results of that type
            var rare = store.SimilaritySearchVector(q, 10, "rare", -1f);
            Assert.Equal(10, rare.Count);
            Assert.All(rare, r => Assert.Equal("rare", r.NodeRef.Type));
            AssertScoresAreExact(store, q, rare);
            AssertOrdered(rare);

            // fewer matches than k: nothing to widen to but the scan, which answers exactly
            var tiny = store.SimilaritySearchVector(q, 10, "tiny", -1f);
            Assert.Equal(StageD.Shape(store.SimilaritySearchVector(q, 10, "tiny", -1f, exact: true)), StageD.Shape(tiny));
            Assert.Equal(5, tiny.Count);

            // a type nobody has
            Assert.Empty(store.SimilaritySearchVector(q, 10, "nope"));
        }
        var recallRare = queries.Select(q =>
        {
            var a = store.SimilaritySearchVector(q, 10, "rare", -1f).Select(r => r.NodeRef).ToHashSet();
            var e = store.SimilaritySearchVector(q, 10, "rare", -1f, exact: true).Select(r => r.NodeRef).ToHashSet();
            return a.Intersect(e).Count() / 10.0;
        }).Average();
        Assert.True(recallRare >= 0.8, $"recall on a selective type: {recallRare}");
        // "tiny" has fewer than k rows: the search can never fill k, so every one ended in the scan
        Assert.True(store.GetAnnStats().Searches.Exact >= queries.Length);

        // threshold: every answer passes it, scores exact; one nobody reaches is empty (the scan says so)
        foreach (var q in queries.Take(10))
        {
            var hits = store.SimilaritySearchVector(q, 10, null, 0.6f);
            Assert.All(hits, r => Assert.True(r.Score >= 0.6f));
            AssertScoresAreExact(store, q, hits);
            Assert.Empty(store.SimilaritySearchVector(q, 10, null, 1.5f));
        }
        // type and threshold together
        var both = store.SimilaritySearchVector(queries[0], 5, "rare", 0.1f);
        Assert.All(both, r => { Assert.Equal("rare", r.NodeRef.Type); Assert.True(r.Score >= 0.1f); });
    }

    private static SemanticGraph Graph(AnnMode ann, int n = 2000, int dim = 32)
    {
        var encoder = new ClusteredEncoder(dim, n);
        var graph = new SemanticGraph("hnsw", encoder, autoEmbed: false, ann: ann, annMinSize: 0);
        for (int i = 0; i < n; i++)
            graph.AddNode("item", i.ToString(), new() { ["group"] = (long)(i % 50) }, embedText: "item " + i);
        return graph;
    }

    [Fact]
    public void Graph_node_filter_returns_k_results_even_when_selective_and_exact_forces_the_scan()
    {
        var graph = Graph(AnnMode.Hnsw);
        var store = graph.EmbeddingStore;
        long Group(Node n) => (long)n.Properties["group"]!;

        var plain = graph.SimilaritySearch("q", 10);
        Assert.Equal(10, plain.Count);
        Assert.True(store.GetAnnStats().Active);

        // one node in 50 passes: ef widens until the gate is satisfied
        var res = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) == 7);
        Assert.Equal(10, res.Count);
        Assert.All(res, r => Assert.Equal(7, int.Parse(r.NodeRef.Id) % 50));
        Assert.Equal(res.Select(r => r.Score).OrderByDescending(s => s), res.Select(r => r.Score));
        var want = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) == 7, exact: true);
        Assert.True(res.Select(r => r.NodeRef).Intersect(want.Select(r => r.NodeRef)).Count() >= 8);
        Assert.True(store.GetAnnStats().Searches.Widened > 0, "a gate passing 1 node in 50 widens ef (64 -> 256 -> 1024)");
        var q = store.EncodeQuery("q");
        var exactScores = store.ScoreMap(q);
        foreach (var r in res) Assert.Equal(Bits(exactScores[$"{r.NodeRef.Type}:{r.NodeRef.Id}"]), Bits(r.Score));

        // nobody passes: the widening ends in the scan and the answer is empty
        Assert.Empty(graph.SimilaritySearch("q", 10, nodeFilter: _ => false));
        // few pass (3 nodes): fewer than k exist, the exact scan answers
        var few = graph.SimilaritySearch("q", 10, threshold: -1f, nodeFilter: n => Group(n) == 7 && int.Parse(n.Id) < 150);
        Assert.Equal(3, few.Count);
        var even = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) % 2 == 0);
        var evenExact = graph.SimilaritySearch("q", 10, nodeFilter: n => Group(n) % 2 == 0, exact: true);
        Assert.True(even.Select(r => r.NodeRef).Intersect(evenExact.Select(r => r.NodeRef)).Count() >= 8);

        long before = store.GetAnnStats().Searches.Approximate;
        graph.SimilaritySearch("q", 5, exact: true);
        graph.SimilaritySearch("q", 5, exact: true, nodeFilter: _ => true);
        Assert.Equal(before, store.GetAnnStats().Searches.Approximate);
        Assert.Equal(5, graph.SimilarToNode(new NodeRef("item", "3"), topK: 5).Count);
        Assert.Equal(3, graph.RecallCheck(new[] { "a", "b", "c" }).Queries);
    }

    // -- writes, deletes, updates, compaction ---------------------------------------------------------------

    [Fact]
    public void Inserts_are_linked_by_the_write_and_found_at_once()
    {
        const int dim = 32;
        var gen = new Clustered(dim, seed: 6);
        var data = gen.Draw(3000);
        var store = Store(data);
        store.BuildAnnIndex();
        Assert.Equal(1, store.GetAnnStats().Searches.Builds);

        var extra = gen.Draw(300);
        for (int i = 0; i < extra.Length; i++)
        {
            store.Add(Ref(3000 + i), "new", extra[i]);
            if (i % 50 == 49) Assert.Null(store.VerifyInvariants());
        }
        var stats = store.GetAnnStats();
        Assert.Equal(1, stats.Searches.Builds);                              // inserts never retrain
        Assert.Equal(3300, stats.GraphNodes);
        Assert.Equal(300, stats.Inserts);
        Assert.Equal(0, stats.Churn);                                        // inserts are not churn
        int self = 0;
        for (int i = 0; i < extra.Length; i++)
        {
            var top = store.SimilaritySearchVector(extra[i], 1);
            if (top[0].NodeRef == Ref(3000 + i)) self++;
        }
        Assert.True(self >= 290, $"{self}/300 new rows were found as their own nearest neighbour");
        Assert.Null(store.VerifyInvariants());
    }

    [Fact]
    public void Deletes_tombstone_then_compaction_renumbers_and_no_dead_id_is_returned()
    {
        const int n = 3200, dim = 32;
        var data = Data(n, dim, clustered: true, seed: 8);
        var store = Store(data);
        var queries = Queries(60, dim, clustered: true, seed: 8);
        store.BuildAnnIndex();
        Assert.True(Recall(store, queries) >= 0.95);

        var dead = new HashSet<int>();
        var rng = new Random(2);
        // below the 25% compaction line the node stays as a tombstone
        while (dead.Count < 400)
        {
            int id = rng.Next(n);
            if (dead.Add(id)) Assert.True(store.Remove(Ref(id)));
        }
        var stats = store.GetAnnStats();
        Assert.Equal(n, stats.GraphNodes);
        Assert.Equal(400, stats.GraphDead);
        Assert.Null(store.VerifyInvariants());
        foreach (var q in queries)
            foreach (var r in store.SimilaritySearchVector(q, 10))
                Assert.DoesNotContain(int.Parse(r.NodeRef.Id), dead);
        Assert.True(Recall(store, queries) >= 0.93);

        // past 25% dead the table compacts and the graph follows (tombstones leave, links are renumbered)
        while (dead.Count < 1000)
        {
            int id = rng.Next(n);
            if (dead.Add(id)) Assert.True(store.Remove(Ref(id)));
            if (dead.Count % 100 == 0) Assert.Null(store.VerifyInvariants());
        }
        stats = store.GetAnnStats();
        // the 801st removal crossed 25% dead (801 * 4 > 3200): the table compacted and the graph followed;
        // the 199 removals after it are tombstones again
        Assert.Equal(n - 801, stats.GraphNodes);
        Assert.Equal(dead.Count - 801, stats.GraphDead);
        store.Compact();
        stats = store.GetAnnStats();
        Assert.Equal(n - dead.Count, stats.GraphNodes);
        Assert.Equal(0, stats.GraphDead);
        Assert.Null(store.VerifyInvariants());
        Assert.False(store.Remove(Ref(dead.First())));
        foreach (var q in queries)
        {
            var hits = store.SimilaritySearchVector(q, 10);
            Assert.Equal(10, hits.Count);
            Assert.All(hits, r => Assert.DoesNotContain(int.Parse(r.NodeRef.Id), dead));
            AssertScoresAreExact(store, q, hits);
        }
        Assert.True(Recall(store, queries) >= 0.93, "recall after churn");
        Assert.Equal(1, store.GetAnnStats().Searches.Builds);                  // bridging, not a rebuild
    }

    [Fact]
    public void Compaction_bridges_over_tombstones_so_the_graph_stays_connected_and_dense()
    {
        const int n = 4000, dim = 12;
        var data = Data(n, dim, clustered: true, seed: 11);
        var store = Store(data, opts: new HnswOptions(M: 12, EfConstruction: 100));
        store.BuildAnnIndex();
        for (int i = 0; i < n; i += 3) store.Remove(Ref(i));
        store.Compact();
        Assert.Null(store.VerifyInvariants());
        int live = store.Count;
        Assert.Equal(2666, live);

        (int Reach, double Degree) Measure(EmbeddingStore s) => s.InspectHnsw((h, t) =>
        {
            var seen = new bool[t.Slots];
            var stack = new Stack<int>();
            stack.Push(h.Entry);
            seen[h.Entry] = true;
            int reach = 0;
            long degree = 0;
            while (stack.Count > 0)
            {
                int cur = stack.Pop();
                if (t.IsLive(cur)) reach++;
                degree += h.Links(cur, 0).Length;
                foreach (int nb in h.Links(cur, 0))
                    if (!seen[nb]) { seen[nb] = true; stack.Push(nb); }
            }
            return (reach, degree / (double)t.Slots);
        });
        var (reach, degree) = Measure(store);
        Assert.True(reach * 100 >= live * 99, $"{reach} of {live} live rows reachable after compaction");

        // lists were refilled from the tombstones' links: degree stays near that of a graph built from scratch
        var fresh = Store(Enumerable.Range(0, n).Where(i => i % 3 != 0).Select(i => data[i]).ToArray(),
                          opts: new HnswOptions(M: 12, EfConstruction: 100));
        fresh.BuildAnnIndex();
        var (_, freshDegree) = Measure(fresh);
        Assert.True(degree >= 0.85 * freshDegree, $"mean layer-0 degree {degree} vs {freshDegree} for a fresh build");
    }

    [Fact]
    public void Updates_relink_the_node_and_find_it_at_its_new_position()
    {
        const int n = 3000, dim = 32;
        var data = Data(n, dim, clustered: true, seed: 12);
        var store = Store(data);
        store.BuildAnnIndex();
        var moved = Data(400, dim, clustered: false, seed: 99);
        for (int i = 0; i < moved.Length; i++) store.Add(Ref(i * 5), "moved", moved[i]);      // same key: update in place
        var stats = store.GetAnnStats();
        Assert.Equal(400, stats.Churn);
        Assert.Equal(n, stats.GraphNodes);
        Assert.Equal(1, stats.Searches.Builds);
        Assert.Null(store.VerifyInvariants());

        int found = 0;
        for (int i = 0; i < moved.Length; i++)
        {
            var top = store.SimilaritySearchVector(moved[i], 1);
            if (top[0].NodeRef == Ref(i * 5) && top[0].Score > 0.9999f) found++;
        }
        Assert.True(found >= 380, $"{found}/400 updated rows found at their new vector");
        // scores are of the current vector, never the old position
        var q = Data(1, dim, clustered: true, seed: 12)[0];
        AssertScoresAreExact(store, q, store.SimilaritySearchVector(q, 10));
    }

    [Fact]
    public void Heavy_churn_marks_the_graph_stale_and_the_next_search_rebuilds_it()
    {
        const int dim = 16;
        var data = Data(1000, dim, clustered: true, seed: 13);
        var store = Store(data);
        store.BuildAnnIndex();
        var other = Data(1000, dim, clustered: false, seed: 14);
        for (int i = 0; i < 999; i++) store.Add(Ref(i), "u", other[i]);        // 999 updates: not yet more than the node count
        Assert.Equal(1, store.GetAnnStats().Searches.Builds);
        store.SimilaritySearchVector(other[0], 5);
        Assert.Equal(1, store.GetAnnStats().Searches.Builds);
        for (int i = 0; i < 5; i++) store.Add(Ref(i), "u", data[i]);           // 1004 > 1000
        Assert.Equal(1, store.GetAnnStats().Searches.Builds);                  // lazily: nothing happens at write time
        var hits = store.SimilaritySearchVector(data[0], 5);
        var stats = store.GetAnnStats();
        Assert.Equal(2, stats.Searches.Builds);
        Assert.Equal(0, stats.Churn);
        Assert.Equal(1000, stats.GraphNodes);
        Assert.Equal(5, hits.Count);
        Assert.Null(store.VerifyInvariants());

        // deletes count too: remove most of the rows, compaction keeps the churn, a rebuild follows
        for (int i = 0; i < 700; i++) store.Remove(Ref(i));
        Assert.True(store.GetAnnStats().Churn >= 300);
        Assert.Equal(5, store.SimilaritySearchVector(other[999], 5).Count);
        Assert.Equal(3, store.GetAnnStats().Searches.Builds);
        Assert.Null(store.VerifyInvariants());
    }

    [Fact]
    public void The_graph_is_dropped_below_min_size_and_by_clear_and_rebuilt_later()
    {
        var data = Data(600, 16, clustered: true, seed: 15);
        var store = Store(data, minSize: 500);
        store.BuildAnnIndex();
        Assert.True(store.GetAnnStats().Active);
        for (int i = 0; i < 150; i++) store.Remove(Ref(i));                    // 450 live < 500
        Assert.False(store.GetAnnStats().Active);
        var q = data[300];
        Assert.Equal(StageD.Shape(store.SimilaritySearchVector(q, 10, exact: true)), StageD.Shape(store.SimilaritySearchVector(q, 10)));
        Assert.Null(store.VerifyInvariants());

        for (int i = 0; i < 100; i++) store.Add(Ref(1000 + i), "new", data[i]);
        store.SimilaritySearchVector(q, 10);
        Assert.True(store.GetAnnStats().Active);
        store.Clear();
        Assert.False(store.GetAnnStats().Active);
        Assert.Empty(store.SimilaritySearchVector(q, 10));
        Assert.Null(store.VerifyInvariants());
    }

    [Fact]
    public void Invariants_hold_through_random_writes_updates_deletes_and_compaction()
    {
        var store = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 300, annSeed: 2, annHnsw: new HnswOptions(M: 6, EfConstruction: 40));
        var rng = new Random(5);
        var live = new Dictionary<string, float[]>();
        for (int step = 0; step < 6000; step++)
        {
            int id = rng.Next(1500);
            var r = new NodeRef(id % 2 == 0 ? "x" : "y", id.ToString());
            if (rng.NextDouble() < 0.35)
                Assert.Equal(live.Remove($"{r.Type}:{r.Id}"), store.Remove(r));
            else
            {
                var v = StageD.Vec(rng, 24);
                store.Add(r, "t", v);
                live[$"{r.Type}:{r.Id}"] = v;
            }
            if (step % 7 == 0)
            {
                var q = StageD.Vec(rng, 24);
                var hits = store.SimilaritySearchVector(q, 8, null, -1f);
                Assert.Equal(Math.Min(8, live.Count), hits.Count);
                Assert.All(hits, h => Assert.True(live.ContainsKey($"{h.NodeRef.Type}:{h.NodeRef.Id}")));
                AssertScoresAreExact(store, q, hits);
            }
            if (step % 50 == 0) Assert.Null(store.VerifyInvariants());
        }
        Assert.Null(store.VerifyInvariants());
        Assert.Equal(live.Count, store.Count);
    }

    // -- determinism, batch, threads --------------------------------------------------------------------------

    [Fact]
    public void The_graph_does_not_depend_on_the_thread_count_and_does_depend_on_the_seed()
    {
        const int dim = 12;
        var data = Data(3000, dim, clustered: true, seed: 16);
        ulong Hash(int threads, int seed = 0)
        {
            var s = Store(data, seed: seed, scanThreads: threads);
            s.BuildAnnIndex();
            Assert.Null(s.VerifyInvariants());
            return s.HnswStructureHash()!.Value;
        }
        ulong one = Hash(1);
        Assert.Equal(one, Hash(0));
        Assert.Equal(one, Hash(3));
        Assert.Equal(one, Hash(8));
        Assert.Equal(one, Hash(1));
        Assert.NotEqual(one, Hash(1, seed: 1));

        // writes after the build are serial but the same routine: any thread setting gives the same graph
        var more = Data(400, dim, clustered: false, seed: 17);
        ulong WithWrites(int threads)
        {
            var s = Store(data, scanThreads: threads);
            s.BuildAnnIndex();
            for (int i = 0; i < more.Length; i++) s.Add(Ref(i % 2 == 0 ? 5000 + i : i), "w", more[i]);   // inserts and updates
            return s.HnswStructureHash()!.Value;
        }
        Assert.Equal(WithWrites(1), WithWrites(0));
        Assert.Equal(WithWrites(1), WithWrites(4));

        // and so are the answers
        var queries = Queries(30, dim, clustered: true, seed: 16);
        var a = Store(data, scanThreads: 1);
        var b = Store(data, scanThreads: 0);
        foreach (var q in queries)
            Assert.Equal(StageD.Shape(a.SimilaritySearchVector(q, 10)), StageD.Shape(b.SimilaritySearchVector(q, 10)));
    }

    [Fact]
    public void Batch_equals_single_bit_for_bit_for_any_parallelism_and_filter()
    {
        const int dim = 32;
        var data = Data(3000, dim, clustered: true, seed: 18);
        string TypeOf(int i) => i % 7 == 0 ? "seven" : "doc";
        var store = Store(data, TypeOf);
        var queries = Queries(150, dim, clustered: true, seed: 18).ToList();
        foreach (var (type, threshold, k) in new (string?, float, int)[] { (null, 0f, 10), ("seven", 0f, 7), (null, 0.5f, 20), ("doc", 0.2f, 1) })
        {
            var single = queries.Select(q => StageD.Shape(store.SimilaritySearchVector(q, k, type, threshold))).ToList();
            foreach (int degree in new[] { 1, 3, 0 })
            {
                var batch = store.SimilaritySearchBatch(queries, k, type, threshold, maxDegreeOfParallelism: degree);
                Assert.Equal(queries.Count, batch.Count);
                for (int i = 0; i < queries.Count; i++) Assert.Equal(single[i], StageD.Shape(batch[i]));
            }
            store.ScanThreads = 2;                                            // the batch honours ScanThreads when no degree is given
            var viaScanThreads = store.SimilaritySearchBatch(queries, k, type, threshold);
            store.ScanThreads = 0;
            for (int i = 0; i < queries.Count; i++) Assert.Equal(single[i], StageD.Shape(viaScanThreads[i]));
        }
        Assert.Empty(store.SimilaritySearchBatch(new List<float[]>(), 5));
        Assert.Throws<DimensionMismatchException>(() => store.SimilaritySearchBatch(new List<float[]> { queries[0], new float[3] }, 5));
        // exact: true bypasses the graph in a batch too
        var exactBatch = store.SimilaritySearchBatch(queries.Take(5).ToList(), 10, exact: true);
        for (int i = 0; i < 5; i++)
            Assert.Equal(StageD.Shape(store.SimilaritySearchVector(queries[i], 10, exact: true)), StageD.Shape(exactBatch[i]));
    }

    [Fact]
    public async Task Concurrent_searches_with_no_writes_all_get_the_single_threaded_answer()
    {
        const int dim = 32;
        var data = Data(3000, dim, clustered: true, seed: 19);
        var store = Store(data);
        var queries = Queries(64, dim, clustered: true, seed: 19);
        var expected = queries.Select(q => StageD.Shape(store.SimilaritySearchVector(q, 10))).ToArray();
        var tasks = Enumerable.Range(0, 8).Select(t => Task.Run(() =>
        {
            for (int round = 0; round < 6; round++)
                for (int i = 0; i < queries.Length; i++)
                {
                    int j = (i * 7 + t * 13 + round) % queries.Length;
                    Assert.Equal(expected[j], StageD.Shape(store.SimilaritySearchVector(queries[j], 10)));
                }
        })).ToArray();
        await Task.WhenAll(tasks);
        // batches from several tasks at once, too
        await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            var batch = store.SimilaritySearchBatch(queries, 10);
            for (int i = 0; i < queries.Length; i++) Assert.Equal(expected[i], StageD.Shape(batch[i]));
        })).ToArray());
    }

    [Fact]
    public async Task Searches_and_writes_interleave_without_torn_reads()
    {
        const int dim = 24;
        var gen = new Clustered(dim, seed: 20);
        var data = gen.Draw(2500);
        var store = Store(data);
        store.BuildAnnIndex();
        var extra = gen.Draw(400);
        var stop = false;
        var searchers = Enumerable.Range(0, 3).Select(t => Task.Run(() =>
        {
            int i = t;
            while (!Volatile.Read(ref stop))
            {
                var q = gen.Query(i++);
                var hits = store.SimilaritySearchVector(q, 10);
                Assert.Equal(10, hits.Count);
                AssertOrdered(hits);
            }
        })).ToArray();
        for (int i = 0; i < extra.Length; i++)
        {
            store.Add(Ref(10_000 + i), "w", extra[i]);
            if (i % 3 == 0) store.Remove(Ref(i));
        }
        Volatile.Write(ref stop, true);
        await Task.WhenAll(searchers);
        Assert.Null(store.VerifyInvariants());
    }

    // -- persistence: nothing changes -------------------------------------------------------------------------------

    [Fact]
    public void Graph_file_sidecar_digest_and_snapshot_are_byte_identical_with_hnsw_on_or_off()
    {
        string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hnsw-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var plain = Graph(AnnMode.None, n: 1200);
            var hnsw = Graph(AnnMode.Hnsw, n: 1200);
            hnsw.EmbeddingStore.BuildAnnIndex();
            Assert.True(hnsw.EmbeddingStore.GetAnnStats().Active);
            hnsw.EmbeddingStore.SimilaritySearchVector(new float[32].Select((_, i) => (float)(i + 1)).ToArray(), 5);

            string pa = System.IO.Path.Combine(dir, "plain.ison"), pb = System.IO.Path.Combine(dir, "hnsw.ison");
            plain.Save(pa, snapshot: true);
            hnsw.Save(pb, snapshot: true);
            Assert.Equal(File.ReadAllBytes(pa), File.ReadAllBytes(pb));
            Assert.Equal(File.ReadAllBytes(SemanticGraph.EmbeddingsPathFor(pa)), File.ReadAllBytes(SemanticGraph.EmbeddingsPathFor(pb)));
            Assert.Equal(File.ReadAllBytes(SemanticGraph.SnapshotPathFor(pa)), File.ReadAllBytes(SemanticGraph.SnapshotPathFor(pb)));
            var sidecar = File.ReadAllText(SemanticGraph.EmbeddingsPathFor(pb));
            Assert.DoesNotContain("hnsw", sidecar, StringComparison.OrdinalIgnoreCase);
            Assert.Equal(plain.EmbeddingStore.ExportEmbeddings("d").GraphDigest, hnsw.EmbeddingStore.ExportEmbeddings("d").GraphDigest);
            Assert.Equal(plain.EmbeddingStore.ToJson("d"), hnsw.EmbeddingStore.ToJson("d"));

            // a JSON-only save is identical too
            string pc = System.IO.Path.Combine(dir, "json.ison");
            hnsw.Save(pc);
            Assert.Equal(File.ReadAllBytes(SemanticGraph.EmbeddingsPathFor(pa)), File.ReadAllBytes(SemanticGraph.EmbeddingsPathFor(pc)));
            Assert.False(File.Exists(SemanticGraph.SnapshotPathFor(pc)));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_loaded_store_builds_the_graph_on_its_first_search(bool snapshot)
    {
        string dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hnsw-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var original = Graph(AnnMode.Hnsw, n: 1500);
            string path = System.IO.Path.Combine(dir, "g.ison");
            original.Save(path, snapshot: snapshot);

            var (loaded, report) = SemanticGraph.LoadWithReport(
                path, new MockEncoder(32), ann: AnnMode.Hnsw, annMinSize: 0, annHnsw: new HnswOptions(M: 12, EfConstruction: 80));
            Assert.Equal(snapshot ? LoadSource.Snapshot : LoadSource.Json, report.Source);
            var store = loaded.EmbeddingStore;
            Assert.Equal(AnnMode.Hnsw, store.Ann);
            Assert.False(store.GetAnnStats().Active);                          // loading builds nothing
            var q = new float[32].Select((_, i) => (float)(i + 1)).ToArray();
            var hits = store.SimilaritySearchVector(q, 10);
            Assert.Equal(10, hits.Count);
            Assert.True(store.GetAnnStats().Active);
            Assert.Equal(1500, store.GetAnnStats().GraphNodes);
            Assert.Equal(1, store.GetAnnStats().Searches.Builds);
            AssertScoresAreExact(store, q, hits);
            Assert.Null(store.VerifyInvariants());
            Assert.True(store.RecallCheck(new[] { q }, 10).Approximate);
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    // -- the Rust reference ----------------------------------------------------------------------------

    private const int XSplitMixSeedClusters = 30;

    /// <summary>
    /// The answers <c>isongraph-vector-rs/examples/hnsw_xcheck.rs</c> gives for
    /// <c>3000 16 12 10 8 50 1 16 600</c> (n, dim, queries, k, m, efConstruction, seed, ef, noise), ids only: the graph
    /// the Rust code builds, grows by inserts, updates and deletes, and compacts is the one this port builds. Scores are
    /// left out on purpose: this port sums in its own order, so a score can differ from Rust's in the last bits.
    /// </summary>
    private const string RustAnswers = """
        built q0 2382 1486 1380 325 72 1207 1695 756 43 2652
        built q1 555 104 2357 1292 428 2482 1537 2045 1347 1561
        built q2 1538 973 1027 2796 1559 1075 1893 2210 1101 1510
        built q3 118 2338 2694 333 2548 613 1588 1067 2740 1253
        built q4 1178 2487 1580 1076 757 2645 2493 1133 1209 170
        built q5 726 302 2709 1592 1866 1724 2364 2620 2813 2695
        built q6 1411 1315 268 2262 1033 1375 1002 1867 2573 2239
        built q7 819 2888 179 337 2856 908 2025 1594 1675 2044
        built q8 701 892 1798 791 2748 727 224 1611 2007 2903
        built q9 35 935 1673 2096 789 964 1100 2021 443 541
        built q10 631 2196 875 2302 969 1918 1571 50 1575 820
        built q11 238 1196 2079 929 2420 1661 2637 2252 1294 322
        built stats nodes=3000 dead=0 level=Some(3) approximate=12
        writes q0 2382 1486 1380 72 1207 1695 3191 756 43 2652
        writes q1 3213 555 2357 1292 356 428 2482 1537 2045 1347
        writes q2 1538 973 2796 1559 1075 1893 2210 1101 1510 330
        writes q3 3170 118 2338 2694 333 2548 613 1588 1067 2740
        writes q4 2487 1580 757 2645 3255 1133 170 110 1040 2558
        writes q5 726 302 2709 1592 1866 1724 2364 2620 2813 2695
        writes q6 1411 1315 268 2262 1326 3005 3085 1033 1375 1002
        writes q7 2888 179 337 2856 908 2025 1594 1675 2044 262
        writes q8 701 892 1798 791 2748 727 3219 224 1742 1611
        writes q9 35 935 1673 2096 104 789 964 1100 2021 443
        writes q10 631 2196 875 2302 455 969 1918 1118 1571 50
        writes q11 238 2079 929 2420 1661 3223 2637 2252 1294 3218
        writes stats nodes=3300 dead=120 level=Some(3) approximate=24
        compacted q0 2382 1380 72 1695 3191 756 2652 776 840 1350
        compacted q1 3213 555 2357 1292 356 428 2045 2864 1347 2733
        compacted q2 1538 2796 1559 1893 2210 1101 330 2015 3187 810
        compacted q3 3170 2694 333 1067 1253 735 1643 1461 3190 2841
        compacted q4 2487 1580 2645 2493 3255 1133 110 1040 2558 128
        compacted q5 726 302 2709 1592 1866 1724 2364 2813 2756 3088
        compacted q6 2262 1326 3005 3085 1002 728 2573 1487 2289 2673
        compacted q7 2888 179 2856 908 2025 2262 507 2054 1428 1034
        compacted q8 701 791 2748 3219 224 1742 1611 2007 909 2903
        compacted q9 35 935 1673 2096 104 789 1100 2021 443 1539
        compacted q10 2196 875 455 969 1571 50 1575 831 329 3046
        compacted q11 2079 929 2420 1661 3223 1520 2637 2252 3218 2639
        compacted stats nodes=2474 dead=254 level=Some(3) approximate=36
        """;

    [Fact]
    public void Answers_equal_the_rust_reference_through_build_writes_and_compaction()
    {
        const int n = 3000, dim = 16, nq = 12, k = 10, noise = 600;
        var centersRng = new SplitMixForTests(1UL ^ 0x51);
        var centers = new int[XSplitMixSeedClusters][];
        for (int c = 0; c < centers.Length; c++)
        {
            centers[c] = new int[dim];
            for (int d = 0; d < dim; d++) centers[c][d] = centersRng.Below(2001) - 1000;
        }
        float[] Draw(SplitMixForTests rng)
        {
            var c = centers[rng.Below(centers.Length)];
            var v = new float[dim];
            for (int d = 0; d < dim; d++) v[d] = (c[d] + rng.Below(2 * noise + 1) - noise) / 1024f;
            return v;
        }
        var data = new SplitMixForTests(1UL ^ 0x52);
        var qrng = new SplitMixForTests(1UL ^ 0x53);
        var queries = Enumerable.Range(0, nq).Select(_ => Draw(qrng)).ToArray();
        var store = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 0, annHnsw: new HnswOptions(M: 8, EfConstruction: 50, EfSearch: 16));
        for (int i = 0; i < n; i++) store.Add(Ref(i), "", Draw(data));
        store.BuildAnnIndex();

        var lines = new List<string>();
        void Dump(string phase)
        {
            for (int i = 0; i < nq; i++)
                lines.Add($"{phase} q{i} " + string.Join(' ', store.SimilaritySearchVector(queries[i], k, null, -2f).Select(r => r.NodeRef.Id)));
            var s = store.GetAnnStats();
            lines.Add($"{phase} stats nodes={s.GraphNodes} dead={s.GraphDead} level=Some({s.MaxLevel}) approximate={s.Searches.Approximate}");
        }
        Dump("built");
        var ops = new SplitMixForTests(1UL ^ 0x54);
        for (int i = 0; i < n / 10; i++) store.Add(Ref(n + i), "", Draw(ops));
        for (int i = 0; i < n / 20; i++) store.Add(Ref(i * 13 % n), "", Draw(ops));
        for (int i = 0; i < n / 25; i++) store.Remove(Ref((i * 17 + 5) % n));
        Dump("writes");
        for (int i = 0; i < n; i++)
            if (i % 3 == 1) store.Remove(Ref(i));
        Assert.Null(store.VerifyInvariants());
        Dump("compacted");

        var expected = RustAnswers.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Assert.Equal(expected, lines);
    }

    /// <summary>The generator of the level draw (and of the Rust example's data): SplitMix64, high-64-bit multiply for below(n).</summary>
    private sealed class SplitMixForTests
    {
        private ulong _s;
        public SplitMixForTests(ulong seed) => _s = seed;
        public int Below(int n)
        {
            unchecked
            {
                _s += 0x9E3779B97F4A7C15UL;
                ulong z = _s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                z ^= z >> 31;
                return (int)Math.BigMul(z, (ulong)n, out _);
            }
        }
    }

    // -- memory -------------------------------------------------------------------------------------------------------------

    [Fact]
    public void The_graph_holds_links_only_about_146_bytes_a_node_at_m_16()
    {
        const int n = 4096, dim = 32;
        var store = Store(Data(n, dim, clustered: true, seed: 21));
        var stats = store.BuildAnnIndex();
        Assert.Equal(0, stats.VectorBytes);                                    // no second copy of any vector
        double perNode = stats.IndexBytes / (double)n;
        Assert.InRange(perNode, 130, 170);
    }
}
