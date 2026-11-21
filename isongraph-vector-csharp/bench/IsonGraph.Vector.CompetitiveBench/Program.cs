// Loader and measurement harness for benchmarks/competitive/ - see the csproj header.
using System.Diagnostics;
using System.Text.Json;
using NodeRef = IsonGraph.NodeRef;
using IsonGraph.Vector;

using J = System.Collections.Generic.Dictionary<string, object?>;

static string Arg(string[] a, string name, string def)
{
    for (int i = 0; i + 1 < a.Length; i++) if (a[i] == name) return a[i + 1];
    return def;
}

static float[] ReadF32(string path, long start, int count, int d)
{
    using var f = File.OpenRead(path);
    f.Seek(start * d * 4, SeekOrigin.Begin);
    var bytes = new byte[(long)count * d * 4];
    f.ReadExactly(bytes);
    var v = new float[(long)count * d];
    Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length);
    return v;
}
static int[] ReadI32(string path)
{
    var b = File.ReadAllBytes(path);
    var v = new int[b.Length / 4];
    Buffer.BlockCopy(b, 0, v, 0, b.Length);
    return v;
}

static double Pct(IEnumerable<double> xs, double p)
{
    var s = xs.OrderBy(x => x).ToArray();
    return s.Length == 0 ? 0 : s[(int)Math.Round((s.Length - 1) * p)];
}
static double Now() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

static J Mem()
{
    var p = Process.GetCurrentProcess();
    p.Refresh();
    return new J { ["rss_mb"] = p.WorkingSet64 / 1048576.0, ["peak_mb"] = p.PeakWorkingSet64 / 1048576.0 };
}

string dir = Arg(args, "--ds-dir", ""), ds = Arg(args, "--ds", ""), label = Arg(args, "--label", "csharp");
int n = int.Parse(Arg(args, "--n", "100000")), nq = int.Parse(Arg(args, "--queries", "1000"));
double budget = double.Parse(Arg(args, "--budget", "15"));
var exps = Arg(args, "--exp", "core").Split(',').ToHashSet();
string outPath = Arg(args, "--out", "");
var meta = JsonDocument.Parse(File.ReadAllText(System.IO.Path.Combine(dir, "meta.json"))).RootElement;
int d = meta.GetProperty("d").GetInt32();
var qflat = ReadF32(System.IO.Path.Combine(dir, "queries.f32"), 0, nq, d);
var queries = Enumerable.Range(0, nq).Select(i => qflat.AsSpan(i * d, d).ToArray()).ToArray();
const int K = 10;
int cpus = Environment.ProcessorCount;
Console.WriteLine($"{label}: ds={ds} n={n} d={d} queries={nq} cpus={cpus}");

float[][] Base(int start, int count)
{
    var flat = ReadF32(System.IO.Path.Combine(dir, "base.f32"), start, count, d);
    return Enumerable.Range(0, count).Select(i => flat.AsSpan(i * d, d).ToArray()).ToArray();
}
int[][] Gt(string kind)
{
    var f = ReadI32(System.IO.Path.Combine(dir, $"gt_{kind}{n}.i32"));
    return Enumerable.Range(0, f.Length / 100).Select(i => f.AsSpan(i * 100, 100).ToArray()).ToArray();
}
double Recall(List<int[]> found, int[][] truth)
{
    long hit = 0;
    for (int i = 0; i < found.Count; i++)
    {
        var want = truth[i].Take(K).ToHashSet();
        hit += found[i].Take(K).Count(x => want.Contains(x));
    }
    return hit / (double)(K * Math.Max(1, found.Count));
}
int[] Search(EmbeddingStore s, float[] q, bool exact, string? ty)
{
    var r = s.SimilaritySearchVector(q, K, ty, 0.0f, exact);
    var ids = new int[r.Count];
    for (int i = 0; i < ids.Length; i++) ids[i] = int.Parse(r[i].NodeRef.Id.ToString()!);
    return ids;
}
J Point(EmbeddingStore s, int[][] truth, bool exact, string? ty)
{
    // Pin to one P-core (logical CPU 8 on the i9-14900HX) for the single-threaded loop: unpinned loops varied up to 3x.
    var proc = Process.GetCurrentProcess();
    var oldMask = proc.ProcessorAffinity;
    int pinCpu = int.Parse(Environment.GetEnvironmentVariable("BENCH_PIN_CPU") ?? "8");
    try { proc.PriorityClass = ProcessPriorityClass.High; proc.ProcessorAffinity = (IntPtr)(1L << pinCpu); } catch { }
    int oldScan = s.ScanThreads;
    s.ScanThreads = 1;
    try { return PointPinned(s, truth, exact, ty); }
    finally { s.ScanThreads = oldScan; try { proc.ProcessorAffinity = oldMask; } catch { } }
}
J PointPinned(EmbeddingStore s, int[][] truth, bool exact, string? ty)
{
    double w0 = Now();
    for (int i = 0; i < Math.Min(50, nq); i++) { Search(s, queries[i], exact, ty); if (Now() - w0 > 3.0) break; }
    var probe = new List<double>();
    for (int i = 0; i < Math.Min(5, nq); i++) { double t = Now(); Search(s, queries[i], exact, ty); probe.Add(Now() - t); }
    double est = Math.Max(Pct(probe, 0.5), 1e-7);
    int m = (int)Math.Min(nq, Math.Max(30, budget / est));
    var runs = new List<double>();
    var lats = new List<double>();
    double rec = 0;
    for (int r = 0; r < 3; r++)
    {
        var lat = new List<double>(m);
        var res = new List<int[]>(m);
        for (int i = 0; i < m; i++)
        {
            double t = Now();
            var ids = Search(s, queries[i], exact, ty);
            lat.Add(Now() - t);
            if (r == 0) res.Add(ids);
        }
        if (r == 0) rec = Recall(res, truth.Take(m).ToArray());
        double total = lat.Sum();
        runs.Add(m / total);
        lats.AddRange(lat);
        if (total > budget * 1.5) break;
    }
    return new J
    {
        ["nq"] = m, ["recall"] = rec, ["qps_1t"] = Pct(runs, 0.5), ["qps_1t_min"] = runs.Min(), ["qps_1t_max"] = runs.Max(), ["qps_1t_runs"] = runs,
        ["median_ms"] = Pct(lats, 0.5) * 1000, ["p95_ms"] = Pct(lats, 0.95) * 1000, ["p99_ms"] = Pct(lats, 0.99) * 1000, ["mean_ms"] = lats.Average() * 1000,
    };
}
J BatchPoint(EmbeddingStore s, int[][] truth, bool exact)
{
    // all logical CPUs: warm up once, then 3 timed runs of the whole query set (ScanThreads left automatic)
    double Run(out IReadOnlyList<IReadOnlyList<SimilarityResult>> r)
    {
        double t = Now();
        r = s.SimilaritySearchBatch(queries, K, null, 0f, exact, 0);
        return queries.Length / (Now() - t);
    }
    Run(out _);
    var qps = new List<double>();
    IReadOnlyList<IReadOnlyList<SimilarityResult>> last = null!;
    for (int i = 0; i < 3; i++) { qps.Add(Run(out last)); }
    var found = last.Select(row => row.Select(x => int.Parse(x.NodeRef.Id.ToString()!)).ToArray()).ToList();
    return new J { ["nq"] = queries.Length, ["threads"] = cpus, ["qps"] = Pct(qps, 0.5), ["qps_runs"] = qps, ["recall"] = Recall(found, truth.Take(found.Count).ToArray()) };
}
int NlistFor(int x) => (int)Math.Clamp(Math.Round(4 * Math.Sqrt(x)), 16, 65536);
List<int> Knobs()
{
    int nl = NlistFor(n);
    var v = new[] { 1, 2, 4, 8, 16, 32, 64, 128, 256 }.Where(x => x < nl).ToList();
    if (n <= 200_000) v.Add(nl);
    return v;
}
double Ingest(EmbeddingStore s, byte[]? grp)
{
    double secs = 0;
    string[] names = { "t10", "t01", "rest" };
    for (int at = 0; at < n; at += 20000)
    {
        int m = Math.Min(20000, n - at);
        var rows = Base(at, m);
        double t = Now();
        for (int i = 0; i < m; i++) s.Add(new NodeRef(grp == null ? "doc" : names[grp[at + i]], (at + i).ToString()), "", rows[i]);
        secs += Now() - t;
    }
    return secs;
}
J Header(string sys) => new J
{
    ["system"] = sys, ["ds"] = ds, ["n"] = n, ["d"] = d, ["k"] = K, ["queries"] = nq, ["all_threads"] = cpus,
    ["version"] = "isongraph-vector-csharp (repo HEAD, .NET 8 Release)",
};

var runsOut = new List<J>();
var memBase = Mem();
if (exps.Contains("core") || exps.Contains("mutation") || exps.Contains("saveload"))
{
    var store = new EmbeddingStore(ann: AnnMode.None);
    double ingestS = Ingest(store, null);
    var memIng = Mem();
    Console.WriteLine($"  ingest {ingestS:F2}s");
    var truth = Gt("");
    double t0 = Now();
    store.SimilaritySearchVector(queries[0], K, null, 0f, true);
    double firstMs = (Now() - t0) * 1000;
    var ex = Header(label + "_exact");
    ex["build"] = new J { ["ingest_s"] = ingestS, ["index_s"] = 0.0, ["total_s"] = ingestS, ["first_query_ms"] = firstMs, ["ready_s"] = ingestS + firstMs / 1000 };
    ex["mem_baseline"] = memBase; ex["mem_after_ingest"] = memIng; ex["mem_after_build"] = Mem(); ex["mem_after_first_query"] = Mem();
    var p1 = Point(store, truth, true, null); p1["knob"] = null;
    Console.WriteLine($"  exact recall={p1["recall"]} qps={p1["qps_1t"]:F1}");
    ex["sweep"] = new List<J> { p1 };
    ex["batch_allthreads"] = BatchPoint(store, truth, true);
    ex["mem_after_sweep"] = Mem();
    runsOut.Add(ex);

    // IVF: a fresh store with the index on (the port enables it at construction)
    store = null!;
    GC.Collect();
    var st = new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: 0, annSeed: 0);
    ingestS = Ingest(st, null);
    memIng = Mem();
    t0 = Now();
    var stats = st.BuildAnnIndex();
    double idxS = Now() - t0;
    Console.WriteLine($"  ivf build {idxS:F2}s nlist={stats.Nlist}");
    var iv = Header(label + "_ivf");
    iv["knob_name"] = "nprobe";
    iv["build"] = new J { ["ingest_s"] = ingestS, ["index_s"] = idxS, ["total_s"] = ingestS + idxS, ["first_query_ms"] = 0.0, ["ready_s"] = ingestS + idxS };
    iv["mem_baseline"] = memBase; iv["mem_after_ingest"] = memIng; iv["mem_after_build"] = Mem();
    iv["index_stats"] = new J { ["nlist"] = stats.Nlist, ["trained_n"] = stats.TrainedN, ["index_bytes"] = stats.IndexBytes, ["vector_bytes"] = stats.VectorBytes, ["build_seconds"] = stats.BuildSeconds };
    var pts = new List<J>();
    int refKnob = -1; double refRecall = 0;
    foreach (int kn in Knobs())
    {
        st.AnnNprobe = kn;
        var p = Point(st, truth, false, null); p["knob"] = kn;
        p["batch_allthreads"] = BatchPoint(st, truth, false);
        Console.WriteLine($"  ivf nprobe={kn} recall={p["recall"]:F4} qps1t={p["qps_1t"]:F0}");
        if (refKnob < 0 && (double)p["recall"]! >= 0.95) { refKnob = kn; refRecall = (double)p["recall"]!; }
        pts.Add(p);
    }
    // No knob reached 0.95: use the widest and report ITS recall (the mutation section's "before" figure
    // used to print 0.000 here because refRecall was only ever assigned when a knob reached 0.95).
    if (refKnob < 0) { refKnob = Knobs().Last(); refRecall = (double)pts[^1]["recall"]!; }
    iv["sweep"] = pts; iv["ref_knob"] = refKnob; iv["ref_recall_pre"] = refRecall; iv["mem_after_sweep"] = Mem();

    if (exps.Contains("mutation"))
    {
        string gtm = System.IO.Path.Combine(dir, $"gt_mut_{n}.i32");
        if (File.Exists(gtm))
        {
            var upd = ReadI32(System.IO.Path.Combine(dir, $"mut_{n}_upd.i32"));
            var del = ReadI32(System.IO.Path.Combine(dir, $"mut_{n}_del.i32"));
            var f = ReadI32(gtm);
            var truthM = Enumerable.Range(0, f.Length / 100).Select(i => f.AsSpan(i * 100, 100).ToArray()).Take(nq).ToArray();
            st.AnnNprobe = refKnob;
            var m = new J { ["ref_knob"] = refKnob, ["recall_before"] = refRecall };
            var ins = Base(n, 10000);
            double t = Now();
            for (int i = 0; i < ins.Length; i++) st.Add(new NodeRef("doc", (n + i).ToString()), "", ins[i]);
            double dt = Now() - t; m["insert_10k"] = new J { ["seconds"] = dt, ["per_s"] = 10000 / dt };
            var nv = Base(n + 10000, 10000);
            t = Now();
            for (int j = 0; j < nv.Length; j++) st.Add(new NodeRef("doc", upd[j].ToString()), "", nv[j]);
            dt = Now() - t; m["update_10k"] = new J { ["seconds"] = dt, ["per_s"] = 10000 / dt };
            t = Now();
            foreach (int id in del) st.Remove(new NodeRef("doc", id.ToString()));
            dt = Now() - t; m["delete_10k"] = new J { ["seconds"] = dt, ["per_s"] = 10000 / dt };
            t = Now();
            Search(st, queries[0], false, null);
            m["first_query_after_ms"] = (Now() - t) * 1000;
            var p = Point(st, truthM, false, null);
            m["recall_after"] = p["recall"]; m["after"] = p; m["mem_after"] = Mem();
            var s2 = st.GetAnnStats();
            m["index_stats_after"] = new J { ["pending"] = s2.Pending, ["churn"] = s2.Churn, ["trained_n"] = s2.TrainedN };
            Console.WriteLine($"  mutation recall_after={p["recall"]}");
            iv["mutation"] = m;
        }
        else iv["mutation"] = new J { ["skipped"] = "dataset has no spare rows for this n" };
    }
    if (exps.Contains("saveload"))
    {
        string path = System.IO.Path.GetFullPath(System.IO.Path.Combine(dir, "..", "..", "work", $"cs_{ds}_{n}.json"));
        Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path)!);
        var sl = new J();
        try
        {
            double t = Now();
            st.SaveEmbeddings(path);
            sl["save_s"] = Now() - t;
            sl["disk_mb"] = new FileInfo(path).Length / 1048576.0;
            var s2 = new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: 0, annNprobe: refKnob);
            t = Now();
            s2.LoadEmbeddings(path);
            double loadS = Now() - t;
            sl["load_s"] = loadS;
            t = Now();
            Search(s2, queries[0], false, null);
            double fq = (Now() - t) * 1000;
            sl["first_query_after_load_ms"] = fq; sl["time_to_ready_s"] = loadS + fq / 1000;
            // the opt-in binary snapshot of the same store (DESIGN.md section 11)
            var snap = System.IO.Path.ChangeExtension(path, ".snap");
            t = Now();
            st.SaveSnapshot(snap);
            sl["snap_save_s"] = Now() - t;
            sl["snap_mb"] = new FileInfo(snap).Length / 1048576.0;
            var s3 = new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: 0, annNprobe: refKnob);
            t = Now();
            s3.LoadSnapshot(snap);
            sl["snap_load_s"] = Now() - t;
            File.Delete(snap);
            File.Delete(path);
        }
        catch (Exception e) { sl["error"] = e.Message; }
        iv["saveload"] = sl;
    }
    runsOut.Add(iv);
}

if (exps.Contains("hnsw"))
{
    // HNSW (AnnMode.Hnsw): its own store, so the memory figure holds only the vectors and the graph
    var hs = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 0, annSeed: 0, annHnsw: new HnswOptions(16, 200, null));
    double ingestS = Ingest(hs, null);
    var memIng = Mem();
    Console.WriteLine($"  hnsw: ingest {ingestS:F2}s");
    var truth = Gt("");
    double t0 = Now();
    var stats = hs.BuildAnnIndex();
    double idxS = Now() - t0;
    var memBuild = Mem();
    Console.WriteLine($"  hnsw build {idxS:F2}s graphNodes={stats.GraphNodes} maxLevel={stats.MaxLevel} indexBytes={stats.IndexBytes}");
    var hn = Header(label + "_hnsw");
    hn["knob_name"] = "ef_search";
    hn["build"] = new J { ["ingest_s"] = ingestS, ["index_s"] = idxS, ["total_s"] = ingestS + idxS, ["first_query_ms"] = 0.0, ["ready_s"] = ingestS + idxS };
    hn["mem_baseline"] = memBase; hn["mem_after_ingest"] = memIng; hn["mem_after_build"] = memBuild;
    hn["index_stats"] = new J { ["graph_nodes"] = stats.GraphNodes, ["max_level"] = stats.MaxLevel, ["index_bytes"] = stats.IndexBytes, ["build_seconds"] = stats.BuildSeconds, ["default_ef_search"] = stats.EfSearch };
    var pts = new List<J>();
    int refKnob = -1; double refRecall = 0;
    var recs = new List<double>();
    foreach (int ef in new[] { 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400, 640 })
    {
        hs.AnnEfSearch = ef;
        var p = Point(hs, truth, false, null); p["knob"] = ef;
        p["batch_allthreads"] = BatchPoint(hs, truth, false);
        double rec = (double)p["recall"]!;
        Console.WriteLine($"  hnsw ef={ef} recall={rec:F4} qps1t={p["qps_1t"]:F0} batch={((J)p["batch_allthreads"]!)["qps"]:F0}");
        if (refKnob < 0 && rec >= 0.95) { refKnob = ef; refRecall = rec; }
        pts.Add(p);
        recs.Add(rec);
        if (recs.Count > 2 && rec >= 0.9995 && recs[^2] >= 0.9995) break;
    }
    if (refKnob < 0) { refKnob = 640; refRecall = (double)pts[^1]["recall"]!; }
    hn["sweep"] = pts; hn["ref_knob"] = refKnob; hn["ref_recall_pre"] = refRecall; hn["mem_after_sweep"] = Mem();
    // the automatic ef (what a user who just enables HNSW gets)
    hs.AnnEfSearch = null;
    var pd = Point(hs, truth, false, null); pd["knob"] = stats.EfSearch; pd["batch_allthreads"] = BatchPoint(hs, truth, false);
    hn["default_ef_point"] = pd;
    Console.WriteLine($"  hnsw default ef={stats.EfSearch} recall={pd["recall"]:F4} qps1t={pd["qps_1t"]:F0}");
    // concurrent clients (each one query at a time on the shared store)
    hs.AnnEfSearch = refKnob;
    hs.ScanThreads = 1;
    var conc = new List<J>();
    foreach (int clients in new[] { 8, Math.Min(16, cpus) })
    {
        int total = Math.Min(nq, 1000);
        double tc = Now();
        var ths = Enumerable.Range(0, clients).Select(w => new Thread(() => { for (int i = w; i < total; i += clients) Search(hs, queries[i], false, null); })).ToList();
        ths.ForEach(x => x.Start()); ths.ForEach(x => x.Join());
        conc.Add(new J { ["clients"] = clients, ["qps"] = total / (Now() - tc) });
    }
    hs.ScanThreads = 0;
    hn["concurrent_clients"] = conc;
    runsOut.Add(hn);
}

if (exps.Contains("filter"))
{
    var g = File.ReadAllBytes(System.IO.Path.Combine(dir, $"grp_{n}.u8"));
    var st = new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: 0, annSeed: 0);
    double ingestS = Ingest(st, g);
    double t0 = Now();
    st.BuildAnnIndex();
    double idxS = Now() - t0;
    var res = Header(label + "_ivf");
    var subsets = new J();
    res["filter"] = new J { ["build_with_metadata_s"] = ingestS + idxS, ["mem"] = Mem(), ["subsets"] = subsets };
    var kn = Knobs();
    var pick = new SortedSet<int> { 1, kn.Count / 3, kn.Count / 2, 2 * kn.Count / 3, kn.Count - 1 };
    foreach (var (ty, lab) in new[] { ("t10", "f10"), ("t01", "f01") })
    {
        var truth = Gt(lab + "_");
        var pts = new List<J>();
        foreach (int i in pick)
        {
            st.AnnNprobe = kn[i];
            var p = Point(st, truth, false, ty); p["knob"] = kn[i];
            Console.WriteLine($"  filter {lab} nprobe={kn[i]} recall={p["recall"]:F4} qps={p["qps_1t"]:F0}");
            pts.Add(p);
        }
        var pe = Point(st, truth, true, ty); pe["knob"] = "exact"; pts.Add(pe);
        subsets[lab] = pts;
    }
    runsOut.Add(res);
}

if (outPath.Length > 0)
{
    File.WriteAllText(outPath, JsonSerializer.Serialize(new J { ["status"] = "ok", ["label"] = label, ["runs"] = runsOut }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"wrote {outPath}");
}
