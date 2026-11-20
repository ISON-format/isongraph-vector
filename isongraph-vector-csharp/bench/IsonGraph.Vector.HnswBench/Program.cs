using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using IsonGraph;
using IsonGraph.Vector;

// HNSW vs IVF vs the exact scan on synthetic data, and a cross-check against the Rust port.
//
//   dotnet run -c Release --project bench/IsonGraph.Vector.HnswBench -- --n 100000
//   dotnet run -c Release --project bench/IsonGraph.Vector.HnswBench -- --n 100000 --data random --ivf
//   dotnet run -c Release --project bench/IsonGraph.Vector.HnswBench -- --mode xcheck --n 20000 --dim 64 --rust rust_topk.txt
//
// Sweep options: --n N  --dim D (384)  --data clustered|random  --noisef F (cluster spread, 1 = Rust's synthetic; larger = harder)  --queries Q (1000)  --ef a,b,c
//                --m M (16)  --efc E (200)  --threads T (build/batch cap, 0 = all)  --ivf (also sweep IVF nprobe; --skip-hnsw leaves HNSW out)  --budget SECONDS (per timed loop)
// xcheck options: --n --dim --noise (0.3) --m --efc --ef (64) --queries (200) --rust FILE (dump written by the Rust driver)
//
//   dotnet run -c Release --project bench/IsonGraph.Vector.HnswBench -- --mode xdump [n dim queries k m efc seed ef noise]
//     (the same output as isongraph-vector-rs/examples/hnsw_xcheck.rs, for diff: ids and score bit patterns after the build,
//      after inserts/updates/deletes, and after a compaction)
//
// The data in the sweep mirrors examples/hnsw_bench.rs (clustered unit vectors, n/100 clusters); the draws use .NET's
// Math functions, so the vectors are the same kind as Rust's, not the same bits. The xcheck data uses integer
// SplitMix64 and exact f32 operations only, so both ports build the same vectors bit for bit.
// Timings are wall-clock on whatever the machine is doing; read them as indicative.

var options = Options.Parse(args);
if (options.Mode == "xcheck") return XCheck.Run(options);
if (options.Mode == "xdump") return XDump.Run(args);
Sweep.Run(options);
return 0;

sealed class Options
{
    public string Mode = "sweep", Data = "clustered", Rust = "";
    public int N = 50_000, Dim = 384, Queries = 1000, M = 16, Efc = 200, Threads = 0, XEf = 64;
    public int[] Efs = { 16, 32, 64, 96, 128, 192, 256 };
    public bool Ivf, SkipHnsw;
    public double Budget = 4, Noise = 0.3, NoiseFactor = 1.0;

    public static Options Parse(string[] a)
    {
        var o = new Options();
        string Get(string n, string d) { for (int i = 0; i + 1 < a.Length; i++) if (a[i] == n) return a[i + 1]; return d; }
        o.Mode = Get("--mode", "sweep");
        o.Data = Get("--data", "clustered");
        o.N = int.Parse(Get("--n", "50000"), CultureInfo.InvariantCulture);
        o.Dim = int.Parse(Get("--dim", "384"), CultureInfo.InvariantCulture);
        o.Queries = int.Parse(Get("--queries", o.Mode == "xcheck" ? "200" : "1000"), CultureInfo.InvariantCulture);
        o.M = int.Parse(Get("--m", "16"), CultureInfo.InvariantCulture);
        o.Efc = int.Parse(Get("--efc", "200"), CultureInfo.InvariantCulture);
        o.Threads = int.Parse(Get("--threads", "0"), CultureInfo.InvariantCulture);
        o.Budget = double.Parse(Get("--budget", "4"), CultureInfo.InvariantCulture);
        o.Noise = double.Parse(Get("--noise", "0.3"), CultureInfo.InvariantCulture);
        o.NoiseFactor = double.Parse(Get("--noisef", "1"), CultureInfo.InvariantCulture);
        if (o.Mode == "xcheck") o.XEf = int.Parse(Get("--ef", "64"), CultureInfo.InvariantCulture);
        o.Rust = Get("--rust", "");
        o.Ivf = a.Contains("--ivf");
        o.SkipHnsw = a.Contains("--skip-hnsw");
        var efs = Get("--ef", "");
        if (o.Mode != "xcheck" && efs.Length > 0) o.Efs = efs.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        return o;
    }
}

struct SplitMix
{
    private ulong _s;
    public SplitMix(ulong seed) => _s = seed;
    public ulong Next()
    {
        unchecked
        {
            _s += 0x9E3779B97F4A7C15UL;
            ulong z = _s;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }
    public int Below(int n) => (int)Math.BigMul(Next(), (ulong)n, out _);
    /// <summary>Uniform in [0, 1), 24 bits.</summary>
    public double Uniform() => ((Next() >> 40) + 0.5) / 16777216.0;
    public double Gauss() => Math.Sqrt(-2.0 * Math.Log(Uniform())) * Math.Cos(2.0 * Math.PI * Uniform());
}

static class Sweep
{
    static void Unit(float[] v)
    {
        double s = 0;
        foreach (float x in v) s += (double)x * x;
        float norm = (float)Math.Sqrt(s);
        if (norm > 0) for (int i = 0; i < v.Length; i++) v[i] /= norm;
    }

    sealed class Source
    {
        private readonly float[][]? _centers;
        private readonly float _noise = 1f;
        public readonly int D;
        public Source(string kind, int n, int d, double noiseFactor = 1.0)
        {
            D = d;
            _noise = (float)noiseFactor;
            if (kind != "clustered") return;
            var rng = new SplitMix(4242);
            int clusters = Math.Max(n / 100, 8);
            _centers = new float[clusters][];
            for (int c = 0; c < clusters; c++)
            {
                var v = new float[d];
                for (int i = 0; i < d; i++) v[i] = (float)rng.Gauss();
                Unit(v);
                _centers[c] = v;
            }
        }

        public float[] Make(ulong key)
        {
            var rng = new SplitMix(unchecked(key * 0x2545F4914F6CDD1DUL));
            var v = new float[D];
            if (_centers is null)
            {
                for (int i = 0; i < D; i++) v[i] = (float)rng.Gauss();
            }
            else
            {
                var c = _centers[rng.Below(_centers.Length)];
                float scale = _noise / MathF.Sqrt(D);
                for (int i = 0; i < D; i++) v[i] = c[i] + (float)rng.Gauss() * scale;
            }
            Unit(v);
            return v;
        }

        public float[][] Base(int start, int count) =>
            Enumerable.Range(start, count).Select(i => Make(0x1000UL + (ulong)i)).ToArray();
        public float[][] Queries(int count) =>
            Enumerable.Range(0, count).Select(i => Make(0xFFFF0000UL + (ulong)i)).ToArray();
    }

    static double Pct(List<double> sorted, double p) =>
        sorted.Count == 0 ? 0 : sorted[(int)Math.Round((sorted.Count - 1) * p)];

    static double Recall(List<int[]> found, List<int[]> truth, int k)
    {
        long hit = 0;
        for (int i = 0; i < found.Count; i++)
        {
            var want = truth[i].Take(k).ToHashSet();
            hit += found[i].Take(k).Count(want.Contains);
        }
        return hit / (double)(k * Math.Max(found.Count, 1));
    }

    static int[] Ids(IReadOnlyList<SimilarityResult> r) => r.Select(x => int.Parse(x.NodeRef.Id, CultureInfo.InvariantCulture)).ToArray();

    static void Fill(EmbeddingStore store, Source src, int n)
    {
        const int step = 20_000;
        for (int at = 0; at < n; at += step)
        {
            var rows = src.Base(at, Math.Min(step, n - at));
            for (int i = 0; i < rows.Length; i++)
                store.Add(new NodeRef("doc", (at + i).ToString(CultureInfo.InvariantCulture)), "", rows[i]);
        }
    }

    record Point(string Label, double Recall, double Qps1, double MedianMs, double P99Ms, double QpsBatch);

    static Point Measure(EmbeddingStore store, float[][] queries, List<int[]> truth, int k, int? ef, double budget, string label)
    {
        IReadOnlyList<SimilarityResult> Search(float[] q) =>
            ef is int e ? store.SimilaritySearchVectorEf(q, k, null, -2f, e) : store.SimilaritySearchVector(q, k, null, -2f);
        var found = queries.Select(q => Ids(Search(q))).ToList();
        double recall = Recall(found, truth, k);
        int saved = store.ScanThreads;
        store.ScanThreads = 1;
        var lat = new List<double>();
        long t0 = Stopwatch.GetTimestamp();
        for (int pass = 0; pass < 3; pass++)
        {
            foreach (var q in queries)
            {
                long s = Stopwatch.GetTimestamp();
                Search(q);
                lat.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
                if (Stopwatch.GetElapsedTime(t0).TotalSeconds > budget) goto done;
            }
        }
    done:
        double total = Stopwatch.GetElapsedTime(t0).TotalSeconds;
        store.ScanThreads = saved;
        lat.Sort();
        long t1 = Stopwatch.GetTimestamp();
        long done2 = 0;
        var qlist = queries.ToList();
        while (Stopwatch.GetElapsedTime(t1).TotalSeconds < Math.Min(budget, 3.0))
        {
            // the batch path uses the configured ef, so a per-call ef is applied by the caller through AnnEfSearch
            store.SimilaritySearchBatch(qlist, k, null, -2f);
            done2 += qlist.Count;
        }
        double qpsBatch = done2 / Stopwatch.GetElapsedTime(t1).TotalSeconds;
        return new Point(label, recall, lat.Count / total, Pct(lat, 0.5), Pct(lat, 0.99), qpsBatch);
    }

    static void Header(string title)
    {
        Console.WriteLine($"\n{title}");
        Console.WriteLine($"  {"setting",-12} {"recall",8} {"qps(1T)",10} {"median ms",10} {"p99 ms",10} {"qps(batch)",12}");
    }

    static void Print(Point p) =>
        Console.WriteLine($"  {p.Label,-12} {p.Recall,8:F4} {p.Qps1,10:F0} {p.MedianMs,10:F3} {p.P99Ms,10:F3} {p.QpsBatch,12:F0}");

    /// <summary>Ingest into an exact store, take the ground truth and the single-thread exact speed. Its own method so the store is garbage before the memory baseline is taken.</summary>
    static (float[][] Queries, List<int[]> Truth, long AfterIngest) ExactPhase(Options o, Source src)
    {
        var store = new EmbeddingStore { ScanThreads = o.Threads };
        var sw = Stopwatch.StartNew();
        long baseline = GC.GetTotalMemory(true);
        Fill(store, src, o.N);
        double ingest = sw.Elapsed.TotalSeconds;
        long afterIngest = GC.GetTotalMemory(true);
        double raw = (double)o.N * o.Dim * 4;
        Console.WriteLine($"ingest {ingest:F2}s ({o.N / ingest:F0} vec/s); managed heap +{(afterIngest - baseline) / 1048576.0:F0} MB = {(afterIngest - baseline) / raw:F3}x raw vectors ({raw / 1048576.0:F0} MB)");
        var queries = src.Queries(o.Queries);
        sw.Restart();
        var truth = queries.Select(q => Ids(store.SimilaritySearchVector(q, 100, null, -2f, exact: true))).ToList();
        Console.WriteLine($"exact ground truth ({sw.Elapsed.TotalSeconds:F1}s, {sw.Elapsed.TotalMilliseconds / queries.Length:F1} ms/query, auto threads)");
        store.ScanThreads = 1;
        var lat = new List<double>();
        foreach (var q in queries.Take(Math.Min(200, queries.Length)))
        {
            long s = Stopwatch.GetTimestamp();
            store.SimilaritySearchVector(q, 10, null, -2f, exact: true);
            lat.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }
        lat.Sort();
        Console.WriteLine($"exact scan, 1 thread: median {Pct(lat, 0.5):F2} ms ({1000 / Pct(lat, 0.5):F0} qps)");
        return (queries, truth, afterIngest);
    }

    public static void Run(Options o)
    {
        var src = new Source(o.Data, o.N, o.Dim, o.NoiseFactor);
        Console.WriteLine($"hnsw_bench (C#): n={o.N} d={o.Dim} queries={o.Queries} cpus={Environment.ProcessorCount} " +
            $"threads={o.Threads} M={o.M} efC={o.Efc} data={o.Data} noisef={o.NoiseFactor} {RuntimeInformation.FrameworkDescription} " +
            $"{RuntimeInformation.OSDescription} Vector<float>.Count={System.Numerics.Vector<float>.Count}");
        Console.WriteLine("(timings are indicative: the machine is shared)");

        long baseline = GC.GetTotalMemory(true);
        var (queries, truth, afterIngest) = ExactPhase(o, src);
        double raw = (double)o.N * o.Dim * 4;
        GC.Collect();
        baseline = GC.GetTotalMemory(true);
        var sw = new Stopwatch();

        // ---- HNSW ----
        if (!o.SkipHnsw)
        {
        var h = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 0, annHnsw: new HnswOptions(o.M, o.Efc)) { ScanThreads = o.Threads };
        Fill(h, src, o.N);
        long beforeBuild = GC.GetTotalMemory(true);
        sw.Restart();
        var stats = h.BuildAnnIndex();
        double buildS = sw.Elapsed.TotalSeconds;
        long afterBuild = GC.GetTotalMemory(true);
        Console.WriteLine($"\nHNSW build {buildS:F1}s ({o.N / buildS:F0} nodes/s), max level {stats.MaxLevel}");
        Console.WriteLine($"  index {stats.IndexBytes / 1048576.0:F1} MB = {(double)stats.IndexBytes / o.N:F1} B/node = {stats.IndexBytes / raw:F3}x the raw vectors; " +
            $"whole store (managed heap) {(afterBuild - baseline) / raw:F3}x raw (store before build {(beforeBuild - baseline) / raw:F3}x); build delta +{(afterBuild - beforeBuild) / 1048576.0:F1} MB");

        Header("HNSW: recall@10 vs speed over ef_search");
        var points = new List<Point>();
        foreach (int ef in o.Efs)
        {
            h.AnnEfSearch = ef;
            var p = Measure(h, queries, truth, 10, ef, o.Budget, $"ef={ef}");
            Print(p);
            points.Add(p);
        }
        h.AnnEfSearch = null;
        var def = Measure(h, queries, truth, 10, null, o.Budget, "default");
        Print(def);
        Console.WriteLine($"  RESULT hnsw_default recall={def.Recall:F4} qps1={def.Qps1:F0} qpsbatch={def.QpsBatch:F0} build_s={buildS:F2} auto_ef={h.GetAnnStats().EfSearch}");
        var first95 = points.FirstOrDefault(p => p.Recall >= 0.95);
        if (first95 is not null)
            Console.WriteLine($"  first setting at recall >= 0.95: {first95.Label} ({first95.Recall:F4}), {first95.Qps1:F0} qps single-thread, {first95.QpsBatch:F0} qps batch");

        // ---- writes into the built graph ----
        int extra = Math.Min(2000, Math.Max(o.Queries, 500));
        var more = src.Base(o.N, extra);
        var ins = new List<double>();
        for (int i = 0; i < more.Length; i++)
        {
            long s = Stopwatch.GetTimestamp();
            h.Add(new NodeRef("doc", (o.N + i).ToString(CultureInfo.InvariantCulture)), "", more[i]);
            ins.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }
        var upd = new List<double>();
        for (int i = 0; i < extra / 2; i++)
        {
            long s = Stopwatch.GetTimestamp();
            h.Add(new NodeRef("doc", i.ToString(CultureInfo.InvariantCulture)), "", more[i]);
            upd.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }
        var del = new List<double>();
        for (int i = 0; i < extra / 2; i++)
        {
            long s = Stopwatch.GetTimestamp();
            h.Remove(new NodeRef("doc", (o.N + i).ToString(CultureInfo.InvariantCulture)));
            del.Add(Stopwatch.GetElapsedTime(s).TotalMilliseconds);
        }
        ins.Sort(); upd.Sort(); del.Sort();
        Console.WriteLine($"\nwrites into the built graph (ms): insert median {Pct(ins, 0.5):F3} p99 {Pct(ins, 0.99):F3} max {Pct(ins, 1):F3} | " +
            $"update median {Pct(upd, 0.5):F3} p99 {Pct(upd, 0.99):F3} | delete median {Pct(del, 0.5):F4} p99 {Pct(del, 0.99):F4}");
        Console.WriteLine($"  RESULT writes insert_med={Pct(ins, 0.5):F3} update_med={Pct(upd, 0.5):F3}");

        }

        // ---- IVF on the same data ----
        if (o.Ivf)
        {
            var ivf = new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: 0) { ScanThreads = o.Threads };
            Fill(ivf, src, o.N);
            long b0 = GC.GetTotalMemory(true);
            sw.Restart();
            var st = ivf.BuildAnnIndex();
            double ivfBuild = sw.Elapsed.TotalSeconds;
            long b1 = GC.GetTotalMemory(true);
            Console.WriteLine($"\nIVF build {ivfBuild:F1}s nlist={st.Nlist}; index {st.IndexBytes / 1048576.0:F1} MB (managed delta {(b1 - b0) / 1048576.0:F1} MB)");
            Header("IVF-flat: recall@10 vs speed over nprobe");
            int nlist = st.Nlist ?? 16;
            foreach (double pct in new[] { 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 45.0 })
            {
                int nprobe = Math.Max(1, (int)Math.Round(nlist * pct / 100.0));
                ivf.AnnNprobe = nprobe;
                var p = Measure(ivf, queries, truth, 10, null, o.Budget, $"nprobe={nprobe}");
                Print(p);
                Console.WriteLine($"  RESULT ivf nprobe={nprobe} recall={p.Recall:F4} qps1={p.Qps1:F0} qpsbatch={p.QpsBatch:F0} build_s={ivfBuild:F2}");
            }
        }
    }
}

static class XCheck
{
    static float U(ref SplitMix rng) => ((rng.Next() >> 40) / 8388608.0f) - 1.0f;

    public static int Run(Options o)
    {
        int n = o.N, dim = o.Dim, clusters = 50;
        float noise = (float)o.Noise;
        var rng = new SplitMix(0xC0FFEE);
        var centers = new float[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centers[c] = new float[dim];
            for (int d = 0; d < dim; d++) centers[c][d] = U(ref rng);
        }
        float[] Make(ref SplitMix r)
        {
            var c = centers[r.Below(clusters)];
            var v = new float[dim];
            for (int d = 0; d < dim; d++) v[d] = c[d] + U(ref r) * noise;
            return v;
        }
        var data = new float[n][];
        for (int i = 0; i < n; i++) data[i] = Make(ref rng);
        var qr = new SplitMix(0xBEEF);
        var queries = new float[o.Queries][];
        for (int i = 0; i < queries.Length; i++) queries[i] = Make(ref qr);

        var store = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 0, annHnsw: new HnswOptions(o.M, o.Efc, o.XEf));
        for (int i = 0; i < n; i++) store.Add(new NodeRef("doc", i.ToString(CultureInfo.InvariantCulture)), "", data[i]);
        var sw = Stopwatch.StartNew();
        var stats = store.BuildAnnIndex();
        Console.WriteLine($"C# build {sw.Elapsed.TotalSeconds:F2}s: max_level {stats.MaxLevel} nodes {stats.GraphNodes} bytes {stats.IndexBytes}");

        var csH = new List<int[]>();
        var csE = new List<int[]>();
        foreach (var q in queries)
        {
            csH.Add(store.SimilaritySearchVector(q, 10, null, -1f).Select(r => int.Parse(r.NodeRef.Id, CultureInfo.InvariantCulture)).ToArray());
            csE.Add(store.SimilaritySearchVector(q, 10, null, -1f, exact: true).Select(r => int.Parse(r.NodeRef.Id, CultureInfo.InvariantCulture)).ToArray());
        }
        double Rec(List<int[]> a, List<int[]> t) => a.Zip(t).Sum(p => p.First.Count(p.Second.Contains)) / (10.0 * a.Count);
        Console.WriteLine($"C# recall@10 vs C# exact = {Rec(csH, csE):F4}");
        if (o.Rust.Length == 0) return 0;

        var rustH = new List<int[]>();
        var rustE = new List<int[]>();
        foreach (var line in File.ReadLines(o.Rust))
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var ids = parts.Skip(1).Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
            (parts[0] == "H" ? rustH : rustE).Add(ids);
        }
        if (rustH.Count != queries.Length) { Console.WriteLine($"rust file has {rustH.Count} queries, expected {queries.Length}"); return 1; }
        Console.WriteLine($"Rust recall@10 vs Rust exact = {Rec(rustH, rustE):F4}");
        int exactSame = csE.Zip(rustE).Count(p => p.First.SequenceEqual(p.Second));
        int exactSetSame = csE.Zip(rustE).Count(p => p.First.ToHashSet().SetEquals(p.Second));
        Console.WriteLine($"exact top-10: C# == Rust (same ids, same order) for {exactSame}/{queries.Length} queries, same id set for {exactSetSame}/{queries.Length} " +
            "(a difference means different data bits or a last-bit tie, not the graph)");
        int hSame = csH.Zip(rustH).Count(p => p.First.SequenceEqual(p.Second));
        int hSet = csH.Zip(rustH).Count(p => p.First.ToHashSet().SetEquals(p.Second));
        double overlap = csH.Zip(rustH).Average(p => p.First.Count(p.Second.Contains) / 10.0);
        Console.WriteLine($"HNSW top-10: C# == Rust (same ids, same order) for {hSame}/{queries.Length}, same id set for {hSet}/{queries.Length}, mean overlap {overlap:F4}");
        // where they differ, who is right?
        int csOnlyRight = 0, rustOnlyRight = 0;
        for (int i = 0; i < queries.Length; i++)
        {
            if (csH[i].SequenceEqual(rustH[i])) continue;
            int c = csH[i].Count(csE[i].Contains), r = rustH[i].Count(rustE[i].Contains);
            if (c > r) csOnlyRight++; else if (r > c) rustOnlyRight++;
        }
        Console.WriteLine($"queries where the answers differ: C# closer to exact on {csOnlyRight}, Rust closer on {rustOnlyRight}");
        return 0;
    }
}

/// <summary>
/// The output of <c>isongraph-vector-rs/examples/hnsw_xcheck.rs</c>, line for line (same data, same operations), so the two
/// dumps can be compared with a plain diff. Positional arguments after <c>--mode xdump</c> as in that example.
/// </summary>
static class XDump
{
    const int Clusters = 30;

    static int[][] Centers(ref SplitMix rng, int dim)
    {
        var c = new int[Clusters][];
        for (int i = 0; i < Clusters; i++)
        {
            c[i] = new int[dim];
            for (int d = 0; d < dim; d++) c[i][d] = rng.Below(2001) - 1000;
        }
        return c;
    }

    static float[] Draw(ref SplitMix rng, int[][] centers, int noise)
    {
        var c = centers[rng.Below(centers.Length)];
        var v = new float[c.Length];
        for (int d = 0; d < c.Length; d++) v[d] = (c[d] + rng.Below(2 * noise + 1) - noise) / 1024f;
        return v;
    }

    static string Line(string phase, char tag, int i, IReadOnlyList<SimilarityResult> hits) =>
        $"{phase} {tag}{i}" + string.Concat(hits.Select(h => $" {h.NodeRef.Id}:{BitConverter.SingleToUInt32Bits(h.Score):x8}"));

    static void Dump(string phase, EmbeddingStore store, float[][] queries, int k, TextWriter w)
    {
        for (int i = 0; i < queries.Length; i++) w.WriteLine(Line(phase, 'q', i, store.SimilaritySearchVector(queries[i], k, null, -2f)));
        for (int i = 0; i < queries.Length; i++) w.WriteLine(Line(phase, 'x', i, store.SimilaritySearchVector(queries[i], k, null, -2f, exact: true)));
        var s = store.GetAnnStats();
        w.WriteLine($"{phase} stats nodes={s.GraphNodes} dead={s.GraphDead} level=Some({s.MaxLevel}) approximate={s.Searches.Approximate}");
    }

    public static int Run(string[] args)
    {
        var pos = args.SkipWhile(a => a != "xdump").Skip(1).ToArray();
        int Get(int i, int d) => i < pos.Length && int.TryParse(pos[i], CultureInfo.InvariantCulture, out int v) ? v : d;
        int n = Get(0, 5000), dim = Get(1, 32), nq = Get(2, 100), k = Get(3, 10), m = Get(4, 16), efc = Get(5, 200);
        ulong seed = (ulong)Get(6, 1);
        int ef = Get(7, 0), noise = Get(8, 500);
        var csRng = new SplitMix(seed ^ 0x51);
        var cs = Centers(ref csRng, dim);
        var data = new SplitMix(seed ^ 0x52);
        var qrng = new SplitMix(seed ^ 0x53);
        var queries = Enumerable.Range(0, nq).Select(_ => Draw(ref qrng, cs, noise)).ToArray();
        var store = new EmbeddingStore(ann: AnnMode.Hnsw, annMinSize: 0, annHnsw: new HnswOptions(m, efc, ef == 0 ? null : ef));
        for (int i = 0; i < n; i++)
            store.Add(new NodeRef("doc", i.ToString(CultureInfo.InvariantCulture)), "", Draw(ref data, cs, noise));
        store.BuildAnnIndex();
        var w = Console.Out;
        w.WriteLine($"params n={n} dim={dim} queries={nq} k={k} m={m} efc={efc} seed={seed} ef={ef} noise={noise}");
        Dump("built", store, queries, k, w);

        var ops = new SplitMix(seed ^ 0x54);
        for (int i = 0; i < n / 10; i++)
            store.Add(new NodeRef("doc", (n + i).ToString(CultureInfo.InvariantCulture)), "", Draw(ref ops, cs, noise));
        for (int i = 0; i < n / 20; i++)
            store.Add(new NodeRef("doc", ((i * 13) % n).ToString(CultureInfo.InvariantCulture)), "", Draw(ref ops, cs, noise));
        for (int i = 0; i < n / 25; i++)
            store.Remove(new NodeRef("doc", ((i * 17 + 5) % n).ToString(CultureInfo.InvariantCulture)));
        Dump("writes", store, queries, k, w);

        for (int i = 0; i < n; i++)
            if (i % 3 == 1) store.Remove(new NodeRef("doc", i.ToString(CultureInfo.InvariantCulture)));
        Dump("compacted", store, queries, k, w);
        return 0;
    }
}
