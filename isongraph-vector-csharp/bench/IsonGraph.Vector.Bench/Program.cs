using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using IsonGraph;
using IsonGraph.Vector;

// Exact scan vs the opt-in IVF index: latency and recall@10 on synthetic
// clustered unit vectors (default 384-d at 10k / 100k / 1M), mirroring
// benchmarks/ann.py so the numbers line up across ports.
//
//   dotnet run -c Release --project bench/IsonGraph.Vector.Bench
//   dotnet run -c Release --project bench/IsonGraph.Vector.Bench -- --sizes 1000000 --regimes clustered,diffuse --out ann_cs.md
//
// Options: --sizes a,b,c   --regimes clustered[,diffuse]   --dim 384   --queries 200
//          --nprobes 1,2,4,...   --out file.md

var options = Options.Parse(args);
var report = new StringBuilder();
void Say(string line = "") { Console.WriteLine(line); report.AppendLine(line); }

Say("# C# approximate index (IVF-flat) benchmark");
Say();
Say($"Machine: {Environment.ProcessorCount} logical CPUs, {RuntimeInformation.OSDescription}, " +
    $"{RuntimeInformation.FrameworkDescription}, Vector<float>.Count = {System.Numerics.Vector<float>.Count}. " +
    $"{options.Dim}-d unit vectors, top_k=10, {options.Queries} queries per cell, warm single-thread medians " +
    "(index builds use all cores). Synthetic clustered data: recall depends on how the data clusters, " +
    "so treat it as a property of these regimes, not of real embeddings.");
Say();

foreach (var n in options.Sizes)
    foreach (var regimeName in options.Regimes)
        RunOne(n, regimeName, options, Say);

if (options.Out is not null) File.WriteAllText(options.Out, report.ToString());
return 0;

static void RunOne(int n, string regimeName, Options o, Action<string> say)
{
    var regime = new Regime(regimeName, n, o.Dim);
    Console.Error.WriteLine($"[{regimeName} {n:N0}] generating ...");
    var data = regime.Sample(n, stream: 1);
    var queries = regime.Sample(o.Queries, stream: 7);

    var store = new EmbeddingStore(ann: AnnMode.Ivf, annMinSize: 0, annSeed: 0);
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < n; i++)
        store.Add(new NodeRef("doc", i.ToString(CultureInfo.InvariantCulture)), "", data[i]);
    double ingest = n / sw.Elapsed.TotalSeconds;
    Console.Error.WriteLine($"[{regimeName} {n:N0}] ingested ({ingest:N0}/s); exact baseline ...");

    var (exactMs, truth) = Timed(store, queries, exact: true);
    Console.Error.WriteLine($"[{regimeName} {n:N0}] exact median {Pct(exactMs, 0.5):F2} ms; building ...");

    var stats = store.BuildAnnIndex();
    long gcAfter = GC.GetTotalMemory(false);
    say($"## {regimeName}, N = {n:N0}");
    say("");
    say($"* exact scan: warm median {Pct(exactMs, 0.5):F2} ms, p95 {Pct(exactMs, 0.95):F2} ms");
    say($"* index build: {stats.BuildSeconds:F2} s for nlist={stats.Nlist} (automatic nprobe={stats.Nprobe}); " +
        $"index {stats.IndexBytes / 1048576.0:F1} MB (centroids and cell lists only: it keeps no copy of the " +
        $"vectors, the store's {(long)n * o.Dim * 4 / 1048576} MB matrix is the only one); " +
        $"managed heap {gcAfter / 1048576} MB");
    say($"* ingest {ingest:N0} vectors/s (the index is built lazily; ingest is unaffected)");
    say("");
    say("| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |");
    say("|---|---|---|---|---|---|---|---|");

    int nlist = stats.Nlist!.Value;
    int auto = stats.Nprobe!.Value;
    var probes = o.Nprobes.Where(p => p < nlist).Append(auto).Distinct().OrderBy(p => p).ToList();
    foreach (int p in probes)
    {
        store.AnnNprobe = p;
        var (ms, found) = Timed(store, queries, exact: false);
        var recalls = found.Select((f, i) => truth[i].Count == 0 ? 1.0
            : (double)f.Intersect(truth[i]).Count() / truth[i].Count).ToList();
        double med = Pct(ms, 0.5);
        say($"| {p}{(p == auto ? " (auto)" : "")} | {100.0 * p / nlist:F1} | {recalls.Average():F3} | " +
            $"{Pct(recalls, 0.05):F2} | {recalls.Min():F2} | {med:F2} | {Pct(ms, 0.95):F2} | {Pct(exactMs, 0.5) / med:F1}x |");
    }
    say("");
    var final = store.GetAnnStats();
    Console.Error.WriteLine(
        $"[{regimeName} {n:N0}] done; searches approximate={final.Searches.Approximate} " +
        $"exact={final.Searches.Exact} widened={final.Searches.Widened} builds={final.Searches.Builds}");
    GC.KeepAlive(data);
}

static (List<double> Ms, List<HashSet<NodeRef>> Found) Timed(EmbeddingStore store, float[][] queries, bool exact)
{
    for (int i = 0; i < Math.Min(3, queries.Length); i++)       // warm up (JIT, caches)
        store.SimilaritySearchVector(queries[i], 10, exact: exact);
    var ms = new List<double>(queries.Length);
    var found = new List<HashSet<NodeRef>>(queries.Length);
    foreach (var q in queries)
    {
        long t0 = Stopwatch.GetTimestamp();
        var res = store.SimilaritySearchVector(q, 10, exact: exact);
        ms.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        found.Add(res.Select(r => r.NodeRef).ToHashSet());
    }
    return (ms, found);
}

static double Pct(List<double> values, double q)
{
    var ordered = values.OrderBy(v => v).ToList();
    return ordered[Math.Min(ordered.Count - 1, (int)Math.Round(q * (ordered.Count - 1)))];
}

/// <summary>Deterministic generator of clustered unit vectors (the "clustered" and "diffuse" regimes of benchmarks/ann.py).</summary>
sealed class Regime
{
    private readonly int _dim;
    private readonly float _noise;
    private readonly float[][] _centers;

    public Regime(string name, int n, int dim)
    {
        _dim = dim;
        bool diffuse = name == "diffuse";
        int perCluster = diffuse ? 40 : 100;
        _noise = diffuse ? 2.0f : 1.0f;
        int clusters = Math.Max(8, n / perCluster);
        var rng = new Random(4242 + name.Length);
        _centers = new float[clusters][];
        float[][]? supers = null;
        if (diffuse)
        {
            supers = new float[20][];
            for (int s = 0; s < 20; s++) supers[s] = UnitGaussian(rng, dim);
        }
        for (int c = 0; c < clusters; c++)
        {
            var v = UnitGaussian(rng, dim);
            if (supers is not null)
            {
                var parent = supers[rng.Next(supers.Length)];
                for (int d = 0; d < dim; d++) v[d] = parent[d] + 1.0f * v[d];
                Normalize(v);
            }
            _centers[c] = v;
        }
    }

    private static float Gauss(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private static float[] UnitGaussian(Random r, int dim)
    {
        var v = new float[dim];
        for (int d = 0; d < dim; d++) v[d] = Gauss(r);
        Normalize(v);
        return v;
    }

    private static void Normalize(float[] v)
    {
        double norm = 0;
        foreach (float x in v) norm += (double)x * x;
        float inv = (float)(1.0 / Math.Sqrt(norm));
        for (int d = 0; d < v.Length; d++) v[d] *= inv;
    }

    public float[][] Sample(int count, int stream)
    {
        var all = new float[count][];
        const int Chunk = 4096;
        Parallel.For(0, (count + Chunk - 1) / Chunk, chunk =>
        {
            var r = new Random(unchecked(stream * 7919 + chunk * 104729 + 13));
            float scale = _noise / MathF.Sqrt(_dim);
            for (int i = chunk * Chunk, end = Math.Min(count, i + Chunk); i < end; i++)
            {
                var center = _centers[r.Next(_centers.Length)];
                var v = new float[_dim];
                for (int d = 0; d < _dim; d++) v[d] = center[d] + scale * Gauss(r);
                Normalize(v);
                all[i] = v;
            }
        });
        return all;
    }
}

sealed record Options(int[] Sizes, string[] Regimes, int Dim, int Queries, int[] Nprobes, string? Out)
{
    public static Options Parse(string[] args)
    {
        var o = new Options(new[] { 10_000, 100_000, 1_000_000 }, new[] { "clustered" }, 384, 200,
                            new[] { 1, 2, 4, 8, 16, 32, 64, 128 }, null);
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            switch (args[i])
            {
                case "--sizes": o = o with { Sizes = Next().Split(',').Select(int.Parse).ToArray() }; break;
                case "--regimes": o = o with { Regimes = Next().Split(',') }; break;
                case "--dim": o = o with { Dim = int.Parse(Next()) }; break;
                case "--queries": o = o with { Queries = int.Parse(Next()) }; break;
                case "--nprobes": o = o with { Nprobes = Next().Split(',').Select(int.Parse).ToArray() }; break;
                case "--out": o = o with { Out = Next() }; break;
                default: throw new ArgumentException($"unknown option {args[i]}");
            }
        }
        return o;
    }
}
