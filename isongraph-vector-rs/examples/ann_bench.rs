//! Exact search against the opt-in IVF index on synthetic clustered vectors.
//!
//! For each size and data regime it prints: the ingest rate, the first query
//! (which builds the scan cache), warm exact latency on one thread and on all
//! of them, the index build time and size, and a sweep over `nprobe` with
//! recall@10 against the exact scan and the speed-up.
//!
//! The data mirror `benchmarks/ann.py` in the repository root (384-d unit
//! vectors; "clustered" = well-separated topics, N/100 clusters; "diffuse" =
//! overlapping, hierarchical, N/40 clusters in 20 super-topics), so the numbers
//! can sit next to the Python ones. They are generated here, with this port's
//! own random numbers, so they are *the same kind* of data, not the same
//! vectors. Synthetic clustered data says what the index does on clustered
//! data and nothing about your embeddings: use `recall_check` on your own
//! queries.
//!
//! ```text
//! cargo run --release --example ann_bench                       # 10k and 100k
//! cargo run --release --example ann_bench -- --sizes 10000,100000,1000000
//! cargo run --release --example ann_bench -- --sizes 1000000 --regimes clustered --json out.json
//! ```
//!
//! Options: `--sizes a,b,c`  `--regimes clustered,diffuse`  `--queries N`
//! `--dim D`  `--nprobe 1,4,16,64,auto`  `--k K`  `--json FILE`  `--no-index`.
//! Needs about 3.1 GB of RAM per million vectors (the store keeps the vectors
//! once for the records and once in the scan cache).

use isongraph_vector_rs::{AnnConfig, EmbeddingStore, NodeId, SimilarityResult};
use std::collections::HashSet;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
    fn fill_gauss(&mut self, out: &mut [f32]) {
        for pair in out.chunks_mut(2) {
            let r = (-2.0 * self.uniform().ln()).sqrt();
            let a = std::f32::consts::TAU * self.uniform();
            pair[0] = r * a.cos();
            if pair.len() == 2 {
                pair[1] = r * a.sin();
            }
        }
    }
}

fn unit(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

struct Regime {
    centers: Vec<Vec<f32>>,
    noise: f32,
    dim: usize,
}

impl Regime {
    fn new(name: &str, n: usize, dim: usize) -> Self {
        let mut rng = Rng::new(4242 ^ (n as u64).wrapping_mul(0x9E37) ^ name.len() as u64);
        let (per_cluster, noise, supers, spread) = match name {
            "clustered" => (100, 1.0, 0, 0.0),
            "diffuse" => (40, 2.0, 20, 1.0),
            other => panic!("unknown regime {other:?} (clustered, diffuse)"),
        };
        let clusters = (n / per_cluster).max(8);
        let gauss_unit = |rng: &mut Rng| {
            let mut v = vec![0.0f32; dim];
            rng.fill_gauss(&mut v);
            unit(&mut v);
            v
        };
        let super_centers: Vec<Vec<f32>> = (0..supers).map(|_| gauss_unit(&mut rng)).collect();
        let centers = (0..clusters)
            .map(|_| {
                let mut c = gauss_unit(&mut rng);
                if supers > 0 {
                    let parent = &super_centers[rng.below(supers)];
                    for (x, p) in c.iter_mut().zip(parent) {
                        *x = p + spread * *x;
                    }
                    unit(&mut c);
                }
                c
            })
            .collect();
        Self {
            centers,
            noise,
            dim,
        }
    }

    fn sample(&self, count: usize, stream: u64) -> Vec<Vec<f32>> {
        let mut rng = Rng::new(0xA5A5_0000 ^ stream.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let scale = self.noise / (self.dim as f32).sqrt();
        (0..count)
            .map(|_| {
                let mut v = vec![0.0f32; self.dim];
                rng.fill_gauss(&mut v);
                let c = &self.centers[rng.below(self.centers.len())];
                for (x, c) in v.iter_mut().zip(c) {
                    *x = c + *x * scale;
                }
                unit(&mut v);
                v
            })
            .collect()
    }
}

struct Opts {
    sizes: Vec<usize>,
    regimes: Vec<String>,
    queries: usize,
    dim: usize,
    k: usize,
    nprobe: Vec<Option<usize>>,
    json: Option<String>,
    index: bool,
}

fn parse() -> Opts {
    let mut o = Opts {
        sizes: vec![10_000, 100_000],
        regimes: vec!["clustered".into(), "diffuse".into()],
        queries: 200,
        dim: 384,
        k: 10,
        nprobe: vec![Some(1), Some(4), Some(16), Some(64), None],
        json: None,
        index: true,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| {
            args.get(i + 1)
                .cloned()
                .unwrap_or_else(|| panic!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--sizes" => {
                o.sizes = value(i)
                    .split(',')
                    .map(|s| s.trim().parse().expect("size"))
                    .collect();
                i += 1;
            }
            "--regimes" => {
                o.regimes = value(i).split(',').map(|s| s.trim().to_string()).collect();
                i += 1;
            }
            "--queries" => {
                o.queries = value(i).parse().expect("queries");
                i += 1;
            }
            "--dim" => {
                o.dim = value(i).parse().expect("dim");
                i += 1;
            }
            "--k" => {
                o.k = value(i).parse().expect("k");
                i += 1;
            }
            "--nprobe" => {
                o.nprobe = value(i)
                    .split(',')
                    .map(|s| match s.trim() {
                        "auto" => None,
                        n => Some(n.parse().expect("nprobe")),
                    })
                    .collect();
                i += 1;
            }
            "--json" => {
                o.json = Some(value(i));
                i += 1;
            }
            "--no-index" => o.index = false,
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("ann_bench.rs")
                        .lines()
                        .take_while(|l| l.starts_with("//!"))
                        .map(|l| l.trim_start_matches("//!").trim_start())
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            other => panic!("unknown option {other}"),
        }
        i += 1;
    }
    o
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let at = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[at]
}

struct Timing {
    median_ms: f64,
    p95_ms: f64,
}

fn time_queries(
    queries: &[Vec<f32>],
    mut run: impl FnMut(&[f32]) -> Vec<SimilarityResult>,
) -> (Timing, Vec<Vec<SimilarityResult>>) {
    let mut times = Vec::with_capacity(queries.len());
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let r = run(q);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        out.push(r);
    }
    times.sort_by(f64::total_cmp);
    (
        Timing {
            median_ms: percentile(&times, 0.5),
            p95_ms: percentile(&times, 0.95),
        },
        out,
    )
}

fn recall(found: &[Vec<SimilarityResult>], truth: &[Vec<SimilarityResult>]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut worst: f64 = 1.0;
    for (f, t) in found.iter().zip(truth) {
        let want: HashSet<&NodeId> = t.iter().map(|r| &r.node_id).collect();
        let r = f.iter().filter(|x| want.contains(&x.node_id)).count() as f64
            / want.len().max(1) as f64;
        sum += r;
        worst = worst.min(r);
    }
    (sum / found.len().max(1) as f64, worst)
}

fn main() {
    let o = parse();
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "isongraph-vector-rs ann_bench: {} logical CPUs, dim {}, top_k {}, {} queries per cell, release={}",
        cpus,
        o.dim,
        o.k,
        o.queries,
        !cfg!(debug_assertions)
    );
    let mut rows = Vec::new();
    for regime_name in &o.regimes {
        for &n in &o.sizes {
            println!("\n## {regime_name}, N = {n}");
            let regime = Regime::new(regime_name, n, o.dim);

            let mut store = EmbeddingStore::new(None);
            let started = Instant::now();
            let chunk = 10_000;
            let mut added = 0;
            while added < n {
                let count = chunk.min(n - added);
                for (i, v) in regime
                    .sample(count, 1000 + (added / chunk) as u64)
                    .into_iter()
                    .enumerate()
                {
                    store
                        .add(NodeId::new("doc", (added + i).to_string()), "", Some(v))
                        .unwrap();
                }
                added += count;
            }
            let ingest = n as f64 / started.elapsed().as_secs_f64();
            println!("* ingest {:.0} vectors/s", ingest);

            let queries = regime.sample(o.queries, 7);
            // first query: builds the scan cache
            let t = Instant::now();
            store
                .similarity_search_vector_exact(&queries[0], o.k, None, 0.0)
                .unwrap();
            let first_ms = t.elapsed().as_secs_f64() * 1000.0;

            store.set_scan_threads(1);
            let (exact_1, truth) = time_queries(&queries, |q| {
                store
                    .similarity_search_vector_exact(q, o.k, None, 0.0)
                    .unwrap()
            });
            // exact latency by thread count, so the automatic choice can be checked
            let mut by_threads = Vec::new();
            for t in [2usize, 4, 8, 16, 32].into_iter().filter(|t| *t <= cpus) {
                store.set_scan_threads(t);
                let (timing, _) = time_queries(&queries[..queries.len().min(50)], |q| {
                    store
                        .similarity_search_vector_exact(q, o.k, None, 0.0)
                        .unwrap()
                });
                by_threads.push((t, timing.median_ms));
            }
            println!(
                "* exact median by thread count: 1 -> {:.2} ms, {}",
                exact_1.median_ms,
                by_threads
                    .iter()
                    .map(|(t, ms)| format!("{t} -> {ms:.2} ms"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            store.set_scan_threads(0);
            let (exact_n, _) = time_queries(&queries, |q| {
                store
                    .similarity_search_vector_exact(q, o.k, None, 0.0)
                    .unwrap()
            });
            println!(
                "* exact scan: first query {first_ms:.0} ms (builds the cache); warm median {:.2} ms / p95 {:.2} ms on 1 thread, {:.2} ms / {:.2} ms on auto threads",
                exact_1.median_ms, exact_1.p95_ms, exact_n.median_ms, exact_n.p95_ms
            );
            let mut row = serde_json::json!({
                "regime": regime_name, "n": n, "dim": o.dim, "k": o.k, "queries": o.queries, "cpus": cpus,
                "ingest_per_s": ingest, "first_query_ms": first_ms,
                "exact_1thread_median_ms": exact_1.median_ms, "exact_1thread_p95_ms": exact_1.p95_ms,
                "exact_auto_median_ms": exact_n.median_ms, "exact_auto_p95_ms": exact_n.p95_ms,
            });

            if o.index {
                store
                    .set_ann(Some(AnnConfig {
                        min_size: 0,
                        ..AnnConfig::default()
                    }))
                    .unwrap();
                let t = Instant::now();
                let stats = store.build_ann_index(false).unwrap();
                let build = t.elapsed().as_secs_f64();
                println!(
                    "* index build {build:.2} s for nlist={} (automatic nprobe={}); index {:.1} MB on top of {:.0} MB of cached vectors",
                    stats.nlist.unwrap_or(0),
                    stats.nprobe.unwrap_or(0),
                    stats.index_bytes as f64 / 1e6,
                    (n * o.dim * 4) as f64 / 1e6
                );
                row["index_build_s"] = build.into();
                row["nlist"] = stats.nlist.into();
                row["index_mb"] = (stats.index_bytes as f64 / 1e6).into();

                println!("\n| nprobe | % cells | recall@{} mean | worst | median ms | p95 ms | vs exact 1T | vs exact auto |", o.k);
                println!("|---|---|---|---|---|---|---|---|");
                let mut sweep = Vec::new();
                for nprobe in &o.nprobe {
                    store.set_ann_nprobe(*nprobe).unwrap();
                    let (ann, found) = time_queries(&queries, |q| {
                        store.similarity_search_vector(q, o.k, None, 0.0).unwrap()
                    });
                    let effective = store.ann_stats().nprobe.unwrap_or(0);
                    let nlist = stats.nlist.unwrap_or(1);
                    let (mean, worst) = recall(&found, &truth);
                    let label = match nprobe {
                        Some(p) => p.to_string(),
                        None => format!("{effective} (auto)"),
                    };
                    println!(
                        "| {label} | {:.1} | {mean:.3} | {worst:.2} | {:.3} | {:.3} | {:.1}x | {:.1}x |",
                        100.0 * effective as f64 / nlist as f64,
                        ann.median_ms,
                        ann.p95_ms,
                        exact_1.median_ms / ann.median_ms,
                        exact_n.median_ms / ann.median_ms
                    );
                    sweep.push(serde_json::json!({
                        "nprobe": effective, "recall_mean": mean, "recall_worst": worst,
                        "median_ms": ann.median_ms, "p95_ms": ann.p95_ms,
                    }));
                }
                row["sweep"] = sweep.into();
                store.set_ann(None).unwrap();
            }
            rows.push(row);
        }
    }
    if let Some(path) = o.json {
        std::fs::write(&path, serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        println!("\nwrote {path}");
    }
}
