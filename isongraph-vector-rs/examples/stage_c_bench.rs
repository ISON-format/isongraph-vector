//! Stage C harness: one process = one (dataset, n, mode, precision) cell of the Rust side of the
//! honest re-benchmark in `benchmarks/rust-stage-c/`. It reads the same raw float32 vectors, queries and
//! float64 ground truth as `benchmarks/competitive/` and writes the same JSON shape for sweep points
//! (`knob`, `recall`, `qps_1t`, `batch.qps`, ...), so one summariser reads Rust and Python results.
//!
//! ```text
//! stage_c_bench --ds-dir D:/.../prep/text384 --ds text384 --n 100000 --mode exact|ivf|hnsw \
//!     [--int8 RF] --exp core,int8,conc,writes,persist --queries 1000 --budget 3 --knobs 16,32,64 --out r.json
//! ```
//! Experiments: `core` (ingest, memory, build, first query, knob sweep: pinned 1-thread loop x3 runs and
//! a batch on every thread), `int8` (mode exact only: rerank factors 1,2,4,8, QPS and recall against the
//! f32 scan and the float64 truth), `conc` (N concurrent clients, one scan thread each, at the reference
//! knob), `writes` (insert / update / delete 10k on the built index, first query after, recall after;
//! ivf and hnsw), `persist` (JSON save/load against the binary snapshot, optional time-to-ready of ivf
//! and hnsw after a load).
//!
//! Single-thread loops pin the process to one P-core (`BENCH_PIN_CPU`, default logical CPU 8) exactly as
//! `competitive_bench` does. Bench-only: nothing in the library is touched.

use isongraph_vector_rs::{
    AnnConfig, EmbeddingStore, NodeId, ScanPrecision, SidecarId, SimilarityResult, SnapshotIdentity,
};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

// ---- process memory + pinning (Windows), no extra crates ---------------------------------------
#[repr(C)]
struct Pmc {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}
#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    fn SetProcessAffinityMask(h: isize, mask: usize) -> i32;
    fn SetPriorityClass(h: isize, class: u32) -> i32;
}

fn pinned<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(windows)]
    {
        let cpu: usize = std::env::var("BENCH_PIN_CPU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let all = usize::MAX
            >> (usize::BITS as usize
                - std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(32));
        // SAFETY: plain kernel32 calls on the current process.
        unsafe {
            SetPriorityClass(GetCurrentProcess(), 0x80); // HIGH_PRIORITY_CLASS
            SetProcessAffinityMask(GetCurrentProcess(), 1usize << cpu);
        }
        let r = f();
        unsafe { SetProcessAffinityMask(GetCurrentProcess(), all) };
        r
    }
    #[cfg(not(windows))]
    f()
}
fn mem() -> Value {
    #[cfg(windows)]
    {
        // SAFETY: Pmc matches PROCESS_MEMORY_COUNTERS and `cb` is its size.
        let p = unsafe {
            let mut p: Pmc = std::mem::zeroed();
            p.cb = std::mem::size_of::<Pmc>() as u32;
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb);
            p
        };
        json!({"rss_mb": p.working_set_size as f64 / 1048576.0, "peak_mb": p.peak_working_set_size as f64 / 1048576.0})
    }
    #[cfg(not(windows))]
    json!({"rss_mb": 0.0, "peak_mb": 0.0})
}
fn rss(v: &Value) -> f64 {
    v["rss_mb"].as_f64().unwrap_or(0.0)
}

// ---- data ---------------------------------------------------------------------------------------
fn read_f32(path: &str, start: usize, count: usize, d: usize) -> Vec<f32> {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    f.seek(SeekFrom::Start((start * d * 4) as u64)).unwrap();
    let mut buf = vec![0u8; count * d * 4];
    f.read_exact(&mut buf).unwrap();
    (0..buf.len() / 4)
        .map(|i| f32::from_le_bytes(buf[4 * i..4 * i + 4].try_into().unwrap()))
        .collect()
}
fn read_i32(path: &str) -> Vec<i32> {
    let mut b = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .read_to_end(&mut b)
        .unwrap();
    (0..b.len() / 4)
        .map(|i| i32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
        .collect()
}

struct Ctx {
    dir: String,
    d: usize,
    n: usize,
    k: usize,
    queries: Vec<Vec<f32>>,
}
impl Ctx {
    fn base(&self, start: usize, count: usize) -> Vec<Vec<f32>> {
        read_f32(&format!("{}/base.f32", self.dir), start, count, self.d)
            .chunks(self.d)
            .map(|c| c.to_vec())
            .collect()
    }
    fn gt(&self, kind: &str) -> Vec<Vec<i32>> {
        read_i32(&format!("{}/gt_{}{}.i32", self.dir, kind, self.n))
            .chunks(100)
            .map(|c| c.to_vec())
            .collect()
    }
}

// ---- stats --------------------------------------------------------------------------------------
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}
fn lat_stats(lat_s: &[f64]) -> Value {
    let mut ms: Vec<f64> = lat_s.iter().map(|x| x * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
    json!({"median_ms": pct(&ms, 0.5), "p95_ms": pct(&ms, 0.95), "p99_ms": pct(&ms, 0.99), "max_ms": ms.last().cloned().unwrap_or(0.0), "mean_ms": mean})
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    pct(&s, 0.5)
}
fn recall(found: &[Vec<usize>], truth: &[Vec<i32>], k: usize) -> f64 {
    let mut hit = 0usize;
    for (f, t) in found.iter().zip(truth) {
        let want: HashSet<i32> = t.iter().take(k).cloned().collect();
        hit += f
            .iter()
            .take(k)
            .filter(|x| want.contains(&(**x as i32)))
            .count();
    }
    hit as f64 / (k * found.len().max(1)) as f64
}
fn ids(r: &[SimilarityResult]) -> Vec<usize> {
    r.iter()
        .map(|x| x.node_id.id.parse::<usize>().unwrap())
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Path {
    Exact,
    Ann,
}
fn search(s: &EmbeddingStore, q: &[f32], k: usize, p: Path) -> Vec<SimilarityResult> {
    match p {
        Path::Exact => s.similarity_search_vector_exact(q, k, None, 0.0).unwrap(),
        Path::Ann => s.similarity_search_vector(q, k, None, 0.0).unwrap(),
    }
}
fn run_loop(s: &EmbeddingStore, qs: &[Vec<f32>], k: usize, p: Path) -> (Vec<f64>, Vec<Vec<usize>>) {
    let mut lat = Vec::with_capacity(qs.len());
    let mut out = Vec::with_capacity(qs.len());
    for q in qs {
        let t = Instant::now();
        let r = search(s, q, k, p);
        lat.push(t.elapsed().as_secs_f64());
        out.push(ids(&r));
    }
    (lat, out)
}

/// Batched call over all the pool's threads: one warm-up run, then 3 timed runs of the whole query set.
fn batch_point(s: &EmbeddingStore, c: &Ctx, truth: &[Vec<i32>], p: Path) -> Value {
    let run = || {
        let t = Instant::now();
        let r = match p {
            Path::Exact => s.similarity_search_vector_exact_batch(&c.queries, c.k, None, 0.0),
            Path::Ann => s.similarity_search_vector_batch(&c.queries, c.k, None, 0.0),
        }
        .unwrap();
        (c.queries.len() as f64 / t.elapsed().as_secs_f64(), r)
    };
    let _ = run();
    let mut qps = Vec::new();
    let mut found = Vec::new();
    for _ in 0..3 {
        let (q, r) = run();
        qps.push(q);
        found = r;
    }
    let found: Vec<Vec<usize>> = found.iter().map(|r| ids(r)).collect();
    json!({"nq": c.queries.len(), "threads": s.search_pool().threads(), "qps_batch": median(&qps), "qps_batch_runs": qps,
           "qps_batch_min": qps.iter().cloned().fold(f64::MAX, f64::min), "qps_batch_max": qps.iter().cloned().fold(0.0, f64::max),
           "recall": recall(&found, &truth[..found.len()], c.k)})
}

/// Warm up, size the query count to `budget` seconds, then 3 timed runs (one-query-at-a-time loop).
fn point(s: &EmbeddingStore, c: &Ctx, truth: &[Vec<i32>], p: Path, budget: f64) -> Value {
    let k = c.k;
    let nqall = c.queries.len();
    let w0 = Instant::now();
    for q in c.queries.iter().take(50) {
        let _ = search(s, q, k, p);
        if w0.elapsed().as_secs_f64() > 3.0 {
            break;
        }
    }
    let (probe, _) = run_loop(s, &c.queries[..5.min(nqall)], k, p);
    let est = median(&probe).max(1e-7);
    let nq = ((budget / est) as usize).clamp(30.min(nqall), nqall);
    let qs = &c.queries[..nq];
    let mut runs = Vec::new();
    let mut lats: Vec<f64> = Vec::new();
    let mut rec = 0.0;
    for r in 0..3 {
        let (lat, res) = run_loop(s, qs, k, p);
        if r == 0 {
            rec = recall(&res, &truth[..nq], k);
        }
        let total: f64 = lat.iter().sum();
        runs.push(nq as f64 / total);
        lats.extend(lat);
        if total > budget * 1.5 {
            break;
        }
    }
    let mut v = lat_stats(&lats);
    v["nq"] = json!(nq);
    v["recall"] = json!(rec);
    v["qps_1t"] = json!(median(&runs));
    v["qps_1t_min"] = json!(runs.iter().cloned().fold(f64::MAX, f64::min));
    v["qps_1t_max"] = json!(runs.iter().cloned().fold(0.0, f64::max));
    v["qps_1t_runs"] = json!(runs);
    v
}

fn ingest(s: &mut EmbeddingStore, c: &Ctx, n: usize) -> f64 {
    let mut secs = 0.0;
    let step = 20000;
    let mut at = 0;
    while at < n {
        let m = step.min(n - at);
        let rows = c.base(at, m);
        let t = Instant::now();
        for (i, v) in rows.into_iter().enumerate() {
            s.add(NodeId::new("doc", (at + i).to_string()), "", Some(v))
                .unwrap();
        }
        secs += t.elapsed().as_secs_f64();
        at += m;
    }
    secs
}

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].clone())
        .unwrap_or(default.to_string())
}
fn list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|x| x.parse().unwrap())
        .collect()
}

fn set_knob(store: &mut EmbeddingStore, mode: &str, kn: usize) {
    match mode {
        "ivf" => store.set_ann_nprobe(Some(kn)).unwrap(),
        "hnsw" => store.set_ann_ef_search(Some(kn)).unwrap(),
        _ => {}
    }
}
fn ann_config(mode: &str, m: usize, efc: usize) -> AnnConfig {
    let mut cfg = if mode == "hnsw" {
        AnnConfig::hnsw()
    } else {
        AnnConfig::default()
    };
    cfg.min_size = 0;
    if mode == "hnsw" {
        cfg.hnsw.m = m;
        cfg.hnsw.ef_construction = efc;
    }
    cfg
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = arg(&args, "--ds-dir", "");
    let ds = arg(&args, "--ds", "");
    let n: usize = arg(&args, "--n", "100000").parse().unwrap();
    let nq: usize = arg(&args, "--queries", "1000").parse().unwrap();
    let budget: f64 = arg(&args, "--budget", "3").parse().unwrap();
    let mode = arg(&args, "--mode", "exact");
    let int8: usize = arg(&args, "--int8", "0").parse().unwrap();
    let exps: Vec<String> = arg(&args, "--exp", "core")
        .split(',')
        .map(|s| s.to_string())
        .collect();
    let out_path = arg(&args, "--out", "");
    let label = arg(&args, "--label", "rust");
    let workdir = arg(&args, "--workdir", "D:/Goldmine/ISON/bench_data/work");
    let ready = arg(&args, "--ready", "");
    let m: usize = arg(&args, "--m", "16").parse().unwrap();
    let efc: usize = arg(&args, "--efc", "200").parse().unwrap();
    let default_knobs = match mode.as_str() {
        "ivf" => "16,32,64,128,256,512",
        "hnsw" => "24,48,64,96,128,192,256,400",
        _ => "",
    };
    let knobs = list(&arg(&args, "--knobs", default_knobs));
    let has = |e: &str| exps.iter().any(|x| x == e);

    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
            .unwrap();
    let d = meta["d"].as_u64().unwrap() as usize;
    let qall = read_f32(&format!("{dir}/queries.f32"), 0, nq, d);
    let c = Ctx {
        dir: dir.clone(),
        d,
        n,
        k: 10,
        queries: qall.chunks(d).map(|x| x.to_vec()).collect(),
    };
    let cpus = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    println!("{label}: ds={ds} n={n} d={d} mode={mode} int8={int8} queries={nq} cpus={cpus} exps={exps:?}");
    let mut out = json!({
        "system": label, "ds": ds, "n": n, "d": d, "k": c.k, "queries": nq, "all_threads": cpus, "mode": mode,
        "int8_rerank": int8, "exps": exps, "budget": budget,
        "version": "isongraph-vector-rs HEAD (cargo release, default flags)",
        "started": format!("{:?}", std::time::SystemTime::now()),
    });
    let mem_base = mem();
    out["mem_baseline"] = mem_base.clone();
    let write_out = |out: &Value| {
        if !out_path.is_empty() {
            std::fs::write(&out_path, serde_json::to_string_pretty(out).unwrap()).unwrap();
        }
    };

    // ---------------------------------------------------------------- persist (own flow)
    if has("persist") {
        persist(&c, n, &workdir, &ready, &mut out, &mode, m, efc);
        write_out(&out);
        return;
    }

    let mut store = EmbeddingStore::new(None);
    let ingest_s = ingest(&mut store, &c, n);
    let mem_after_ingest = mem();
    println!(
        "  ingest {ingest_s:.2}s  rss {:.0} MB",
        rss(&mem_after_ingest)
    );
    out["mem_after_ingest"] = mem_after_ingest.clone();
    let truth = c.gt("");
    let raw_mb = (n * d * 4) as f64 / 1048576.0;
    out["raw_vectors_mb"] = json!(raw_mb);

    // ---------------------------------------------------------------- int8 study (mode exact)
    // runs first on a pristine store; the f32 baseline is the same process, same pinning
    if has("int8") && mode == "exact" {
        store.set_scan_threads(1);
        let _ = store.similarity_search_vector_exact(&c.queries[0], c.k, None, 0.0);
        let mut study = json!({});
        let f32_pin = pinned(|| point(&store, &c, &truth, Path::Exact, budget));
        store.set_scan_threads(0);
        let f32_auto = point(&store, &c, &truth, Path::Exact, budget);
        let f32_batch = batch_point(&store, &c, &truth, Path::Exact);
        let mem_f32 = mem();
        println!(
            "  int8 study: f32 1T {:.1} qps, auto {:.1} qps, batch {:.0} qps",
            f32_pin["qps_1t"].as_f64().unwrap(),
            f32_auto["qps_1t"].as_f64().unwrap(),
            f32_batch["qps_batch"].as_f64().unwrap()
        );
        study["f32"] = json!({"one_thread": f32_pin, "auto_threads": f32_auto, "batch": f32_batch, "mem": mem_f32.clone()});
        let t = Instant::now();
        store.set_scan_precision(ScanPrecision::int8()).unwrap();
        study["quant_build_s"] = json!(t.elapsed().as_secs_f64());
        let mem_i8 = mem();
        study["mem_after_int8"] = mem_i8.clone();
        study["quant_bytes_mb"] = json!(store.quant_bytes() as f64 / 1048576.0);
        study["rss_delta_mb"] = json!(rss(&mem_i8) - rss(&mem_f32));
        // recall of int8 + rerank against the f32 scan, on the real queries (cap for time at 1M)
        let rq: Vec<Vec<f32>> = c
            .queries
            .iter()
            .take(if n >= 500_000 { 300 } else { 1000 })
            .cloned()
            .collect();
        store.set_scan_threads(0);
        let rep = store
            .int8_recall_report(&rq, c.k, None, 0.0, &[1, 2, 4, 8])
            .unwrap();
        let mut by = Vec::new();
        for r in &rep {
            println!(
                "  rerank x{}: recall vs f32 {:.4} (worst {:.2}) exact_scores={} applies={} int8 {:.3} ms f32 {:.3} ms",
                r.rerank_factor, r.recall, r.min_recall, r.scores_exact, r.applies, r.int8_ms, r.f32_ms
            );
            store
                .set_scan_precision(ScanPrecision::Int8 {
                    rerank_factor: r.rerank_factor,
                })
                .unwrap();
            store.set_scan_threads(1);
            let p1 = pinned(|| point(&store, &c, &truth, Path::Exact, budget));
            store.set_scan_threads(0);
            let pa = point(&store, &c, &truth, Path::Exact, budget);
            let pb = batch_point(&store, &c, &truth, Path::Exact);
            println!(
                "    1T {:.1} qps (recall vs float64 truth {:.4}), auto {:.1} qps, batch {:.0} qps",
                p1["qps_1t"].as_f64().unwrap(),
                p1["recall"].as_f64().unwrap(),
                pa["qps_1t"].as_f64().unwrap(),
                pb["qps_batch"].as_f64().unwrap()
            );
            by.push(json!({"rerank_factor": r.rerank_factor, "recall_vs_f32_scan": r.recall, "min_recall_vs_f32_scan": r.min_recall,
                "scores_exact": r.scores_exact, "applies": r.applies, "report_queries": r.queries, "report_int8_ms": r.int8_ms, "report_f32_ms": r.f32_ms,
                "one_thread": p1, "auto_threads": pa, "batch": pb}));
        }
        study["by_factor"] = json!(by);
        out["int8_study"] = study;
        store.set_scan_precision(ScanPrecision::F32).unwrap();
        write_out(&out);
    }

    // ---------------------------------------------------------------- core
    if has("core") {
        if int8 > 0 {
            let before = mem();
            let t = Instant::now();
            store
                .set_scan_precision(ScanPrecision::Int8 {
                    rerank_factor: int8,
                })
                .unwrap();
            out["quant_build_s"] = json!(t.elapsed().as_secs_f64());
            out["quant_bytes_mb"] = json!(store.quant_bytes() as f64 / 1048576.0);
            out["rss_delta_int8_mb"] = json!(rss(&mem()) - rss(&before));
        }
        store.set_scan_threads(1);
        let t = Instant::now();
        store
            .similarity_search_vector_exact(&c.queries[0], c.k, None, 0.0)
            .unwrap();
        let first_exact_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mem_after_first = mem();
        out["first_exact_query_ms"] = json!(first_exact_ms);
        out["mem_after_first_exact_query"] = mem_after_first.clone();
        out["build"] = json!({"ingest_s": ingest_s, "first_exact_query_ms": first_exact_ms});

        let mut pts = Vec::new();
        if mode == "exact" {
            let mut p1 = pinned(|| point(&store, &c, &truth, Path::Exact, budget));
            p1["knob"] = Value::Null;
            store.set_scan_threads(0);
            let pa = point(&store, &c, &truth, Path::Exact, budget);
            p1["auto_threads"] = json!({"qps": pa["qps_1t"], "median_ms": pa["median_ms"], "p99_ms": pa["p99_ms"], "runs": pa["qps_1t_runs"]});
            p1["batch"] = batch_point(&store, &c, &truth, Path::Exact);
            println!(
                "  exact{} 1T recall={:.4} qps={:.1} median={:.3}ms | auto {:.1} qps | batch {:.0} qps",
                if int8 > 0 { format!("+int8x{int8}") } else { String::new() },
                p1["recall"].as_f64().unwrap(),
                p1["qps_1t"].as_f64().unwrap(),
                p1["median_ms"].as_f64().unwrap(),
                pa["qps_1t"].as_f64().unwrap(),
                p1["batch"]["qps_batch"].as_f64().unwrap()
            );
            pts.push(p1);
            out["ref_knob"] = Value::Null;
        } else {
            store.set_scan_threads(0);
            store.set_ann(Some(ann_config(&mode, m, efc))).unwrap();
            let t = Instant::now();
            let stats = store.build_ann_index(false).unwrap();
            let idx_s = t.elapsed().as_secs_f64();
            let mem_after_build = mem();
            println!(
                "  {mode} build {idx_s:.2}s (stats {:?}) rss {:.0} MB (peak {:.0})",
                stats.build_seconds,
                rss(&mem_after_build),
                mem_after_build["peak_mb"].as_f64().unwrap()
            );
            out["build"] = json!({"ingest_s": ingest_s, "index_s": idx_s, "total_s": ingest_s + idx_s,
                "stats_build_seconds": stats.build_seconds});
            out["mem_after_build"] = mem_after_build;
            out["index_stats"] = json!({"nlist": stats.nlist, "trained_n": stats.trained_n, "index_bytes": stats.index_bytes,
                "max_level": stats.max_level, "graph_nodes": stats.graph_nodes, "ef_search_default": stats.ef_search});
            store.set_scan_threads(1);
            let t = Instant::now();
            let _ = store
                .similarity_search_vector(&c.queries[0], c.k, None, 0.0)
                .unwrap();
            out["build"]["first_query_ms"] = json!(t.elapsed().as_secs_f64() * 1000.0);
            let ms = store.memory_stats();
            out["memory_stats"] = json!({"matrix_bytes": ms.matrix_bytes, "metadata_bytes": ms.metadata_bytes, "index_bytes": ms.index_bytes,
                "total_bytes": ms.total_bytes(), "quant_bytes": store.quant_bytes()});
            let mut ref_knob: Option<usize> = None;
            for &kn in &knobs {
                set_knob(&mut store, &mode, kn);
                store.set_scan_threads(1);
                let mut p = pinned(|| point(&store, &c, &truth, Path::Ann, budget));
                store.set_scan_threads(0);
                p["knob"] = json!(kn);
                p["batch"] = batch_point(&store, &c, &truth, Path::Ann);
                println!(
                    "  {mode}{} knob={kn} recall={:.4} qps1t={:.0} median={:.3}ms batch={:.0}",
                    if int8 > 0 {
                        format!("+int8x{int8}")
                    } else {
                        String::new()
                    },
                    p["recall"].as_f64().unwrap(),
                    p["qps_1t"].as_f64().unwrap(),
                    p["median_ms"].as_f64().unwrap(),
                    p["batch"]["qps_batch"].as_f64().unwrap()
                );
                if ref_knob.is_none() && p["recall"].as_f64().unwrap() >= 0.95 {
                    ref_knob = Some(kn);
                    out["ref_recall_pre"] = p["recall"].clone();
                }
                pts.push(p);
            }
            if ref_knob.is_none() {
                ref_knob = knobs.last().cloned();
                out["ref_recall_pre"] = pts
                    .last()
                    .map(|p| p["recall"].clone())
                    .unwrap_or(Value::Null);
            }
            out["ref_knob"] = json!(ref_knob);
        }
        out["knob_name"] = json!(match mode.as_str() {
            "ivf" => "nprobe",
            "hnsw" => "ef_search",
            _ => "",
        });
        out["sweep"] = json!(pts);
        out["mem_after_sweep"] = mem();
        write_out(&out);

        let ref_knob = out["ref_knob"].as_u64().map(|x| x as usize);
        if let Some(rk) = ref_knob {
            set_knob(&mut store, &mode, rk);
        }

        // ------------------------------------------------------------ concurrent clients
        if has("conc") {
            let secs = 3.0;
            let mut res = Vec::new();
            store.set_scan_threads(1);
            for clients in [1usize, 2, 4, 8, 16] {
                let path = if mode == "exact" {
                    Path::Exact
                } else {
                    Path::Ann
                };
                // warm
                for q in c.queries.iter().take(20) {
                    let _ = search(&store, q, c.k, path);
                }
                let t0 = Instant::now();
                let done: usize = std::thread::scope(|sc| {
                    let hs: Vec<_> = (0..clients)
                        .map(|w| {
                            let st = &store;
                            let qs = &c.queries;
                            sc.spawn(move || {
                                let mut cnt = 0usize;
                                let mut i = w;
                                while t0.elapsed().as_secs_f64() < secs {
                                    let _ = search(st, &qs[i % qs.len()], 10, path);
                                    i += clients;
                                    cnt += 1;
                                }
                                cnt
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).sum()
                });
                let qps = done as f64 / t0.elapsed().as_secs_f64();
                println!("  concurrent clients={clients} (1 scan thread each): {qps:.1} qps");
                res.push(json!({"clients": clients, "qps": qps}));
            }
            out["concurrent_clients"] = json!(res);
            write_out(&out);
        }

        // ------------------------------------------------------------ writes
        if has("writes") && mode != "exact" {
            let gtm = format!("{dir}/gt_mut_{n}.i32");
            if std::path::Path::new(&gtm).exists() {
                let upd: Vec<i32> = read_i32(&format!("{dir}/mut_{n}_upd.i32"));
                let del: Vec<i32> = read_i32(&format!("{dir}/mut_{n}_del.i32"));
                let truth_m: Vec<Vec<i32>> = read_i32(&gtm)
                    .chunks(100)
                    .map(|c| c.to_vec())
                    .take(nq)
                    .collect();
                let mut w = json!({"ref_knob": ref_knob, "recall_before": out["ref_recall_pre"].clone(), "mode": mode});
                store.set_scan_threads(0);
                let lat_stats_of = |lat: &[f64]| lat_stats(lat);
                let ins = c.base(n, 10000);
                let mut lat = Vec::with_capacity(10000);
                let t = Instant::now();
                for (i, v) in ins.into_iter().enumerate() {
                    let t1 = Instant::now();
                    store
                        .add(NodeId::new("doc", (n + i).to_string()), "", Some(v))
                        .unwrap();
                    lat.push(t1.elapsed().as_secs_f64());
                }
                let dt = t.elapsed().as_secs_f64();
                w["insert_10k"] =
                    json!({"seconds": dt, "per_s": 10000.0 / dt, "lat": lat_stats_of(&lat)});
                let newv = c.base(n + 10000, 10000);
                lat.clear();
                let t = Instant::now();
                for (j, v) in newv.into_iter().enumerate() {
                    let t1 = Instant::now();
                    store
                        .add(NodeId::new("doc", upd[j].to_string()), "", Some(v))
                        .unwrap();
                    lat.push(t1.elapsed().as_secs_f64());
                }
                let dt = t.elapsed().as_secs_f64();
                w["update_10k"] =
                    json!({"seconds": dt, "per_s": 10000.0 / dt, "lat": lat_stats_of(&lat)});
                lat.clear();
                let t = Instant::now();
                for id in &del {
                    let t1 = Instant::now();
                    store.remove(&NodeId::new("doc", id.to_string()));
                    lat.push(t1.elapsed().as_secs_f64());
                }
                let dt = t.elapsed().as_secs_f64();
                w["delete_10k"] =
                    json!({"seconds": dt, "per_s": 10000.0 / dt, "lat": lat_stats_of(&lat)});
                store.set_scan_threads(1);
                let t = Instant::now();
                let _ = store
                    .similarity_search_vector(&c.queries[0], c.k, None, 0.0)
                    .unwrap();
                w["first_query_after_ms"] = json!(t.elapsed().as_secs_f64() * 1000.0);
                let p = pinned(|| point(&store, &c, &truth_m, Path::Ann, budget));
                w["recall_after"] = p["recall"].clone();
                w["qps_after"] = p["qps_1t"].clone();
                w["after"] = p;
                let st = store.ann_stats();
                w["index_stats_after"] = json!({"pending": st.pending, "churn": st.churn, "trained_n": st.trained_n,
                    "graph_nodes": st.graph_nodes, "graph_dead": st.graph_dead, "inserts": st.inserts});
                w["mem_after"] = mem();
                println!(
                    "  writes: ins {:.0}/s upd {:.0}/s del {:.0}/s first-query {:.1} ms recall_after={}",
                    w["insert_10k"]["per_s"].as_f64().unwrap(),
                    w["update_10k"]["per_s"].as_f64().unwrap(),
                    w["delete_10k"]["per_s"].as_f64().unwrap(),
                    w["first_query_after_ms"].as_f64().unwrap(),
                    w["recall_after"]
                );
                out["mutation"] = w;
            } else {
                out["mutation"] = json!({"skipped": "no spare rows for this n"});
            }
            write_out(&out);
        }
    }
    out["mem_end"] = mem();
    out["status"] = json!("ok");
    write_out(&out);
}

#[allow(clippy::too_many_arguments)]
fn persist(
    c: &Ctx,
    n: usize,
    workdir: &str,
    ready: &str,
    out: &mut Value,
    _mode: &str,
    m: usize,
    efc: usize,
) {
    std::fs::create_dir_all(workdir).unwrap();
    let json_path = format!("{workdir}/stagec_rust_{n}.embeddings.json");
    let snap_path = format!("{workdir}/stagec_rust_{n}.embeddings.snap");
    let mut store = EmbeddingStore::new(None);
    let ingest_s = ingest(&mut store, c, n);
    out["ingest_s"] = json!(ingest_s);
    out["mem_after_ingest"] = mem();
    let raw_mb = (n * c.d * 4) as f64 / 1048576.0;
    out["raw_vectors_mb"] = json!(raw_mb);
    let mut p = json!({});
    let t = Instant::now();
    store
        .save_embeddings_with(&json_path, Some("stagec"))
        .unwrap();
    p["json_save_s"] = json!(t.elapsed().as_secs_f64());
    p["json_mb"] = json!(std::fs::metadata(&json_path).unwrap().len() as f64 / 1048576.0);
    let t = Instant::now();
    let sc = SidecarId::of_file(&json_path).unwrap();
    p["sidecar_fingerprint_s"] = json!(t.elapsed().as_secs_f64());
    let t = Instant::now();
    let info = store
        .save_snapshot(
            &snap_path,
            &SnapshotIdentity {
                graph_digest: Some("stagec".to_string()),
                sidecar: Some(sc),
            },
        )
        .unwrap();
    p["snapshot_save_s"] = json!(t.elapsed().as_secs_f64());
    p["snapshot_mb"] = json!(info.bytes as f64 / 1048576.0);
    println!(
        "  save: JSON {:.2}s {:.0} MB | snapshot {:.2}s {:.0} MB (+ sidecar fingerprint {:.2}s)",
        p["json_save_s"].as_f64().unwrap(),
        p["json_mb"].as_f64().unwrap(),
        p["snapshot_save_s"].as_f64().unwrap(),
        p["snapshot_mb"].as_f64().unwrap(),
        p["sidecar_fingerprint_s"].as_f64().unwrap()
    );
    let q0 = c.queries[0].clone();
    let ref_ids: Vec<Vec<usize>> = c
        .queries
        .iter()
        .take(20)
        .map(|q| {
            ids(&store
                .similarity_search_vector_exact(q, 10, None, 0.0)
                .unwrap())
        })
        .collect();
    drop(store);
    let mut runs_json = Vec::new();
    let mut runs_snap = Vec::new();
    for rep in 0..3 {
        let mut a = EmbeddingStore::new(None);
        let t = Instant::now();
        a.load_embeddings(&json_path, true).unwrap();
        let jl = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = a
            .similarity_search_vector_exact(&q0, 10, None, 0.0)
            .unwrap();
        let jf = t.elapsed().as_secs_f64() * 1000.0;
        let agree_a: Vec<Vec<usize>> = c
            .queries
            .iter()
            .take(20)
            .map(|q| ids(&a.similarity_search_vector_exact(q, 10, None, 0.0).unwrap()))
            .collect();
        if rep == 0 {
            p["json_load_peak_mb"] = mem()["peak_mb"].clone();
        }
        drop(a);
        let mut b = EmbeddingStore::new(None);
        let t = Instant::now();
        b.load_snapshot(&snap_path).unwrap();
        let sl = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = b
            .similarity_search_vector_exact(&q0, 10, None, 0.0)
            .unwrap();
        let sf = t.elapsed().as_secs_f64() * 1000.0;
        let agree_b: Vec<Vec<usize>> = c
            .queries
            .iter()
            .take(20)
            .map(|q| ids(&b.similarity_search_vector_exact(q, 10, None, 0.0).unwrap()))
            .collect();
        println!("  load rep {rep}: JSON {jl:.2}s (first query {jf:.1} ms) | snapshot {sl:.2}s (first query {sf:.1} ms) | answers equal ref: json={} snap={}",
            agree_a == ref_ids, agree_b == ref_ids);
        runs_json.push(jl);
        runs_snap.push(sl);
        p["json_first_query_ms"] = json!(jf);
        p["snapshot_first_query_ms"] = json!(sf);
        p["json_answers_equal_original"] = json!(agree_a == ref_ids);
        p["snapshot_answers_equal_original"] = json!(agree_b == ref_ids);
        if rep == 2 && !ready.is_empty() {
            // time to ready: load (snapshot) + index build + first indexed query, per requested ANN mode
            let mut rd = json!({});
            for md in ready.split(',') {
                let mut s = EmbeddingStore::new(None);
                let t = Instant::now();
                s.load_snapshot(&snap_path).unwrap();
                let l = t.elapsed().as_secs_f64();
                s.set_ann(Some(ann_config(md, m, efc))).unwrap();
                let t = Instant::now();
                s.build_ann_index(false).unwrap();
                let bs = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let _ = s.similarity_search_vector(&q0, 10, None, 0.0).unwrap();
                let fq = t.elapsed().as_secs_f64();
                println!(
                    "  ready {md}: load {l:.2}s + build {bs:.2}s + first query {:.1} ms = {:.2}s",
                    fq * 1000.0,
                    l + bs + fq
                );
                rd[md] = json!({"load_s": l, "build_s": bs, "first_query_ms": fq * 1000.0, "ready_s": l + bs + fq});
            }
            p["time_to_ready_after_snapshot_load"] = rd;
        }
        drop(b);
    }
    p["json_load_s_runs"] = json!(runs_json);
    p["snapshot_load_s_runs"] = json!(runs_snap);
    p["json_load_s"] = json!(median(&runs_json));
    p["snapshot_load_s"] = json!(median(&runs_snap));
    out["persist"] = p;
    out["mem_end"] = mem();
    out["status"] = json!("ok");
    let _ = std::fs::remove_file(&json_path);
    let _ = std::fs::remove_file(&snap_path);
}
