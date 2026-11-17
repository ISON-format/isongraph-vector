//! Loader and measurement harness for `benchmarks/competitive/` (the cross-library comparison).
//!
//! It reads the same raw float32 vectors, queries and float64 ground truth that the Python harness
//! feeds the other systems, so every system sees identical data. Bench-only: nothing in the library
//! is touched.
//!
//! ```text
//! cargo run --release --example competitive_bench -- --ds-dir D:/.../prep/text384 --ds text384 --n 100000 \
//!     --exp core,mutation,saveload --queries 2000 --out result.json
//! cargo run --release --example competitive_bench -- ... --exp filter
//! ```
//! `core` measures the exact scan (1 thread, then automatic threads) and the opt-in IVF index
//! (nprobe sweep, 3 timed runs each); `mutation` inserts, updates and removes 10k vectors and
//! measures recall again; `saveload` times the JSON sidecar; `filter` restricts to a node type
//! holding a random 10% / 1% of the vectors.

use isongraph_vector_rs::{AnnConfig, EmbeddingStore, NodeId, SimilarityResult};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

// ---- process memory (Windows working set), no extra crates -------------------------------------
#[cfg(windows)]
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

/// Pin the process to one P-core (logical CPU 8 on the i9-14900HX; 0-15 are P-cores) for a single-threaded timed
/// loop, then release it. Unpinned single-thread loops varied up to 3x between processes on this hybrid CPU.
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
fn read_u8(path: &str) -> Vec<u8> {
    let mut b = Vec::new();
    File::open(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .read_to_end(&mut b)
        .unwrap();
    b
}

struct Ctx {
    dir: String,
    #[allow(dead_code)]
    ds: String,
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
    json!({"median_ms": pct(&ms, 0.5), "p95_ms": pct(&ms, 0.95), "p99_ms": pct(&ms, 0.99), "mean_ms": mean})
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
fn search(
    s: &EmbeddingStore,
    q: &[f32],
    k: usize,
    p: Path,
    ty: Option<&str>,
) -> Vec<SimilarityResult> {
    match p {
        Path::Exact => s.similarity_search_vector_exact(q, k, ty, 0.0).unwrap(),
        Path::Ann => s.similarity_search_vector(q, k, ty, 0.0).unwrap(),
    }
}
fn run_loop(
    s: &EmbeddingStore,
    qs: &[Vec<f32>],
    k: usize,
    p: Path,
    ty: Option<&str>,
) -> (Vec<f64>, Vec<Vec<usize>>) {
    let mut lat = Vec::with_capacity(qs.len());
    let mut out = Vec::with_capacity(qs.len());
    for q in qs {
        let t = Instant::now();
        let r = search(s, q, k, p, ty);
        lat.push(t.elapsed().as_secs_f64());
        out.push(ids(&r));
    }
    (lat, out)
}

/// The batched call over all the pool's threads: warm up once, then 3 timed runs of the whole query set.
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
    json!({"nq": c.queries.len(), "threads": s.search_pool().threads(), "qps": median(&qps), "qps_runs": qps, "recall": recall(&found, &truth[..found.len()], c.k)})
}

/// Warm up, size the query count to `budget` seconds, then 3 timed runs.
fn point(
    s: &EmbeddingStore,
    c: &Ctx,
    truth: &[Vec<i32>],
    p: Path,
    ty: Option<&str>,
    budget: f64,
) -> Value {
    let k = c.k;
    let nqall = c.queries.len();
    let w0 = Instant::now();
    for q in c.queries.iter().take(50) {
        let _ = search(s, q, k, p, ty);
        if w0.elapsed().as_secs_f64() > 3.0 {
            break;
        }
    }
    let (probe, _) = run_loop(s, &c.queries[..5.min(nqall)], k, p, ty);
    let est = median(&probe).max(1e-7);
    let nq = ((budget / est) as usize).clamp(30.min(nqall), nqall);
    let qs = &c.queries[..nq];
    let mut runs = Vec::new();
    let mut lats: Vec<f64> = Vec::new();
    let mut rec = 0.0;
    for r in 0..3 {
        let (lat, res) = run_loop(s, qs, k, p, ty);
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

fn nlist_for(n: usize) -> usize {
    ((4.0 * (n as f64).sqrt()).round() as usize).clamp(16, 65536)
}
fn knobs(n: usize) -> Vec<usize> {
    let nl = nlist_for(n);
    let mut v: Vec<usize> = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        .into_iter()
        .filter(|x| *x < nl)
        .collect();
    if n <= 200_000 {
        v.push(nl); // nprobe == nlist is the exact scan; measured separately at scale
    }
    v
}

fn ingest(s: &mut EmbeddingStore, c: &Ctx, n: usize, grp: Option<&[u8]>) -> f64 {
    let mut secs = 0.0;
    let step = 20000;
    let mut at = 0;
    while at < n {
        let m = step.min(n - at);
        let rows = c.base(at, m);
        let t = Instant::now();
        for (i, v) in rows.into_iter().enumerate() {
            let id = at + i;
            let ty = match grp {
                None => "doc",
                Some(g) => ["t10", "t01", "rest"][g[id] as usize],
            };
            s.add(NodeId::new(ty, id.to_string()), "", Some(v)).unwrap();
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = arg(&args, "--ds-dir", "");
    let ds = arg(&args, "--ds", "");
    let n: usize = arg(&args, "--n", "100000").parse().unwrap();
    let nq: usize = arg(&args, "--queries", "1000").parse().unwrap();
    let budget: f64 = arg(&args, "--budget", "15").parse().unwrap();
    let exps: Vec<String> = arg(&args, "--exp", "core")
        .split(',')
        .map(|s| s.to_string())
        .collect();
    let out_path = arg(&args, "--out", "");
    let label = arg(&args, "--label", "rust");
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
            .unwrap();
    let d = meta["d"].as_u64().unwrap() as usize;
    let qall = read_f32(&format!("{dir}/queries.f32"), 0, nq, d);
    let c = Ctx {
        dir: dir.clone(),
        ds: ds.clone(),
        d,
        n,
        k: 10,
        queries: qall.chunks(d).map(|x| x.to_vec()).collect(),
    };
    let cpus = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let mut runs: Vec<Value> = Vec::new();
    println!(
        "{label}: ds={ds} n={n} d={d} queries={nq} cpus={cpus} release={}",
        !cfg!(debug_assertions)
    );
    let mem_base = mem();

    if exps
        .iter()
        .any(|e| e == "core" || e == "mutation" || e == "saveload")
    {
        let mut store = EmbeddingStore::new(None);
        let ingest_s = ingest(&mut store, &c, n, None);
        let mem_after_ingest = mem();
        println!("  ingest {:.2}s", ingest_s);
        let truth = c.gt("");
        // ---- exact ----
        store.set_scan_threads(1);
        let t = Instant::now();
        store
            .similarity_search_vector_exact(&c.queries[0], c.k, None, 0.0)
            .unwrap();
        let first_ms = t.elapsed().as_secs_f64() * 1000.0;
        let mem_after_first = mem();
        let mut exact = json!({
            "system": format!("{label}_exact"), "ds": ds, "n": n, "d": d, "k": c.k, "queries": nq, "all_threads": cpus,
            "version": "isongraph-vector-rs (repo HEAD, cargo release)",
            "build": {"ingest_s": ingest_s, "index_s": 0.0, "total_s": ingest_s, "first_query_ms": first_ms, "ready_s": ingest_s + first_ms / 1000.0},
            "mem_baseline": mem_base, "mem_after_ingest": mem_after_ingest, "mem_after_build": mem_after_first, "mem_after_first_query": mem_after_first,
        });
        let mut p1 = pinned(|| point(&store, &c, &truth, Path::Exact, None, budget));
        p1["knob"] = Value::Null;
        println!(
            "  exact 1T recall={} qps={:.1} median={:.3}ms",
            p1["recall"],
            p1["qps_1t"].as_f64().unwrap(),
            p1["median_ms"].as_f64().unwrap()
        );
        exact["sweep"] = json!([p1]);
        store.set_scan_threads(0);
        let pa = point(&store, &c, &truth, Path::Exact, None, budget);
        println!(
            "  exact auto-threads qps={:.1} median={:.3}ms",
            pa["qps_1t"].as_f64().unwrap(),
            pa["median_ms"].as_f64().unwrap()
        );
        exact["loop_allthreads"] = json!({"nq": pa["nq"], "threads": "auto", "qps": pa["qps_1t"], "median_ms": pa["median_ms"], "p95_ms": pa["p95_ms"], "p99_ms": pa["p99_ms"], "mean_ms": pa["mean_ms"], "qps_runs": pa["qps_1t_runs"]});
        exact["batch_allthreads"] = batch_point(&store, &c, &truth, Path::Exact);
        println!(
            "  exact batch qps={:.1}",
            exact["batch_allthreads"]["qps"].as_f64().unwrap()
        );
        // concurrent clients, each running single-threaded scans against the same store: measured, not assumed
        store.set_scan_threads(1);
        let conc = 8usize.min(cpus);
        let qs = &c.queries[..c.queries.len().min(400)];
        let t = Instant::now();
        std::thread::scope(|sc| {
            for w in 0..conc {
                let st = &store;
                sc.spawn(move || {
                    for q in qs.iter().skip(w).step_by(conc) {
                        st.similarity_search_vector_exact(q, 10, None, 0.0).unwrap();
                    }
                });
            }
        });
        let conc_qps = qs.len() as f64 / t.elapsed().as_secs_f64();
        exact["concurrent_clients"] = json!({"clients": conc, "scan_threads": 1, "qps": conc_qps});
        println!(
            "  exact {conc} concurrent clients (scan threads 1): {:.1} qps",
            conc_qps
        );
        exact["mem_after_sweep"] = mem();
        runs.push(exact);

        // ---- ivf ----
        store.set_scan_threads(0);
        store
            .set_ann(Some(AnnConfig {
                min_size: 0,
                ..AnnConfig::default()
            }))
            .unwrap();
        let t = Instant::now();
        let stats = store.build_ann_index(false).unwrap();
        let idx_s = t.elapsed().as_secs_f64();
        let mem_after_build = mem();
        println!("  ivf build {:.2}s nlist={:?}", idx_s, stats.nlist);
        let mut ivf = json!({
            "system": format!("{label}_ivf"), "ds": ds, "n": n, "d": d, "k": c.k, "queries": nq, "all_threads": cpus,
            "version": "isongraph-vector-rs (repo HEAD, cargo release)", "knob_name": "nprobe",
            "build": {"ingest_s": ingest_s, "cache_s": first_ms / 1000.0, "index_s": idx_s, "total_s": ingest_s + first_ms / 1000.0 + idx_s,
                      "first_query_ms": 0.0, "ready_s": ingest_s + first_ms / 1000.0 + idx_s},
            "mem_baseline": mem_base, "mem_after_ingest": mem_after_ingest, "mem_after_build": mem_after_build,
            "index_stats": {"nlist": stats.nlist, "trained_n": stats.trained_n, "index_bytes": stats.index_bytes, "build_seconds": stats.build_seconds},
        });
        let mut pts = Vec::new();
        let mut ref_knob: Option<usize> = None;
        let mut ref_recall = Value::Null;
        for kn in knobs(n) {
            store.set_ann_nprobe(Some(kn)).unwrap();
            store.set_scan_threads(1);
            let mut p = pinned(|| point(&store, &c, &truth, Path::Ann, None, budget));
            // all-cores: the same loop with the library's automatic thread choice
            store.set_scan_threads(0);
            let pa = point(&store, &c, &truth, Path::Ann, None, budget.min(5.0));
            p["knob"] = json!(kn);
            p["batch_allthreads"] = batch_point(&store, &c, &truth, Path::Ann);
            p["allthreads"] =
                json!({"qps": pa["qps_1t"], "median_ms": pa["median_ms"], "recall": pa["recall"]});
            println!(
                "  ivf nprobe={kn} recall={:.4} qps1t={:.0} median={:.3}ms allthreads_qps={:.0} batch_qps={:.0}",
                p["recall"].as_f64().unwrap(),
                p["qps_1t"].as_f64().unwrap(),
                p["median_ms"].as_f64().unwrap(),
                pa["qps_1t"].as_f64().unwrap(),
                p["batch_allthreads"]["qps"].as_f64().unwrap()
            );
            if ref_knob.is_none() && p["recall"].as_f64().unwrap() >= 0.95 {
                ref_knob = Some(kn);
                ref_recall = p["recall"].clone();
            }
            pts.push(p);
        }
        if ref_knob.is_none() {
            ref_knob = knobs(n).last().cloned();
        }
        ivf["sweep"] = json!(pts);
        ivf["ref_knob"] = json!(ref_knob);
        ivf["ref_recall_pre"] = ref_recall.clone();
        ivf["mem_after_sweep"] = mem();

        // ---- mutation ----
        if exps.iter().any(|e| e == "mutation") {
            let gtm = format!("{dir}/gt_mut_{n}.i32");
            if std::path::Path::new(&gtm).exists() {
                let upd: Vec<i32> = read_i32(&format!("{dir}/mut_{n}_upd.i32"));
                let del: Vec<i32> = read_i32(&format!("{dir}/mut_{n}_del.i32"));
                let truth_m: Vec<Vec<i32>> = read_i32(&gtm)
                    .chunks(100)
                    .map(|c| c.to_vec())
                    .take(nq)
                    .collect();
                store.set_ann_nprobe(ref_knob).unwrap();
                store.set_scan_threads(0);
                let mut m = json!({"ref_knob": ref_knob, "recall_before": ref_recall});
                let ins = c.base(n, 10000);
                let t = Instant::now();
                for (i, v) in ins.into_iter().enumerate() {
                    store
                        .add(NodeId::new("doc", (n + i).to_string()), "", Some(v))
                        .unwrap();
                }
                let dt = t.elapsed().as_secs_f64();
                m["insert_10k"] = json!({"seconds": dt, "per_s": 10000.0 / dt});
                let newv = c.base(n + 10000, 10000);
                let t = Instant::now();
                for (j, v) in newv.into_iter().enumerate() {
                    store
                        .add(NodeId::new("doc", upd[j].to_string()), "", Some(v))
                        .unwrap();
                }
                let dt = t.elapsed().as_secs_f64();
                m["update_10k"] = json!({"seconds": dt, "per_s": 10000.0 / dt});
                let t = Instant::now();
                for id in &del {
                    store.remove(&NodeId::new("doc", id.to_string()));
                }
                let dt = t.elapsed().as_secs_f64();
                m["delete_10k"] = json!({"seconds": dt, "per_s": 10000.0 / dt});
                store.set_scan_threads(1);
                let t = Instant::now();
                let _ = store
                    .similarity_search_vector(&c.queries[0], c.k, None, 0.0)
                    .unwrap();
                m["first_query_after_ms"] = json!(t.elapsed().as_secs_f64() * 1000.0);
                let p = pinned(|| point(&store, &c, &truth_m, Path::Ann, None, budget));
                m["recall_after"] = p["recall"].clone();
                m["after"] = p;
                let st = store.ann_stats();
                m["index_stats_after"] = json!({"pending": st.pending, "churn": st.churn, "trained_n": st.trained_n, "builds": st.searches.approximate});
                m["mem_after"] = mem();
                println!(
                    "  mutation: ins {:.2}s upd {:.2}s del {:.2}s recall_after={}",
                    m["insert_10k"]["seconds"].as_f64().unwrap(),
                    m["update_10k"]["seconds"].as_f64().unwrap(),
                    m["delete_10k"]["seconds"].as_f64().unwrap(),
                    m["recall_after"]
                );
                ivf["mutation"] = m;
            } else {
                ivf["mutation"] = json!({"skipped": "dataset has no spare rows for this n"});
            }
        }

        // ---- save / load (JSON sidecar, the only persistence this port has for vectors) ----
        if exps.iter().any(|e| e == "saveload") {
            let path = format!("{dir}/../../work/rust_{ds}_{n}.json");
            let _ = std::fs::create_dir_all(format!("{dir}/../../work"));
            let mut sl = json!({});
            let t = Instant::now();
            match store.save_embeddings(&path) {
                Ok(_) => {
                    sl["save_s"] = json!(t.elapsed().as_secs_f64());
                    sl["disk_mb"] =
                        json!(std::fs::metadata(&path).unwrap().len() as f64 / 1048576.0);
                    drop(store);
                    let mut s2 = EmbeddingStore::new(None);
                    let t = Instant::now();
                    s2.load_embeddings(&path, false).unwrap();
                    sl["load_s"] = json!(t.elapsed().as_secs_f64());
                    s2.set_ann(Some(AnnConfig {
                        min_size: 0,
                        nprobe: ref_knob,
                        ..AnnConfig::default()
                    }))
                    .unwrap();
                    let t = Instant::now();
                    s2.similarity_search_vector(&c.queries[0], c.k, None, 0.0)
                        .unwrap();
                    sl["first_query_after_load_ms"] = json!(t.elapsed().as_secs_f64() * 1000.0);
                    sl["time_to_ready_s"] = json!(
                        sl["load_s"].as_f64().unwrap()
                            + sl["first_query_after_load_ms"].as_f64().unwrap() / 1000.0
                    );
                    sl["note"] = json!("load = JSON parse + insert; first query includes the scan-cache build and the IVF index build (the index is never persisted)");
                    let _ = std::fs::remove_file(&path);
                }
                Err(e) => sl["error"] = json!(format!("{e:?}")),
            }
            ivf["saveload"] = sl;
        }
        runs.push(ivf);
    }

    if exps.iter().any(|e| e == "filter") {
        let g = read_u8(&format!("{dir}/grp_{n}.u8"));
        let mut store = EmbeddingStore::new(None);
        let ingest_s = ingest(&mut store, &c, n, Some(&g));
        store.set_scan_threads(1);
        store
            .set_ann(Some(AnnConfig {
                min_size: 0,
                ..AnnConfig::default()
            }))
            .unwrap();
        let t = Instant::now();
        store.build_ann_index(false).unwrap();
        let idx_s = t.elapsed().as_secs_f64();
        let mut res = json!({"system": format!("{label}_ivf"), "ds": ds, "n": n, "d": d, "filter": {"build_with_metadata_s": ingest_s + idx_s, "mem": mem(), "subsets": {}}});
        let kn_all = knobs(n);
        let pick: Vec<usize> = [
            1usize,
            kn_all.len() / 3,
            kn_all.len() / 2,
            2 * kn_all.len() / 3,
            kn_all.len() - 1,
        ]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|i| kn_all[i])
        .collect();
        for (ty, lab) in [("t10", "f10"), ("t01", "f01")] {
            let truth = c.gt(&format!("{lab}_"));
            let mut pts = Vec::new();
            for kn in &pick {
                store.set_ann_nprobe(Some(*kn)).unwrap();
                let mut p = pinned(|| point(&store, &c, &truth, Path::Ann, Some(ty), budget));
                p["knob"] = json!(kn);
                println!(
                    "  filter {lab} nprobe={kn} recall={:.4} qps={:.0}",
                    p["recall"].as_f64().unwrap(),
                    p["qps_1t"].as_f64().unwrap()
                );
                pts.push(p);
            }
            let mut pe = pinned(|| point(&store, &c, &truth, Path::Exact, Some(ty), budget));
            pe["knob"] = json!("exact");
            pts.push(pe);
            res["filter"]["subsets"][lab] = json!(pts);
        }
        runs.push(res);
    }

    if !out_path.is_empty() {
        let doc = json!({"runs": runs, "status": "ok", "label": label});
        std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {out_path}");
    }
}
