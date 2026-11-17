//! A/B harness for the engine work (concurrent and batch search, one copy of the
//! vectors, no stall after writes, deletes). Bench-only; reads the same raw
//! float32 data and float64 ground truth as `competitive_bench`.
//!
//! It uses only the API that exists both before and after the change, plus (with
//! `RUSTFLAGS="--cfg has_batch"`) the batch search, so the very same source is
//! built against the old commit and the new tree and run back to back.
//!
//! ```text
//! engine_ab --ds-dir D:/.../prep/text384 --n 100000 --exp scaling,batch --queries 2000
//! engine_ab --ds-dir D:/.../prep/text384 --n 1000000 --exp writes,deletes
//! ```
//! Experiments: `scaling` (N concurrent single-query callers), `batch` (batch
//! throughput by thread count, equality with per-query answers, recall),
//! `writes` (insert/update/delete throughput and write pauses on a built IVF
//! index, the first query after them), `deletes` (delete cost, by store state).
#![allow(unexpected_cfgs, dead_code)]

use isongraph_vector_rs::{AnnConfig, EmbeddingStore, NodeId, SimilarityResult};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

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

/// Pin to one P-core for a single-threaded timed loop (see README_method.md).
fn pinned<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(windows)]
    unsafe {
        let cpu: usize = std::env::var("BENCH_PIN_CPU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        SetPriorityClass(GetCurrentProcess(), 0x80);
        SetProcessAffinityMask(GetCurrentProcess(), 1usize << cpu);
        let r = f();
        SetProcessAffinityMask(
            GetCurrentProcess(),
            usize::MAX
                >> (usize::BITS as usize
                    - std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(32)),
        );
        return r;
    }
    #[allow(unreachable_code)]
    f()
}
fn rss_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut p: Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<Pmc>() as u32;
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb);
        return p.working_set_size as f64 / 1048576.0;
    }
    #[allow(unreachable_code)]
    0.0
}
fn peak_mb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut p: Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<Pmc>() as u32;
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb);
        return p.peak_working_set_size as f64 / 1048576.0;
    }
    #[allow(unreachable_code)]
    0.0
}

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

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].clone())
        .unwrap_or(default.to_string())
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        0.0
    } else {
        s[s.len() / 2]
    }
}
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() {
        0.0
    } else {
        s[((s.len() - 1) as f64 * p).round() as usize]
    }
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
    fn gt(&self) -> Vec<Vec<i32>> {
        read_i32(&format!("{}/gt_{}.i32", self.dir, self.n))
            .chunks(100)
            .map(|c| c.to_vec())
            .collect()
    }
}

fn ids(r: &[SimilarityResult]) -> Vec<usize> {
    r.iter()
        .map(|x| x.node_id.id.parse::<usize>().unwrap())
        .collect()
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

fn built(c: &Ctx, ivf: bool, nprobe: usize) -> EmbeddingStore {
    let mut s = EmbeddingStore::new(None);
    ingest(&mut s, c, c.n);
    if ivf {
        s.set_ann(Some(AnnConfig {
            min_size: 0,
            nprobe: Some(nprobe),
            ..AnnConfig::default()
        }))
        .unwrap();
        s.build_ann_index(false).unwrap();
    }
    s
}

/// Total QPS of `clients` threads each calling the single-query search in a loop for `secs`.
fn clients_qps(
    s: &EmbeddingStore,
    qs: &[Vec<f32>],
    k: usize,
    clients: usize,
    secs: f64,
    exact: bool,
) -> f64 {
    let stop = AtomicBool::new(false);
    let total = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|sc| {
        for w in 0..clients {
            let (stop, total) = (&stop, &total);
            sc.spawn(move || {
                let mut i = w * 7919;
                let mut done = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let q = &qs[i % qs.len()];
                    if exact {
                        s.similarity_search_vector_exact(q, k, None, 0.0).unwrap();
                    } else {
                        s.similarity_search_vector(q, k, None, 0.0).unwrap();
                    }
                    i += 1;
                    done += 1;
                }
                total.fetch_add(done, Ordering::Relaxed);
            });
        }
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
    });
    total.load(Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = arg(&args, "--ds-dir", "");
    let n: usize = arg(&args, "--n", "100000").parse().unwrap();
    let nq: usize = arg(&args, "--queries", "2000").parse().unwrap();
    let nprobe: usize = arg(&args, "--nprobe", "128").parse().unwrap();
    let label = arg(&args, "--label", "run");
    let secs: f64 = arg(&args, "--secs", "3").parse().unwrap();
    let exps: Vec<String> = arg(&args, "--exp", "scaling")
        .split(',')
        .map(|s| s.to_string())
        .collect();
    let out_path = arg(&args, "--out", "");
    let clients_list: Vec<usize> = arg(&args, "--clients", "1,2,4,8,16,24,32")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
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
    let mut out = json!({"label": label, "n": n, "d": d, "nprobe": nprobe});
    println!(
        "{label}: n={n} d={d} queries={nq} nprobe={nprobe} has_batch={}",
        cfg!(has_batch)
    );
    let rss0 = rss_mb();

    if exps.iter().any(|e| e == "scaling") {
        let mut res = json!({});
        for (mode, ivf) in [("exact", false), ("ivf", true)] {
            let mut s = built(&c, ivf, nprobe);
            s.set_scan_threads(1);
            let _ = clients_qps(&s, &c.queries, c.k, 1, 0.5, !ivf); // warm
            let mut rows = Vec::new();
            let mut base = 0.0;
            for &cl in &clients_list {
                let runs: Vec<f64> = (0..3)
                    .map(|_| clients_qps(&s, &c.queries, c.k, cl, secs, !ivf))
                    .collect();
                let q = median(&runs);
                if cl == 1 {
                    base = q;
                }
                println!(
                    "  scaling {mode} clients={cl:2} qps={q:9.1} efficiency={:.2}",
                    q / (base * cl as f64).max(1e-9)
                );
                rows.push(json!({"clients": cl, "qps": q, "runs": runs, "efficiency": q / (base * cl as f64).max(1e-9)}));
            }
            res[mode] = json!(rows);
        }
        out["scaling"] = res;
    }

    if exps.iter().any(|e| e == "mem") {
        // working set after each stage, minus the baseline before any data
        let base = rss_mb();
        let raw_mb = (n * d * 4) as f64 / 1048576.0;
        let mut s = EmbeddingStore::new(None);
        let ingest_s = ingest(&mut s, &c, n);
        let after_ingest = rss_mb() - base;
        let peak_ingest = peak_mb() - base;
        s.similarity_search_vector_exact(&c.queries[0], c.k, None, 0.0)
            .unwrap();
        let after_first = rss_mb() - base;
        s.set_ann(Some(AnnConfig {
            min_size: 0,
            nprobe: Some(nprobe),
            ..AnnConfig::default()
        }))
        .unwrap();
        s.build_ann_index(false).unwrap();
        let after_index = rss_mb() - base;
        let peak_index = peak_mb() - base;
        println!("  mem raw={raw_mb:.0} MB: after ingest {after_ingest:.0} ({:.2}x raw, peak {peak_ingest:.0}), after first query {after_first:.0}, after IVF build {after_index:.0} ({:.2}x raw, peak {peak_index:.0}); ingest {ingest_s:.1}s", after_ingest / raw_mb, after_index / raw_mb);
        out["mem"] = json!({"raw_mb": raw_mb, "after_ingest_mb": after_ingest, "peak_after_ingest_mb": peak_ingest, "after_first_query_mb": after_first, "after_ivf_build_mb": after_index, "peak_after_build_mb": peak_index, "ingest_s": ingest_s});
    }

    if exps.iter().any(|e| e == "loop") {
        // one-query-at-a-time loops by thread cap: latency of a single scan
        let mut s = built(&c, false, nprobe);
        let mut rows = Vec::new();
        for threads in [1usize, 2, 4, 8, 0] {
            s.set_scan_threads(threads);
            let run = |s: &EmbeddingStore| {
                let t = Instant::now();
                for q in c.queries.iter().take(150) {
                    s.similarity_search_vector_exact(q, c.k, None, 0.0).unwrap();
                }
                150.0 / t.elapsed().as_secs_f64()
            };
            let _ = run(&s);
            let runs: Vec<f64> = (0..5).map(|_| run(&s)).collect();
            println!(
                "  exact loop threads={threads} qps={:.1} (min {:.1} max {:.1})",
                median(&runs),
                runs.iter().cloned().fold(f64::MAX, f64::min),
                runs.iter().cloned().fold(0.0, f64::max)
            );
            rows.push(json!({"threads": threads, "qps": median(&runs), "runs": runs}));
        }
        out["loop"] = json!(rows);
    }

    if exps.iter().any(|e| e == "batch") {
        #[cfg(has_batch)]
        {
            let truth = c.gt();
            let mut res = json!({});
            for (mode, ivf) in [("exact", false), ("ivf", true)] {
                let mut s = built(&c, ivf, nprobe);
                let mut rows = Vec::new();
                let per_query: Vec<Vec<usize>> = c
                    .queries
                    .iter()
                    .take(200)
                    .map(|q| {
                        ids(&if ivf {
                            s.similarity_search_vector(q, c.k, None, 0.0)
                        } else {
                            s.similarity_search_vector_exact(q, c.k, None, 0.0)
                        }
                        .unwrap())
                    })
                    .collect();
                let pool = s.search_pool().threads();
                for threads in [1usize, 2, 4, 8, 16, 24, 32] {
                    if threads > pool {
                        continue;
                    }
                    s.set_scan_threads(threads);
                    let run = |s: &EmbeddingStore| {
                        let t = Instant::now();
                        let r = if ivf {
                            s.similarity_search_vector_batch(&c.queries, c.k, None, 0.0)
                        } else {
                            s.similarity_search_vector_exact_batch(&c.queries, c.k, None, 0.0)
                        }
                        .unwrap();
                        (t.elapsed().as_secs_f64(), r)
                    };
                    let _ = run(&s); // warm
                    let mut times = Vec::new();
                    let mut last = Vec::new();
                    for _ in 0..3 {
                        let (t, r) = run(&s);
                        times.push(c.queries.len() as f64 / t);
                        last = r;
                    }
                    let found: Vec<Vec<usize>> = last.iter().map(|r| ids(r)).collect();
                    let identical = found.iter().take(200).zip(&per_query).all(|(a, b)| a == b);
                    let rec = recall(&found, &truth, c.k);
                    let qps = median(&times);
                    println!("  batch {mode} threads={threads:2} qps={qps:9.1} recall={rec:.4} equals_single(first200)={identical}");
                    rows.push(json!({"threads": threads, "qps": qps, "runs": times, "recall": rec, "equals_single": identical}));
                }
                res[mode] = json!(rows);
            }
            out["batch"] = res;
        }
        #[cfg(not(has_batch))]
        println!("  batch: not available in this build");
    }

    if exps.iter().any(|e| e == "writes") {
        // an IVF store at ref nprobe; then 30k pure inserts, 10k updates, 10k deletes
        let rss_before = rss_mb();
        let mut s = built(&c, true, nprobe);
        let rss_built = rss_mb();
        let mut res = json!({"rss_built_mb": rss_built - rss_before, "peak_mb": peak_mb() - rss0});
        let probe = |s: &EmbeddingStore, label: &str| {
            let t = Instant::now();
            let _ = s
                .similarity_search_vector(&c.queries[0], c.k, None, 0.0)
                .unwrap();
            let first = t.elapsed().as_secs_f64() * 1000.0;
            let lat: Vec<f64> = c
                .queries
                .iter()
                .take(300)
                .map(|q| {
                    let t = Instant::now();
                    let _ = s.similarity_search_vector(q, c.k, None, 0.0).unwrap();
                    t.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            println!(
                "  {label}: first query {first:.2} ms, then median {:.3} ms p99 {:.3} ms",
                median(&lat),
                pct(&lat, 0.99)
            );
            json!({"first_ms": first, "median_ms": median(&lat), "p99_ms": pct(&lat, 0.99)})
        };
        res["before"] = probe(&s, "built");
        s.set_scan_threads(1);
        res["before_1t"] = probe(&s, "built (1 thread)");
        s.set_scan_threads(0);
        let upd: Vec<i32> = read_i32(&format!("{dir}/mut_{n}_upd.i32"));
        let del: Vec<i32> = read_i32(&format!("{dir}/mut_{n}_del.i32"));
        let batch = 30000usize;
        let ins = c.base(n, batch);
        let mut per_write = Vec::with_capacity(batch);
        let t = Instant::now();
        for (i, v) in ins.into_iter().enumerate() {
            let t1 = Instant::now();
            s.add(NodeId::new("doc", (n + i).to_string()), "", Some(v))
                .unwrap();
            per_write.push(t1.elapsed().as_secs_f64() * 1000.0);
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "  insert 30k: {:.2}s = {:.0}/s, per-write median {:.4} ms p99 {:.4} ms max {:.3} ms",
            dt,
            batch as f64 / dt,
            median(&per_write),
            pct(&per_write, 0.99),
            pct(&per_write, 1.0)
        );
        res["insert_30k"] = json!({"seconds": dt, "per_s": batch as f64 / dt, "median_ms": median(&per_write), "p99_ms": pct(&per_write, 0.99), "max_ms": pct(&per_write, 1.0)});
        s.set_scan_threads(1);
        res["after_30k_inserts"] = probe(&s, "after 30k inserts (1 thread)");
        s.set_scan_threads(0);
        let newv = c.base(n + 40000, 10000);
        let t = Instant::now();
        let mut pw = Vec::new();
        for (j, v) in newv.into_iter().enumerate() {
            let t1 = Instant::now();
            s.add(NodeId::new("doc", upd[j].to_string()), "", Some(v))
                .unwrap();
            pw.push(t1.elapsed().as_secs_f64() * 1000.0);
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "  update 10k: {:.0}/s max {:.3} ms",
            10000.0 / dt,
            pct(&pw, 1.0)
        );
        res["update_10k"] = json!({"per_s": 10000.0 / dt, "max_ms": pct(&pw, 1.0)});
        let t = Instant::now();
        for id in &del {
            s.remove(&NodeId::new("doc", id.to_string()));
        }
        let dt = t.elapsed().as_secs_f64();
        println!("  delete 10k: {:.0}/s", 10000.0 / dt);
        res["delete_10k"] = json!({"per_s": 10000.0 / dt});
        s.set_scan_threads(1);
        res["after_all"] = probe(&s, "after 30k writes total (1 thread)");
        res["rss_after_mb"] = json!(rss_mb() - rss0);
        out["writes"] = res;
    }

    if exps.iter().any(|e| e == "deletes") {
        let mut res = json!({});
        // random delete order over the whole store, so every delete is a cache miss
        let mut order: Vec<usize> = (0..n).collect();
        let mut x = 0x9E3779B97F4A7C15u64;
        for i in (1..order.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            order.swap(i, (x % (i as u64 + 1)) as usize);
        }
        for (label, search_first, ivf) in [
            ("never_searched", false, false),
            ("after_exact_search", true, false),
            ("ivf_built", true, true),
        ] {
            let mut s = built(&c, ivf, nprobe);
            if search_first {
                s.similarity_search_vector_exact(&c.queries[0], c.k, None, 0.0)
                    .unwrap();
            }
            let ndel = 10000usize.min(n / 4);
            let keys: Vec<NodeId> = order
                .iter()
                .take(ndel)
                .map(|i| NodeId::new("doc", i.to_string()))
                .collect();
            let t = Instant::now();
            for k in &keys {
                s.remove(k);
            }
            let dt = t.elapsed().as_secs_f64();
            println!(
                "  delete {ndel} ({label}): {:.0}/s = {:.0} ns each",
                ndel as f64 / dt,
                dt / ndel as f64 * 1e9
            );
            res[label] = json!({"per_s": ndel as f64 / dt, "ns_each": dt / ndel as f64 * 1e9});
        }
        out["deletes"] = res;
    }

    if !out_path.is_empty() {
        std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
        println!("wrote {out_path}");
    }
}
