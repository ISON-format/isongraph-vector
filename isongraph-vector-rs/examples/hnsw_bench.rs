//! The opt-in HNSW index: recall@10 versus queries/second across `ef_search`,
//! build time, memory, and the cost of one insert / update / delete.
//!
//! Data: with `--ds-dir DIR` the real vectors, queries and float64 ground truth
//! of `benchmarks/competitive/` (the `prep/text384` layout: `base.f32`,
//! `queries.f32`, `gt_<n>.i32`, `meta.json`); without it, synthetic clustered
//! unit vectors (ground truth from the exact scan).
//!
//! ```text
//! cargo run --release --example hnsw_bench -- --n 100000
//! cargo run --release --example hnsw_bench -- --ds-dir D:/.../prep/text384 --n 100000 --ivf
//! cargo run --release --example hnsw_bench -- --ds-dir D:/.../prep/text384 --n 1000000 --ef 16,32,64,128,256
//! ```
//!
//! Options: `--n N` `--dim D` (synthetic) `--queries Q` `--ef a,b,c` `--m M`
//! `--efc E` `--threads T` (build/batch pool cap, 0 = all) `--ivf` (also sweep
//! IVF `nprobe` on the same data; `--ivf-only` skips HNSW) `--budget SECONDS` (per timed loop) `--pin`
//! (single-thread loops on one pinned CPU, Windows).
//!
//! Timings are wall-clock on whatever the machine is doing; when other work
//! runs, read them as indicative. Recall and memory are not affected.

use isongraph_vector_rs::{AnnConfig, EmbeddingStore, HnswConfig, NodeId};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

// ---- process memory (Windows working set), no extra crates ------------------------------------
#[cfg(windows)]
mod winmem {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        rest: [usize; 6],
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
    /// (working set, peak working set) in MB.
    pub fn mb() -> (f64, f64) {
        // SAFETY: Pmc matches PROCESS_MEMORY_COUNTERS and `cb` is its size.
        let p = unsafe {
            let mut p: Pmc = std::mem::zeroed();
            p.cb = std::mem::size_of::<Pmc>() as u32;
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb);
            p
        };
        (
            p.working_set_size as f64 / 1048576.0,
            p.peak_working_set_size as f64 / 1048576.0,
        )
    }
}
#[cfg(not(windows))]
mod winmem {
    pub fn mb() -> (f64, f64) {
        (0.0, 0.0)
    }
}

/// With `--pin`, run `f` on one logical CPU (`BENCH_PIN_CPU`, default 8) at high
/// priority: single-thread loops vary a lot between runs on a hybrid CPU
/// otherwise. Windows only; elsewhere it just runs `f`.
fn pinned<T>(pin: bool, f: impl FnOnce() -> T) -> T {
    #[cfg(windows)]
    if pin {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn SetProcessAffinityMask(h: isize, mask: usize) -> i32;
            fn SetPriorityClass(h: isize, class: u32) -> i32;
        }
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
        return r;
    }
    let _ = pin;
    f()
}

// ---- data -------------------------------------------------------------------------------------
struct Rng(u64);

impl Rng {
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
    fn gauss(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}

fn unit(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
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

/// Where the vectors come from.
enum Source {
    Files { dir: String, d: usize },
    Synthetic { centers: Vec<Vec<f32>>, d: usize },
}

impl Source {
    fn dim(&self) -> usize {
        match self {
            Source::Files { d, .. } | Source::Synthetic { d, .. } => *d,
        }
    }

    /// Base vectors `start..start + count`.
    fn base(&self, start: usize, count: usize) -> Vec<Vec<f32>> {
        match self {
            Source::Files { dir, d } => read_f32(&format!("{dir}/base.f32"), start, count, *d)
                .chunks(*d)
                .map(|c| c.to_vec())
                .collect(),
            Source::Synthetic { centers, d } => (start..start + count)
                .map(|i| Self::synth(centers, *d, 0x1000 + i as u64))
                .collect(),
        }
    }

    fn queries(&self, count: usize) -> Vec<Vec<f32>> {
        match self {
            Source::Files { dir, d } => read_f32(&format!("{dir}/queries.f32"), 0, count, *d)
                .chunks(*d)
                .map(|c| c.to_vec())
                .collect(),
            Source::Synthetic { centers, d } => (0..count)
                .map(|i| Self::synth(centers, *d, 0xFFFF_0000 + i as u64))
                .collect(),
        }
    }

    fn synth(centers: &[Vec<f32>], d: usize, key: u64) -> Vec<f32> {
        let mut rng = Rng(key.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let c = &centers[rng.below(centers.len())];
        let scale = 1.0 / (d as f32).sqrt();
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.gauss() * scale).collect();
        unit(&mut v);
        v
    }
}

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn ingest(store: &mut EmbeddingStore, src: &Source, n: usize) -> f64 {
    let mut secs = 0.0;
    let step = 20_000;
    let mut at = 0;
    while at < n {
        let m = step.min(n - at);
        let rows = src.base(at, m);
        let t = Instant::now();
        for (i, v) in rows.into_iter().enumerate() {
            store
                .add(NodeId::new("doc", (at + i).to_string()), "", Some(v))
                .unwrap();
        }
        secs += t.elapsed().as_secs_f64();
        at += m;
    }
    secs
}

fn ids_of(store: &EmbeddingStore, q: &[f32], k: usize, ef: Option<usize>) -> Vec<usize> {
    let r = match ef {
        Some(ef) => store.similarity_search_vector_ef(q, k, None, 0.0, ef),
        None => store.similarity_search_vector(q, k, None, 0.0),
    }
    .unwrap();
    r.iter().map(|x| x.node_id.id.parse().unwrap()).collect()
}

fn recall(found: &[Vec<usize>], truth: &[Vec<usize>], k: usize) -> f64 {
    let mut hit = 0usize;
    for (f, t) in found.iter().zip(truth) {
        let want: HashSet<usize> = t.iter().take(k).copied().collect();
        hit += f.iter().take(k).filter(|x| want.contains(x)).count();
    }
    hit as f64 / (k * found.len().max(1)) as f64
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

struct Point {
    label: String,
    recall: f64,
    qps_1t: f64,
    median_ms: f64,
    p99_ms: f64,
    qps_batch: f64,
}

/// One accuracy setting: recall, single-thread loop (latency + QPS), batch QPS.
#[allow(clippy::too_many_arguments)]
fn measure(
    store: &mut EmbeddingStore,
    threads: usize,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    k: usize,
    ef: Option<usize>,
    budget: f64,
    pin: bool,
    label: String,
) -> Point {
    let found: Vec<Vec<usize>> = queries.iter().map(|q| ids_of(store, q, k, ef)).collect();
    let rec = recall(&found, truth, k);
    // single-threaded one-query-at-a-time loop, each query timed
    let mut lat = Vec::new();
    store.set_scan_threads(1);
    let loop_start = Instant::now();
    pinned(pin, || {
        'outer: for _ in 0..3 {
            for q in queries {
                let t = Instant::now();
                let _ = ids_of(store, q, k, ef);
                lat.push(t.elapsed().as_secs_f64());
                if loop_start.elapsed().as_secs_f64() > budget {
                    break 'outer;
                }
            }
        }
    });
    let total = loop_start.elapsed().as_secs_f64();
    store.set_scan_threads(threads);
    let mut ms: Vec<f64> = lat.iter().map(|x| x * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    // batch over all threads (HNSW has no per-call ef in a batch: it uses the configured one)
    let t = Instant::now();
    let mut done = 0usize;
    while t.elapsed().as_secs_f64() < budget.min(3.0) {
        let _ = store
            .similarity_search_vector_batch(queries, k, None, 0.0)
            .unwrap();
        done += queries.len();
    }
    let qps_batch = done as f64 / t.elapsed().as_secs_f64();
    Point {
        label,
        recall: rec,
        qps_1t: lat.len() as f64 / total,
        median_ms: pct(&ms, 0.5),
        p99_ms: pct(&ms, 0.99),
        qps_batch,
    }
}

fn print_header(title: &str) {
    println!("\n{title}");
    println!(
        "  {:<12} {:>8} {:>10} {:>10} {:>10} {:>12}",
        "setting", "recall", "qps(1T)", "median ms", "p99 ms", "qps(batch)"
    );
}

fn print_point(p: &Point) {
    println!(
        "  {:<12} {:>8.4} {:>10.0} {:>10.3} {:>10.3} {:>12.0}",
        p.label, p.recall, p.qps_1t, p.median_ms, p.p99_ms, p.qps_batch
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = arg(&args, "--n", "50000").parse().unwrap();
    let nq: usize = arg(&args, "--queries", "1000").parse().unwrap();
    let k = 10usize;
    let budget: f64 = arg(&args, "--budget", "6").parse().unwrap();
    let m: usize = arg(&args, "--m", "16").parse().unwrap();
    let efc: usize = arg(&args, "--efc", "200").parse().unwrap();
    let threads: usize = arg(&args, "--threads", "0").parse().unwrap();
    let efs: Vec<usize> = arg(&args, "--ef", "10,16,24,32,48,64,96,128,256")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let with_ivf = args.iter().any(|a| a == "--ivf");
    let pin = args.iter().any(|a| a == "--pin");
    let ivf_only = args.iter().any(|a| a == "--ivf-only");
    let dir = arg(&args, "--ds-dir", "");
    let cpus = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);

    let src = if dir.is_empty() {
        let d: usize = arg(&args, "--dim", "384").parse().unwrap();
        let mut rng = Rng(4242);
        let clusters = (n / 100).max(8);
        let centers = (0..clusters)
            .map(|_| {
                let mut c: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
                unit(&mut c);
                c
            })
            .collect();
        Source::Synthetic { centers, d }
    } else {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
                .unwrap();
        Source::Files {
            dir: dir.clone(),
            d: meta["d"].as_u64().unwrap() as usize,
        }
    };
    let d = src.dim();
    println!(
        "hnsw_bench: n={n} d={d} queries={nq} cpus={cpus} threads={threads} M={m} efC={efc} data={} release={}",
        if dir.is_empty() { "synthetic clustered" } else { &dir },
        !cfg!(debug_assertions)
    );
    println!("(timings are indicative: the machine is shared)");

    let mut store = EmbeddingStore::new(None);
    store.set_scan_threads(threads);
    let rss0 = winmem::mb();
    let ingest_s = ingest(&mut store, &src, n);
    let rss_ingest = winmem::mb();
    println!(
        "ingest {:.2}s ({:.0} vec/s); working set {:.0} MB (raw vectors {:.0} MB)",
        ingest_s,
        n as f64 / ingest_s,
        rss_ingest.0 - rss0.0,
        (n * d * 4) as f64 / 1048576.0
    );
    let queries = src.queries(nq);

    // ground truth: the files' float64 truth, or the exact scan
    let truth: Vec<Vec<usize>> = if dir.is_empty() {
        let t = Instant::now();
        let truth = queries
            .iter()
            .map(|q| {
                store
                    .similarity_search_vector_exact(q, 100, None, -2.0)
                    .unwrap()
                    .iter()
                    .map(|r| r.node_id.id.parse().unwrap())
                    .collect()
            })
            .collect();
        println!(
            "exact ground truth ({:.1}s, {:.1} ms/query)",
            t.elapsed().as_secs_f64(),
            t.elapsed().as_secs_f64() * 1000.0 / nq as f64
        );
        truth
    } else {
        read_i32(&format!("{dir}/gt_{n}.i32"))
            .chunks(100)
            .take(nq)
            .map(|c| c.iter().map(|&x| x as usize).collect())
            .collect()
    };

    // ---- HNSW ----
    if !ivf_only {
        store
            .set_ann(Some(AnnConfig {
                min_size: 0,
                hnsw: HnswConfig {
                    m,
                    ef_construction: efc,
                    ef_search: None,
                },
                ..AnnConfig::hnsw()
            }))
            .unwrap();
        let rss_before = winmem::mb();
        let t = Instant::now();
        let stats = store.build_ann_index(false).unwrap();
        let build_s = t.elapsed().as_secs_f64();
        let rss_after = winmem::mb();
        let mem = store.memory_stats();
        println!(
            "\nHNSW build {:.1}s ({:.0} nodes/s), max level {:?}",
            build_s,
            n as f64 / build_s,
            stats.max_level
        );
        println!(
        "  index {:.1} MB = {:.1} B/node = {:.3}x the raw vectors; whole store {:.3}x raw; working set +{:.0} MB, peak {:.0} MB",
        mem.index_bytes as f64 / 1048576.0,
        mem.index_bytes as f64 / n as f64,
        mem.index_bytes as f64 / mem.live_vector_bytes as f64,
        mem.total_bytes() as f64 / mem.live_vector_bytes as f64,
        rss_after.0 - rss_before.0,
        rss_after.1
    );

        print_header("HNSW: recall@10 vs speed over ef_search");
        let mut points = Vec::new();
        for &ef in &efs {
            // the batch path uses the configured ef, so set it for the batch too
            store.set_ann_ef_search(Some(ef)).unwrap();
            let p = measure(
                &mut store,
                threads,
                &queries,
                &truth,
                k,
                Some(ef),
                budget,
                pin,
                format!("ef={ef}"),
            );
            print_point(&p);
            points.push(p);
        }
        store.set_ann_ef_search(None).unwrap();
        let default = measure(
            &mut store,
            threads,
            &queries,
            &truth,
            k,
            None,
            budget,
            pin,
            "default".into(),
        );
        print_point(&default);
        if let Some(p) = points.iter().find(|p| p.recall >= 0.95) {
            println!(
            "  first setting at recall >= 0.95: {} ({:.4}), {:.0} qps single-thread, {:.0} qps batch",
            p.label, p.recall, p.qps_1t, p.qps_batch
        );
        }

        // ---- writes into the built graph ----
        let extra = 2000usize.min(nq.max(500));
        let more = src.base(n, extra);
        let mut ins = Vec::new();
        for (i, v) in more.iter().enumerate() {
            let t = Instant::now();
            store
                .add(NodeId::new("doc", (n + i).to_string()), "", Some(v.clone()))
                .unwrap();
            ins.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut upd = Vec::new();
        for (i, v) in more.iter().enumerate().take(extra / 2) {
            let t = Instant::now();
            store
                .add(NodeId::new("doc", i.to_string()), "", Some(v.clone()))
                .unwrap();
            upd.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut del = Vec::new();
        for i in 0..extra / 2 {
            let t = Instant::now();
            store.remove(&NodeId::new("doc", (n + i).to_string()));
            del.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        for v in [&mut ins, &mut upd, &mut del] {
            v.sort_by(f64::total_cmp);
        }
        println!(
        "\nwrites into the built graph (ms): insert median {:.3} p99 {:.3} max {:.3} | update median {:.3} p99 {:.3} | delete median {:.4} p99 {:.4}",
        pct(&ins, 0.5), pct(&ins, 0.99), pct(&ins, 1.0), pct(&upd, 0.5), pct(&upd, 0.99), pct(&del, 0.5), pct(&del, 0.99)
    );
        store.verify_invariants().unwrap();
    }

    // ---- IVF on the same data, for the same comparison ----
    if with_ivf || ivf_only {
        let mut store2 = EmbeddingStore::new(None);
        store2.set_scan_threads(threads);
        ingest(&mut store2, &src, n);
        store2
            .set_ann(Some(AnnConfig {
                min_size: 0,
                ..AnnConfig::default()
            }))
            .unwrap();
        let t = Instant::now();
        let st = store2.build_ann_index(false).unwrap();
        println!(
            "\nIVF build {:.1}s nlist={:?}; index {:.1} MB",
            t.elapsed().as_secs_f64(),
            st.nlist,
            store2.memory_stats().index_bytes as f64 / 1048576.0
        );
        print_header("IVF-flat: recall@10 vs speed over nprobe");
        let nlist = st.nlist.unwrap_or(16);
        for pct_cells in [0.5f64, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0] {
            let nprobe = ((nlist as f64 * pct_cells / 100.0).round() as usize).max(1);
            store2.set_ann_nprobe(Some(nprobe)).unwrap();
            let p = measure(
                &mut store2,
                threads,
                &queries,
                &truth,
                k,
                None,
                budget,
                pin,
                format!("nprobe={nprobe}"),
            );
            print_point(&p);
        }
    }
}
