//! Indicative measurements for the binary snapshot and the int8 scan.
//!
//! ```text
//! cargo run --release --example stage_b_bench -- snapshot 100000 <dir>
//! cargo run --release --example stage_b_bench -- quant 1000000
//! ```
//!
//! Timings are wall clock on whatever machine this runs on, with other work
//! going on: read them as indicative, not as a benchmark result.

use isongraph_vector_rs::{
    AnnConfig, EmbeddingStore, NodeId, ScanPrecision, SidecarId, SnapshotIdentity,
};
use std::path::PathBuf;
use std::time::Instant;

const DIM: usize = 384;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Clustered unit vectors (200 centres, noise about one unit norm).
fn vectors(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    let centers: Vec<Vec<f32>> = (0..200)
        .map(|_| {
            let mut c: Vec<f32> = (0..DIM).map(|_| rng.gauss()).collect();
            let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            c.iter_mut().for_each(|x| *x /= norm);
            c
        })
        .collect();
    let scale = 1.0 / (DIM as f32).sqrt();
    (0..n)
        .map(|_| {
            let c = &centers[(rng.next_u64() % 200) as usize];
            c.iter().map(|x| x + rng.gauss() * scale).collect()
        })
        .collect()
}

fn filled(n: usize) -> EmbeddingStore {
    let data = vectors(n, 1);
    let mut store = EmbeddingStore::new(None);
    for (i, v) in data.into_iter().enumerate() {
        store
            .add(
                NodeId::new("doc", i.to_string()),
                &format!("document {i}"),
                Some(v),
            )
            .unwrap();
    }
    store
}

/// Current and peak working set in MiB (0 where unknown).
fn rss_mib() -> (f64, f64) {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_working_set: usize,
            working_set: usize,
            rest: [usize; 6],
        }
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(h: isize, c: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters {
            cb: std::mem::size_of::<Counters>() as u32,
            page_faults: 0,
            peak_working_set: 0,
            working_set: 0,
            rest: [0; 6],
        };
        // SAFETY: `c` is a correctly sized PROCESS_MEMORY_COUNTERS and `cb` says so.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        if ok != 0 {
            return (
                c.working_set as f64 / 1048576.0,
                c.peak_working_set as f64 / 1048576.0,
            );
        }
        (0.0, 0.0)
    }
    #[cfg(not(windows))]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let kb = |key: &str| {
            status
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0)
                / 1024.0
        };
        (kb("VmRSS:"), kb("VmHWM:"))
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / 1048576.0
}

fn snapshot(n: usize, dir: PathBuf) {
    std::fs::create_dir_all(&dir).unwrap();
    println!("== snapshot: {n} x {DIM} ==");
    let t = Instant::now();
    let store = filled(n);
    println!("build store            {:8.2} s", t.elapsed().as_secs_f64());
    let json = dir.join("bench.embeddings.json");
    let snap = dir.join("bench.embeddings.snap");

    let t = Instant::now();
    store
        .save_embeddings_with(&json, Some("bench-digest"))
        .unwrap();
    let json_save = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let identity = SnapshotIdentity {
        graph_digest: Some("bench-digest".to_string()),
        sidecar: Some(SidecarId::of_file(&json).unwrap()),
    };
    let sidecar_id = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let info = store.save_snapshot(&snap, &identity).unwrap();
    let snap_save = t.elapsed().as_secs_f64();
    let json_bytes = std::fs::metadata(&json).unwrap().len();
    println!(
        "JSON     save {json_save:8.2} s   size {:9.1} MiB",
        mib(json_bytes)
    );
    println!(
        "snapshot save {snap_save:8.2} s   size {:9.1} MiB  ({:.1}x smaller; sidecar identity {:.3} s)",
        mib(info.bytes),
        json_bytes as f64 / info.bytes as f64,
        sidecar_id
    );
    let raw = (n * DIM * 4) as f64 / 1048576.0;
    println!("raw f32 matrix         {raw:9.1} MiB");
    drop(store);

    let mut a = EmbeddingStore::new(None);
    let t = Instant::now();
    a.load_embeddings(&json, true).unwrap();
    let json_load = t.elapsed().as_secs_f64();
    let mut b = EmbeddingStore::new(None);
    let t = Instant::now();
    b.load_snapshot(&snap).unwrap();
    let snap_load = t.elapsed().as_secs_f64();
    println!("JSON     load {json_load:8.2} s");
    println!(
        "snapshot load {snap_load:8.2} s   ({:.1}x faster)",
        json_load / snap_load
    );
    // same store?
    assert_eq!(a.count(), b.count());
    let q = vectors(5, 99);
    for v in &q {
        let (ra, rb) = (
            a.similarity_search_vector(v, 10, None, -1.0).unwrap(),
            b.similarity_search_vector(v, 10, None, -1.0).unwrap(),
        );
        assert_eq!(
            ra.iter()
                .map(|r| (r.node_id.id.clone(), r.score.to_bits()))
                .collect::<Vec<_>>(),
            rb.iter()
                .map(|r| (r.node_id.id.clone(), r.score.to_bits()))
                .collect::<Vec<_>>()
        );
    }
    println!("loads agree on count and on top-10 bits of 5 queries");
    let (now, peak) = rss_mib();
    println!("working set now {now:.0} MiB, peak {peak:.0} MiB");
    let _ = std::fs::remove_file(&json);
    let _ = std::fs::remove_file(&snap);
}

fn quant(n: usize, ivf_only: bool) {
    println!("== int8 scan: {n} x {DIM} ==");
    let mut store = filled(n);
    let queries = vectors(128, 4242);
    let (rss_f32, _) = rss_mib();
    println!("working set, f32 store only: {rss_f32:.0} MiB");

    // the fastest of 180 searches / of 5 batches: the machine is shared, and
    // the minimum is the least disturbed measurement (medians swing 3x here)
    let time = |store: &EmbeddingStore, _threads: usize, exact: bool| -> f64 {
        (0..3)
            .map(|_| {
                let mut ts = Vec::new();
                for q in queries.iter().cycle().take(60) {
                    let t = Instant::now();
                    if exact {
                        store
                            .similarity_search_vector_exact(q, 10, None, -1.0)
                            .unwrap();
                    } else {
                        store.similarity_search_vector(q, 10, None, -1.0).unwrap();
                    }
                    ts.push(t.elapsed().as_secs_f64() * 1000.0);
                }
                ts.into_iter().fold(f64::INFINITY, f64::min)
            })
            .fold(f64::INFINITY, f64::min)
    };
    let batch = |store: &EmbeddingStore| -> f64 {
        (0..5)
            .map(|_| {
                let t = Instant::now();
                store
                    .similarity_search_vector_batch(&queries, 10, None, -1.0)
                    .unwrap();
                t.elapsed().as_secs_f64() * 1000.0
            })
            .fold(f64::INFINITY, f64::min)
    };

    for threads in if ivf_only { vec![] } else { vec![1usize, 0] } {
        store.set_scan_threads(threads);
        let label = if threads == 1 {
            "1 thread "
        } else {
            "auto thr "
        };
        store.set_scan_precision(ScanPrecision::F32).unwrap();
        let f = time(&store, threads, true);
        let fb = batch(&store);
        store.set_scan_precision(ScanPrecision::int8()).unwrap();
        let i = time(&store, threads, true);
        let ib = batch(&store);
        println!(
            "exact scan, {label}: f32 {f:8.2} ms/query  int8(x4) {i:8.2} ms/query  ({:.2}x)   batch of 128: f32 {fb:8.1} ms  int8 {ib:8.1} ms  ({:.2}x)",
            f / i,
            fb / ib
        );
        store.set_scan_precision(ScanPrecision::F32).unwrap();
    }
    store.set_scan_threads(0);

    // recall and memory
    if !ivf_only {
        store.set_scan_precision(ScanPrecision::int8()).unwrap();
        let (rss_int8, peak) = rss_mib();
        println!(
        "working set with int8 copy: {rss_int8:.0} MiB (+{:.0} MiB; copy holds {:.0} MiB, f32 matrix {:.0} MiB); peak {peak:.0} MiB",
        rss_int8 - rss_f32,
        store.quant_bytes() as f64 / 1048576.0,
        store.memory_stats().matrix_bytes as f64 / 1048576.0
    );
        let report = store
            .int8_recall_report(&queries, 10, None, -1.0, &[1, 2, 4, 8, 16])
            .unwrap();
        for r in &report {
            println!(
            "  rerank x{:<2}: recall@10 {:.4} (worst {:.2}) exact scores {} | {:7.2} ms vs f32 {:7.2} ms",
            r.rerank_factor, r.recall, r.min_recall, r.scores_exact, r.int8_ms, r.f32_ms
        );
        }
    }

    // IVF
    store.set_scan_precision(ScanPrecision::F32).unwrap();
    store
        .set_ann(Some(AnnConfig {
            seed: 1,
            ..AnnConfig::default()
        }))
        .unwrap();
    let t = Instant::now();
    let stats = store.build_ann_index(false).unwrap();
    println!(
        "IVF build {:.1} s (nlist {:?}, nprobe {:?})",
        t.elapsed().as_secs_f64(),
        stats.nlist,
        stats.nprobe
    );
    for threads in [0usize, 1] {
        store.set_scan_threads(threads);
        store.set_scan_precision(ScanPrecision::F32).unwrap();
        let f = time(&store, 0, false);
        let fb = batch(&store);
        store.set_scan_precision(ScanPrecision::int8()).unwrap();
        let i = time(&store, 0, false);
        let ib = batch(&store);
        println!(
            "IVF search, threads {threads}: f32 {f:.3} ms/query  int8(x4) {i:.3} ms/query ({:.2}x)   batch of 128: f32 {fb:.1} ms  int8 {ib:.1} ms ({:.2}x)",
            f / i,
            fb / ib
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(100_000);
    match args.first().map(String::as_str) {
        Some("snapshot") => snapshot(
            n,
            args.get(2)
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("stage_b_bench")),
        ),
        Some("quant") => quant(n, args.get(2).is_some_and(|a| a == "ivf-only")),
        _ => eprintln!("usage: stage_b_bench snapshot|quant <rows> [dir]"),
    }
}
