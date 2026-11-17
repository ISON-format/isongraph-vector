//! The engine guarantees: batch search equals per-query search exactly,
//! readers run concurrently and never see a half-applied write, a write leaves
//! no backlog for the next search, and deletes stay deleted. DESIGN.md states
//! the guarantees; these tests hold the code to them.

use isongraph_vector_rs::{
    AnnConfig, EmbeddingStore, MockEncoder, NodeId, SearchPool, SharedEmbeddingStore,
    SimilarityResult, FOLD_BATCH,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// -- data ---------------------------------------------------------------------

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
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.unit() * 2.0 - 1.0).collect()
    }
}

fn id(i: usize) -> NodeId {
    let ty = ["alpha", "beta", "gamma"][i % 3];
    NodeId::new(ty, i.to_string())
}

fn key(r: &SimilarityResult) -> (String, String) {
    (r.node_id.node_type.clone(), r.node_id.id.clone())
}

fn keys(rs: &[SimilarityResult]) -> Vec<(String, String)> {
    rs.iter().map(key).collect()
}

fn scores_bits(rs: &[SimilarityResult]) -> Vec<u32> {
    rs.iter().map(|r| r.score.to_bits()).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

/// A store with `n` vectors in three node types, some identical (ties), some
/// updated and some deleted, so tombstones and rewritten rows are in play.
fn messy_store(n: usize, dim: usize, ann: Option<AnnConfig>, seed: u64) -> EmbeddingStore {
    let mut rng = Rng(seed);
    let mut store = EmbeddingStore::new(None);
    for i in 0..n {
        let v = if i % 17 == 0 && i > 0 {
            // an exact duplicate of an earlier row: equal scores for every query
            store.get_vector(&id(i - 1)).unwrap().to_vec()
        } else {
            rng.vector(dim)
        };
        store.add(id(i), &format!("text {i}"), Some(v)).unwrap();
    }
    for i in (0..n).step_by(11) {
        store
            .add(id(i), "rewritten", Some(rng.vector(dim)))
            .unwrap();
    }
    for i in (3..n).step_by(13) {
        store.remove(&id(i));
    }
    if let Some(cfg) = ann {
        store.set_ann(Some(cfg)).unwrap();
        store.build_ann_index(false).unwrap();
        // writes after the build leave rows pending and rows folded
        for i in n..n + 300 {
            store.add(id(i), "late", Some(rng.vector(dim))).unwrap();
        }
        for i in (0..n).step_by(29) {
            store.remove(&id(i));
        }
    }
    store.verify_invariants().unwrap();
    store
}

fn queries(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    let mut qs: Vec<Vec<f32>> = (0..count).map(|_| rng.vector(dim)).collect();
    qs[1] = vec![0.0; dim]; // a query with no direction
    qs[2] = qs[3].clone(); // two identical queries
    qs
}

// -- batch search equals per-query search --------------------------------------

#[test]
fn batch_answers_equal_single_answers_bit_for_bit() {
    let ann = AnnConfig {
        nlist: Some(24),
        nprobe: Some(3),
        min_size: 0,
        ..AnnConfig::default()
    };
    for dim in [8usize, 37, 100] {
        for use_ann in [false, true] {
            let mut store = messy_store(1500, dim, use_ann.then(|| ann.clone()), 7 + dim as u64);
            let qs = queries(37, dim, 99);
            for threads in [1usize, 0, 8] {
                store.set_scan_threads(threads);
                for exact in [false, true] {
                    for (k, ty, thr) in [
                        (1usize, None, -1.0f32),
                        (10, None, -1.0),
                        (10, Some("beta"), -1.0),
                        (25, None, 0.15),
                        (7, Some("nowhere"), -1.0),
                        (100_000, None, -1.0), // "everything"
                    ] {
                        let batch = if exact {
                            store.similarity_search_vector_exact_batch(&qs, k, ty, thr)
                        } else {
                            store.similarity_search_vector_batch(&qs, k, ty, thr)
                        }
                        .unwrap();
                        assert_eq!(batch.len(), qs.len());
                        for (i, q) in qs.iter().enumerate() {
                            let single = if exact {
                                store.similarity_search_vector_exact(q, k, ty, thr)
                            } else {
                                store.similarity_search_vector(q, k, ty, thr)
                            }
                            .unwrap();
                            let ctx = format!(
                                "dim {dim} ann {use_ann} threads {threads} exact {exact} k {k} type {ty:?} q {i}"
                            );
                            assert_eq!(keys(&batch[i]), keys(&single), "{ctx}");
                            assert_eq!(scores_bits(&batch[i]), scores_bits(&single), "{ctx}");
                            assert_eq!(
                                batch[i].iter().map(|r| &r.text).collect::<Vec<_>>(),
                                single.iter().map(|r| &r.text).collect::<Vec<_>>(),
                                "{ctx}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn text_batches_encode_together_and_match_single_searches() {
    let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(32))));
    for i in 0..400 {
        store
            .add(
                id(i),
                &format!("document number {i} about topic {}", i % 9),
                None,
            )
            .unwrap();
    }
    let texts = [
        "topic 3",
        "document number 77",
        "nothing like it",
        "topic 3",
    ];
    let batch = store
        .similarity_search_batch(&texts, 5, None, -1.0)
        .unwrap();
    assert_eq!(batch.len(), texts.len());
    for (t, got) in texts.iter().zip(&batch) {
        let single = store.similarity_search(t, 5, None, -1.0).unwrap();
        assert_eq!(keys(got), keys(&single));
        assert_eq!(scores_bits(got), scores_bits(&single));
    }
    assert!(store
        .similarity_search_batch(&[], 5, None, -1.0)
        .unwrap()
        .is_empty());
    let none = EmbeddingStore::new(None);
    assert!(none.similarity_search_batch(&["x"], 5, None, -1.0).is_err());
}

#[test]
fn batch_edge_cases() {
    let store = messy_store(200, 12, None, 5);
    let qs = queries(5, 12, 1);
    // zero results asked for, and queries of the wrong width
    let none = store
        .similarity_search_vector_batch(&qs, 0, None, -1.0)
        .unwrap();
    assert!(none.iter().all(|r| r.is_empty()));
    let bad = vec![vec![1.0f32; 3]];
    assert!(store
        .similarity_search_vector_batch(&bad, 5, None, -1.0)
        .is_err());
    let empty: Vec<Vec<f32>> = Vec::new();
    assert!(store
        .similarity_search_vector_batch(&empty, 5, None, -1.0)
        .unwrap()
        .is_empty());
    // an empty store answers every query with nothing
    let blank = EmbeddingStore::new(None);
    let got = blank.similarity_search_vector_batch(&[vec![1.0f32; 4]], 5, None, -1.0);
    assert!(got.unwrap()[0].is_empty());
    // more queries than one pass handles
    let many: Vec<Vec<f32>> = queries(4200, 12, 3);
    let big = store
        .similarity_search_vector_batch(&many, 3, None, -1.0)
        .unwrap();
    assert_eq!(big.len(), 4200);
    for i in [0usize, 1, 2049, 4095, 4096, 4199] {
        let single = store
            .similarity_search_vector(&many[i], 3, None, -1.0)
            .unwrap();
        assert_eq!(keys(&big[i]), keys(&single), "query {i}");
    }
}

#[test]
fn a_dedicated_pool_gives_the_same_answers() {
    let mut store = messy_store(1200, 24, None, 3);
    let qs = queries(20, 24, 4);
    let want = store
        .similarity_search_vector_batch(&qs, 10, None, -1.0)
        .unwrap();
    store.set_search_pool(Some(Arc::new(SearchPool::new(3))));
    assert_eq!(store.search_pool().threads(), 3);
    let got = store
        .similarity_search_vector_batch(&qs, 10, None, -1.0)
        .unwrap();
    for (a, b) in want.iter().zip(&got) {
        assert_eq!(keys(a), keys(b));
        assert_eq!(scores_bits(a), scores_bits(b));
    }
}

// -- readers run concurrently ---------------------------------------------------

/// Queries per second from `threads` callers of the single-query search.
fn throughput(store: &EmbeddingStore, qs: &[Vec<f32>], threads: usize, window: Duration) -> f64 {
    let stop = AtomicBool::new(false);
    let done = AtomicUsize::new(0);
    let started = Instant::now();
    std::thread::scope(|s| {
        for t in 0..threads {
            let (stop, done) = (&stop, &done);
            s.spawn(move || {
                let mut i = t * 31;
                let mut mine = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    store
                        .similarity_search_vector_exact(&qs[i % qs.len()], 10, None, -1.0)
                        .unwrap();
                    i += 1;
                    mine += 1;
                }
                done.fetch_add(mine, Ordering::Relaxed);
            });
        }
        std::thread::sleep(window);
        stop.store(true, Ordering::Relaxed);
    });
    done.load(Ordering::Relaxed) as f64 / started.elapsed().as_secs_f64()
}

#[test]
fn concurrent_searches_scale_instead_of_queueing_on_one_lock() {
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    if cpus < 4 {
        eprintln!("skipped: needs 4 CPUs, have {cpus}");
        return;
    }
    // small enough to sit in cache, so the scan is compute-bound and a lock
    // (not memory bandwidth) is the only thing that could stop it scaling
    let mut store = EmbeddingStore::new(None);
    let mut rng = Rng(21);
    for i in 0..6000 {
        store.add(id(i), "", Some(rng.vector(64))).unwrap();
    }
    store.set_scan_threads(1);
    let qs = queries(64, 64, 8);
    throughput(&store, &qs, 2, Duration::from_millis(100)); // warm up
    let mut best = 0.0f64;
    for _ in 0..4 {
        let one = throughput(&store, &qs, 1, Duration::from_millis(400));
        let four = throughput(&store, &qs, 4, Duration::from_millis(400));
        best = best.max(four / one);
        if best >= 2.0 {
            break;
        }
    }
    // serialised callers give 1.0x; independent ones approach 4x
    eprintln!("4 concurrent callers: {best:.2}x one caller");
    assert!(
        best >= 2.0,
        "4 concurrent callers ran at {best:.2}x one caller"
    );
}

// -- readers and a writer: no torn, duplicated or deleted results --------------

#[derive(Clone)]
enum Op {
    Put(usize, Vec<f32>),
    Remove(usize),
}

/// For each id, `(first state it holds this value in, value)`; `None` is a delete.
type History = HashMap<usize, Vec<(usize, Option<Vec<f32>>)>>;

/// The serial oracle: the state after each prefix of the write sequence.
struct Oracle {
    versions: History,
}

impl Oracle {
    fn new(initial: &[(usize, Vec<f32>)], ops: &[Op]) -> Self {
        let mut versions: History = HashMap::new();
        for (i, v) in initial {
            versions.entry(*i).or_default().push((0, Some(v.clone())));
        }
        for (n, op) in ops.iter().enumerate() {
            let state = n + 1;
            match op {
                Op::Put(i, v) => versions
                    .entry(*i)
                    .or_default()
                    .push((state, Some(v.clone()))),
                Op::Remove(i) => {
                    if versions
                        .get(i)
                        .and_then(|h| h.last())
                        .is_some_and(|(_, v)| v.is_some())
                    {
                        versions.entry(*i).or_default().push((state, None));
                    }
                }
            }
        }
        Self { versions }
    }

    fn value_at(&self, i: usize, state: usize) -> Option<&Vec<f32>> {
        let history = self.versions.get(&i)?;
        history
            .iter()
            .rev()
            .find(|(from, _)| *from <= state)
            .and_then(|(_, v)| v.as_ref())
    }

    /// The exact top-k ids of a query in `state`.
    fn top(&self, q: &[f32], k: usize, state: usize) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = self
            .versions
            .keys()
            .filter_map(|&i| self.value_at(i, state).map(|v| (i, cosine(q, v))))
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }
}

struct Seen {
    query: Vec<f32>,
    lo: usize,
    hi: usize,
    exact: bool,
    k: usize,
    result: Vec<(usize, f32)>,
}

fn raw_id(r: &SimilarityResult) -> usize {
    r.node_id.id.parse().unwrap()
}

fn run_stress(approximate: bool) {
    const DIM: usize = 16;
    const N0: usize = 400;
    const OPS: usize = 2500;
    let mut rng = Rng(if approximate { 5 } else { 6 });
    let initial: Vec<(usize, Vec<f32>)> = (0..N0).map(|i| (i, rng.vector(DIM))).collect();
    let mut ops = Vec::new();
    let mut live: Vec<usize> = (0..N0).collect();
    let mut next_id = N0;
    for _ in 0..OPS {
        match rng.below(10) {
            0..=3 => {
                // a new id
                ops.push(Op::Put(next_id, rng.vector(DIM)));
                live.push(next_id);
                next_id += 1;
            }
            4..=6 => {
                let i = live[rng.below(live.len())];
                ops.push(Op::Put(i, rng.vector(DIM)));
            }
            _ => {
                if live.len() > 60 {
                    let at = rng.below(live.len());
                    ops.push(Op::Remove(live.swap_remove(at)));
                }
            }
        }
    }
    let oracle = Oracle::new(&initial, &ops);

    let mut base = EmbeddingStore::new(None);
    for (i, v) in &initial {
        base.add(NodeId::new("n", i.to_string()), "", Some(v.clone()))
            .unwrap();
    }
    if approximate {
        base.set_ann(Some(AnnConfig {
            nlist: Some(12),
            nprobe: Some(4),
            min_size: 0,
            ..AnnConfig::default()
        }))
        .unwrap();
        base.build_ann_index(false).unwrap();
    }
    let shared = SharedEmbeddingStore::new(base);

    let started = AtomicUsize::new(0); // ops begun
    let finished = AtomicUsize::new(0); // ops completed
    let stop = AtomicBool::new(false);
    let mut seen: Vec<Seen> = Vec::new();

    std::thread::scope(|s| {
        let writer = s.spawn(|| {
            for (n, op) in ops.iter().enumerate() {
                started.store(n + 1, Ordering::SeqCst);
                match op {
                    Op::Put(i, v) => {
                        shared
                            .add(NodeId::new("n", i.to_string()), "", Some(v.clone()))
                            .unwrap();
                    }
                    Op::Remove(i) => {
                        shared.remove(&NodeId::new("n", i.to_string()));
                    }
                }
                finished.store(n + 1, Ordering::SeqCst);
                if n % 50 == 0 {
                    std::thread::yield_now();
                }
            }
            stop.store(true, Ordering::SeqCst);
        });
        let readers: Vec<_> = (0..3)
            .map(|r| {
                let (shared, started, finished, stop) = (&shared, &started, &finished, &stop);
                s.spawn(move || {
                    let mut rng = Rng(1000 + r as u64);
                    let mut mine = Vec::new();
                    while !stop.load(Ordering::SeqCst) && mine.len() < 700 {
                        let k = 1 + rng.below(12);
                        let batch = r == 2;
                        let exact = !approximate || rng.below(2) == 0;
                        let qs: Vec<Vec<f32>> = (0..if batch { 3 } else { 1 })
                            .map(|_| rng.vector(DIM))
                            .collect();
                        let lo = finished.load(Ordering::SeqCst);
                        let results: Vec<Vec<SimilarityResult>> = if batch {
                            let guard = shared.read();
                            if exact {
                                guard.similarity_search_vector_exact_batch(&qs, k, None, -2.0)
                            } else {
                                guard.similarity_search_vector_batch(&qs, k, None, -2.0)
                            }
                            .unwrap()
                        } else if exact {
                            vec![shared
                                .read()
                                .similarity_search_vector_exact(&qs[0], k, None, -2.0)
                                .unwrap()]
                        } else {
                            vec![shared
                                .similarity_search_vector(&qs[0], k, None, -2.0)
                                .unwrap()]
                        };
                        let hi = started.load(Ordering::SeqCst);
                        for (q, res) in qs.into_iter().zip(results) {
                            mine.push(Seen {
                                query: q,
                                lo,
                                hi,
                                exact,
                                k,
                                result: res.iter().map(|x| (raw_id(x), x.score)).collect(),
                            });
                        }
                    }
                    mine
                })
            })
            .collect();
        writer.join().unwrap();
        for r in readers {
            seen.extend(r.join().unwrap());
        }
    });

    assert!(
        seen.len() > 200,
        "readers only managed {} searches",
        seen.len()
    );
    let mut straddled = 0usize;
    for (n, obs) in seen.iter().enumerate() {
        let states = obs.lo..=obs.hi.min(ops.len());
        if obs.hi >= obs.lo + 2 {
            straddled += 1;
        }
        // no duplicates, best first, no more than asked for
        let ids: HashSet<usize> = obs.result.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids.len(), obs.result.len(), "search {n}: duplicate ids");
        assert!(obs.result.len() <= obs.k, "search {n}: too many results");
        assert!(
            obs.result.windows(2).all(|w| w[0].1 >= w[1].1),
            "search {n}: not sorted"
        );
        if obs.exact {
            // exactly the serial oracle's answer in *some* state the write
            // sequence passed through while the search ran
            let ok = states.clone().any(|state| {
                let want = oracle.top(&obs.query, obs.k, state);
                want.len() == obs.result.len()
                    && want
                        .iter()
                        .zip(&obs.result)
                        .all(|((wi, ws), (gi, gs))| wi == gi && (ws - *gs as f64).abs() < 1e-4)
            });
            assert!(
                ok,
                "search {n} (states {}..={}): not any oracle state's top-{}",
                obs.lo, obs.hi, obs.k
            );
        } else {
            // every hit is a live row with that exact value in some state of the window
            for (i, score) in &obs.result {
                let ok = states.clone().any(|state| {
                    oracle
                        .value_at(*i, state)
                        .is_some_and(|v| (cosine(&obs.query, v) - *score as f64).abs() < 1e-4)
                });
                assert!(
                    ok,
                    "search {n}: hit {i} (score {score}) matches no state in {}..={}",
                    obs.lo, obs.hi
                );
            }
        }
    }
    eprintln!(
        "stress approximate={approximate}: {} searches, {straddled} overlapped writes",
        seen.len()
    );

    // the end state is the oracle's end state, and the store is internally sound
    let guard = shared.read();
    guard.verify_invariants().unwrap();
    let final_state = ops.len();
    let mut want: Vec<usize> = oracle
        .versions
        .keys()
        .copied()
        .filter(|&i| oracle.value_at(i, final_state).is_some())
        .collect();
    want.sort_unstable();
    assert_eq!(guard.count(), want.len());
    let mut got: Vec<usize> = guard
        .similarity_search_vector_exact(&[1.0; DIM], usize::MAX, None, -2.0)
        .unwrap()
        .iter()
        .map(raw_id)
        .collect();
    got.sort_unstable();
    assert_eq!(got, want);
}

#[test]
fn exact_readers_and_a_writer_match_a_serial_oracle() {
    run_stress(false);
}

#[test]
fn indexed_readers_and_a_writer_never_see_torn_or_deleted_rows() {
    run_stress(true);
}

// -- no backlog after writes ----------------------------------------------------

#[test]
fn writes_leave_no_backlog_for_the_next_search() {
    let dim = 48;
    let mut rng = Rng(77);
    let mut store = EmbeddingStore::with_ann(
        None,
        AnnConfig {
            min_size: 0,
            ..AnnConfig::default()
        },
    )
    .unwrap();
    for i in 0..20_000 {
        store.add(id(i), "", Some(rng.vector(dim))).unwrap();
    }
    store.build_ann_index(false).unwrap();
    let q = rng.vector(dim);
    let steady = {
        let t = Instant::now();
        for _ in 0..20 {
            store.similarity_search_vector(&q, 10, None, -1.0).unwrap();
        }
        t.elapsed() / 20
    };

    // inserts, updates and deletes: more than the whole index's 2% policy
    let before = store.ann_stats();
    for i in 20_000..25_000 {
        store.add(id(i), "", Some(rng.vector(dim))).unwrap();
    }
    for i in (0..20_000).step_by(7) {
        store.add(id(i), "", Some(rng.vector(dim))).unwrap();
    }
    for i in (1..20_000).step_by(11) {
        store.remove(&id(i));
    }
    let after = store.ann_stats();
    assert!(
        after.pending < FOLD_BATCH,
        "{} rows are waiting to be folded after the writes",
        after.pending
    );
    assert!(after.folds > before.folds, "the writes folded nothing");
    assert_eq!(
        after.searches.builds, before.searches.builds,
        "writes retrained"
    );

    // the first search folds nothing, rebuilds nothing, and is not slow
    let folds = store.ann_stats().folds;
    let t = Instant::now();
    let first = store.similarity_search_vector(&q, 10, None, -1.0).unwrap();
    let first_took = t.elapsed();
    assert_eq!(first.len(), 10);
    let now = store.ann_stats();
    assert_eq!(now.folds, folds);
    assert_eq!(now.pending, after.pending);
    assert_eq!(now.searches.builds, before.searches.builds);
    // generous: an order of magnitude over a steady query, plus a fixed allowance
    assert!(
        first_took < steady * 20 + Duration::from_millis(100),
        "first search {first_took:?} against a steady {steady:?}"
    );
    store.verify_invariants().unwrap();
}

// -- deletes stay deleted ---------------------------------------------------------

#[test]
fn deleted_rows_never_come_back_through_any_path() {
    let dim = 10;
    for use_ann in [false, true] {
        let mut rng = Rng(if use_ann { 31 } else { 32 });
        let mut store = EmbeddingStore::new(None);
        if use_ann {
            store
                .set_ann(Some(AnnConfig {
                    nlist: Some(10),
                    nprobe: Some(4),
                    min_size: 0,
                    ..AnnConfig::default()
                }))
                .unwrap();
        }
        let mut model: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut ever_deleted: HashSet<usize> = HashSet::new();
        let mut next = 0usize;
        for step in 0..4000 {
            match rng.below(10) {
                0..=4 => {
                    let v = rng.vector(dim);
                    store.add(id(next), "", Some(v.clone())).unwrap();
                    model.insert(next, v);
                    next += 1;
                }
                5 | 6 if next > 0 => {
                    let i = rng.below(next);
                    let v = rng.vector(dim);
                    store.add(id(i), "", Some(v.clone())).unwrap();
                    model.insert(i, v);
                    ever_deleted.remove(&i);
                }
                _ if next > 0 => {
                    let i = rng.below(next);
                    let was = model.remove(&i).is_some();
                    assert_eq!(store.remove(&id(i)), was);
                    if was {
                        ever_deleted.insert(i);
                    }
                }
                _ => {}
            }
            if step % 97 == 0 {
                store.verify_invariants().unwrap();
                assert_eq!(store.count(), model.len());
                let q = rng.vector(dim);
                // the exact path returns precisely the live rows
                let all = store
                    .similarity_search_vector_exact(&q, usize::MAX, None, -2.0)
                    .unwrap();
                let got: HashSet<usize> = all.iter().map(raw_id).collect();
                let want: HashSet<usize> = model.keys().copied().collect();
                assert_eq!(got, want, "step {step}");
                // no path returns a row deleted since it was last written
                for res in [
                    store.similarity_search_vector(&q, 20, None, -2.0).unwrap(),
                    store
                        .similarity_search_vector_exact(&q, 20, None, -2.0)
                        .unwrap(),
                    store
                        .similarity_search_vector_batch(std::slice::from_ref(&q), 20, None, -2.0)
                        .unwrap()
                        .remove(0),
                ] {
                    for r in &res {
                        let i = raw_id(r);
                        assert!(
                            model.contains_key(&i),
                            "step {step}: deleted row {i} came back"
                        );
                        assert!(!ever_deleted.contains(&i));
                        // and it comes back with its current vector's score
                        assert!(
                            (cosine(&q, &model[&i]) - r.score as f64).abs() < 1e-4,
                            "step {step}: stale score for {i}"
                        );
                    }
                }
            }
        }
        assert!(store.memory_stats().dead < store.memory_stats().rows / 4 + 1);
    }
}

#[test]
fn compaction_keeps_answers_and_the_index() {
    let dim = 12;
    let mut rng = Rng(8);
    let mut store = EmbeddingStore::new(None);
    store
        .set_ann(Some(AnnConfig {
            nlist: Some(16),
            nprobe: Some(16),
            min_size: 0,
            ..AnnConfig::default()
        }))
        .unwrap();
    for i in 0..3000 {
        store.add(id(i), "", Some(rng.vector(dim))).unwrap();
    }
    store.build_ann_index(false).unwrap();
    let q = rng.vector(dim);
    for i in (0..3000).filter(|i| i % 3 != 0).take(1200) {
        store.remove(&id(i));
    }
    let before = store
        .similarity_search_vector_exact(&q, 15, None, -2.0)
        .unwrap();
    store.compact();
    store.verify_invariants().unwrap();
    assert_eq!(store.memory_stats().dead, 0);
    let after = store
        .similarity_search_vector_exact(&q, 15, None, -2.0)
        .unwrap();
    assert_eq!(keys(&before), keys(&after));
    assert_eq!(scores_bits(&before), scores_bits(&after));
    let indexed = store.similarity_search_vector(&q, 15, None, -2.0).unwrap();
    assert_eq!(keys(&indexed), keys(&after)); // nprobe covers every cell
}

// -- the memory account ---------------------------------------------------------

#[test]
fn memory_stats_account_for_one_copy_of_the_matrix() {
    let (n, dim) = (5000usize, 64usize);
    let store = messy_store(n, dim, None, 12);
    let stats = store.memory_stats();
    assert_eq!(stats.live, store.count());
    assert_eq!(stats.live_vector_bytes, store.count() * dim * 4);
    // the matrix is the vectors (dead rows and a chunk's tail included): one copy
    assert!(stats.matrix_bytes >= stats.rows * dim * 4);
    assert!((stats.matrix_bytes as f64) < (stats.rows * dim * 4) as f64 * 1.1 + 1e5);
    assert!(stats.chunks >= 1 && stats.dead <= stats.rows / 4 + 1);
    // a search builds nothing that holds vectors
    let before = store.memory_stats().total_bytes();
    store
        .similarity_search_vector(&[0.5; 64], 10, None, -1.0)
        .unwrap();
    assert_eq!(store.memory_stats().total_bytes(), before);
    // borrowed and copied vectors are the same row
    let someone = (0..n).find(|&i| store.get(&id(i)).is_some()).unwrap();
    assert_eq!(
        store.get(&id(someone)).unwrap().vector,
        store.get_vector(&id(someone)).unwrap()
    );
    assert!(store.get_vector(&NodeId::new("nobody", "1")).is_none());
    store.verify_invariants().unwrap();
}
