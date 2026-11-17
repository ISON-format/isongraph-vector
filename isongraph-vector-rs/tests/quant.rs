//! The opt-in int8 scan with exact rerank: recall, exactness of returned
//! scores, batch == single bit for bit, consistency of the int8 copy under
//! writes (against a twin store and a model), concurrency, filters and
//! tombstones, and that nothing it keeps reaches a sidecar or digest.

#![allow(clippy::manual_is_multiple_of)]

use isongraph_vector_rs::{
    AnnConfig, EmbeddingStore, NodeId, ScanPrecision, SearchPool, SharedEmbeddingStore,
    SimilarityResult, DEFAULT_RERANK_FACTOR,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Clustered unit-ish vectors from a seeded generator (same recipe as the IVF
/// tests): 40 centres, noise of about one unit norm.
struct Clusters {
    centers: Vec<Vec<f32>>,
    dim: usize,
}

impl Clusters {
    fn new(dim: usize, clusters: usize, seed: u64) -> Self {
        let mut rng = Rng(seed);
        let centers = (0..clusters)
            .map(|_| {
                let mut c: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
                let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
                c.iter_mut().for_each(|x| *x /= norm);
                c
            })
            .collect();
        Self { centers, dim }
    }

    fn draw(&self, count: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
        let scale = 1.0 / (self.dim as f32).sqrt();
        (0..count)
            .map(|_| {
                let c = &self.centers[rng.below(self.centers.len())];
                c.iter().map(|x| x + rng.gauss() * scale).collect()
            })
            .collect()
    }
}

fn doc(ty: &str, i: usize) -> NodeId {
    NodeId::new(ty, i.to_string())
}

fn key(r: &SimilarityResult) -> String {
    format!("{}:{}", r.node_id.node_type, r.node_id.id)
}

fn keys(results: &[SimilarityResult]) -> Vec<String> {
    results.iter().map(key).collect()
}

fn bits(results: &[SimilarityResult]) -> Vec<(String, u32)> {
    results
        .iter()
        .map(|r| (key(r), r.score.to_bits()))
        .collect()
}

fn filled(data: &[Vec<f32>], type_of: impl Fn(usize) -> &'static str) -> EmbeddingStore {
    let mut store = EmbeddingStore::new(None);
    for (i, v) in data.iter().enumerate() {
        store
            .add(doc(type_of(i), i), "", Some(v.clone()))
            .expect("add");
    }
    store
}

fn int8(factor: usize) -> ScanPrecision {
    ScanPrecision::Int8 {
        rerank_factor: factor,
    }
}

#[test]
fn default_rerank_reaches_99_percent_recall_on_clustered_data() {
    let (dim, n) = (128, 12_000);
    let clusters = Clusters::new(dim, 40, 1);
    let mut rng = Rng(77);
    let data = clusters.draw(n, &mut rng);
    let queries = clusters.draw(200, &mut rng);
    let mut store = filled(&data, |_| "doc");
    store.set_scan_precision(ScanPrecision::int8()).unwrap();
    assert_eq!(
        store.scan_precision(),
        int8(DEFAULT_RERANK_FACTOR),
        "default factor"
    );
    let report = store
        .int8_recall_report(&queries, 10, None, -1.0, &[1, 2, DEFAULT_RERANK_FACTOR, 8])
        .unwrap();
    for r in &report {
        eprintln!(
            "factor {}: recall {:.4} (worst {:.2}) int8 {:.3} ms f32 {:.3} ms",
            r.rerank_factor, r.recall, r.min_recall, r.int8_ms, r.f32_ms
        );
        assert!(r.scores_exact, "factor {}", r.rerank_factor);
        assert!(r.applies);
    }
    let at_default = report
        .iter()
        .find(|r| r.rerank_factor == DEFAULT_RERANK_FACTOR)
        .unwrap();
    assert!(at_default.recall >= 0.99, "recall {}", at_default.recall);
    assert!(
        report[3].recall >= at_default.recall - 1e-12 && report[3].recall >= 0.995,
        "a larger factor does not lose recall"
    );
}

#[test]
fn returned_scores_are_the_exact_f32_cosines_and_ties_keep_slot_order() {
    let (dim, n) = (100, 3000);
    let clusters = Clusters::new(dim, 12, 3);
    let mut rng = Rng(5);
    let mut data = clusters.draw(n, &mut rng);
    // exact duplicates, to exercise the tie order
    for i in 0..40 {
        data[n - 1 - i] = data[i].clone();
    }
    let exact = filled(&data, |_| "doc");
    let mut quant = filled(&data, |_| "doc");
    quant.set_scan_precision(int8(4)).unwrap();
    for q in clusters.draw(50, &mut rng).iter().chain(&data[..10]) {
        let want = exact.score_map(q).unwrap();
        let got = quant.similarity_search_vector(q, 10, None, -1.0).unwrap();
        assert_eq!(got.len(), 10);
        for r in &got {
            assert_eq!(
                r.score.to_bits(),
                want[&key(r)].to_bits(),
                "{} is not the exact cosine",
                key(r)
            );
        }
        assert!(got.windows(2).all(|w| {
            w[0].score > w[1].score
                || (w[0].score == w[1].score
                    && w[0].node_id.id.parse::<usize>().unwrap()
                        < w[1].node_id.id.parse::<usize>().unwrap())
        }));
    }
    // a vector queried with itself: its duplicate ties with it, lower slot first
    let got = quant
        .similarity_search_vector(&data[3], 2, None, -1.0)
        .unwrap();
    assert_eq!(
        keys(&got),
        vec!["doc:3".to_string(), format!("doc:{}", n - 1 - 3)]
    );
    assert_eq!(got[0].score.to_bits(), got[1].score.to_bits());
}

#[test]
fn batch_equals_single_bit_for_bit_exact_and_ivf_and_across_threads() {
    let (dim, n) = (96, 6000);
    let clusters = Clusters::new(dim, 30, 9);
    let mut rng = Rng(11);
    let data = clusters.draw(n, &mut rng);
    let queries = clusters.draw(70, &mut rng);
    let ann = AnnConfig {
        nlist: Some(48),
        nprobe: Some(6),
        min_size: 0,
        ..AnnConfig::default()
    };
    for config in [None, Some(ann)] {
        let mut store = filled(&data, |i| match i {
            _ if i % 500 == 0 => "rare", // a dozen rows: probes come up short
            _ if i % 3 == 0 => "a",
            _ => "b",
        });
        if let Some(c) = config.clone() {
            store.set_ann(Some(c)).unwrap();
        }
        store.set_scan_precision(int8(4)).unwrap();
        for filter in [None, Some("a"), Some("rare")] {
            let singles: Vec<_> = queries
                .iter()
                .map(|q| store.similarity_search_vector(q, 10, filter, -1.0).unwrap())
                .collect();
            for threads in [1usize, 3, 0] {
                store.set_scan_threads(threads);
                if threads == 3 {
                    store.set_search_pool(Some(Arc::new(SearchPool::new(3))));
                }
                let batch = store
                    .similarity_search_vector_batch(&queries, 10, filter, -1.0)
                    .unwrap();
                for (i, (b, s)) in batch.iter().zip(&singles).enumerate() {
                    assert_eq!(bits(b), bits(s), "query {i}, threads {threads}, {config:?}");
                }
                // and one query is the same on any thread count
                let alone = store
                    .similarity_search_vector(&queries[0], 10, filter, -1.0)
                    .unwrap();
                assert_eq!(bits(&alone), bits(&singles[0]));
            }
            store.set_scan_threads(0);
            store.set_search_pool(None);
        }
    }
}

#[test]
fn ivf_with_int8_keeps_the_ivf_recall_and_exact_scores() {
    let (dim, n) = (96, 8000);
    let clusters = Clusters::new(dim, 40, 2);
    let mut rng = Rng(21);
    let data = clusters.draw(n, &mut rng);
    let queries = clusters.draw(100, &mut rng);
    let ann = AnnConfig {
        nlist: Some(64),
        nprobe: Some(8),
        min_size: 0,
        ..AnnConfig::default()
    };
    let mut plain = filled(&data, |_| "doc");
    plain.set_ann(Some(ann.clone())).unwrap();
    let mut quant = filled(&data, |_| "doc");
    quant.set_ann(Some(ann)).unwrap();
    quant.set_scan_precision(ScanPrecision::int8()).unwrap();
    let (mut same, mut total) = (0usize, 0usize);
    for q in &queries {
        let a = plain.similarity_search_vector(q, 10, None, -1.0).unwrap();
        let b = quant.similarity_search_vector(q, 10, None, -1.0).unwrap();
        let wanted: HashSet<String> = keys(&a).into_iter().collect();
        total += wanted.len();
        same += keys(&b).iter().filter(|k| wanted.contains(*k)).count();
        let scores = plain.score_map(q).unwrap();
        for r in &b {
            assert_eq!(r.score.to_bits(), scores[&key(r)].to_bits());
        }
    }
    let recall = same as f64 / total as f64;
    eprintln!("IVF f32 vs IVF int8 set overlap: {recall:.4}");
    assert!(recall >= 0.99, "{recall}");
    let stats = quant.ann_stats();
    assert!(stats.active && stats.searches.approximate >= queries.len() as u64);
}

#[test]
fn int8_copy_follows_writes_updates_deletes_and_compaction() {
    let (dim, n) = (50, 600); // 50 is not a multiple of 32: the scalar tail runs
    let clusters = Clusters::new(dim, 8, 4);
    let mut rng = Rng(8);
    let data = clusters.draw(n + 2000, &mut rng);
    let mut twin = EmbeddingStore::new(None);
    let mut store = EmbeddingStore::new(None);
    store.set_scan_precision(int8(8)).unwrap(); // enabled while empty
    let mut model: HashMap<usize, usize> = HashMap::new(); // id -> data index
    let mut next = 0usize;
    let (mut hit, mut want_total) = (0usize, 0usize);
    for step in 0..2500 {
        match rng.below(10) {
            0..=4 => {
                // insert
                if next < data.len() {
                    for s in [&mut twin, &mut store] {
                        s.add(doc("doc", next), "", Some(data[next].clone()))
                            .unwrap();
                    }
                    model.insert(next, next);
                    next += 1;
                }
            }
            5 | 6 => {
                // update an existing id with another vector
                if let Some(&id) = model.keys().next() {
                    let v = rng.below(data.len());
                    for s in [&mut twin, &mut store] {
                        s.add(doc("doc", id), "", Some(data[v].clone())).unwrap();
                    }
                    model.insert(id, v);
                }
            }
            7 | 8 => {
                // delete
                if let Some(&id) = model.keys().nth(rng.below(model.len().max(1))) {
                    for s in [&mut twin, &mut store] {
                        assert!(s.remove(&doc("doc", id)));
                    }
                    model.remove(&id);
                }
            }
            _ => {
                if step % 4 == 0 {
                    twin.compact();
                    store.compact();
                }
            }
        }
        if step % 25 == 0 {
            store.verify_invariants().unwrap();
            assert_eq!(store.count(), model.len());
            let q = &data[rng.below(data.len())];
            let want = twin.score_map(q).unwrap();
            let got = store.similarity_search_vector(q, 3, None, -1.0).unwrap();
            let live: HashSet<String> = model.keys().map(|i| format!("doc:{i}")).collect();
            for r in &got {
                assert!(
                    live.contains(&key(r)),
                    "{} was deleted or never stored",
                    key(r)
                );
                assert_eq!(
                    r.score.to_bits(),
                    want[&key(r)].to_bits(),
                    "stale or inexact score"
                );
            }
            let truth = twin
                .similarity_search_vector_exact(q, 3, None, -1.0)
                .unwrap();
            want_total += truth.len();
            let wanted: HashSet<String> = keys(&truth).into_iter().collect();
            hit += keys(&got).iter().filter(|k| wanted.contains(*k)).count();
            assert_eq!(got.len(), truth.len());
        }
    }
    store.verify_invariants().unwrap();
    let recall = hit as f64 / want_total as f64;
    eprintln!("recall against the twin during the churn: {recall:.4}");
    assert!(recall >= 0.95, "{recall}");
    // every surviving row: its int8 copy was verified above; its stored vector
    // is the model's
    for (&id, &v) in &model {
        assert_eq!(store.get_vector(&doc("doc", id)).unwrap(), &data[v][..]);
    }
}

#[test]
fn switching_precision_is_clean_and_costs_a_quarter_of_the_matrix() {
    let (dim, n) = (384, 3000);
    let clusters = Clusters::new(dim, 20, 6);
    let mut rng = Rng(2);
    let data = clusters.draw(n, &mut rng);
    let mut store = filled(&data, |_| "doc");
    assert_eq!(store.quant_bytes(), 0);
    let q = &data[7];
    let before = store.similarity_search_vector(q, 5, None, -1.0).unwrap();
    store.set_scan_precision(int8(4)).unwrap();
    store.verify_invariants().unwrap();
    let matrix = store.memory_stats().matrix_bytes as f64;
    let ratio = store.quant_bytes() as f64 / matrix;
    eprintln!("int8 copy / f32 matrix = {ratio:.4}");
    assert!((0.25..0.27).contains(&ratio), "{ratio}");
    let with = store.similarity_search_vector(q, 5, None, -1.0).unwrap();
    assert_eq!(
        bits(&before),
        bits(&with),
        "the query's own neighbours survive"
    );
    store.set_scan_precision(ScanPrecision::F32).unwrap();
    assert_eq!(store.quant_bytes(), 0);
    assert_eq!(store.scan_precision(), ScanPrecision::F32);
    let after = store.similarity_search_vector(q, 5, None, -1.0).unwrap();
    assert_eq!(bits(&before), bits(&after));
    assert!(store.set_scan_precision(int8(0)).is_err());
    // clearing and refilling with another width keeps the copy in step
    store.set_scan_precision(int8(2)).unwrap();
    store.clear();
    store.add(doc("doc", 1), "", Some(vec![1.0; 40])).unwrap();
    store.add(doc("doc", 2), "", Some(vec![-1.0; 40])).unwrap();
    store.verify_invariants().unwrap();
}

#[test]
fn filters_thresholds_and_tombstones() {
    let (dim, n) = (64, 4000);
    let clusters = Clusters::new(dim, 16, 12);
    let mut rng = Rng(31);
    let data = clusters.draw(n, &mut rng);
    let kind = |i: usize| match i % 4 {
        0 => "alpha",
        1 => "beta",
        _ => "gamma",
    };
    let mut twin = filled(&data, kind);
    let mut store = filled(&data, kind);
    store.set_scan_precision(int8(6)).unwrap();
    // tombstone a third of the rows, but stay under the compaction threshold
    // for part of the test
    let mut dead = HashSet::new();
    for i in (0..n).step_by(5) {
        let id = doc(kind(i), i);
        twin.remove(&id);
        store.remove(&id);
        dead.insert(format!("{}:{}", kind(i), i));
    }
    store.verify_invariants().unwrap();
    for ty in [None, Some("alpha"), Some("gamma"), Some("missing")] {
        for threshold in [-1.0f32, 0.3, 0.6] {
            for q in clusters.draw(15, &mut rng) {
                let got = store
                    .similarity_search_vector(&q, 8, ty, threshold)
                    .unwrap();
                let want = twin
                    .similarity_search_vector_exact(&q, 8, ty, threshold)
                    .unwrap();
                let scores = twin.score_map(&q).unwrap();
                for r in &got {
                    assert!(!dead.contains(&key(r)));
                    assert!(ty.is_none_or(|t| r.node_id.node_type == t));
                    assert!(r.score >= threshold);
                    assert_eq!(r.score.to_bits(), scores[&key(r)].to_bits());
                }
                if ty == Some("missing") {
                    assert!(got.is_empty());
                }
                // the same rows when the threshold leaves few candidates, and a
                // large overlap otherwise
                let wanted: HashSet<String> = keys(&want).into_iter().collect();
                let hit = keys(&got).iter().filter(|k| wanted.contains(*k)).count();
                assert!(
                    hit as f64 >= 0.75 * wanted.len() as f64,
                    "{ty:?} thr {threshold}: {hit}/{}",
                    wanted.len()
                );
                if want.len() < 8 {
                    assert_eq!(keys(&got), keys(&want), "short answers are exact");
                }
            }
        }
    }
}

#[test]
fn odd_inputs_fall_back_or_work() {
    let mut store = EmbeddingStore::new(None);
    store.set_scan_precision(int8(4)).unwrap();
    for i in 0..50 {
        let mut v = vec![0.0f32; 7];
        v[i % 7] = 1.0 + (i / 7) as f32;
        store.add(doc("doc", i), "", Some(v)).unwrap();
    }
    store.add(doc("zero", 0), "", Some(vec![0.0; 7])).unwrap();
    store.verify_invariants().unwrap();
    // a zero query has no direction: every score is 0, as in the f32 scan
    let z = store
        .similarity_search_vector(&[0.0; 7], 3, None, -1.0)
        .unwrap();
    assert_eq!(z.len(), 3);
    assert!(z.iter().all(|r| r.score == 0.0));
    // k >= live: the f32 path, everything comes back
    let all = store
        .similarity_search_vector(&[1.0; 7], usize::MAX, None, -1.0)
        .unwrap();
    assert_eq!(all.len(), 51);
    let some = store.similarity_search_vector(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, None, -1.0);
    assert_eq!(some.unwrap().len(), 5);
}

#[test]
fn concurrent_readers_and_a_writer_stay_consistent() {
    let (dim, n) = (64, 3000);
    let clusters = Clusters::new(dim, 16, 14);
    let mut rng = Rng(3);
    let data = Arc::new(clusters.draw(n + 4000, &mut rng));
    let mut base = filled(&data[..n], |_| "doc");
    base.set_scan_precision(int8(4)).unwrap();
    base.set_ann(Some(AnnConfig {
        nlist: Some(32),
        nprobe: Some(4),
        min_size: 0,
        ..AnnConfig::default()
    }))
    .unwrap();
    let shared = SharedEmbeddingStore::new(base);
    let queries = Arc::new(clusters.draw(64, &mut rng));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let writer = {
        let (shared, data, stop) = (shared.clone(), data.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut rng = Rng(99);
            for i in 0..4000usize {
                match rng.below(3) {
                    0 | 1 => {
                        let id = n + i;
                        shared
                            .add(doc("doc", id), "", Some(data[id].clone()))
                            .unwrap();
                    }
                    _ => {
                        shared.remove(&doc("doc", rng.below(n + i)));
                    }
                }
                if i % 500 == 0 {
                    shared.write().compact();
                }
            }
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let readers: Vec<_> = (0..4)
        .map(|t| {
            let (shared, queries, stop) = (shared.clone(), queries.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut searches = 0usize;
                while !stop.load(std::sync::atomic::Ordering::SeqCst) || searches < 20 {
                    let q = &queries[(searches + t) % queries.len()];
                    let results = if searches % 3 == 0 {
                        shared
                            .similarity_search_vector_batch(&queries[..8], 5, None, -1.0)
                            .unwrap()
                    } else {
                        vec![shared.similarity_search_vector(q, 5, None, -1.0).unwrap()]
                    };
                    for r in results {
                        let ks = keys(&r);
                        let unique: HashSet<&String> = ks.iter().collect();
                        assert_eq!(unique.len(), ks.len(), "a row twice");
                        assert!(r.windows(2).all(|w| w[0].score >= w[1].score));
                        assert!(r
                            .iter()
                            .all(|x| x.score.is_finite() && x.score.abs() <= 1.0001));
                    }
                    searches += 1;
                }
                searches
            })
        })
        .collect();
    writer.join().unwrap();
    let ran: usize = readers.into_iter().map(|h| h.join().unwrap()).sum();
    assert!(ran >= 80);
    shared.read().verify_invariants().unwrap();
}

#[test]
fn the_int8_copy_never_reaches_a_sidecar_or_the_digest() {
    let (dim, n) = (48, 200);
    let clusters = Clusters::new(dim, 5, 15);
    let mut rng = Rng(4);
    let data = clusters.draw(n, &mut rng);
    let plain = filled(&data, |_| "doc");
    let mut quant = filled(&data, |_| "doc");
    quant.set_scan_precision(ScanPrecision::int8()).unwrap();
    assert_eq!(
        plain.export_embeddings_with(Some("d")).embeddings.len(),
        quant.export_embeddings_with(Some("d")).embeddings.len()
    );
    assert_eq!(
        serde_json::to_string(&plain.export_embeddings_with(Some("d"))).unwrap(),
        serde_json::to_string(&quant.export_embeddings_with(Some("d"))).unwrap()
    );
}
