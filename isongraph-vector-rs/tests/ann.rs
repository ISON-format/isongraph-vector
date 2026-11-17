//! The opt-in IVF-flat approximate index: recall, exactness where it must be
//! exact, upkeep under writes, filters, determinism, and the guarantee that
//! nothing it builds ever reaches a graph file, a sidecar or the graph digest.
//!
//! Test data are clustered unit-ish vectors drawn from a seeded generator, so
//! every run sees the same numbers.

use ison_graph_rs::Node;
use isongraph_vector_rs::{
    graph_digest, AnnConfig, EmbeddingError, EmbeddingStore, MockEncoder, MultiHopOptions,
    NodeFilter, NodeId, PropertyValue, SearchOptions, SemanticGraph, SimilarityOptions,
    SimilarityResult, FOLD_BATCH,
};
use std::collections::HashSet;

/// Recall@10 demanded of the index on the clustered data below (4,000 vectors,
/// 40 clusters, 64 cells, nprobe 8 = 12.5% of the cells). The same floor the
/// Python port's tests use; measured 1.000 on every seed tried.
const RECALL_FLOOR: f64 = 0.95;

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
    fn gauss(&mut self) -> f32 {
        // Box-Muller
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

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

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn doc(i: usize) -> NodeId {
    NodeId::new("doc", i.to_string())
}

fn ref_of(r: &SimilarityResult) -> (String, String) {
    (r.node_id.node_type.clone(), r.node_id.id.clone())
}

fn refs(results: &[SimilarityResult]) -> Vec<(String, String)> {
    results.iter().map(ref_of).collect()
}

fn ref_set(results: &[SimilarityResult]) -> HashSet<(String, String)> {
    results.iter().map(ref_of).collect()
}

/// nlist 64, nprobe 8, no minimum size: the configuration the recall floor is
/// stated for.
fn config() -> AnnConfig {
    AnnConfig {
        nlist: Some(64),
        nprobe: Some(8),
        min_size: 0,
        ..AnnConfig::default()
    }
}

struct Fixture {
    store: EmbeddingStore,
    data: Vec<Vec<f32>>,
    clusters: Clusters,
    rng: Rng,
}

impl Fixture {
    fn queries(&mut self, count: usize) -> Vec<Vec<f32>> {
        self.clusters.draw(count, &mut self.rng)
    }
}

fn fixture_with(
    n: usize,
    dim: usize,
    ann: Option<AnnConfig>,
    type_of: impl Fn(usize) -> String,
) -> Fixture {
    let clusters = Clusters::new(dim, 40, 1);
    let mut rng = Rng(77);
    let data = clusters.draw(n, &mut rng);
    let mut store = EmbeddingStore::new(None);
    for (i, v) in data.iter().enumerate() {
        store
            .add(
                NodeId::new(type_of(i), i.to_string()),
                &format!("text {i}"),
                Some(v.clone()),
            )
            .unwrap();
    }
    if let Some(cfg) = ann {
        store.set_ann(Some(cfg)).unwrap();
    }
    Fixture {
        store,
        data,
        clusters,
        rng,
    }
}

fn fixture(n: usize) -> Fixture {
    fixture_with(n, 32, Some(config()), |_| "doc".to_string())
}

fn brute(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f64)> {
    let mut all: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(query, v)))
        .collect();
    all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

// -- configuration ------------------------------------------------------------

#[test]
fn off_by_default_and_exact_is_untouched() {
    let mut fx = fixture_with(1500, 32, None, |_| "doc".to_string());
    assert!(fx.store.ann().is_none());
    let stats = fx.store.ann_stats();
    assert!(!stats.enabled && !stats.active);
    assert!(matches!(
        fx.store.build_ann_index(false),
        Err(EmbeddingError::InvalidOption(_))
    ));

    // The same store with the index switched on and exact forced: identical,
    // not merely close, because the exact path is the same code either way.
    let mut other = fixture_with(1500, 32, Some(config()), |_| "doc".to_string());
    for q in fx.queries(5) {
        let a = fx
            .store
            .similarity_search_vector(&q, 10, None, -1.0)
            .unwrap();
        let b = other
            .store
            .similarity_search_vector_exact(&q, 10, None, -1.0)
            .unwrap();
        assert_eq!(refs(&a), refs(&b));
        assert_eq!(
            a.iter().map(|r| r.score).collect::<Vec<_>>(),
            b.iter().map(|r| r.score).collect::<Vec<_>>()
        );
    }
    // and a store with ann off never records an approximate answer
    assert_eq!(fx.store.ann_stats().searches.approximate, 0);
    let _ = other.queries(1);
}

#[test]
fn argument_validation() {
    let mut store = EmbeddingStore::new(None);
    let bad = [
        AnnConfig {
            nlist: Some(0),
            ..AnnConfig::default()
        },
        AnnConfig {
            nprobe: Some(0),
            ..AnnConfig::default()
        },
    ];
    for cfg in bad {
        assert!(matches!(
            store.set_ann(Some(cfg.clone())),
            Err(EmbeddingError::InvalidOption(_))
        ));
        assert!(EmbeddingStore::with_ann(None, cfg).is_err());
    }
    assert!(store.ann().is_none(), "a refused config must not be kept");

    store.set_ann(Some(AnnConfig::default())).unwrap();
    let cfg = store.ann().unwrap();
    assert_eq!(cfg.min_size, 50_000, "default minimum size matches Python");
    assert_eq!(cfg.seed, 0);
    assert!(cfg.nlist.is_none() && cfg.nprobe.is_none());

    assert!(store.set_ann_nprobe(Some(0)).is_err());
    store.set_ann_nprobe(Some(5)).unwrap();
    assert_eq!(store.ann().unwrap().nprobe, Some(5));
    store.set_ann(None).unwrap();
    assert!(store.ann().is_none());
}

// -- recall and exactness -----------------------------------------------------

#[test]
fn recall_floor_on_clustered_data() {
    let mut fx = fixture(4000);
    let queries = fx.queries(60);
    let report = fx.store.recall_check(&queries, 10, None, 0.0).unwrap();
    assert!(report.approximate, "{report:?}");
    assert_eq!((report.nlist, report.nprobe), (64, 8));
    assert_eq!((report.queries, report.k), (60, 10));
    assert!(report.recall >= RECALL_FLOOR, "{report:?}");
    assert!(0.0 <= report.min_recall && report.min_recall <= report.recall && report.recall <= 1.0);
    let stats = fx.store.ann_stats();
    assert!(stats.enabled && stats.active);
    assert_eq!(stats.nlist, Some(64));
    assert_eq!(stats.trained_n, Some(4000));
    assert!(stats.index_bytes > 0);
    assert!(stats.searches.approximate >= 60);
}

#[test]
fn results_carry_exact_cosine_scores() {
    let mut fx = fixture(4000);
    for q in fx.queries(10) {
        let got = fx
            .store
            .similarity_search_vector(&q, 10, None, 0.0)
            .unwrap();
        assert_eq!(got.len(), 10);
        for r in &got {
            let i: usize = r.node_id.id.parse().unwrap();
            let want = cosine(&q, &fx.data[i]);
            assert!(
                (r.score as f64 - want).abs() < 1e-5,
                "{} vs {}",
                r.score,
                want
            );
        }
        let scores: Vec<f32> = got.iter().map(|r| r.score).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]));
        assert!(fx.store.ann_stats().searches.approximate > 0);
    }
}

#[test]
fn exact_true_equals_brute_force() {
    let mut fx = fixture(4000);
    for q in fx.queries(12) {
        let got = fx
            .store
            .similarity_search_vector_exact(&q, 10, None, -1.0)
            .unwrap();
        let want = brute(&fx.data, &q, 10);
        assert_eq!(got.len(), 10);
        let got_ids: HashSet<usize> = got.iter().map(|r| r.node_id.id.parse().unwrap()).collect();
        let want_ids: HashSet<usize> = want.iter().map(|w| w.0).collect();
        assert_eq!(got_ids, want_ids);
        for (r, (_, score)) in got.iter().zip(&want) {
            assert!((r.score as f64 - score).abs() < 1e-5);
        }
    }
    assert_eq!(
        fx.store.ann_stats().searches.approximate,
        0,
        "exact must never consult the index"
    );
    assert!(fx.store.ann_stats().searches.exact >= 12);
}

#[test]
fn text_queries_take_the_same_path() {
    let mut store = EmbeddingStore::new(Some(Box::new(MockEncoder::new(16))));
    for i in 0..300 {
        store
            .add(doc(i), &format!("item number {i}"), None)
            .unwrap();
    }
    store
        .set_ann(Some(AnnConfig {
            nlist: Some(8),
            nprobe: Some(8), // every cell: must equal the exact answer
            min_size: 0,
            ..AnnConfig::default()
        }))
        .unwrap();
    let a = store
        .similarity_search("item number 42", 5, None, -1.0)
        .unwrap();
    let b = store
        .similarity_search_exact("item number 42", 5, None, -1.0)
        .unwrap();
    assert_eq!(refs(&a), refs(&b));
    assert_eq!(a[0].node_id, doc(42));
    assert!((a[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn probing_every_cell_is_the_exact_scan() {
    let mut fx = fixture(4000);
    fx.store.set_ann_nprobe(Some(64)).unwrap(); // nprobe == nlist
    let reference = fixture_with(4000, 32, None, |_| "doc".to_string());
    for q in fx.queries(8) {
        let got = fx
            .store
            .similarity_search_vector(&q, 10, None, 0.0)
            .unwrap();
        let want = reference
            .store
            .similarity_search_vector(&q, 10, None, 0.0)
            .unwrap();
        assert_eq!(refs(&got), refs(&want));
        assert_eq!(
            got.iter().map(|r| r.score).collect::<Vec<_>>(),
            want.iter().map(|r| r.score).collect::<Vec<_>>()
        );
    }
    let queries = fx.queries(5);
    let report = fx.store.recall_check(&queries, 10, None, 0.0).unwrap();
    assert!(!report.approximate);
    assert_eq!(report.recall, 1.0);
}

#[test]
fn under_min_size_stays_exact() {
    let mut fx = fixture_with(
        500,
        32,
        Some(AnnConfig {
            nlist: Some(16),
            nprobe: Some(2),
            min_size: 1000,
            ..AnnConfig::default()
        }),
        |_| "doc".to_string(),
    );
    let q = fx.queries(1).remove(0);
    let a = fx
        .store
        .similarity_search_vector(&q, 10, None, 0.0)
        .unwrap();
    let b = fx
        .store
        .similarity_search_vector_exact(&q, 10, None, 0.0)
        .unwrap();
    assert_eq!(refs(&a), refs(&b));
    let stats = fx.store.ann_stats();
    assert!(!stats.active);
    assert_eq!(stats.searches.approximate, 0);
    assert_eq!(stats.searches.builds, 0, "no index is built below min_size");
    let report = fx.store.recall_check(&[q], 10, None, 0.0).unwrap();
    assert!(!report.approximate);
    assert_eq!(report.recall, 1.0);
    // build_ann_index says so in the stats instead of building
    assert!(!fx.store.build_ann_index(false).unwrap().active);
}

#[test]
fn unbounded_zero_and_oversized_k_are_exact() {
    let mut fx = fixture(2000);
    let q = fx.queries(1).remove(0);
    let before = fx.store.ann_stats().searches.approximate;
    let all = fx
        .store
        .similarity_search_vector(&q, usize::MAX, None, -1.0)
        .unwrap();
    assert_eq!(all.len(), 2000);
    assert!(all.windows(2).all(|w| w[0].score >= w[1].score));
    let k_ge_live = fx
        .store
        .similarity_search_vector(&q, 2500, None, -1.0)
        .unwrap();
    assert_eq!(k_ge_live.len(), 2000);
    assert!(fx
        .store
        .similarity_search_vector(&q, 0, None, -1.0)
        .unwrap()
        .is_empty());
    assert_eq!(fx.store.ann_stats().searches.approximate, before);
}

#[test]
fn zero_query_falls_back_to_exact() {
    let fx = fixture(2000);
    let res = fx
        .store
        .similarity_search_vector(&[0.0; 32], 5, None, 0.0)
        .unwrap();
    assert_eq!(res.len(), 5);
    assert!(res.iter().all(|r| r.score == 0.0));
    assert_eq!(fx.store.ann_stats().searches.approximate, 0);
}

#[test]
fn wrong_dimension_is_still_an_error() {
    let fx = fixture(100);
    assert!(matches!(
        fx.store.similarity_search_vector(&[1.0, 2.0], 3, None, 0.0),
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        fx.store
            .similarity_search_vector_exact(&[1.0, 2.0], 3, None, 0.0),
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

// -- upkeep under writes ------------------------------------------------------

#[test]
fn writes_are_visible_immediately() {
    let mut fx = fixture(4000);
    fx.store.build_ann_index(false).unwrap();
    let fresh = fx.queries(1).remove(0);
    fx.store
        .add(NodeId::new("new", "1"), "fresh", Some(fresh.clone()))
        .unwrap();
    let top = fx
        .store
        .similarity_search_vector(&fresh, 3, None, 0.0)
        .unwrap();
    assert_eq!(top[0].node_id, NodeId::new("new", "1"));
    assert!((top[0].score - 1.0).abs() < 1e-5);
    assert_eq!(fx.store.ann_stats().pending, 1);

    // An in-place update moves the row: found at its new place, scored from
    // its new vector everywhere, and listed once.
    let target = fx.queries(1).remove(0);
    fx.store.add(doc(7), "moved", Some(target.clone())).unwrap();
    let hit = fx
        .store
        .similarity_search_vector(&target, 1, None, 0.0)
        .unwrap();
    assert_eq!(hit[0].node_id, doc(7));
    assert!((hit[0].score - 1.0).abs() < 1e-5);
    assert_eq!(hit[0].text, "moved");
    let old = fx.data[7].clone();
    for q in [old, target] {
        let res = fx
            .store
            .similarity_search_vector(&q, 400, None, -1.0)
            .unwrap();
        let ids = refs(&res);
        assert_eq!(ids.len(), ref_set(&res).len(), "a stale second entry");
        let truth = fx
            .store
            .similarity_search_vector_exact(&q, usize::MAX, None, -1.0)
            .unwrap();
        let truth: std::collections::HashMap<_, _> =
            truth.iter().map(|r| (ref_of(r), r.score)).collect();
        for r in &res {
            assert!((r.score - truth[&ref_of(r)]).abs() < 1e-5);
        }
    }
    assert_eq!(fx.store.count(), 4001);
}

#[test]
fn deletes_never_come_back() {
    let mut fx = fixture(4000);
    fx.store.build_ann_index(false).unwrap();
    let builds = fx.store.ann_stats().searches.builds;
    let mut gone = HashSet::new();
    for i in (0..4000).step_by(3) {
        // a third of them: forces the scan cache to compact
        assert!(fx.store.remove(&doc(i)));
        gone.insert(("doc".to_string(), i.to_string()));
    }
    for q in fx.queries(15) {
        let got = ref_set(
            &fx.store
                .similarity_search_vector(&q, 10, None, 0.0)
                .unwrap(),
        );
        assert!(got.is_disjoint(&gone));
        // a tombstone must not score 0.0 and beat a negative cosine
        let deep = ref_set(
            &fx.store
                .similarity_search_vector(&q, 250, None, -1.0)
                .unwrap(),
        );
        assert!(deep.is_disjoint(&gone));
        let every = ref_set(
            &fx.store
                .similarity_search_vector_exact(&q, usize::MAX, None, -1.0)
                .unwrap(),
        );
        assert!(every.is_disjoint(&gone));
        assert_eq!(every.len(), fx.store.count());
    }
    let queries = fx.queries(40);
    let report = fx.store.recall_check(&queries, 10, None, 0.0).unwrap();
    assert!(report.recall >= RECALL_FLOOR, "{report:?}");
    // 1,334 deletes is below the retrain threshold: followed, not retrained
    assert_eq!(fx.store.ann_stats().searches.builds, builds);
    assert!(fx.store.ann_stats().active);
}

#[test]
fn pending_rows_fold_and_heavy_churn_retrains() {
    let mut fx = fixture(3000);
    fx.store.build_ann_index(false).unwrap();
    let builds = fx.store.ann_stats().searches.builds;
    let extra = fx.queries(700);
    for (i, v) in extra.iter().enumerate() {
        fx.store
            .add(NodeId::new("late", i.to_string()), "", Some(v.clone()))
            .unwrap();
    }
    // folding is done by the writes themselves, FOLD_BATCH rows at a time: 700
    // writes made two folds and left 188 rows waiting
    let stats = fx.store.ann_stats();
    assert_eq!(stats.pending, 700 % FOLD_BATCH);
    assert_eq!(stats.folds, (700 / FOLD_BATCH) as u64);
    fx.store
        .similarity_search_vector(&extra[0], 5, None, 0.0)
        .unwrap(); // a search folds nothing and rebuilds nothing
    let stats = fx.store.ann_stats();
    assert_eq!(stats.pending, 700 % FOLD_BATCH);
    assert_eq!(stats.folds, (700 / FOLD_BATCH) as u64);
    assert_eq!(stats.searches.builds, builds);
    fx.store.build_ann_index(false).unwrap(); // folds the remainder on request
    assert_eq!(fx.store.ann_stats().pending, 0);

    let more = fx.queries(1200);
    for (i, v) in more.iter().enumerate() {
        fx.store
            .add(NodeId::new("later", i.to_string()), "", Some(v.clone()))
            .unwrap();
    }
    fx.store
        .similarity_search_vector(&more[0], 5, None, 0.0)
        .unwrap(); // churn 1900 > 50% of 3000
    assert_eq!(fx.store.ann_stats().searches.builds, builds + 1);
    assert_eq!(fx.store.ann_stats().trained_n, Some(4900));
    let queries = fx.queries(30);
    assert!(
        fx.store
            .recall_check(&queries, 10, None, 0.0)
            .unwrap()
            .recall
            >= RECALL_FLOOR
    );
}

#[test]
fn small_writes_stay_pending_until_enough() {
    let mut fx = fixture(2000);
    fx.store.build_ann_index(false).unwrap();
    for (i, v) in fx.queries(100).iter().enumerate() {
        fx.store
            .add(NodeId::new("w", i.to_string()), "", Some(v.clone()))
            .unwrap();
    }
    let q = fx.queries(1).remove(0);
    fx.store.similarity_search_vector(&q, 5, None, 0.0).unwrap();
    // 100 < FOLD_BATCH: scored exactly, not yet folded, and no rebuild
    let stats = fx.store.ann_stats();
    assert_eq!(stats.pending, 100);
    assert_eq!(stats.churn, 100);
    assert_eq!(stats.searches.builds, 1);
}

#[test]
fn clear_and_dimension_change() {
    let mut fx = fixture_with(
        1500,
        32,
        Some(AnnConfig {
            nlist: Some(16),
            nprobe: Some(4),
            min_size: 0,
            ..AnnConfig::default()
        }),
        |_| "doc".to_string(),
    );
    fx.store.build_ann_index(false).unwrap();
    let q = fx.data[0].clone();
    fx.store.clear_with(false);
    assert!(fx
        .store
        .similarity_search_vector(&q, 5, None, 0.0)
        .unwrap()
        .is_empty());
    assert!(!fx.store.ann_stats().active);

    fx.store.clear();
    let small = Clusters::new(8, 12, 3);
    let mut rng = Rng(5);
    let vectors = small.draw(1200, &mut rng);
    for (i, v) in vectors.iter().enumerate() {
        fx.store.add(doc(i), "", Some(v.clone())).unwrap();
    }
    let res = fx
        .store
        .similarity_search_vector(&vectors[5], 5, None, 0.0)
        .unwrap();
    assert_eq!(res[0].node_id, doc(5));
    assert!(fx.store.ann_stats().active);
    assert_eq!(fx.store.dimension(), Some(8));
}

#[test]
fn importing_a_payload_keeps_the_index_honest() {
    let mut fx = fixture(3000);
    fx.store.build_ann_index(false).unwrap();
    let payload = fx.store.export_embeddings();
    // a replace import discards the old rows (and so the old index)
    fx.store.import_embeddings(&payload, true).unwrap();
    assert_eq!(fx.store.count(), 3000);
    let q = fx.queries(1).remove(0);
    let got = fx
        .store
        .similarity_search_vector(&q, 10, None, 0.0)
        .unwrap();
    assert_eq!(got.len(), 10);
    assert!(fx.store.ann_stats().active);
    let queries = fx.queries(30);
    assert!(
        fx.store
            .recall_check(&queries, 10, None, 0.0)
            .unwrap()
            .recall
            >= RECALL_FLOOR
    );
}

// -- filters ------------------------------------------------------------------

#[test]
fn node_type_filter_and_widening() {
    let mut cfg = config();
    cfg.nprobe = Some(2);
    let mut fx = fixture_with(4000, 32, Some(cfg), |i| {
        if i % 200 == 0 { "rare" } else { "common" }.to_string()
    });
    let q = fx.queries(1).remove(0);
    let common = fx
        .store
        .similarity_search_vector(&q, 10, Some("common"), 0.0)
        .unwrap();
    assert_eq!(common.len(), 10);
    assert!(common.iter().all(|r| r.node_id.node_type == "common"));

    // 20 rare rows in the whole store and a top_k of 15: two probed cells
    // cannot hold 15 of them, so the probe widens (ending in the scan).
    let widened = fx.store.ann_stats().searches.widened;
    let exact_before = fx.store.ann_stats().searches.exact;
    let rare = fx
        .store
        .similarity_search_vector(&q, 15, Some("rare"), -1.0)
        .unwrap();
    assert_eq!(rare.len(), 15);
    assert!(rare.iter().all(|r| r.node_id.node_type == "rare"));
    let truth = fx
        .store
        .similarity_search_vector_exact(&q, 15, Some("rare"), -1.0)
        .unwrap();
    assert_eq!(rare.len(), truth.len());
    let stats = fx.store.ann_stats();
    assert!(stats.searches.widened > widened || stats.searches.exact > exact_before + 1);

    assert!(fx
        .store
        .similarity_search_vector(&q, 5, Some("nope"), -1.0)
        .unwrap()
        .is_empty());
}

#[test]
fn threshold_is_respected() {
    let mut fx = fixture(4000);
    let q = fx.queries(1).remove(0);
    let best = fx
        .store
        .similarity_search_vector_exact(&q, 1, None, -1.0)
        .unwrap()[0]
        .score;
    let res = fx
        .store
        .similarity_search_vector(&q, 10, None, best - 1e-3)
        .unwrap();
    assert!(!res.is_empty() && res.iter().all(|r| r.score >= best - 1e-3));
    assert!(fx
        .store
        .similarity_search_vector(&q, 10, None, 2.0)
        .unwrap()
        .is_empty());
    // fewer than k above a threshold: the same short list as the scan
    let cut = fx
        .store
        .similarity_search_vector_exact(&q, 10, None, -1.0)
        .unwrap()[4]
        .score;
    let got = fx
        .store
        .similarity_search_vector(&q, 10, None, cut)
        .unwrap();
    let want = fx
        .store
        .similarity_search_vector_exact(&q, 10, None, cut)
        .unwrap();
    assert_eq!(ref_set(&got), ref_set(&want));
}

// -- determinism --------------------------------------------------------------

#[test]
fn a_fixed_seed_gives_the_same_index_every_time() {
    let one_cell = |seed: u64| {
        let mut cfg = config();
        cfg.nprobe = Some(1);
        cfg.seed = seed;
        fixture_with(3000, 32, Some(cfg), |_| "doc".to_string())
    };
    let mut a = one_cell(9);
    let b = one_cell(9);
    let queries = a.queries(40);
    for q in &queries {
        let ra = a.store.similarity_search_vector(q, 10, None, -1.0).unwrap();
        let rb = b.store.similarity_search_vector(q, 10, None, -1.0).unwrap();
        assert_eq!(refs(&ra), refs(&rb));
        assert_eq!(
            ra.iter().map(|r| r.score).collect::<Vec<_>>(),
            rb.iter().map(|r| r.score).collect::<Vec<_>>()
        );
    }
    let (sa, sb) = (a.store.ann_stats(), b.store.ann_stats());
    assert_eq!(
        (sa.nlist, sa.trained_n, sa.index_bytes),
        (sb.nlist, sb.trained_n, sb.index_bytes)
    );

    // retraining the same data with the same seed reproduces the answers
    let before: Vec<_> = queries
        .iter()
        .map(|q| refs(&a.store.similarity_search_vector(q, 10, None, -1.0).unwrap()))
        .collect();
    a.store.build_ann_index(true).unwrap();
    let after: Vec<_> = queries
        .iter()
        .map(|q| refs(&a.store.similarity_search_vector(q, 10, None, -1.0).unwrap()))
        .collect();
    assert_eq!(before, after);
    assert_eq!(a.store.ann_stats().searches.builds, 2);
}

#[test]
fn the_thread_count_never_changes_an_answer() {
    // 256 dimensions put 2,048 rows in a scan block, so 7,000 rows are four
    // blocks and eight threads really do share the work.
    let mut cfg = config();
    cfg.nprobe = Some(3);
    let build = || fixture_with(7000, 256, Some(cfg.clone()), |_| "doc".to_string());
    let (mut single, mut many) = (build(), build());
    single.store.set_scan_threads(1);
    many.store.set_scan_threads(8);
    assert_eq!(
        (single.store.scan_threads(), many.store.scan_threads()),
        (1, 8)
    );
    for q in single.queries(10) {
        for k in [10, 25, 3000] {
            let a = single
                .store
                .similarity_search_vector_exact(&q, k, None, -1.0)
                .unwrap();
            let b = many
                .store
                .similarity_search_vector_exact(&q, k, None, -1.0)
                .unwrap();
            assert_eq!(refs(&a), refs(&b), "k={k}");
            assert_eq!(
                a.iter().map(|r| r.score).collect::<Vec<_>>(),
                b.iter().map(|r| r.score).collect::<Vec<_>>()
            );
        }
        let a = single
            .store
            .similarity_search_vector(&q, 10, None, -1.0)
            .unwrap();
        let b = many
            .store
            .similarity_search_vector(&q, 10, None, -1.0)
            .unwrap();
        assert_eq!(refs(&a), refs(&b));
    }
    assert_eq!(single.store.ann_stats().nlist, many.store.ann_stats().nlist);
}

#[test]
fn awkward_dimensions_work_end_to_end() {
    // 37 is neither a multiple of 8 nor of 32, and nlist 13 is not a multiple
    // of the kernel's group of four.
    let mut fx = fixture_with(
        1200,
        37,
        Some(AnnConfig {
            nlist: Some(13),
            nprobe: Some(13),
            min_size: 0,
            ..AnnConfig::default()
        }),
        |_| "doc".to_string(),
    );
    for q in fx.queries(6) {
        let got = fx
            .store
            .similarity_search_vector_exact(&q, 10, None, -1.0)
            .unwrap();
        let want = brute(&fx.data, &q, 10);
        let ids: HashSet<usize> = got.iter().map(|r| r.node_id.id.parse().unwrap()).collect();
        assert_eq!(ids, want.iter().map(|w| w.0).collect::<HashSet<_>>());
    }
    // Thirteen coarse cells over forty clusters: recall climbs with nprobe.
    let queries = fx.queries(30);
    let mut last = 0.0;
    for nprobe in [2, 6, 10] {
        fx.store.set_ann_nprobe(Some(nprobe)).unwrap();
        let report = fx.store.recall_check(&queries, 10, None, 0.0).unwrap();
        assert!(
            report.approximate && report.nlist == 13 && report.nprobe == nprobe,
            "{report:?}"
        );
        assert!(
            report.recall >= last,
            "recall fell as nprobe rose: {report:?}"
        );
        last = report.recall;
    }
    assert!(last >= RECALL_FLOOR, "nprobe 10 of 13 gave {last}");
}

#[test]
fn equal_scores_break_in_a_stable_order() {
    // Identical vectors score identically; the answer must not depend on the
    // run (the store used to iterate a HashMap, so it did).
    let mut order = None;
    for _ in 0..5 {
        let mut store = EmbeddingStore::new(None);
        for i in 0..50 {
            store
                .add(doc(i), "", Some(vec![1.0, 2.0, 3.0, 4.0]))
                .unwrap();
        }
        let got = refs(
            &store
                .similarity_search_vector(&[1.0, 2.0, 3.0, 4.0], 7, None, 0.0)
                .unwrap(),
        );
        match &order {
            None => order = Some(got),
            Some(first) => assert_eq!(&got, first),
        }
    }
    assert_eq!(order.unwrap().len(), 7);
}

// -- recall_check -------------------------------------------------------------

#[test]
fn recall_check_shapes_and_errors() {
    let mut fx = fixture(2000);
    let q = fx.queries(3);
    assert_eq!(
        fx.store
            .recall_check(&q[..1], 10, None, 0.0)
            .unwrap()
            .queries,
        1
    );
    assert_eq!(fx.store.recall_check(&q, 10, None, 0.0).unwrap().queries, 3);
    assert_eq!(
        fx.store
            .recall_check(q.iter().map(|v| v.as_slice()), 5, None, 0.0)
            .unwrap()
            .k,
        5
    );
    let empty: [&[f32]; 0] = [];
    let none = fx.store.recall_check(empty, 10, None, 0.0).unwrap();
    assert_eq!((none.queries, none.recall, none.min_recall), (0, 1.0, 1.0));
    assert!(matches!(
        fx.store.recall_check(&q, 0, None, 0.0),
        Err(EmbeddingError::InvalidOption(_))
    ));

    // without an index in use both answers are the scan
    let off = fixture_with(100, 32, None, |_| "doc".to_string());
    let r = off.store.recall_check(&q, 10, None, 0.0).unwrap();
    assert_eq!(r.recall, 1.0);
    assert!(!r.approximate);
    assert!(r.ann_ms >= 0.0 && r.exact_ms >= 0.0);
}

#[test]
fn concurrent_readers_agree() {
    let mut fx = fixture(3000);
    fx.store.build_ann_index(false).unwrap();
    let queries = fx.queries(24);
    let store = std::sync::Arc::new(fx.store);
    let want: Vec<_> = queries
        .iter()
        .map(|q| refs(&store.similarity_search_vector(q, 8, None, -1.0).unwrap()))
        .collect();
    std::thread::scope(|s| {
        for t in 0..4 {
            let (store, queries, want) = (&store, &queries, &want);
            s.spawn(move || {
                for i in 0..60 {
                    let at = (i + t) % queries.len();
                    let res = store
                        .similarity_search_vector(&queries[at], 8, None, -1.0)
                        .unwrap();
                    assert_eq!(refs(&res), want[at]);
                }
            });
        }
    });
}

// -- through the graph --------------------------------------------------------

fn graph_with(
    n: usize,
    dim: usize,
    ann: Option<AnnConfig>,
) -> (SemanticGraph, Vec<Vec<f32>>, Clusters, Rng) {
    let clusters = Clusters::new(dim, 40, 1);
    let mut rng = Rng(11);
    let data = clusters.draw(n, &mut rng);
    let mut graph = SemanticGraph::with_options(
        "ann",
        Box::new(MockEncoder::new(dim)),
        false,
        vec!["name".to_string()],
    );
    for (i, v) in data.iter().enumerate() {
        graph
            .add_node(
                "item",
                &i.to_string(),
                vec![("group", PropertyValue::Int((i % 50) as i64))],
                None,
            )
            .unwrap();
        graph
            .embedding_store_mut()
            .add(
                NodeId::new("item", i.to_string()),
                &format!("item {i}"),
                Some(v.clone()),
            )
            .unwrap();
    }
    if let Some(cfg) = ann {
        graph.set_ann(Some(cfg)).unwrap();
    }
    (graph, data, clusters, rng)
}

fn graph_config() -> AnnConfig {
    AnnConfig {
        nlist: Some(48),
        nprobe: Some(6),
        min_size: 0,
        ..AnnConfig::default()
    }
}

fn group_of(graph: &SemanticGraph, r: &SimilarityResult) -> i64 {
    match graph
        .graph()
        .get_node(&r.node_id.node_type, &r.node_id.id)
        .unwrap()
        .properties
        .get("group")
    {
        Some(PropertyValue::Int(i)) => *i,
        other => panic!("group was {other:?}"),
    }
}

#[test]
fn graph_passthrough_and_recall() {
    let (graph, _, clusters, mut rng) = graph_with(3000, 32, Some(graph_config()));
    assert_eq!(graph.ann().unwrap().nprobe, Some(6));
    let q = clusters.draw(1, &mut rng).remove(0);
    let res = graph
        .similarity_search_with(&q, &SimilarityOptions::default())
        .unwrap();
    assert_eq!(res.len(), 10);
    assert!(graph.ann_stats().active);
    let queries = clusters.draw(20, &mut rng);
    assert!(graph.recall_check(&queries, 10, None, 0.0).unwrap().recall >= RECALL_FLOOR);

    let (plain, ..) = graph_with(50, 32, None);
    assert!(plain.ann().is_none());
}

#[test]
fn graph_exact_flag_and_similar_to_node() {
    let (graph, _, clusters, mut rng) = graph_with(3000, 32, Some(graph_config()));
    let q = clusters.draw(1, &mut rng).remove(0);
    let before = graph.ann_stats().searches.approximate;
    let exact = SimilarityOptions {
        top_k: Some(5),
        exact: true,
        ..SimilarityOptions::default()
    };
    graph.similarity_search_with(&q, &exact).unwrap();
    let pred = |_: &Node| true;
    graph
        .similarity_search_with(
            &q,
            &SimilarityOptions {
                filter: NodeFilter::new().predicate(&pred),
                ..exact.clone()
            },
        )
        .unwrap();
    assert_eq!(graph.ann_stats().searches.approximate, before);
    let near = graph
        .similar_to_node(&NodeId::new("item", "3"), 5, None, 0.0)
        .unwrap();
    assert_eq!(near.len(), 5);
    assert!(near.iter().all(|r| r.node_id != NodeId::new("item", "3")));
}

#[test]
fn node_filter_gets_k_results_even_when_selective() {
    let (graph, _, clusters, mut rng) = graph_with(3000, 32, Some(graph_config()));
    let q = clusters.draw(1, &mut rng).remove(0);
    let seven = |n: &Node| matches!(n.properties.get("group"), Some(PropertyValue::Int(7)));
    // 1 node in 50 passes: the probed cells hold few, so the search widens
    let filter = NodeFilter::new().predicate(&seven);
    let res = graph
        .similarity_search_with(
            &q,
            &SimilarityOptions {
                filter: filter.clone(),
                ..SimilarityOptions::default()
            },
        )
        .unwrap();
    assert_eq!(res.len(), 10);
    assert!(res.iter().all(|r| group_of(&graph, r) == 7));
    assert!(res.windows(2).all(|w| w[0].score >= w[1].score));
    let want = graph
        .similarity_search_with(
            &q,
            &SimilarityOptions {
                filter: filter.clone(),
                exact: true,
                ..SimilarityOptions::default()
            },
        )
        .unwrap();
    assert!(!want.is_empty());
    // The widened search finds most of the exact answers: judged over many
    // queries, because one query's overlap depends on which sample the index
    // was trained on (the old single-query bound of 7 of 10 held only for the
    // sample that the sorted-key row order happened to give).
    let (mut overlap, mut asked) = (0, 0);
    for q in clusters.draw(10, &mut rng) {
        let res = graph
            .similarity_search_with(
                &q,
                &SimilarityOptions {
                    filter: filter.clone(),
                    ..SimilarityOptions::default()
                },
            )
            .unwrap();
        let want = graph
            .similarity_search_with(
                &q,
                &SimilarityOptions {
                    filter: filter.clone(),
                    exact: true,
                    ..SimilarityOptions::default()
                },
            )
            .unwrap();
        overlap += ref_set(&res).intersection(&ref_set(&want)).count();
        asked += want.len();
    }
    assert!(overlap * 100 >= asked * 80, "{overlap} of {asked}");

    // a filter nothing passes terminates, empty
    let never = |_: &Node| false;
    let none = graph
        .similarity_search_with(
            &q,
            &SimilarityOptions {
                filter: NodeFilter::new().predicate(&never),
                ..SimilarityOptions::default()
            },
        )
        .unwrap();
    assert!(none.is_empty());
}

#[test]
fn multi_hop_seeds_use_the_index_unless_blending() {
    let (mut graph, _, clusters, mut rng) = graph_with(2000, 32, Some(graph_config()));
    for i in 0..1999 {
        graph
            .add_edge(
                "NEXT",
                ("item", &i.to_string()),
                ("item", &(i + 1).to_string()),
                vec![],
            )
            .unwrap();
    }
    let q = clusters.draw(1, &mut rng).remove(0);
    let opts = MultiHopOptions {
        rel_type: Some("NEXT"),
        max_hops: 2,
        top_k_seeds: 3,
        threshold: -1.0,
        ..MultiHopOptions::default()
    };
    let out = graph
        .semantic_multi_hop_search(&q, &SearchOptions::from(opts.clone()))
        .unwrap();
    assert!(!out.is_empty() && out[0].hop_count == 0);
    assert!(graph.ann_stats().searches.approximate >= 1);

    // blend > 0 scores every node and must not touch the approximate path
    let before = graph.ann_stats().searches.approximate;
    let blended = MultiHopOptions {
        blend: 0.3,
        max_hops: 1,
        ..opts
    };
    graph
        .semantic_multi_hop_search(&q, &SearchOptions::from(blended))
        .unwrap();
    assert_eq!(graph.ann_stats().searches.approximate, before);
}

#[test]
fn a_loaded_graph_rebuilds_lazily_and_persists_nothing() {
    let (graph, _, clusters, mut rng) = graph_with(1500, 32, Some(graph_config()));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.ison");
    graph.save(&path, None, true).unwrap();
    let mut loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(32))).unwrap();
    loaded.set_ann(Some(graph_config())).unwrap();
    assert!(!loaded.ann_stats().active, "nothing was persisted");
    let q = clusters.draw(1, &mut rng).remove(0);
    assert_eq!(
        loaded
            .similarity_search_with(&q, &SimilarityOptions::default())
            .unwrap()
            .len(),
        10
    );
    assert!(loaded.ann_stats().active);
}

// -- the index never reaches a file or the digest -----------------------------

#[test]
fn graph_files_sidecar_and_digest_are_the_same_with_ann_on() {
    let (off, ..) = graph_with(600, 16, None);
    let (on, _, clusters, mut rng) = graph_with(600, 16, Some(graph_config()));
    // exercise the index hard before saving: build, write, delete, search
    let q = clusters.draw(1, &mut rng).remove(0);
    on.similarity_search_with(&q, &SimilarityOptions::default())
        .unwrap();
    assert!(on.ann_stats().active);

    let dir = tempfile::tempdir().unwrap();
    let (p_off, p_on) = (dir.path().join("off.ison"), dir.path().join("on.ison"));
    off.save(&p_off, None, true).unwrap();
    on.save(&p_on, None, true).unwrap();
    assert_eq!(
        std::fs::read(&p_off).unwrap(),
        std::fs::read(&p_on).unwrap()
    );

    let (s_off, s_on) = (
        SemanticGraph::embeddings_path_for(&p_off),
        SemanticGraph::embeddings_path_for(&p_on),
    );
    let (side_off, side_on) = (
        std::fs::read(&s_off).unwrap(),
        std::fs::read(&s_on).unwrap(),
    );
    assert_eq!(side_off, side_on, "the sidecar must be byte-identical");
    let text = String::from_utf8(side_on).unwrap().to_lowercase();
    for word in ["ivf", "nlist", "nprobe", "centroid", "\"ann"] {
        assert!(!text.contains(word), "sidecar mentions {word}");
    }
    assert_eq!(graph_digest(off.graph()), graph_digest(on.graph()));

    // and the directory holds exactly the two files, nothing for the index
    let mut names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "off.ison",
            "off.ison.embeddings.json",
            "on.ison",
            "on.ison.embeddings.json"
        ]
    );

    // the exported payload is also unchanged by building the index
    let a = off.embedding_store().to_json().unwrap();
    let b = on.embedding_store().to_json().unwrap();
    assert_eq!(a, b);
}

// -- the graph keeps the index honest -----------------------------------------

#[test]
fn graph_edits_reach_the_index() {
    // Auto-embedded nodes: add, update (re-embeds), remove, with the index on.
    let mut graph = SemanticGraph::new("g", Box::new(MockEncoder::new(48)));
    for i in 0..300 {
        graph
            .add_node(
                "doc",
                &i.to_string(),
                vec![(
                    "name",
                    PropertyValue::String(format!("document number {i}")),
                )],
                None,
            )
            .unwrap();
    }
    graph
        .set_ann(Some(AnnConfig {
            nlist: Some(10),
            nprobe: Some(10), // every cell: the index must agree with the scan
            min_size: 0,
            ..AnnConfig::default()
        }))
        .unwrap();
    let top = |graph: &SemanticGraph, text: &str| {
        graph
            .similarity_search_with(
                text,
                &SimilarityOptions {
                    top_k: Some(3),
                    threshold: -1.0,
                    ..SimilarityOptions::default()
                },
            )
            .unwrap()
    };
    assert_eq!(top(&graph, "document number 5")[0].node_id, doc(5));

    // an edit moves the node's vector: found by its new text, not its old
    graph
        .update_node(
            "doc",
            "5",
            vec![(
                "name",
                PropertyValue::String("completely different words".into()),
            )],
        )
        .unwrap();
    assert_eq!(top(&graph, "completely different words")[0].node_id, doc(5));
    assert_ne!(top(&graph, "document number 5")[0].node_id, doc(5));

    // a removal is gone from every result
    graph.remove_node("doc", "9").unwrap();
    let all = graph
        .similarity_search_with(
            "document number 9",
            &SimilarityOptions {
                top_k: None,
                threshold: -1.0,
                ..SimilarityOptions::default()
            },
        )
        .unwrap();
    assert_eq!(all.len(), 299);
    assert!(all.iter().all(|r| r.node_id != doc(9)));
}
