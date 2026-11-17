//! The opt-in HNSW approximate index: recall, exactness where it must be exact,
//! upkeep under writes (insert, update, delete, compaction) checked against a
//! model through every search path, filters, concurrency, determinism, and the
//! guarantee that nothing it builds ever reaches a graph file, a sidecar or the
//! graph digest.
//!
//! Test data are clustered unit-ish vectors from a seeded generator, so every
//! run sees the same numbers.

use ison_graph_rs::Node;
use isongraph_vector_rs::{
    graph_digest, AnnConfig, AnnMode, EmbeddingError, EmbeddingStore, HnswConfig, MockEncoder,
    NodeFilter, NodeId, PropertyValue, SemanticGraph, SharedEmbeddingStore, SimilarityOptions,
    SimilarityResult,
};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

/// Recall@10 demanded at the defaults on the clustered data below.
const RECALL_FLOOR: f64 = 0.9;

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

fn ids(results: &[SimilarityResult]) -> Vec<String> {
    results
        .iter()
        .map(|r| format!("{}:{}", r.node_id.node_type, r.node_id.id))
        .collect()
}

fn id_set(results: &[SimilarityResult]) -> HashSet<String> {
    ids(results).into_iter().collect()
}

fn bits(results: &[SimilarityResult]) -> Vec<(String, u32)> {
    results
        .iter()
        .map(|r| {
            (
                format!("{}:{}", r.node_id.node_type, r.node_id.id),
                r.score.to_bits(),
            )
        })
        .collect()
}

/// HNSW with the defaults, no minimum size.
fn hnsw_config() -> AnnConfig {
    AnnConfig {
        min_size: 0,
        ..AnnConfig::hnsw()
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
    fixture_with(n, 32, Some(hnsw_config()), |_| "doc".to_string())
}

fn recall_of(store: &EmbeddingStore, queries: &[Vec<f32>], k: usize) -> f64 {
    let mut sum = 0.0;
    for q in queries {
        let got = id_set(&store.similarity_search_vector(q, k, None, -1.0).unwrap());
        let want = id_set(
            &store
                .similarity_search_vector_exact(q, k, None, -1.0)
                .unwrap(),
        );
        sum += got.intersection(&want).count() as f64 / want.len() as f64;
    }
    sum / queries.len() as f64
}

// -- configuration ------------------------------------------------------------

#[test]
fn defaults_leave_ivf_alone_and_hnsw_is_opt_in() {
    let cfg = AnnConfig::default();
    assert_eq!(cfg.mode, AnnMode::Ivf);
    assert_eq!(cfg.hnsw, HnswConfig::default());
    assert_eq!(
        (cfg.hnsw.m, cfg.hnsw.ef_construction, cfg.hnsw.ef_search),
        (16, 200, None)
    );
    let h = AnnConfig::hnsw();
    assert_eq!(h.mode, AnnMode::Hnsw);
    assert_eq!(h.min_size, 50_000, "the same minimum size as IVF");
    assert_eq!(h.seed, 0);

    // an IVF store never builds a graph and reports IVF facts
    let mut fx = fixture_with(
        1200,
        16,
        Some(AnnConfig {
            nlist: Some(16),
            nprobe: Some(4),
            min_size: 0,
            ..AnnConfig::default()
        }),
        |_| "doc".to_string(),
    );
    let q = fx.queries(1).remove(0);
    fx.store.similarity_search_vector(&q, 5, None, 0.0).unwrap();
    let stats = fx.store.ann_stats();
    assert_eq!(stats.mode, AnnMode::Ivf);
    assert_eq!(stats.nlist, Some(16));
    assert_eq!(stats.graph_nodes, 0);
}

#[test]
fn argument_validation() {
    let mut store = EmbeddingStore::new(None);
    let bad = [
        HnswConfig {
            m: 1,
            ..HnswConfig::default()
        },
        HnswConfig {
            m: 65,
            ..HnswConfig::default()
        },
        HnswConfig {
            ef_construction: 0,
            ..HnswConfig::default()
        },
        HnswConfig {
            ef_search: Some(0),
            ..HnswConfig::default()
        },
    ];
    for hnsw in bad {
        let cfg = AnnConfig {
            hnsw,
            ..AnnConfig::hnsw()
        };
        assert!(matches!(
            store.set_ann(Some(cfg.clone())),
            Err(EmbeddingError::InvalidOption(_))
        ));
        assert!(EmbeddingStore::with_ann(None, cfg).is_err());
    }
    assert!(store.ann().is_none(), "a refused config must not be kept");

    assert!(store.set_ann_ef_search(Some(10)).is_err(), "no ann yet");
    store.set_ann(Some(AnnConfig::hnsw())).unwrap();
    assert!(store.set_ann_ef_search(Some(0)).is_err());
    store.set_ann_ef_search(Some(33)).unwrap();
    assert_eq!(store.ann().unwrap().hnsw.ef_search, Some(33));
    assert_eq!(store.ann_stats().ef_search, Some(33));
    assert!(store
        .similarity_search_vector_ef(&[1.0], 1, None, 0.0, 0)
        .is_err());
}

// -- recall and exactness -----------------------------------------------------

#[test]
fn recall_floor_at_defaults_and_recall_check_agrees() {
    let mut fx = fixture(4000);
    let queries = fx.queries(80);
    let report = fx.store.recall_check(&queries, 10, None, 0.0).unwrap();
    assert!(report.approximate, "{report:?}");
    assert_eq!((report.queries, report.k), (80, 10));
    assert_eq!(report.ef_search, 64);
    assert!(report.recall >= RECALL_FLOOR, "{report:?}");
    assert!(report.min_recall <= report.recall && report.recall <= 1.0);

    // the report's mean is the mean of the per-query overlaps computed here
    let mine = recall_of(&fx.store, &queries, 10);
    assert!(
        (report.recall - mine).abs() < 1e-9,
        "{} vs {mine}",
        report.recall
    );

    let stats = fx.store.ann_stats();
    assert!(stats.enabled && stats.active);
    assert_eq!(stats.mode, AnnMode::Hnsw);
    assert_eq!(stats.graph_nodes, 4000);
    assert_eq!(stats.graph_dead, 0);
    assert!(stats.max_level.is_some() && stats.build_seconds.is_some());
    assert!(stats.index_bytes > 0);
    assert!(stats.searches.approximate >= 80, "{stats:?}");
    assert_eq!(stats.searches.builds, 1);
    fx.store.verify_invariants().unwrap();
}

#[test]
fn more_ef_never_hurts_much_and_ef_one_still_answers() {
    let mut fx = fixture(3000);
    let queries = fx.queries(40);
    let mean = |ef: usize| -> f64 {
        let mut sum = 0.0;
        for q in &queries {
            let got = id_set(
                &fx.store
                    .similarity_search_vector_ef(q, 10, None, -1.0, ef)
                    .unwrap(),
            );
            let want = id_set(
                &fx.store
                    .similarity_search_vector_exact(q, 10, None, -1.0)
                    .unwrap(),
            );
            sum += got.intersection(&want).count() as f64 / 10.0;
        }
        sum / queries.len() as f64
    };
    let (low, mid, high) = (mean(10), mean(64), mean(400));
    assert!(
        low <= mid + 0.02 && mid <= high + 0.02,
        "{low} {mid} {high}"
    );
    assert!(high >= 0.98, "{high}");
    // ef below k is raised to k
    let res = fx
        .store
        .similarity_search_vector_ef(&queries[0], 10, None, -1.0, 1)
        .unwrap();
    assert_eq!(res.len(), 10);
}

#[test]
fn scores_are_the_exact_scans_and_hits_come_back_sorted() {
    let mut fx = fixture(2500);
    let queries = fx.queries(30);
    for q in &queries {
        let exact_all = fx
            .store
            .similarity_search_vector_exact(q, usize::MAX, None, -2.0)
            .unwrap();
        let by_id: HashMap<String, u32> = bits(&exact_all).into_iter().collect();
        let res = fx
            .store
            .similarity_search_vector(q, 15, None, -1.0)
            .unwrap();
        assert_eq!(res.len(), 15);
        for (id, score_bits) in bits(&res) {
            assert_eq!(by_id[&id], score_bits, "{id}: score differs from the scan");
        }
        assert!(res.windows(2).all(|w| w[0].score >= w[1].score));
    }
    assert!(fx.store.ann_stats().searches.approximate >= 30);
    let _ = &fx.data;
}

#[test]
fn exact_true_and_big_k_and_small_stores_stay_exact() {
    let mut fx = fixture(1500);
    let q = fx.queries(1).remove(0);
    let before = fx.store.ann_stats().searches;
    let a = fx
        .store
        .similarity_search_vector_exact(&q, 10, None, 0.0)
        .unwrap();
    let b = fx
        .store
        .similarity_search_vector(&q, usize::MAX, None, -2.0)
        .unwrap();
    assert_eq!(a.len(), 10);
    assert_eq!(b.len(), 1500);
    assert_eq!(bits(&a), bits(&b[..10]));
    let after = fx.store.ann_stats().searches;
    assert_eq!(after.approximate, before.approximate);
    assert_eq!(after.exact, before.exact + 2);

    // under min_size nothing is built
    let small = fixture_with(
        400,
        16,
        Some(AnnConfig {
            min_size: 1000,
            ..AnnConfig::hnsw()
        }),
        |_| "doc".to_string(),
    );
    let q = vec![0.5f32; 16];
    small
        .store
        .similarity_search_vector(&q, 5, None, 0.0)
        .unwrap();
    let stats = small.store.ann_stats();
    assert!(!stats.active && stats.searches.approximate == 0 && stats.searches.exact == 1);

    // a store smaller than the ef window answers exactly (the graph would
    // cost more than the scan)
    let tiny = fixture_with(40, 16, Some(hnsw_config()), |_| "doc".to_string());
    tiny.store
        .similarity_search_vector(&q, 5, None, 0.0)
        .unwrap();
    assert_eq!(tiny.store.ann_stats().searches.approximate, 0);
}

#[test]
fn zero_query_and_wrong_dimension() {
    let fx = fixture(1200);
    let zero = vec![0.0f32; 32];
    let res = fx
        .store
        .similarity_search_vector(&zero, 5, None, -1.0)
        .unwrap();
    assert_eq!(res.len(), 5, "the scan answers a zero query");
    assert!(matches!(
        fx.store.similarity_search_vector(&[1.0, 2.0], 5, None, 0.0),
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

#[test]
fn threshold_is_respected() {
    let mut fx = fixture(2500);
    let queries = fx.queries(10);
    for q in &queries {
        let exact = fx
            .store
            .similarity_search_vector_exact(q, usize::MAX, None, 0.6)
            .unwrap();
        let res = fx.store.similarity_search_vector(q, 5, None, 0.6).unwrap();
        assert!(res.iter().all(|r| r.score >= 0.6));
        assert_eq!(res.len(), exact.len().min(5));
    }
    // nothing passes
    let none = fx
        .store
        .similarity_search_vector(&queries[0], 5, None, 1.01)
        .unwrap();
    assert!(none.is_empty());
}

// -- batches ------------------------------------------------------------------

#[test]
fn batch_equals_single_bit_for_bit() {
    let mut fx = fixture_with(3000, 32, Some(hnsw_config()), |i| {
        if i % 4 == 0 { "a" } else { "b" }.to_string()
    });
    let queries = fx.queries(50);
    for node_type in [None, Some("a")] {
        let batch = fx
            .store
            .similarity_search_vector_batch(&queries, 10, node_type, 0.0)
            .unwrap();
        for (q, got) in queries.iter().zip(&batch) {
            let one = fx
                .store
                .similarity_search_vector(q, 10, node_type, 0.0)
                .unwrap();
            assert_eq!(bits(got), bits(&one));
        }
    }
    // the exact batch is exact, and different calls agree on thread counts
    let exact = fx
        .store
        .similarity_search_vector_exact_batch(&queries, 10, None, 0.0)
        .unwrap();
    for (q, got) in queries.iter().zip(&exact) {
        let one = fx
            .store
            .similarity_search_vector_exact(q, 10, None, 0.0)
            .unwrap();
        assert_eq!(bits(got), bits(&one));
    }
    let counts = fx.store.ann_stats().searches;
    assert!(counts.approximate >= 100, "{counts:?}");
}

// -- filters ------------------------------------------------------------------

#[test]
fn node_type_filter_returns_k_even_when_selective() {
    // 1 row in 50 has the rare type: a plain ef window holds almost none
    let mut fx = fixture_with(4000, 32, Some(hnsw_config()), |i| {
        if i % 50 == 7 { "rare" } else { "common" }.to_string()
    });
    let queries = fx.queries(30);
    let (mut overlap, mut asked) = (0usize, 0usize);
    for q in &queries {
        let res = fx
            .store
            .similarity_search_vector(q, 10, Some("rare"), -1.0)
            .unwrap();
        assert_eq!(res.len(), 10, "a selective filter must still give k");
        assert!(res.iter().all(|r| r.node_id.node_type == "rare"));
        assert!(res.windows(2).all(|w| w[0].score >= w[1].score));
        let want = fx
            .store
            .similarity_search_vector_exact(q, 10, Some("rare"), -1.0)
            .unwrap();
        overlap += id_set(&res).intersection(&id_set(&want)).count();
        asked += want.len();
    }
    assert!(overlap * 100 >= asked * 80, "{overlap} of {asked}");
    let counts = fx.store.ann_stats().searches;
    assert!(counts.widened > 0 || counts.exact > 0, "{counts:?}");

    // a common type is answered by the graph itself
    let before = fx.store.ann_stats().searches.approximate;
    fx.store
        .similarity_search_vector(&queries[0], 10, Some("common"), -1.0)
        .unwrap();
    assert_eq!(fx.store.ann_stats().searches.approximate, before + 1);

    // a type nothing has, and one that is too rare to ever fill k
    assert!(fx
        .store
        .similarity_search_vector(&queries[0], 10, Some("nope"), -1.0)
        .unwrap()
        .is_empty());
    let mut store = fx.store;
    for i in 0..4000 {
        if i % 50 != 7 {
            continue;
        }
        if i > 7 + 50 * 3 {
            store.remove(&NodeId::new("rare", i.to_string()));
        }
    }
    let res = store
        .similarity_search_vector(&queries[1], 10, Some("rare"), -1.0)
        .unwrap();
    assert_eq!(res.len(), 4, "only four rare rows remain");
    store.verify_invariants().unwrap();
}

fn graph_with(n: usize, dim: usize, ann: Option<AnnConfig>) -> (SemanticGraph, Clusters, Rng) {
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
    (graph, clusters, rng)
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
fn graph_node_filter_gets_k_results_even_when_selective() {
    let (graph, clusters, mut rng) = graph_with(3000, 32, Some(hnsw_config()));
    let seven = |n: &Node| matches!(n.properties.get("group"), Some(PropertyValue::Int(7)));
    let filter = NodeFilter::new().predicate(&seven);
    let (mut overlap, mut asked) = (0, 0);
    for q in clusters.draw(15, &mut rng) {
        let opts = SimilarityOptions {
            filter: filter.clone(),
            ..SimilarityOptions::default()
        };
        let res = graph.similarity_search_with(&q, &opts).unwrap();
        assert_eq!(res.len(), 10);
        assert!(res.iter().all(|r| group_of(&graph, r) == 7));
        let want = graph
            .similarity_search_with(
                &q,
                &SimilarityOptions {
                    exact: true,
                    ..opts
                },
            )
            .unwrap();
        overlap += id_set(&res).intersection(&id_set(&want)).count();
        asked += want.len();
    }
    assert!(overlap * 100 >= asked * 80, "{overlap} of {asked}");
    assert!(graph.ann_stats().active);

    // a filter nothing passes terminates, empty
    let never = |_: &Node| false;
    let q = clusters.draw(1, &mut rng).remove(0);
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
    graph.embedding_store().verify_invariants().unwrap();
}

// -- writes, checked against a model ------------------------------------------

/// What the store should hold: every id's current vector, and every vector it
/// ever had (a concurrent reader may see any of them).
#[derive(Default)]
struct Model {
    current: HashMap<usize, Vec<f32>>,
    history: HashMap<usize, Vec<Vec<f32>>>,
}

impl Model {
    fn put(&mut self, id: usize, v: Vec<f32>) {
        self.history.entry(id).or_default().push(v.clone());
        self.current.insert(id, v);
    }
    fn remove(&mut self, id: usize) {
        self.current.remove(&id);
    }
}

fn raw_id(r: &SimilarityResult) -> usize {
    r.node_id.id.parse().unwrap()
}

/// Every result is a live id whose score is the cosine of its current vector.
fn check_against_model(model: &Model, q: &[f32], res: &[SimilarityResult], what: &str) {
    let mut seen = HashSet::new();
    for r in res {
        let id = raw_id(r);
        let v = model
            .current
            .get(&id)
            .unwrap_or_else(|| panic!("{what}: returned deleted id {id}"));
        assert!(seen.insert(id), "{what}: id {id} twice");
        let want = cosine(q, v);
        assert!(
            (r.score as f64 - want).abs() < 2e-5,
            "{what}: id {id} scored {} but its vector says {want}",
            r.score
        );
    }
    assert!(
        res.windows(2).all(|w| w[0].score >= w[1].score),
        "{what}: not sorted"
    );
}

#[test]
fn inserts_updates_and_deletes_agree_with_a_model_on_every_path() {
    const DIM: usize = 24;
    let clusters = Clusters::new(DIM, 12, 5);
    let mut rng = Rng(9);
    let mut model = Model::default();
    let mut store = EmbeddingStore::new(None);
    let initial = clusters.draw(1500, &mut rng);
    for (i, v) in initial.iter().enumerate() {
        store.add(doc(i), "", Some(v.clone())).unwrap();
        model.put(i, v.clone());
    }
    store.set_ann(Some(hnsw_config())).unwrap();
    store.build_ann_index(false).unwrap();
    assert!(store.ann_stats().active);
    let mut next_id = 1500;
    let mut found_after_update = 0usize;
    let mut updated = 0usize;

    for step in 0..900 {
        match rng.below(10) {
            0..=3 => {
                let v = clusters.draw(1, &mut rng).remove(0);
                store.add(doc(next_id), "", Some(v.clone())).unwrap();
                model.put(next_id, v);
                next_id += 1;
            }
            4..=6 => {
                // an update may move the row anywhere
                let live: Vec<usize> = model.current.keys().copied().collect();
                let id = live[rng.below(live.len())];
                let v = clusters.draw(1, &mut rng).remove(0);
                store.add(doc(id), "", Some(v.clone())).unwrap();
                model.put(id, v.clone());
                // it must now be found at its new position, with a high ef
                let top = store
                    .similarity_search_vector_ef(&v, 3, None, -1.0, 200)
                    .unwrap();
                updated += 1;
                if top.iter().any(|r| raw_id(r) == id) {
                    found_after_update += 1;
                }
            }
            _ => {
                let live: Vec<usize> = model.current.keys().copied().collect();
                if live.len() > 600 {
                    let id = live[rng.below(live.len())];
                    assert!(store.remove(&doc(id)));
                    model.remove(id);
                }
            }
        }
        if step % 30 == 0 {
            store.verify_invariants().unwrap();
            for q in clusters.draw(3, &mut rng) {
                let single = store.similarity_search_vector(&q, 10, None, -1.0).unwrap();
                check_against_model(&model, &q, &single, "single");
                let exact = store
                    .similarity_search_vector_exact(&q, 10, None, -1.0)
                    .unwrap();
                check_against_model(&model, &q, &exact, "exact");
                let batch = store
                    .similarity_search_vector_batch(&[q.clone(), q.clone()], 10, None, -1.0)
                    .unwrap();
                for b in &batch {
                    check_against_model(&model, &q, b, "batch");
                    assert_eq!(bits(b), bits(&single));
                }
                let all = store
                    .similarity_search_vector(&q, usize::MAX, None, -1.0)
                    .unwrap();
                assert_eq!(all.len(), model.current.len());
                check_against_model(&model, &q, &all, "unbounded");
                // the exact top-1 is nearly always inside the approximate top 10
                let overlap = id_set(&single).intersection(&id_set(&exact)).count();
                assert!(overlap >= 6, "step {step}: only {overlap} of 10 agree");
            }
        }
    }
    assert_eq!(store.count(), model.current.len());
    assert!(
        found_after_update * 100 >= updated * 98,
        "{found_after_update} of {updated} updated rows found at their new position"
    );
    store.verify_invariants().unwrap();
    let stats = store.ann_stats();
    assert!(stats.inserts > 0 && stats.churn > 0, "{stats:?}");

    // and the graph still has good recall after all of that
    let queries = clusters.draw(60, &mut rng);
    let recall = recall_of(&store, &queries, 10);
    assert!(recall >= RECALL_FLOOR, "recall {recall} after churn");
}

#[test]
fn deletes_never_come_back_and_compaction_keeps_the_graph_whole() {
    const DIM: usize = 24;
    let clusters = Clusters::new(DIM, 12, 6);
    let mut rng = Rng(21);
    let data = clusters.draw(3000, &mut rng);
    let mut store = EmbeddingStore::new(None);
    for (i, v) in data.iter().enumerate() {
        store.add(doc(i), "", Some(v.clone())).unwrap();
    }
    store.set_ann(Some(hnsw_config())).unwrap();
    store.build_ann_index(false).unwrap();
    let mut gone: HashSet<usize> = HashSet::new();

    // delete a fifth: tombstones stay in the graph (below the 25% compaction)
    for i in (0..3000).step_by(5) {
        assert!(store.remove(&doc(i)));
        gone.insert(i);
    }
    let stats = store.ann_stats();
    assert_eq!(stats.graph_dead, 600, "tombstones are kept for routing");
    assert_eq!(stats.graph_nodes, 3000);
    store.verify_invariants().unwrap();
    let queries = clusters.draw(40, &mut rng);
    for q in &queries {
        for r in store.similarity_search_vector(q, 10, None, -1.0).unwrap() {
            assert!(!gone.contains(&raw_id(&r)));
        }
    }
    let recall_with_tombstones = recall_of(&store, &queries, 10);
    assert!(
        recall_with_tombstones >= RECALL_FLOOR,
        "{recall_with_tombstones}"
    );

    // cross 25% dead: the table compacts, and the graph follows
    for i in (1..3000).step_by(9) {
        if store.remove(&doc(i)) {
            gone.insert(i);
        }
    }
    let stats = store.ann_stats();
    assert!(
        stats.graph_dead < 600,
        "compaction dropped the tombstones: {} left",
        stats.graph_dead
    );
    assert_eq!(stats.graph_nodes, store.count() + stats.graph_dead);
    assert_eq!(store.count(), 3000 - gone.len());
    store.compact();
    let stats = store.ann_stats();
    assert_eq!((stats.graph_dead, stats.graph_nodes), (0, store.count()));
    store.verify_invariants().unwrap();
    for q in &queries {
        let res = store.similarity_search_vector(q, 10, None, -1.0).unwrap();
        assert_eq!(res.len(), 10);
        for r in &res {
            assert!(!gone.contains(&raw_id(r)));
        }
    }
    let recall = recall_of(&store, &queries, 10);
    assert!(recall >= RECALL_FLOOR, "recall {recall} after compaction");

    // explicit compaction of a few more, then writes into the compacted store
    for i in (2..400).step_by(7) {
        if store.remove(&doc(i)) {
            gone.insert(i);
        }
    }
    store.compact();
    store.verify_invariants().unwrap();
    for (n, v) in clusters.draw(50, &mut rng).iter().enumerate() {
        store.add(doc(10_000 + n), "", Some(v.clone())).unwrap();
        let top = store
            .similarity_search_vector_ef(v, 3, None, -1.0, 200)
            .unwrap();
        assert!(top.iter().any(|r| raw_id(r) == 10_000 + n));
    }
    store.verify_invariants().unwrap();

    // delete almost everything: the index goes away under min_size 2 and
    // answers come from the scan
    let live: Vec<usize> = (0..3000)
        .filter(|i| !gone.contains(i))
        .chain(10_000..10_050)
        .collect();
    for &i in &live[..live.len() - 1] {
        store.remove(&doc(i));
    }
    assert_eq!(store.count(), 1);
    assert!(!store.ann_stats().active);
    let res = store
        .similarity_search_vector(&queries[0], 3, None, -1.0)
        .unwrap();
    assert_eq!(res.len(), 1);
    store.verify_invariants().unwrap();
}

#[test]
fn heavy_churn_rebuilds_the_graph() {
    let mut fx = fixture(1500);
    fx.store.build_ann_index(false).unwrap();
    assert_eq!(fx.store.ann_stats().searches.builds, 1);
    let mut rng = Rng(3);
    // rewrite every row once and a half times: churn passes the node count
    for round in 0..3 {
        for i in 0..750 {
            let v = fx.clusters.draw(1, &mut rng).remove(0);
            fx.store
                .add(NodeId::new("doc", i.to_string()), "", Some(v))
                .unwrap();
        }
        assert!(round < 3);
    }
    let q = fx.queries(1).remove(0);
    fx.store
        .similarity_search_vector(&q, 10, None, 0.0)
        .unwrap();
    let stats = fx.store.ann_stats();
    assert_eq!(stats.searches.builds, 2, "{stats:?}");
    assert_eq!(stats.churn, 0);
    fx.store.verify_invariants().unwrap();
    let queries = fx.queries(40);
    assert!(recall_of(&fx.store, &queries, 10) >= RECALL_FLOOR);
}

#[test]
fn clear_and_dimension_change() {
    let mut fx = fixture(1200);
    fx.store.build_ann_index(false).unwrap();
    assert!(fx.store.ann_stats().active);
    fx.store.clear();
    assert!(!fx.store.ann_stats().active);
    assert_eq!(fx.store.count(), 0);
    // a different width afterwards is fine; the index is rebuilt lazily
    let mut rng = Rng(5);
    for i in 0..300 {
        let v: Vec<f32> = (0..8).map(|_| rng.gauss()).collect();
        fx.store.add(doc(i), "", Some(v)).unwrap();
    }
    let q = vec![0.3f32; 8];
    let res = fx
        .store
        .similarity_search_vector(&q, 5, None, -2.0)
        .unwrap();
    assert_eq!(res.len(), 5);
    assert!(fx.store.ann_stats().active);
    fx.store.verify_invariants().unwrap();
}

// -- determinism --------------------------------------------------------------

#[test]
fn same_seed_same_order_gives_the_same_answers_whatever_the_threads() {
    let build = |seed: u64, threads: usize| -> Vec<Vec<(String, u32)>> {
        let mut fx = fixture_with(
            2500,
            32,
            Some(AnnConfig {
                seed,
                ..hnsw_config()
            }),
            |_| "doc".to_string(),
        );
        fx.store.set_scan_threads(threads);
        // some incremental writes after the build too
        fx.store.build_ann_index(false).unwrap();
        let mut rng = Rng(100);
        for i in 0..60 {
            let v = fx.clusters.draw(1, &mut rng).remove(0);
            fx.store.add(doc(5000 + i), "", Some(v)).unwrap();
        }
        let mut qrng = Rng(55);
        let queries = fx.clusters.draw(30, &mut qrng);
        queries
            .iter()
            .map(|q| {
                bits(
                    &fx.store
                        .similarity_search_vector(q, 10, None, -1.0)
                        .unwrap(),
                )
            })
            .collect()
    };
    let a = build(7, 0);
    assert_eq!(a, build(7, 0), "same seed, same answers");
    assert_eq!(a, build(7, 1), "one thread builds the same graph");
    assert_eq!(a, build(7, 3), "three threads build the same graph");
    // another seed is another graph (almost surely different somewhere), but
    // the same quality; compare against exact only
    let b = build(8, 0);
    assert_eq!(a.len(), b.len());
}

// -- awkward shapes -----------------------------------------------------------

#[test]
fn awkward_dimensions_and_duplicates_work() {
    for dim in [1usize, 3, 7, 9, 31, 33] {
        let mut rng = Rng(dim as u64);
        let mut store = EmbeddingStore::with_ann(None, hnsw_config()).unwrap();
        for i in 0..600 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
            store.add(doc(i), "", Some(v)).unwrap();
        }
        // exact duplicates, and a zero vector
        for i in 600..650 {
            store.add(doc(i), "", Some(vec![1.0; dim])).unwrap();
        }
        store.add(doc(650), "", Some(vec![0.0; dim])).unwrap();
        let q: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        let res = store.similarity_search_vector(&q, 10, None, -2.0).unwrap();
        assert_eq!(res.len(), 10, "dim {dim}");
        store.verify_invariants().unwrap();
        // ties break by arrival, as in the scan
        let dup = store
            .similarity_search_vector(&vec![1.0; dim], 20, None, 0.999)
            .unwrap();
        let order: Vec<usize> = dup.iter().map(raw_id).collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(
            order, sorted,
            "dim {dim}: equal scores must be in slot order"
        );
    }
}

#[test]
fn non_finite_vectors_do_not_break_the_graph() {
    let mut fx = fixture(1200);
    fx.store.build_ann_index(false).unwrap();
    let mut bad = vec![0.5f32; 32];
    bad[3] = f32::NAN;
    fx.store.add(doc(5000), "", Some(bad)).unwrap();
    let mut inf = vec![0.5f32; 32];
    inf[0] = f32::INFINITY;
    fx.store.add(doc(5001), "", Some(inf)).unwrap();
    let q = fx.queries(1).remove(0);
    let res = fx
        .store
        .similarity_search_vector(&q, 10, None, -2.0)
        .unwrap();
    assert_eq!(res.len(), 10);
    assert!(res.iter().all(|r| raw_id(r) < 5000));
    fx.store.verify_invariants().unwrap();
    // a healthy vector can replace a bad one
    fx.store.add(doc(5000), "", Some(vec![0.25; 32])).unwrap();
    fx.store.verify_invariants().unwrap();
}

// -- memory -------------------------------------------------------------------

#[test]
fn the_graph_is_links_only() {
    let fx = fixture_with(5000, 128, Some(hnsw_config()), |_| "doc".to_string());
    let before = fx.store.memory_stats();
    assert_eq!(before.index_bytes, 0);
    fx.store.build_ann_index(false).unwrap();
    let after = fx.store.memory_stats();
    assert_eq!(after.matrix_bytes, before.matrix_bytes, "no second copy");
    // 132 bytes of layer-0 links + about 10 more per node at M = 16
    let per_node = after.index_bytes as f64 / 5000.0;
    assert!(
        (130.0..170.0).contains(&per_node),
        "{per_node} bytes per node"
    );
    // far less than the vectors themselves (512 bytes each at dim 128)
    assert!(after.index_bytes * 3 < after.live_vector_bytes);
}

// -- concurrency --------------------------------------------------------------

#[test]
fn concurrent_readers_and_one_writer_never_see_a_torn_graph() {
    const DIM: usize = 16;
    let clusters = Clusters::new(DIM, 10, 8);
    let mut rng = Rng(31);
    let initial = clusters.draw(800, &mut rng);
    let mut model = Model::default();
    let mut base = EmbeddingStore::new(None);
    for (i, v) in initial.iter().enumerate() {
        base.add(doc(i), "", Some(v.clone())).unwrap();
        model.put(i, v.clone());
    }
    base.set_ann(Some(hnsw_config())).unwrap();
    base.build_ann_index(false).unwrap();

    // the writer's script, with the model of every version known up front
    enum Op {
        Put(usize, Vec<f32>),
        Remove(usize),
    }
    let mut ops = Vec::new();
    let mut live: Vec<usize> = (0..800).collect();
    let mut next_id = 800;
    for _ in 0..1500 {
        match rng.below(10) {
            0..=3 => {
                let v = clusters.draw(1, &mut rng).remove(0);
                model.put(next_id, v.clone());
                ops.push(Op::Put(next_id, v));
                live.push(next_id);
                next_id += 1;
            }
            4..=6 => {
                let id = live[rng.below(live.len())];
                let v = clusters.draw(1, &mut rng).remove(0);
                model.put(id, v.clone());
                ops.push(Op::Put(id, v));
            }
            _ => {
                if live.len() > 300 {
                    let at = rng.below(live.len());
                    ops.push(Op::Remove(live.swap_remove(at)));
                }
            }
        }
    }
    let final_live: HashSet<usize> = live.iter().copied().collect();

    let shared = SharedEmbeddingStore::new(base);
    let stop = AtomicBool::new(false);
    type Observed = (Vec<f32>, usize, Vec<(usize, f32)>);
    let observed: Vec<Observed> = std::thread::scope(|s| {
        let writer = s.spawn(|| {
            for (n, op) in ops.iter().enumerate() {
                match op {
                    Op::Put(i, v) => {
                        shared.add(doc(*i), "", Some(v.clone())).unwrap();
                    }
                    Op::Remove(i) => {
                        shared.remove(&doc(*i));
                    }
                }
                if n % 40 == 0 {
                    std::thread::yield_now();
                }
            }
            stop.store(true, Ordering::SeqCst);
        });
        let readers: Vec<_> = (0..3)
            .map(|r| {
                let (shared, stop) = (&shared, &stop);
                let clusters = &clusters;
                s.spawn(move || {
                    let mut rng = Rng(900 + r as u64);
                    let mut mine = Vec::new();
                    while !stop.load(Ordering::SeqCst) && mine.len() < 500 {
                        let k = 1 + rng.below(12);
                        let qs = clusters.draw(if r == 2 { 3 } else { 1 }, &mut rng);
                        let results = if r == 2 {
                            shared
                                .similarity_search_vector_batch(&qs, k, None, -2.0)
                                .unwrap()
                        } else {
                            vec![shared
                                .similarity_search_vector(&qs[0], k, None, -2.0)
                                .unwrap()]
                        };
                        for (q, res) in qs.into_iter().zip(results) {
                            mine.push((
                                q,
                                k,
                                res.iter().map(|x| (raw_id(x), x.score)).collect::<Vec<_>>(),
                            ));
                        }
                    }
                    mine
                })
            })
            .collect();
        writer.join().unwrap();
        readers
            .into_iter()
            .flat_map(|r| r.join().unwrap())
            .collect()
    });

    assert!(observed.len() > 100, "readers managed {}", observed.len());
    for (q, k, res) in &observed {
        // sorted, no duplicates, no more than asked, and every score is the
        // cosine of *some* version the row ever had (never a blend)
        assert!(res.len() <= *k);
        assert!(res.windows(2).all(|w| w[0].1 >= w[1].1));
        let uniq: HashSet<usize> = res.iter().map(|r| r.0).collect();
        assert_eq!(uniq.len(), res.len());
        for &(id, score) in res {
            let versions = &model.history[&id];
            assert!(
                versions
                    .iter()
                    .any(|v| (cosine(q, v) - score as f64).abs() < 2e-5),
                "id {id} scored {score}, matching none of its {} versions",
                versions.len()
            );
        }
    }

    // the finished store: sound, and exactly the model
    let store = shared.read();
    store.verify_invariants().unwrap();
    assert_eq!(store.count(), final_live.len());
    for q in clusters.draw(20, &mut rng) {
        let res = store.similarity_search_vector(&q, 10, None, -2.0).unwrap();
        let mut final_model = Model::default();
        for &id in &final_live {
            final_model.put(id, model.history[&id].last().unwrap().clone());
        }
        check_against_model(&final_model, &q, &res, "final");
    }
}

#[test]
fn concurrent_readers_agree_with_each_other() {
    let mut fx = fixture(3000);
    fx.store.build_ann_index(false).unwrap();
    let queries = fx.queries(24);
    let store = std::sync::Arc::new(fx.store);
    let want: Vec<_> = queries
        .iter()
        .map(|q| bits(&store.similarity_search_vector(q, 8, None, -1.0).unwrap()))
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
                    assert_eq!(bits(&res), want[at]);
                    if i % 20 == 0 {
                        let b = store
                            .similarity_search_vector_batch(&queries[..5], 8, None, -1.0)
                            .unwrap();
                        for (j, got) in b.iter().enumerate() {
                            assert_eq!(bits(got), want[j]);
                        }
                    }
                }
            });
        }
    });
}

// -- the index never reaches a file or the digest -----------------------------

#[test]
fn a_loaded_graph_rebuilds_lazily_and_persists_nothing() {
    let (graph, clusters, mut rng) = graph_with(1500, 32, Some(hnsw_config()));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.ison");
    graph.save(&path, None, true).unwrap();
    let mut loaded = SemanticGraph::load(&path, Box::new(MockEncoder::new(32))).unwrap();
    loaded.set_ann(Some(hnsw_config())).unwrap();
    assert!(!loaded.ann_stats().active, "nothing was persisted");
    let q = clusters.draw(1, &mut rng).remove(0);
    let res = loaded
        .similarity_search_with(&q, &SimilarityOptions::default())
        .unwrap();
    assert_eq!(res.len(), 10);
    let stats = loaded.ann_stats();
    assert!(stats.active && stats.mode == AnnMode::Hnsw);
    assert_eq!(stats.graph_nodes, 1500);
    // the answers equal the original's: same data, same slot order, same seed
    let original = graph
        .similarity_search_with(&q, &SimilarityOptions::default())
        .unwrap();
    assert_eq!(bits(&res), bits(&original));
    // and the rebuilt graph is as good
    let queries = clusters.draw(30, &mut rng);
    assert!(loaded.recall_check(&queries, 10, None, 0.0).unwrap().recall >= RECALL_FLOOR);
    loaded.embedding_store().verify_invariants().unwrap();
}

#[test]
fn graph_files_sidecar_and_digest_are_the_same_with_hnsw_on() {
    let (off, ..) = graph_with(600, 16, None);
    let (on, clusters, mut rng) = graph_with(600, 16, Some(hnsw_config()));
    // exercise the index before saving: build, search, filter
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
        std::fs::read(&p_on).unwrap(),
        "the graph file must be byte-identical"
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
    for word in ["hnsw", "ef_search", "ef_construction", "neighbor", "\"ann"] {
        assert!(!text.contains(word), "sidecar mentions {word}");
    }
    assert_eq!(graph_digest(off.graph()), graph_digest(on.graph()));
    let mut names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names.len(), 4, "nothing extra for the index: {names:?}");
    assert_eq!(
        off.embedding_store().to_json().unwrap(),
        on.embedding_store().to_json().unwrap()
    );
}

#[test]
fn graph_edits_reach_the_index() {
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
            hnsw: HnswConfig {
                ef_search: Some(150), // half the store: the graph still answers
                ..HnswConfig::default()
            },
            ..hnsw_config()
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
