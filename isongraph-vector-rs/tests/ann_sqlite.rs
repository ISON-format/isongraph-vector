//! The approximate index and the SQLite store, behind the `sqlite` feature:
//! `cargo test --features sqlite`.
//!
//! `SqliteEmbeddingStore` is a separate, persistent store with no approximate
//! index of its own (its search is a scan, or sqlite-vec's exact KNN). What is
//! worth pinning down is the seam between the two: vectors that live in SQLite
//! can be loaded into an in-memory store with `ann` on, the index answers
//! from the same scores the SQLite store gives, and nothing about the index
//! leaks back into the portable payload.

#![cfg(feature = "sqlite")]

use ison_graph_rs::NodeId;
use isongraph_vector_rs::{AnnConfig, EmbeddingStore, ExportedEmbedding, SqliteEmbeddingStore};

fn vector(i: usize, dim: usize) -> Vec<f32> {
    // Ten loose clusters: a fixed direction per cluster plus a small,
    // deterministic wobble per vector.
    let mut state = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let cluster = i % 10;
    (0..dim)
        .map(|d| if d % 10 == cluster { 3.0 } else { 0.0 } + 0.4 * next())
        .collect()
}

#[test]
fn vectors_from_sqlite_are_searched_by_the_index_with_the_same_scores() {
    let dim = 40;
    let mut sqlite = SqliteEmbeddingStore::open_in_memory(None, false).unwrap();
    for i in 0..1500 {
        sqlite
            .add(NodeId::new("doc", i.to_string()), "", Some(vector(i, dim)))
            .unwrap();
    }
    let payload = sqlite.export_embeddings().unwrap();

    let mut indexed = EmbeddingStore::with_ann(
        None,
        AnnConfig {
            nlist: Some(20),
            nprobe: Some(4),
            min_size: 0,
            ..AnnConfig::default()
        },
    )
    .unwrap();
    assert_eq!(indexed.import_embeddings(&payload, true).unwrap(), 1500);
    assert_eq!(indexed.count(), sqlite.count().unwrap());

    let queries: Vec<Vec<f32>> = (0..20).map(|i| vector(5000 + i, dim)).collect();
    let report = indexed.recall_check(&queries, 10, None, 0.0).unwrap();
    assert!(report.approximate, "{report:?}");
    assert!(report.recall >= 0.95, "{report:?}");

    for q in &queries {
        let want = sqlite.similarity_search_vector(q, 10, None, 0.0).unwrap();
        // the full scan agrees with SQLite on the whole ranking, scores included
        let exact = indexed
            .similarity_search_vector_exact(q, 10, None, 0.0)
            .unwrap();
        assert_eq!(want.len(), exact.len());
        for (a, b) in want.iter().zip(&exact) {
            assert!(
                (a.score - b.score).abs() < 1e-5,
                "{} vs {}",
                a.score,
                b.score
            );
        }
        // and every approximate hit carries the score SQLite gives that node
        let truth: std::collections::HashMap<_, _> = sqlite
            .similarity_search_vector(q, usize::MAX, None, -1.0)
            .unwrap()
            .into_iter()
            .map(|r| (r.node_id, r.score))
            .collect();
        for r in indexed.similarity_search_vector(q, 10, None, 0.0).unwrap() {
            assert!((r.score - truth[&r.node_id]).abs() < 1e-5);
        }
    }
}

#[test]
fn the_index_changes_nothing_that_is_exported() {
    let dim = 24;
    let mut sqlite = SqliteEmbeddingStore::open_in_memory(None, false).unwrap();
    let mut plain = EmbeddingStore::new(None);
    let mut indexed = EmbeddingStore::with_ann(
        None,
        AnnConfig {
            nlist: Some(8),
            nprobe: Some(2),
            min_size: 0,
            ..AnnConfig::default()
        },
    )
    .unwrap();
    for i in 0..400 {
        let id = NodeId::new("doc", i.to_string());
        let v = vector(i, dim);
        sqlite.add(id.clone(), "t", Some(v.clone())).unwrap();
        plain.add(id.clone(), "t", Some(v.clone())).unwrap();
        indexed.add(id, "t", Some(v)).unwrap();
    }
    // build the index, then write and delete through it
    indexed
        .similarity_search_vector(&vector(3, dim), 5, None, 0.0)
        .unwrap();
    assert!(indexed.ann_stats().active);
    for store in [&mut plain, &mut indexed] {
        store
            .add(NodeId::new("doc", "7"), "t", Some(vector(900, dim)))
            .unwrap();
        store.remove(&NodeId::new("doc", "8"));
    }
    sqlite
        .add(NodeId::new("doc", "7"), "t", Some(vector(900, dim)))
        .unwrap();
    sqlite.remove(&NodeId::new("doc", "8")).unwrap();

    assert_eq!(plain.to_json().unwrap(), indexed.to_json().unwrap());
    // the SQLite store exports the same set of vectors
    let (a, b) = (
        indexed.export_embeddings(),
        sqlite.export_embeddings().unwrap(),
    );
    assert_eq!(a.count, b.count);
    let rows = |e: &[ExportedEmbedding]| {
        let mut rows: Vec<_> = e
            .iter()
            .map(|e| ((e.node_type.clone(), e.id.as_string()), e.vector.clone()))
            .collect();
        rows.sort_by(|x, y| x.0.cmp(&y.0));
        rows
    };
    assert_eq!(rows(&a.embeddings), rows(&b.embeddings));
}

#[test]
fn ann_and_the_sqlite_vec_index_are_independent() {
    // The Python port refuses ann together with sqlite_vec because one store
    // would own two indexes. Here they are different types: a SQLite store
    // (with or without vec0) never consults the in-memory index, and turning
    // the index on in memory does not touch it.
    let mut sqlite = SqliteEmbeddingStore::open_in_memory(None, true).unwrap();
    let mut indexed = EmbeddingStore::new(None);
    for i in 0..300 {
        let id = NodeId::new("doc", i.to_string());
        sqlite.add(id.clone(), "", Some(vector(i, 16))).unwrap();
        indexed.add(id, "", Some(vector(i, 16))).unwrap();
    }
    indexed
        .set_ann(Some(AnnConfig {
            min_size: 0,
            nlist: Some(10),
            nprobe: Some(10),
            ..AnnConfig::default()
        }))
        .unwrap();
    let q = vector(12, 16);
    let a = sqlite.similarity_search_vector(&q, 5, None, 0.0).unwrap();
    let b = indexed.similarity_search_vector(&q, 5, None, 0.0).unwrap();
    assert_eq!(
        a.iter().map(|r| r.node_id.clone()).collect::<Vec<_>>(),
        b.iter().map(|r| r.node_id.clone()).collect::<Vec<_>>()
    );
}
