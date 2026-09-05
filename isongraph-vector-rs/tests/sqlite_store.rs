//! SQLite-backed store, behind the `sqlite` feature:
//! `cargo test --features sqlite`

#![cfg(feature = "sqlite")]

use isongraph_vector_rs::{
    EmbeddingEncoder, EmbeddingError, MockEncoder, SqliteEmbeddingStore,
};
use ison_graph_rs::NodeId;

fn store(sqlite_vec: bool) -> SqliteEmbeddingStore {
    SqliteEmbeddingStore::open_in_memory(Some(Box::new(MockEncoder::new(32))), sqlite_vec).unwrap()
}

#[test]
fn adds_and_reads_back() {
    let mut s = store(false);
    s.add(NodeId::new("person", "1"), "Alice is an engineer", None).unwrap();

    let record = s.get(&NodeId::new("person", "1")).unwrap().unwrap();
    assert_eq!(record.text, "Alice is an engineer");
    assert_eq!(record.vector.len(), 32);
    assert_eq!(s.count().unwrap(), 1);
}

#[test]
fn removes_and_clears() {
    let mut s = store(false);
    s.add(NodeId::new("a", "1"), "one", None).unwrap();
    s.add(NodeId::new("a", "2"), "two", None).unwrap();

    assert!(s.remove(&NodeId::new("a", "1")).unwrap());
    assert!(!s.remove(&NodeId::new("a", "1")).unwrap());
    assert_eq!(s.count().unwrap(), 1);

    s.clear().unwrap();
    assert_eq!(s.count().unwrap(), 0);
}

#[test]
fn overwrites_rather_than_duplicating() {
    let mut s = store(false);
    s.add(NodeId::new("a", "1"), "first", None).unwrap();
    s.add(NodeId::new("a", "1"), "second", None).unwrap();
    assert_eq!(s.count().unwrap(), 1);
    assert_eq!(s.get(&NodeId::new("a", "1")).unwrap().unwrap().text, "second");
}

#[test]
fn validates_dimensions() {
    let mut s = store(false);
    s.add(NodeId::new("a", "1"), "full width", None).unwrap();

    assert!(matches!(
        s.add(NodeId::new("a", "2"), "short", Some(vec![1.0, 0.0])),
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        s.add(NodeId::new("a", "3"), "empty", Some(vec![])),
        Err(EmbeddingError::EmptyVector(_))
    ));
    assert!(matches!(
        s.similarity_search_vector(&[1.0, 0.0], 1, None, -1.0),
        Err(EmbeddingError::DimensionMismatch { .. })
    ));
}

#[test]
fn batches_in_one_transaction() {
    let mut s = store(false);
    let entries: Vec<(NodeId, &str)> = (0..10)
        .map(|i| (NodeId::new("doc", &i.to_string()), "a document"))
        .collect();
    assert_eq!(s.add_batch(&entries).unwrap(), 10);
    assert_eq!(s.count().unwrap(), 10);
}

#[test]
fn survives_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");

    let probe = {
        let mut s = SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), false)
            .unwrap();
        s.add(NodeId::new("person", "1"), "Alice", None).unwrap();
        s.get(&NodeId::new("person", "1")).unwrap().unwrap().vector
    };

    let reopened =
        SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), false).unwrap();
    assert_eq!(reopened.count().unwrap(), 1);
    assert_eq!(reopened.get(&NodeId::new("person", "1")).unwrap().unwrap().vector, probe);
    // dimension is recovered from the stored rows
    assert_eq!(reopened.dimension(), Some(32));
}

// =============================================================================
// sqlite-vec index
// =============================================================================

#[test]
fn reports_the_backend() {
    assert!(!store(false).sqlite_vec());
    assert!(store(true).sqlite_vec());
}

#[test]
fn index_matches_the_scan_exactly() {
    let encoder = MockEncoder::new(32);
    let entries: Vec<(NodeId, String)> = (0..200)
        .map(|i| (NodeId::new("n", &i.to_string()), format!("text {i}")))
        .collect();
    let refs: Vec<(NodeId, &str)> = entries
        .iter()
        .map(|(id, t)| (id.clone(), t.as_str()))
        .collect();

    let mut scan = store(false);
    let mut indexed = store(true);
    scan.add_batch(&refs).unwrap();
    indexed.add_batch(&refs).unwrap();

    for probe in ["text 0", "text 57", "text 199"] {
        let query = encoder.encode(probe);
        let want = scan.similarity_search_vector(&query, 10, None, -1.0).unwrap();
        let got = indexed.similarity_search_vector(&query, 10, None, -1.0).unwrap();

        assert_eq!(want.len(), got.len());
        for (a, b) in want.iter().zip(got.iter()) {
            assert_eq!(a.node_id, b.node_id, "ordering differs for {probe}");
            assert!((a.score - b.score).abs() < 1e-5, "{} vs {}", a.score, b.score);
        }
    }
}

#[test]
fn type_filter_stays_exact() {
    // Filtering after a top-k fetch would drop the rare node entirely; the
    // type is a vec0 metadata column so the search itself is narrowed.
    let mut s = store(true);
    for i in 0..50 {
        s.add(NodeId::new("crowd", &i.to_string()), &format!("crowd {i}"), None)
            .unwrap();
    }
    s.add(NodeId::new("rare", "1"), "the only rare node", None).unwrap();

    let results = s.similarity_search("anything", 5, Some("rare"), -1.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, NodeId::new("rare", "1"));
}

#[test]
fn updates_and_deletes_reach_the_index() {
    let mut s = store(true);
    s.add(NodeId::new("a", "1"), "first", None).unwrap();
    s.add(NodeId::new("a", "2"), "second", None).unwrap();

    let second = s.get(&NodeId::new("a", "2")).unwrap().unwrap().vector;
    s.add(NodeId::new("a", "1"), "first", Some(second.clone())).unwrap();

    let top = s.similarity_search_vector(&second, 2, None, -1.0).unwrap();
    assert_eq!(top.len(), 2);
    assert!(top.iter().all(|r| (r.score - 1.0).abs() < 1e-5));

    s.remove(&NodeId::new("a", "1")).unwrap();
    let after = s.similarity_search_vector(&second, 10, None, -1.0).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].node_id, NodeId::new("a", "2"));
}

#[test]
fn indexes_a_database_written_without_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("backfill.db");

    let probe = {
        let mut plain =
            SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), false).unwrap();
        for i in 0..20 {
            plain
                .add(NodeId::new("n", &i.to_string()), &format!("text {i}"), None)
                .unwrap();
        }
        plain.get(&NodeId::new("n", "7")).unwrap().unwrap().vector
    };

    let mut indexed =
        SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), true).unwrap();
    assert_eq!(indexed.count().unwrap(), 20);

    let results = indexed.similarity_search_vector(&probe, 3, None, -1.0).unwrap();
    assert_eq!(results[0].node_id, NodeId::new("n", "7"));
    assert!((results[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn score_map_still_covers_everything() {
    let mut s = store(true);
    for i in 0..30 {
        s.add(NodeId::new("n", &i.to_string()), &format!("text {i}"), None)
            .unwrap();
    }
    let query = MockEncoder::new(32).encode("anything");
    assert_eq!(s.score_map(&query).unwrap().len(), 30);
}

#[test]
fn zero_query_vector_falls_back_to_the_scan() {
    let mut s = SqliteEmbeddingStore::open_in_memory(
        Some(Box::new(MockEncoder::new(4))),
        true,
    )
    .unwrap();
    s.add(NodeId::new("a", "1"), "x", Some(vec![1.0, 0.0, 0.0, 0.0])).unwrap();

    let results = s.similarity_search_vector(&[0.0, 0.0, 0.0, 0.0], 1, None, -1.0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].score, 0.0);
}

#[test]
fn round_trips_the_portable_payload() {
    let mut source = store(false);
    source.add(NodeId::new("person", "1"), "Alice", None).unwrap();
    source.add(NodeId::new("account", "007"), "Bond", None).unwrap();

    let json = source.to_json().unwrap();
    let mut target = store(true);
    assert_eq!(target.from_json(&json, false).unwrap(), 2);
    assert_eq!(
        target.get(&NodeId::new("account", "007")).unwrap().unwrap().text,
        "Bond"
    );
}
