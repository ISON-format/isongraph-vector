//! SQLite-backed store, behind the `sqlite` feature:
//! `cargo test --features sqlite`

#![cfg(feature = "sqlite")]

use ison_graph_rs::NodeId;
use isongraph_vector_rs::{EmbeddingEncoder, EmbeddingError, MockEncoder, SqliteEmbeddingStore};

fn store(sqlite_vec: bool) -> SqliteEmbeddingStore {
    SqliteEmbeddingStore::open_in_memory(Some(Box::new(MockEncoder::new(32))), sqlite_vec).unwrap()
}

#[test]
fn adds_and_reads_back() {
    let mut s = store(false);
    s.add(NodeId::new("person", "1"), "Alice is an engineer", None)
        .unwrap();

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
    assert_eq!(
        s.get(&NodeId::new("a", "1")).unwrap().unwrap().text,
        "second"
    );
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
        .map(|i| (NodeId::new("doc", i.to_string()), "a document"))
        .collect();
    assert_eq!(s.add_batch(&entries).unwrap(), 10);
    assert_eq!(s.count().unwrap(), 10);
}

#[test]
fn survives_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");

    let probe = {
        let mut s =
            SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), false).unwrap();
        s.add(NodeId::new("person", "1"), "Alice", None).unwrap();
        s.get(&NodeId::new("person", "1")).unwrap().unwrap().vector
    };

    let reopened =
        SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), false).unwrap();
    assert_eq!(reopened.count().unwrap(), 1);
    assert_eq!(
        reopened
            .get(&NodeId::new("person", "1"))
            .unwrap()
            .unwrap()
            .vector,
        probe
    );
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
        .map(|i| (NodeId::new("n", i.to_string()), format!("text {i}")))
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
        let want = scan
            .similarity_search_vector(&query, 10, None, -1.0)
            .unwrap();
        let got = indexed
            .similarity_search_vector(&query, 10, None, -1.0)
            .unwrap();

        assert_eq!(want.len(), got.len());
        for (a, b) in want.iter().zip(got.iter()) {
            assert_eq!(a.node_id, b.node_id, "ordering differs for {probe}");
            assert!(
                (a.score - b.score).abs() < 1e-5,
                "{} vs {}",
                a.score,
                b.score
            );
        }
    }
}

#[test]
fn type_filter_stays_exact() {
    // Filtering after a top-k fetch would drop the rare node entirely; the
    // type is a vec0 metadata column so the search itself is narrowed.
    let mut s = store(true);
    for i in 0..50 {
        s.add(
            NodeId::new("crowd", i.to_string()),
            &format!("crowd {i}"),
            None,
        )
        .unwrap();
    }
    s.add(NodeId::new("rare", "1"), "the only rare node", None)
        .unwrap();

    let results = s
        .similarity_search("anything", 5, Some("rare"), -1.0)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, NodeId::new("rare", "1"));
}

#[test]
fn updates_and_deletes_reach_the_index() {
    let mut s = store(true);
    s.add(NodeId::new("a", "1"), "first", None).unwrap();
    s.add(NodeId::new("a", "2"), "second", None).unwrap();

    let second = s.get(&NodeId::new("a", "2")).unwrap().unwrap().vector;
    s.add(NodeId::new("a", "1"), "first", Some(second.clone()))
        .unwrap();

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
                .add(NodeId::new("n", i.to_string()), &format!("text {i}"), None)
                .unwrap();
        }
        plain.get(&NodeId::new("n", "7")).unwrap().unwrap().vector
    };

    let mut indexed =
        SqliteEmbeddingStore::open(&path, Some(Box::new(MockEncoder::new(32))), true).unwrap();
    assert_eq!(indexed.count().unwrap(), 20);

    let results = indexed
        .similarity_search_vector(&probe, 3, None, -1.0)
        .unwrap();
    assert_eq!(results[0].node_id, NodeId::new("n", "7"));
    assert!((results[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn score_map_still_covers_everything() {
    let mut s = store(true);
    for i in 0..30 {
        s.add(NodeId::new("n", i.to_string()), &format!("text {i}"), None)
            .unwrap();
    }
    let query = MockEncoder::new(32).encode("anything");
    assert_eq!(s.score_map(&query).unwrap().len(), 30);
}

#[test]
fn zero_query_vector_falls_back_to_the_scan() {
    let mut s =
        SqliteEmbeddingStore::open_in_memory(Some(Box::new(MockEncoder::new(4))), true).unwrap();
    s.add(NodeId::new("a", "1"), "x", Some(vec![1.0, 0.0, 0.0, 0.0]))
        .unwrap();

    let results = s
        .similarity_search_vector(&[0.0, 0.0, 0.0, 0.0], 1, None, -1.0)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].score, 0.0);
}

#[test]
fn round_trips_the_portable_payload() {
    let mut source = store(false);
    source
        .add(NodeId::new("person", "1"), "Alice", None)
        .unwrap();
    source
        .add(NodeId::new("account", "007"), "Bond", None)
        .unwrap();

    let json = source.to_json().unwrap();
    let mut target = store(true);
    assert_eq!(target.from_json(&json, false).unwrap(), 2);
    assert_eq!(
        target
            .get(&NodeId::new("account", "007"))
            .unwrap()
            .unwrap()
            .text,
        "Bond"
    );
}

// -- model lifecycle ---------------------------------------------------------

use isongraph_vector_rs::{ModelMismatch, Query, EMBEDDING_FORMAT};
use std::collections::HashMap;

fn models(pairs: &[(&str, usize)]) -> HashMap<String, usize> {
    pairs.iter().map(|(m, n)| (m.to_string(), *n)).collect()
}

fn model_store(model: &str, dim: usize, sqlite_vec: bool) -> SqliteEmbeddingStore {
    SqliteEmbeddingStore::open_in_memory(
        Some(Box::new(MockEncoder::new(dim).with_model_name(model))),
        sqlite_vec,
    )
    .unwrap()
}

#[test]
fn stamps_the_encoders_model_and_counts_them() {
    let mut s = model_store("model-a", 8, false);
    s.add(NodeId::new("n", "1"), "one", None).unwrap();
    s.add_batch(&[(NodeId::new("n", "2"), "two")]).unwrap();
    assert_eq!(
        s.get(&NodeId::new("n", "1")).unwrap().unwrap().model,
        "model-a"
    );
    assert_eq!(s.models().unwrap(), models(&[("model-a", 2)]));

    // A swap of encoder does not rewrite what is stored; new writes carry the
    // new name, and the counts follow overwrites and removals.
    s.set_encoder(Box::new(MockEncoder::new(8).with_model_name("model-b")));
    s.add(NodeId::new("n", "1"), "one again", None).unwrap();
    assert_eq!(
        s.models().unwrap(),
        models(&[("model-a", 1), ("model-b", 1)])
    );
    s.remove(&NodeId::new("n", "2")).unwrap();
    assert_eq!(s.models().unwrap(), models(&[("model-b", 1)]));
    s.clear().unwrap();
    assert!(s.models().unwrap().is_empty());

    // No encoder: a vector handed in is "provided".
    let mut bare = SqliteEmbeddingStore::open_in_memory(None, false).unwrap();
    bare.add(NodeId::new("n", "1"), "x", Some(vec![1.0, 0.0]))
        .unwrap();
    assert_eq!(bare.models().unwrap(), models(&[("provided", 1)]));
}

#[test]
fn stale_by_model_dimension_and_unknown() {
    let mut s = model_store("model-a", 8, false);
    s.add(NodeId::new("n", "2"), "two", None).unwrap();
    s.add(NodeId::new("n", "1"), "one", None).unwrap();
    assert!(s.stale(None, None, false).unwrap().is_empty());

    s.set_encoder(Box::new(MockEncoder::new(8).with_model_name("model-b")));
    assert_eq!(
        s.stale(None, None, false).unwrap(),
        vec![NodeId::new("n", "1"), NodeId::new("n", "2")]
    );
    assert!(s.stale(Some("model-a"), None, false).unwrap().is_empty());

    // Dimension counts even for an unknown model.
    s.set_encoder(Box::new(MockEncoder::new(16).with_model_name("model-a")));
    assert_eq!(s.stale(None, None, false).unwrap().len(), 2);

    let json = format!(
        r#"{{"format":"{EMBEDDING_FORMAT}","version":1,"dimension":2,"count":1,
            "embeddings":[{{"type":"n","id":"9","vector":[1.0,0.0]}}]}}"#
    );
    let mut u = model_store("model-a", 2, false);
    u.from_json(&json, true).unwrap();
    assert_eq!(u.models().unwrap(), models(&[("unknown", 1)]));
    assert!(u.stale(None, None, false).unwrap().is_empty());
    assert_eq!(
        u.stale(None, None, true).unwrap(),
        vec![NodeId::new("n", "9")]
    );

    // Nothing to judge by without an encoder or arguments.
    let bare = SqliteEmbeddingStore::open_in_memory(None, false).unwrap();
    assert!(bare.stale(None, None, true).unwrap().is_empty());
}

#[test]
fn text_query_against_another_model_warns_once_or_errors() {
    let mut s = model_store("model-a", 8, false);
    s.add(NodeId::new("n", "1"), "one", None).unwrap();
    assert!(s.warnings().is_empty());

    s.set_encoder(Box::new(MockEncoder::new(8).with_model_name("model-b")));
    s.similarity_search("one", 1, None, -1.0).unwrap();
    s.similarity_search("one", 1, None, -1.0).unwrap();
    assert_eq!(s.warnings().len(), 1);
    assert!(s.warnings()[0].contains("'model-a'"));

    // A vector query is never judged.
    s.clear_warnings();
    s.similarity_search_vector(&[1.0; 8], 1, None, -1.0)
        .unwrap();
    s.resolve_query(Query::Vector(&[1.0; 8])).unwrap();
    assert!(s.warnings().is_empty());

    s.set_on_model_mismatch(ModelMismatch::Error);
    assert_eq!(s.on_model_mismatch(), ModelMismatch::Error);
    assert!(matches!(
        s.similarity_search("one", 1, None, -1.0),
        Err(EmbeddingError::ModelMismatch(_))
    ));
    s.set_on_model_mismatch(ModelMismatch::Ignore);
    s.clear_warnings();
    s.similarity_search("one", 1, None, -1.0).unwrap();
    assert!(s.warnings().is_empty());

    // An encoder that does not name its model never triggers a mismatch.
    struct Nameless;
    impl EmbeddingEncoder for Nameless {
        fn dimension(&self) -> usize {
            8
        }
        fn encode(&self, text: &str) -> Vec<f32> {
            MockEncoder::new(8).encode(text)
        }
    }
    s.set_on_model_mismatch(ModelMismatch::Error);
    s.set_encoder(Box::new(Nameless));
    s.similarity_search("one", 1, None, -1.0).unwrap();
}

#[test]
fn clear_with_keeps_or_resets_the_dimension_and_the_index() {
    for sqlite_vec in [false, true] {
        let mut s = model_store("model-a", 8, sqlite_vec);
        s.add(NodeId::new("n", "1"), "one", None).unwrap();

        s.clear_with(false).unwrap();
        assert_eq!(s.dimension(), Some(8));
        assert!(matches!(
            s.add(NodeId::new("n", "1"), "x", Some(vec![1.0, 0.0])),
            Err(EmbeddingError::DimensionMismatch { .. })
        ));

        s.clear_with(true).unwrap();
        assert_eq!(s.dimension(), None);
        s.set_encoder(Box::new(MockEncoder::new(4).with_model_name("model-b")));
        s.add(NodeId::new("n", "1"), "one", None).unwrap();
        assert_eq!(s.dimension(), Some(4));
        let hits = s.similarity_search("one", 1, None, -1.0).unwrap();
        assert_eq!(hits.len(), 1, "sqlite_vec={sqlite_vec}");
        assert_eq!(s.models().unwrap(), models(&[("model-b", 1)]));
    }
}

#[test]
fn export_carries_encoder_and_digest_metadata_and_imports_it_back() {
    let mut s = model_store("model-a", 8, false);
    s.set_version(Some("rev-2".to_string()));
    s.add(NodeId::new("n", "1"), "one", None).unwrap();

    let plain = s.export_embeddings().unwrap();
    assert_eq!(plain.encoder_model.as_deref(), Some("model-a"));
    assert_eq!(plain.encoder_version.as_deref(), Some("rev-2"));
    assert_eq!(plain.graph_digest, None);
    assert_eq!(plain.embeddings[0].model.as_deref(), Some("model-a"));

    let tied = s.export_embeddings_with(Some("abc123")).unwrap();
    assert_eq!(tied.graph_digest.as_deref(), Some("abc123"));
    assert!(tied.graph_digest_basis.is_some());

    let mut t = model_store("model-a", 8, false);
    assert_eq!(t.last_import_meta().encoder_model, None);
    t.import_embeddings(&tied, true).unwrap();
    let meta = t.last_import_meta();
    assert_eq!(meta.encoder_model.as_deref(), Some("model-a"));
    assert_eq!(meta.encoder_version.as_deref(), Some("rev-2"));
    assert_eq!(meta.graph_digest.as_deref(), Some("abc123"));
    assert_eq!(t.models().unwrap(), models(&[("model-a", 1)]));
}

#[test]
fn saves_and_loads_the_sidecar_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("side.embeddings.json");
    let mut s = model_store("model-a", 8, false);
    s.add(NodeId::new("n", "1"), "one", None).unwrap();
    s.add(NodeId::new("n", "2"), "two", None).unwrap();
    assert_eq!(s.save_embeddings_with(&path, Some("d1")).unwrap(), 2);
    assert_eq!(s.save_embeddings(&path).unwrap(), 2);

    // No temp files left behind.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(leftovers.len(), 1, "{leftovers:?}");

    let mut t = model_store("model-a", 8, false);
    assert_eq!(t.load_embeddings(&path, true).unwrap(), 2);
    assert_eq!(t.count().unwrap(), 2);
    assert_eq!(
        t.last_import_meta().encoder_model.as_deref(),
        Some("model-a")
    );
    assert!(matches!(
        t.load_embeddings(dir.path().join("missing.json"), true),
        Err(EmbeddingError::Io(_))
    ));
}

// -- opening a database an older release or another port made ----------------

/// Table as an older release made it: no `node_id_type`, and optionally no
/// `model`.
fn legacy_db(path: &std::path::Path, with_model: bool) {
    let conn = rusqlite::Connection::open(path).unwrap();
    let model_col = if with_model { "model TEXT," } else { "" };
    conn.execute_batch(&format!(
        "CREATE TABLE embeddings (
             id TEXT PRIMARY KEY, node_type TEXT NOT NULL, node_id TEXT NOT NULL,
             vector BLOB NOT NULL, text TEXT NOT NULL, {model_col}
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
             UNIQUE(node_type, node_id));"
    ))
    .unwrap();
    let blob: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    conn.execute(
        "INSERT INTO embeddings (id, node_type, node_id, vector, text)
         VALUES ('a', 'n', '1', ?, 't')",
        [blob],
    )
    .unwrap();
}

#[test]
fn opens_a_database_with_no_model_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.db");
    legacy_db(&path, false);

    let mut s = SqliteEmbeddingStore::open(
        &path,
        Some(Box::new(MockEncoder::new(2).with_model_name("model-a"))),
        false,
    )
    .unwrap();
    assert_eq!(s.models().unwrap(), models(&[("unknown", 1)]));
    let rec = s.get(&NodeId::new("n", "1")).unwrap().unwrap();
    assert_eq!(rec.model, "unknown");
    // unknown vectors are not judged stale by model, and text search is quiet.
    assert!(s.stale(None, None, false).unwrap().is_empty());
    assert_eq!(s.stale(None, None, true).unwrap().len(), 1);
    s.similarity_search("t", 1, None, -1.0).unwrap();
    assert!(s.warnings().is_empty());
    assert_eq!(s.export_embeddings().unwrap().embeddings[0].id_type, "str");

    // New writes land beside the old row, stamped properly, and survive reopen.
    s.add(NodeId::new("n", "2"), "two", None).unwrap();
    drop(s);
    let s = SqliteEmbeddingStore::open(&path, None, false).unwrap();
    assert_eq!(
        s.models().unwrap(),
        models(&[("model-a", 1), ("unknown", 1)])
    );
}

#[test]
fn a_null_model_reads_as_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("null.db");
    legacy_db(&path, true);

    let s = SqliteEmbeddingStore::open(&path, None, false).unwrap();
    assert_eq!(s.models().unwrap(), models(&[("unknown", 1)]));
    assert_eq!(
        s.export_embeddings().unwrap().embeddings[0]
            .model
            .as_deref(),
        Some("unknown")
    );
}
