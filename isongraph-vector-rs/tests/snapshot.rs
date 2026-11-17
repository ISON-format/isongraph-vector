//! The optional binary snapshot beside the JSON sidecar: a load from it is the
//! JSON load bit for bit; anything doubtful (corrupt, truncated, stale, for a
//! different graph) is refused and the JSON is read instead; the graph file and
//! the sidecar bytes are what they always were.

use isongraph_vector_rs::{
    snapshot_path_for, AnnConfig, EmbeddingError, EmbeddingStore, GraphMismatch, LoadOptions,
    LoadSource, MockEncoder, NodeId, ScanPrecision, SemanticGraph, SidecarId, SnapshotIdentity,
    SNAPSHOT_SUFFIX,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

const DIM: usize = 12;

fn encoder() -> Box<dyn isongraph_vector_rs::EmbeddingEncoder> {
    Box::new(MockEncoder::new(DIM))
}

/// A graph whose store holds `n` rows of three types, ids with colons and
/// unicode, text of every shape (empty, quotes, newlines), exact duplicates
/// (ties), two models, updates, and tombstoned rows left behind by deletes.
fn graph(n: usize, seed: u64) -> SemanticGraph {
    let mut g = SemanticGraph::new("g", encoder());
    g.add_node_with_embed("person", "1", vec![("name", "Alice")], None)
        .unwrap();
    g.add_node_with_embed("person", "2", vec![("name", "Bob")], None)
        .unwrap();
    g.add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])
        .unwrap();
    let mut rng = Rng(seed);
    let types = ["doc", "note", "a:b"];
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for i in 0..n {
        let v: Vec<f32> = if i % 17 == 5 && !vectors.is_empty() {
            vectors[i / 2].clone() // an exact duplicate: a tie
        } else if i % 41 == 7 {
            vec![0.0; DIM] // a zero vector
        } else {
            (0..DIM).map(|_| rng.f()).collect()
        };
        vectors.push(v.clone());
        if i == n / 2 {
            g.set_encoder(Box::new(
                MockEncoder::new(DIM).with_model_name("second-model"),
            ));
        }
        let text = match i % 5 {
            0 => String::new(),
            1 => format!("quote \" and \\ and\nnewline {i}"),
            2 => format!("unicode \u{1F600} \u{00E9}\u{4E2D} {i}"),
            _ => format!("text {i}"),
        };
        let id = if i % 9 == 0 {
            format!("id:with:colons:{i}")
        } else {
            format!("{i:05}")
        };
        g.embedding_store_mut()
            .add(NodeId::new(types[i % 3], id), &text, Some(v))
            .unwrap();
    }
    // updates and deletes leave rewritten rows and tombstones behind
    for i in (0..n).step_by(11) {
        let id = if i % 9 == 0 {
            format!("id:with:colons:{i}")
        } else {
            format!("{i:05}")
        };
        let v: Vec<f32> = (0..DIM).map(|_| rng.f()).collect();
        g.embedding_store_mut()
            .add(NodeId::new(types[i % 3], id), "updated", Some(v))
            .unwrap();
    }
    for i in (3..n).step_by(7) {
        let id = if i % 9 == 0 {
            format!("id:with:colons:{i}")
        } else {
            format!("{i:05}")
        };
        g.embedding_store_mut()
            .remove(&NodeId::new(types[i % 3], id));
    }
    g
}

fn path_in(dir: &tempfile::TempDir, name: &str) -> PathBuf {
    dir.path().join(name)
}

/// Copy a saved pair (graph + sidecar, no snapshot) to a fresh name, so it can
/// only be loaded from the JSON.
fn json_only_copy(dir: &tempfile::TempDir, from: &Path, name: &str) -> PathBuf {
    let to = path_in(dir, name);
    std::fs::copy(from, &to).unwrap();
    std::fs::copy(
        SemanticGraph::embeddings_path_for(from),
        SemanticGraph::embeddings_path_for(&to),
    )
    .unwrap();
    to
}

fn load(path: &Path) -> (SemanticGraph, isongraph_vector_rs::LoadReport) {
    SemanticGraph::load_with_report(path, encoder(), &LoadOptions::default()).unwrap()
}

fn queries(count: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng(555);
    (0..count)
        .map(|_| (0..DIM).map(|_| rng.f()).collect())
        .collect()
}

fn search_bits(
    store: &EmbeddingStore,
    q: &[f32],
    k: usize,
    ty: Option<&str>,
) -> Vec<(String, u32, String)> {
    store
        .similarity_search_vector(q, k, ty, -1.0)
        .unwrap()
        .into_iter()
        .map(|r| {
            (
                format!("{}:{}", r.node_id.node_type, r.node_id.id),
                r.score.to_bits(),
                r.text,
            )
        })
        .collect()
}

/// Everything observable about two stores is the same.
fn assert_same_store(a: &EmbeddingStore, b: &EmbeddingStore) {
    a.verify_invariants().unwrap();
    b.verify_invariants().unwrap();
    assert_eq!(a.count(), b.count());
    assert_eq!(a.dimension(), b.dimension());
    assert_eq!(a.models(), b.models());
    assert_eq!(a.last_import_meta(), b.last_import_meta());
    assert_eq!(
        serde_json::to_string(&a.export_embeddings_with(Some("x"))).unwrap(),
        serde_json::to_string(&b.export_embeddings_with(Some("x"))).unwrap(),
        "export differs"
    );
    let ea = a.export_embeddings();
    for e in &ea.embeddings {
        let id = NodeId::new(&e.node_type, e.id.as_string());
        let (ra, rb) = (a.get(&id).unwrap(), b.get(&id).unwrap());
        assert_eq!(ra.text, rb.text);
        assert_eq!(ra.model, rb.model);
        let (va, vb): (Vec<u32>, Vec<u32>) = (
            ra.vector.iter().map(|x| x.to_bits()).collect(),
            rb.vector.iter().map(|x| x.to_bits()).collect(),
        );
        assert_eq!(va, vb, "vector bits of {:?}", id);
    }
    for q in queries(30) {
        for ty in [None, Some("doc"), Some("a:b")] {
            for k in [1, 5, 40, usize::MAX] {
                assert_eq!(
                    search_bits(a, &q, k, ty),
                    search_bits(b, &q, k, ty),
                    "k {k} {ty:?}"
                );
            }
        }
        let (ma, mb) = (a.score_map(&q).unwrap(), b.score_map(&q).unwrap());
        assert_eq!(ma.len(), mb.len());
        for (k, v) in ma {
            assert_eq!(v.to_bits(), mb[&k].to_bits());
        }
    }
}

#[test]
fn a_snapshot_load_equals_the_json_load_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    // 2600 rows: two full blocks of 1024 and a partial one, with holes and ties
    let g = graph(2600, 1);
    assert!(g.embedding_store().verify_invariants().is_ok());
    let path = path_in(&dir, "g.ison");
    g.save_with_snapshot(&path, None).unwrap();
    assert!(snapshot_path_for(&path).is_file());
    assert!(snapshot_path_for(&path)
        .to_string_lossy()
        .ends_with(SNAPSHOT_SUFFIX));

    let (from_snap, report) = load(&path);
    assert_eq!(
        report.source,
        LoadSource::Snapshot,
        "{:?}",
        report.snapshot_skipped
    );
    assert_eq!(report.snapshot_skipped, None);
    assert_eq!(report.embeddings, g.embedding_store().count());

    let json_path = json_only_copy(&dir, &path, "copy.ison");
    let (from_json, jreport) = load(&json_path);
    assert_eq!(jreport.source, LoadSource::Json);
    assert_eq!(
        jreport.snapshot_skipped, None,
        "no snapshot beside the copy"
    );

    assert_same_store(from_snap.embedding_store(), from_json.embedding_store());
    // the loaded graph is the same too
    assert_eq!(
        isongraph_vector_rs::graph_digest(from_snap.graph()),
        isongraph_vector_rs::graph_digest(from_json.graph())
    );
}

#[test]
fn snapshot_loaded_stores_search_the_same_with_ivf_and_int8_on() {
    let dir = tempfile::tempdir().unwrap();
    let g = graph(2200, 2);
    let path = path_in(&dir, "g.ison");
    g.save_with_snapshot(&path, None).unwrap();
    let (mut a, ra) = load(&path);
    let (mut b, rb) = load(&json_only_copy(&dir, &path, "c.ison"));
    assert_eq!(
        (ra.source, rb.source),
        (LoadSource::Snapshot, LoadSource::Json)
    );
    let ann = AnnConfig {
        nlist: Some(16),
        nprobe: Some(4),
        min_size: 0,
        ..AnnConfig::default()
    };
    for g in [&mut a, &mut b] {
        g.set_ann(Some(ann.clone())).unwrap();
        g.embedding_store_mut()
            .set_scan_precision(ScanPrecision::int8())
            .unwrap();
    }
    for q in queries(40) {
        assert_eq!(
            search_bits(a.embedding_store(), &q, 7, None),
            search_bits(b.embedding_store(), &q, 7, None)
        );
    }
    assert_same_store(a.embedding_store(), b.embedding_store());
}

#[test]
fn the_graph_file_and_the_json_sidecar_are_byte_identical_with_or_without_a_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let g = graph(300, 3);
    let plain = path_in(&dir, "plain.ison");
    let snap = path_in(&dir, "snap.ison");
    g.save(&plain, None, true).unwrap();
    g.save_with_snapshot(&snap, None).unwrap();
    assert!(!snapshot_path_for(&plain).exists());
    assert_eq!(
        std::fs::read(&plain).unwrap(),
        std::fs::read(&snap).unwrap()
    );
    let (sj_plain, sj_snap) = (
        std::fs::read(SemanticGraph::embeddings_path_for(&plain)).unwrap(),
        std::fs::read(SemanticGraph::embeddings_path_for(&snap)).unwrap(),
    );
    // the digest differs only through the file name (graph name), so compare
    // after normalising: the sidecars are otherwise the same bytes
    assert_eq!(sj_plain.len(), sj_snap.len());
    let text = |b: &[u8]| String::from_utf8(b.to_vec()).unwrap();
    assert_eq!(text(&sj_plain), text(&sj_snap));
    // the snapshot is much smaller than the JSON
    let snap_len = std::fs::metadata(snapshot_path_for(&snap)).unwrap().len();
    eprintln!("json {} bytes, snapshot {} bytes", sj_snap.len(), snap_len);
    assert!(snap_len < sj_snap.len() as u64);
}

fn saved(dir: &tempfile::TempDir, n: usize) -> (SemanticGraph, PathBuf) {
    let g = graph(n, 4);
    let path = path_in(dir, "g.ison");
    g.save_with_snapshot(&path, None).unwrap();
    (g, path)
}

/// Load after `damage` was applied to the snapshot file: the JSON must be
/// used, with a reason, and equal the undamaged load.
fn assert_falls_back(dir: &tempfile::TempDir, path: &Path, reason_has: &str) {
    let (loaded, report) = load(path);
    assert_eq!(report.source, LoadSource::Json, "damage accepted");
    let why = report.snapshot_skipped.clone().expect("a reason");
    assert!(
        why.contains(reason_has),
        "reason {why:?} should mention {reason_has:?}"
    );
    let reference = json_only_copy(dir, path, &format!("ref-{}.ison", reason_has.len()));
    let (expected, _) = load(&reference);
    assert_same_store(loaded.embedding_store(), expected.embedding_store());
}

#[test]
fn a_flipped_byte_anywhere_is_caught_by_the_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let (_g, path) = saved(&dir, 1500);
    let snap = snapshot_path_for(&path);
    let original = std::fs::read(&snap).unwrap();
    let mut at = vec![
        0usize,
        7,
        9,
        13,
        20,
        40,
        original.len() / 3,
        original.len() / 2,
    ];
    at.extend([
        original.len() - 40,
        original.len() - 33,
        original.len() - 32,
        original.len() - 1,
    ]);
    for position in at {
        let mut damaged = original.clone();
        damaged[position] ^= 0x10;
        std::fs::write(&snap, &damaged).unwrap();
        let (_, report) = load(&path);
        assert_eq!(
            report.source,
            LoadSource::Json,
            "flip at {position} accepted"
        );
        assert!(report.snapshot_skipped.is_some());
    }
    std::fs::write(&snap, &original).unwrap();
    let (_, report) = load(&path);
    assert_eq!(
        report.source,
        LoadSource::Snapshot,
        "{:?}",
        report.snapshot_skipped
    );
}

#[test]
fn a_float_flip_reports_a_checksum_mismatch_and_loads_the_json() {
    let dir = tempfile::tempdir().unwrap();
    let (_g, path) = saved(&dir, 1500);
    let snap = snapshot_path_for(&path);
    let mut bytes = std::fs::read(&snap).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    std::fs::write(&snap, &bytes).unwrap();
    assert_falls_back(&dir, &path, "checksum");
}

#[test]
fn truncated_snapshots_are_refused_at_every_length() {
    let dir = tempfile::tempdir().unwrap();
    let (_g, path) = saved(&dir, 1300);
    let snap = snapshot_path_for(&path);
    let original = std::fs::read(&snap).unwrap();
    let len = original.len();
    for cut in [
        0,
        1,
        7,
        8,
        11,
        31,
        33,
        100,
        500,
        len / 2,
        len - 33,
        len - 32,
        len - 31,
        len - 1,
    ] {
        std::fs::write(&snap, &original[..cut]).unwrap();
        let (_, report) = load(&path);
        assert_eq!(
            report.source,
            LoadSource::Json,
            "cut at {cut} of {len} accepted"
        );
        assert!(report.snapshot_skipped.is_some(), "cut {cut}");
    }
    // and trailing bytes
    let mut longer = original.clone();
    longer.push(0);
    std::fs::write(&snap, &longer).unwrap();
    assert_eq!(load(&path).1.source, LoadSource::Json);
}

fn reseal(mut bytes: Vec<u8>) -> Vec<u8> {
    let n = bytes.len() - 32;
    let sum = Sha256::digest(&bytes[..n]);
    bytes[n..].copy_from_slice(&sum);
    bytes
}

#[test]
fn a_well_formed_looking_but_wrong_snapshot_is_refused_even_with_a_valid_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let (_g, path) = saved(&dir, 1200);
    let snap = snapshot_path_for(&path);
    let original = std::fs::read(&snap).unwrap();
    let patch = |offset: usize, with: &[u8]| {
        let mut b = original.clone();
        b[offset..offset + with.len()].copy_from_slice(with);
        reseal(b)
    };
    // header: magic 0..8, version 8..12, flags 12..16, dim 16..20, block 20..24, count 24..32
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("bad magic", patch(0, b"NOTASNAP")),
        ("version 2", patch(8, &2u32.to_le_bytes())),
        ("unknown flag", patch(12, &0x8000_0001u32.to_le_bytes())),
        ("zero block size", patch(20, &0u32.to_le_bytes())),
        ("huge count", patch(24, &u64::MAX.to_le_bytes())),
        (
            "count past the data",
            patch(24, &(5_000_000u64).to_le_bytes()),
        ),
        ("count short by one", {
            let count = u64::from_le_bytes(original[24..32].try_into().unwrap()) - 1;
            patch(24, &count.to_le_bytes())
        }),
        ("dim changed", patch(16, &((DIM as u32) + 1).to_le_bytes())),
    ];
    for (name, bytes) in cases {
        std::fs::write(&snap, bytes).unwrap();
        let (_, report) = load(&path);
        assert_eq!(report.source, LoadSource::Json, "{name} accepted");
        assert!(report.snapshot_skipped.is_some(), "{name}");
    }
    // a store loaded straight from a damaged file is untouched by the failure
    let mut store = EmbeddingStore::new(None);
    store
        .add(NodeId::new("keep", "me"), "t", Some(vec![1.0; 4]))
        .unwrap();
    let mut b = original.clone();
    let mid = b.len() / 2;
    b[mid] ^= 0xFF;
    std::fs::write(&snap, b).unwrap();
    assert!(matches!(
        store.load_snapshot(&snap),
        Err(EmbeddingError::Payload(_))
    ));
    assert_eq!(store.count(), 1);
    assert!(store.get(&NodeId::new("keep", "me")).is_some());
    store.verify_invariants().unwrap();
    assert!(matches!(
        store.load_snapshot(path_in(&dir, "missing.snap")),
        Err(EmbeddingError::Io(_))
    ));
}

#[test]
fn a_stale_snapshot_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (g, path) = saved(&dir, 800);
    let snap = snapshot_path_for(&path);
    let sidecar = SemanticGraph::embeddings_path_for(&path);

    // 1. another save of the same graph with other vectors rewrote the JSON
    //    only (as another port or tool might)
    let mut other = EmbeddingStore::new(Some(encoder()));
    let mut rng = Rng(9);
    for e in g.embedding_store().export_embeddings().embeddings {
        let v: Vec<f32> = (0..DIM).map(|_| rng.f()).collect();
        other
            .add(
                NodeId::new(&e.node_type, e.id.as_string()),
                &e.text,
                Some(v),
            )
            .unwrap();
    }
    let old: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
    let recorded = old["graph_digest"].as_str().unwrap().to_string();
    other
        .save_embeddings_with(&sidecar, Some(&recorded))
        .unwrap();
    let (loaded, report) = load(&path);
    assert_eq!(report.source, LoadSource::Json);
    assert!(
        report
            .snapshot_skipped
            .as_deref()
            .unwrap()
            .contains("stale"),
        "{:?}",
        report.snapshot_skipped
    );
    // the new vectors were used, not the snapshot's
    let mut expected = EmbeddingStore::new(Some(encoder()));
    expected.load_embeddings(&sidecar, true).unwrap();
    assert_same_store(loaded.embedding_store(), &expected);

    // 2. a graph saved without a snapshot removes the old snapshot
    g.save_with_snapshot(&path, None).unwrap();
    assert!(snap.is_file());
    assert_eq!(load(&path).1.source, LoadSource::Snapshot);
    g.save(&path, None, true).unwrap();
    assert!(
        !snap.exists(),
        "a plain save must not leave a stale snapshot"
    );

    // 3. a snapshot copied beside another graph's file is for a different graph
    let (_g2, path2) = {
        let mut g2 = graph(800, 4);
        g2.add_node_with_embed("person", "3", vec![("name", "Carol")], None)
            .unwrap();
        let p = path_in(&dir, "other.ison");
        g2.save_with_snapshot(&p, None).unwrap();
        (g2, p)
    };
    g.save_with_snapshot(&path, None).unwrap();
    std::fs::copy(snapshot_path_for(&path), snapshot_path_for(&path2)).unwrap();
    let (_, report) = load(&path2);
    assert_eq!(report.source, LoadSource::Json);
    assert!(
        report
            .snapshot_skipped
            .as_deref()
            .unwrap()
            .contains("different graph"),
        "{:?}",
        report.snapshot_skipped
    );
}

#[test]
fn an_edited_graph_still_follows_the_graph_mismatch_policy() {
    let dir = tempfile::tempdir().unwrap();
    let (_g, path) = saved(&dir, 400);
    let text = std::fs::read_to_string(&path)
        .unwrap()
        .replace("Bob", "Rob");
    std::fs::write(&path, text).unwrap();
    // default: the JSON digest check fails the load, as before
    assert!(matches!(
        SemanticGraph::load_with_report(&path, encoder(), &LoadOptions::default()),
        Err(EmbeddingError::GraphEmbeddingMismatch(_))
    ));
    // warn: loads the JSON, the snapshot (for the old graph) is not used
    let opts = LoadOptions {
        on_mismatch: GraphMismatch::Warn,
        ..LoadOptions::default()
    };
    let (g, report) = SemanticGraph::load_with_report(&path, encoder(), &opts).unwrap();
    assert_eq!(report.source, LoadSource::Json);
    assert!(report.snapshot_skipped.unwrap().contains("different graph"));
    assert!(!g.embedding_store().warnings().is_empty());
}

#[test]
fn a_snapshot_that_cannot_be_written_leaves_the_json_pair_and_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let g = graph(100, 5);
    let path = path_in(&dir, "g.ison");
    // a directory sits where the snapshot goes: the rename must fail
    let snap = snapshot_path_for(&path);
    std::fs::create_dir(&snap).unwrap();
    std::fs::write(snap.join("inside"), "x").unwrap();
    let err = g.save_with_snapshot(&path, None).unwrap_err();
    assert!(matches!(err, EmbeddingError::Io(_)), "{err}");
    assert!(snap.join("inside").is_file(), "what was there is untouched");
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    // the JSON pair is complete and loads
    let (loaded, report) = load(&path);
    assert_eq!(report.source, LoadSource::Json);
    assert_eq!(
        loaded.embedding_store().count(),
        g.embedding_store().count()
    );
}

#[test]
fn a_failed_snapshot_write_keeps_the_previous_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = EmbeddingStore::new(None);
    for i in 0..40 {
        store
            .add(
                NodeId::new("doc", i.to_string()),
                "t",
                Some(vec![i as f32 + 1.0; 6]),
            )
            .unwrap();
    }
    let snap = path_in(&dir, "s.snap");
    let info = store
        .save_snapshot(&snap, &SnapshotIdentity::default())
        .unwrap();
    assert_eq!(info.count, 40);
    assert_eq!(info.bytes, std::fs::metadata(&snap).unwrap().len());
    let before = std::fs::read(&snap).unwrap();
    // writing into a directory that does not exist fails before touching it
    let bad = dir.path().join("no-such-dir").join("s.snap");
    assert!(store
        .save_snapshot(&bad, &SnapshotIdentity::default())
        .is_err());
    assert_eq!(std::fs::read(&snap).unwrap(), before);
    // and a rename onto a non-empty directory fails after the temp was written
    let blocked = path_in(&dir, "blocked");
    std::fs::create_dir(&blocked).unwrap();
    std::fs::write(blocked.join("inside"), "x").unwrap();
    assert!(store
        .save_snapshot(&blocked, &SnapshotIdentity::default())
        .is_err());
    assert!(blocked.join("inside").is_file());
    assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
}

#[test]
fn readers_never_see_a_torn_snapshot_while_it_is_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let snap = path_in(&dir, "s.snap");
    let build = |scale: f32, n: usize| {
        let mut s = EmbeddingStore::new(None);
        for i in 0..n {
            s.add(
                NodeId::new("doc", i.to_string()),
                "t",
                Some(vec![scale * (i as f32 + 1.0); 16]),
            )
            .unwrap();
        }
        s
    };
    let (a, b) = (build(1.0, 3000), build(-2.0, 2100));
    a.save_snapshot(&snap, &SnapshotIdentity::default())
        .unwrap();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let writer = {
        let (snap, stop) = (snap.clone(), stop.clone());
        std::thread::spawn(move || {
            for i in 0..40 {
                let s = if i % 2 == 0 { &b } else { &a };
                s.save_snapshot(&snap, &SnapshotIdentity::default())
                    .unwrap();
            }
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let readers: Vec<_> = (0..3)
        .map(|_| {
            let (snap, stop) = (snap.clone(), stop.clone());
            std::thread::spawn(move || {
                let (mut ok, mut transient) = (0usize, 0usize);
                while !stop.load(std::sync::atomic::Ordering::SeqCst) || ok == 0 {
                    let mut s = EmbeddingStore::new(None);
                    match s.load_snapshot(&snap) {
                        Ok(info) => {
                            // wholly one of the two stores
                            let (scale, n) = match info.count {
                                3000 => (1.0f32, 3000usize),
                                2100 => (-2.0, 2100),
                                other => panic!("a torn count {other}"),
                            };
                            assert_eq!(s.count(), n);
                            let v = s
                                .get_vector(&NodeId::new("doc", (n - 1).to_string()))
                                .unwrap();
                            assert_eq!(v[0], scale * n as f32);
                            s.verify_invariants().unwrap();
                            ok += 1;
                        }
                        // a sharing violation while the file is swapped is not a torn read
                        Err(EmbeddingError::Io(_)) => transient += 1,
                        Err(e) => panic!("a reader saw an invalid snapshot: {e}"),
                    }
                }
                (ok, transient)
            })
        })
        .collect();
    writer.join().unwrap();
    for r in readers {
        let (ok, transient) = r.join().unwrap();
        eprintln!("reader: {ok} verified loads, {transient} transient open errors");
        assert!(ok > 0);
    }
}

#[test]
fn empty_and_tiny_stores_round_trip_and_options_are_respected() {
    let dir = tempfile::tempdir().unwrap();
    // empty embeddings
    let mut g = SemanticGraph::new("g", encoder());
    g.add_node_with_embed("person", "1", vec![("name", "Alice")], None)
        .unwrap();
    g.embedding_store_mut().clear();
    let path = path_in(&dir, "e.ison");
    g.save_with_snapshot(&path, None).unwrap();
    let (loaded, report) = load(&path);
    assert_eq!(
        report.source,
        LoadSource::Snapshot,
        "{:?}",
        report.snapshot_skipped
    );
    assert_eq!(loaded.embedding_store().count(), 0);
    loaded.embedding_store().verify_invariants().unwrap();
    // one row, jsonl format
    let mut g = SemanticGraph::new("g", encoder());
    g.embedding_store_mut()
        .add(NodeId::new("t", "only"), "one", Some(vec![0.5; DIM]))
        .unwrap();
    let path = path_in(&dir, "one.isonl");
    g.save_with_snapshot(&path, None).unwrap();
    let (loaded, report) = load(&path);
    assert_eq!(
        report.source,
        LoadSource::Snapshot,
        "{:?}",
        report.snapshot_skipped
    );
    assert_eq!(loaded.embedding_store().count(), 1);
    // embeddings off: nothing restored, no snapshot read
    let opts = LoadOptions {
        embeddings: false,
        ..LoadOptions::default()
    };
    let (none, report) = SemanticGraph::load_with_report(&path, encoder(), &opts).unwrap();
    assert_eq!(report.source, LoadSource::NoEmbeddings);
    assert_eq!(none.embedding_store().count(), 0);
    // plain load() / load_with() use the snapshot too
    let plain = SemanticGraph::load(&path, encoder()).unwrap();
    assert_eq!(plain.embedding_store().count(), 1);
}

#[test]
fn the_sidecar_identity_sees_a_rewritten_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = path_in(&dir, "f.json");
    std::fs::write(&p, vec![b'a'; 1_000_000]).unwrap();
    let id = SidecarId::of_file(&p).unwrap();
    assert_eq!(id, SidecarId::of_file(&p).unwrap());
    assert_eq!(id.len, 1_000_000);
    let mut changed = vec![b'a'; 1_000_000];
    changed[999_999] = b'b'; // the last byte
    std::fs::write(&p, &changed).unwrap();
    assert_ne!(id, SidecarId::of_file(&p).unwrap());
    changed[999_999] = b'a';
    changed[0] = b'b'; // the first byte
    std::fs::write(&p, &changed).unwrap();
    assert_ne!(id, SidecarId::of_file(&p).unwrap());
    std::fs::write(&p, vec![b'a'; 1_000_001]).unwrap();
    assert_ne!(id, SidecarId::of_file(&p).unwrap());
    // a tiny file
    std::fs::write(&p, b"x").unwrap();
    SidecarId::of_file(&p).unwrap();
}

#[test]
fn a_store_keeps_its_scan_precision_and_index_settings_across_a_snapshot_load() {
    let dir = tempfile::tempdir().unwrap();
    let g = graph(1500, 6);
    let snap = path_in(&dir, "s.snap");
    g.embedding_store()
        .save_snapshot(&snap, &SnapshotIdentity::default())
        .unwrap();

    let mut store = EmbeddingStore::new(None);
    store.set_scan_precision(ScanPrecision::int8()).unwrap();
    store
        .set_ann(Some(AnnConfig {
            nlist: Some(16),
            nprobe: Some(4),
            min_size: 0,
            ..AnnConfig::default()
        }))
        .unwrap();
    store
        .add(NodeId::new("old", "row"), "t", Some(vec![1.0; DIM]))
        .unwrap();
    let info = store.load_snapshot(&snap).unwrap();
    assert_eq!(info.count, g.embedding_store().count());
    assert!(store.get(&NodeId::new("old", "row")).is_none(), "replaced");
    assert_eq!(store.scan_precision(), ScanPrecision::int8());
    assert!(store.quant_bytes() > 0);
    assert!(store.ann().is_some());
    store.verify_invariants().unwrap();
    // and it searches like a JSON-loaded store with the same settings
    let mut json = EmbeddingStore::new(None);
    let side = path_in(&dir, "s.json");
    g.embedding_store().save_embeddings(&side).unwrap();
    json.load_embeddings(&side, true).unwrap();
    json.set_scan_precision(ScanPrecision::int8()).unwrap();
    json.set_ann(store.ann().cloned()).unwrap();
    for q in queries(20) {
        assert_eq!(
            search_bits(&store, &q, 6, None),
            search_bits(&json, &q, 6, None)
        );
    }
}
