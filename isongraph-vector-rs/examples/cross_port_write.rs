//! Write a SQLite store and a portable JSON payload for another port to read.
//!
//! The six ports claim to share one SQLite schema and one payload format. This
//! example is the Rust half of the check that runs in CI; see
//! `scripts/cross_port/` for the rest. It is an example rather than a binary
//! so it stays out of the published crate's artifacts.
//!
//! Usage: cargo run --example cross_port_write --features sqlite -- <db> <json>

use isongraph_vector_rs::{MockEncoder, NodeId, SqliteEmbeddingStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: cross_port_write <db-path> <json-path>");
        std::process::exit(2);
    }
    let (db_path, json_path) = (&args[1], &args[2]);

    // Matches FIXTURES in node_write.mjs and py_write.py, except for the id
    // type: NodeId is string-typed in this port, so Rust writes id_type "str"
    // for every row. It reads and preserves "int" written by another port -
    // there is a test for that - but cannot originate one, so the integer-id
    // fixture is checked in the Node-to-Python direction instead.
    let fixtures = [
        (NodeId::new("person", "1"), "Alice is a software engineer"),
        (NodeId::new("person", "2"), "Bob analyses ML models"),
        (
            NodeId::new("doc", "r\u{e9}sum\u{e9}"),
            "Carol plans roadmaps for \u{fc}ber-teams",
        ),
    ];

    let mut store =
        SqliteEmbeddingStore::open(db_path, Some(Box::new(MockEncoder::new(384))), false)?;
    store.add_batch(&fixtures)?;
    std::fs::write(json_path, store.to_json()?)?;

    println!(
        "rust wrote {} embeddings to {db_path} and {json_path}",
        fixtures.len()
    );
    Ok(())
}
