//! Check a graph saved by any port: is its binary snapshot accepted, and is the
//! store it yields the same as a JSON-only load?
//!
//! Usage: cargo run --example snapshot_check -- <graph file> [dim]
//!
//! Prints `source=<Snapshot|Json|NoEmbeddings> skipped=<reason|none> count=<n> equal_to_json=<bool>`
//! and exits non-zero if the snapshot was not used or the stores differ. The
//! comparison is the store's JSON export, which carries every vector bit for
//! bit (shortest round-trip float text).

use isongraph_vector_rs::{
    snapshot_path_for, LoadOptions, LoadSource, MockEncoder, SemanticGraph,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let graph = args.next().ok_or("usage: snapshot_check <graph file> [dim]")?;
    let dim: usize = args.next().map(|d| d.parse()).transpose()?.unwrap_or(12);
    let enc = || Box::new(MockEncoder::new(dim));

    let (g, report) = SemanticGraph::load_with_report(&graph, enc(), &LoadOptions::default())?;

    // A JSON-only twin: copy graph + sidecar, leave the snapshot behind.
    let twin = format!("{graph}.jsononly.ison");
    std::fs::copy(&graph, &twin)?;
    std::fs::copy(
        SemanticGraph::embeddings_path_for(&graph),
        SemanticGraph::embeddings_path_for(&twin),
    )?;
    let _ = std::fs::remove_file(snapshot_path_for(&twin));
    let (j, jreport) = SemanticGraph::load_with_report(&twin, enc(), &LoadOptions::default())?;
    let equal = g.embedding_store().to_json()? == j.embedding_store().to_json()?;
    for p in [
        twin.clone(),
        SemanticGraph::embeddings_path_for(&twin).display().to_string(),
    ] {
        let _ = std::fs::remove_file(p);
    }

    println!(
        "source={:?} skipped={} count={} json_count={} equal_to_json={}",
        report.source,
        report.snapshot_skipped.as_deref().unwrap_or("none"),
        report.embeddings,
        jreport.embeddings,
        equal
    );
    if report.source != LoadSource::Snapshot || !equal {
        std::process::exit(1);
    }
    Ok(())
}
