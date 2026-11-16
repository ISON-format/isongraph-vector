//! Rust half of the graph-file cross-port check (see ../graph_files.py).
//!
//!     cross_port_graph write    <dir>
//!     cross_port_graph read     <dir> <observation.json>
//!     cross_port_graph mismatch <dir>
//!
//! `write` saves the shared fixture graph as <dir>/graph.ison plus its
//! embeddings sidecar. `read` loads it with GraphMismatch::Error and writes what
//! it found to <observation.json>. `mismatch` loads a graph whose file was
//! edited after the save and exits 0 only if the load fails with
//! EmbeddingError::GraphEmbeddingMismatch.
//!
//! NodeId is string-typed in this port, so ids 1 and 2 are written as "1" and
//! "2" (the other typed-id ports write integers); the orchestrator allows for
//! that.

use isongraph_vector_rs::{
    EmbeddingEncoder, EmbeddingError, GraphMismatch, LoadOptions, MockEncoder, NodeId,
    PropertyValue, SemanticGraph,
};
use serde_json::{json, Map, Value as Json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const DIM: usize = 384;

fn doc_text() -> String {
    // A backslash, spelled so no source-level escaping can mangle it.
    let bs = char::from(92u8);
    format!("Résumé of 日本語 engineers, \"quoted\" and back{bs}slash")
}

fn queries() -> Vec<String> {
    vec![
        "Bob analyses ML models".to_string(),
        "Carol plans roadmaps for über-teams".to_string(),
        doc_text(),
    ]
}

type Props = Vec<(&'static str, PropertyValue)>;

fn s(v: &str) -> PropertyValue {
    PropertyValue::String(v.to_string())
}

// Keep in step with the other graph_* helpers.
fn nodes() -> Vec<(&'static str, &'static str, Props)> {
    vec![
        (
            "person",
            "1",
            vec![
                ("text", s("Alice is a software engineer")),
                ("age", PropertyValue::Int(30)),
                ("score", PropertyValue::Float(0.5)),
                ("active", PropertyValue::Bool(true)),
                ("nickname", PropertyValue::Null),
            ],
        ),
        (
            "person",
            "2",
            vec![
                ("text", s("Bob analyses ML models")),
                ("age", PropertyValue::Int(41)),
                ("score", PropertyValue::Float(2.75)),
                ("active", PropertyValue::Bool(false)),
                ("nickname", PropertyValue::Null),
            ],
        ),
        (
            "person",
            "carol",
            vec![
                ("text", s("Carol plans roadmaps for über-teams")),
                ("age", PropertyValue::Int(35)),
                ("score", PropertyValue::Float(-1.25)),
                ("active", PropertyValue::Bool(true)),
                ("nickname", s("Caz")),
            ],
        ),
        (
            "doc",
            "résumé",
            vec![("text", PropertyValue::String(doc_text()))],
        ),
        (
            "topic",
            "ml",
            vec![
                ("text", s("machine learning models")),
                ("weight", PropertyValue::Float(0.125)),
            ],
        ),
    ]
}

type EdgeSpec = (
    &'static str,
    (&'static str, &'static str),
    (&'static str, &'static str),
    Props,
);

fn edges() -> Vec<EdgeSpec> {
    vec![
        (
            "KNOWS",
            ("person", "1"),
            ("person", "2"),
            vec![
                ("since", PropertyValue::Int(2020)),
                ("weight", PropertyValue::Float(0.75)),
            ],
        ),
        (
            "WROTE",
            ("person", "carol"),
            ("doc", "résumé"),
            vec![("note", s("first draft"))],
        ),
        ("ABOUT", ("doc", "résumé"), ("topic", "ml"), vec![]),
    ]
}

fn graph_path(dir: &str) -> PathBuf {
    Path::new(dir).join("graph.ison")
}

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn text_of(props: &[(&str, PropertyValue)]) -> String {
    match props.iter().find(|(k, _)| *k == "text") {
        Some((_, PropertyValue::String(t))) => t.clone(),
        _ => String::new(),
    }
}

fn write(dir: &str) -> Res<()> {
    let mut graph = SemanticGraph::new("fixture", Box::new(MockEncoder::new(DIM)));
    for (node_type, id, props) in nodes() {
        let text = text_of(&props);
        graph.add_node(node_type, id, props, Some(&text))?;
    }
    for (rel, source, target, props) in edges() {
        graph.add_edge(rel, source, target, props)?;
    }
    std::fs::create_dir_all(dir)?;
    graph.save(graph_path(dir), None, true)?;
    Ok(())
}

fn json_value(v: &PropertyValue) -> Json {
    match v {
        PropertyValue::Null => Json::Null,
        PropertyValue::Bool(b) => json!(b),
        PropertyValue::Int(i) => json!(i),
        PropertyValue::Float(f) => json!(f),
        PropertyValue::String(t) => json!(t),
        other => json!(format!("{other:?}")),
    }
}

fn json_props(props: &HashMap<String, PropertyValue>) -> Json {
    let mut keys: Vec<&String> = props.keys().collect();
    keys.sort();
    let mut map = Map::new();
    for k in keys {
        map.insert(k.clone(), json_value(&props[k]));
    }
    Json::Object(map)
}

fn observe(graph: &SemanticGraph, encoder: &MockEncoder) -> Json {
    let mut node_list = Vec::new();
    let mut missing = Vec::new();
    let mut worst = 0.0f64;
    for node in graph.graph().nodes() {
        node_list.push(json!({
            "type": node.node_type,
            "id": node.id,
            "props": json_props(&node.properties),
        }));
        let id = NodeId::new(node.node_type.clone(), node.id.clone());
        match graph.get_embedding(&id) {
            None => missing.push(json!([node.node_type, node.id])),
            Some(record) => {
                let text = match node.properties.get("text") {
                    Some(PropertyValue::String(t)) => t.clone(),
                    _ => String::new(),
                };
                let expected = encoder.encode(&text);
                if expected.len() != record.vector.len() {
                    worst = 1e9;
                } else {
                    for (a, b) in record.vector.iter().zip(expected.iter()) {
                        worst = worst.max((f64::from(*a) - f64::from(*b)).abs());
                    }
                }
            }
        }
    }
    let mut edge_list = Vec::new();
    let mut rels: Vec<&String> = graph.graph().edge_types();
    rels.sort();
    for rel in rels {
        for edge in graph.graph().edges_of_type(rel) {
            edge_list.push(json!({
                "rel": edge.rel_type,
                "source": [edge.source.node_type, edge.source.id],
                "target": [edge.target.node_type, edge.target.id],
                "props": json_props(&edge.properties),
            }));
        }
    }
    let searches: Vec<Json> = queries()
        .iter()
        .map(|query| {
            let hits = graph
                .similarity_search(query, 1, None, 0.0)
                .unwrap_or_default();
            match hits.first() {
                Some(hit) => json!({
                    "query": query,
                    "top": [hit.node_id.node_type, hit.node_id.id],
                    "score": f64::from(hit.score),
                }),
                None => json!({"query": query, "top": null, "score": null}),
            }
        })
        .collect();
    json!({
        "port": "rust",
        "nodes": node_list,
        "edges": edge_list,
        "embedding_count": graph.embedding_store().count(),
        "missing_embeddings": missing,
        "max_vec_diff": worst,
        "searches": searches,
    })
}

fn load(dir: &str) -> Result<SemanticGraph, EmbeddingError> {
    let options = LoadOptions {
        on_mismatch: GraphMismatch::Error,
        ..LoadOptions::default()
    };
    SemanticGraph::load_with(graph_path(dir), Box::new(MockEncoder::new(DIM)), &options)
}

fn run() -> Res<i32> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize| args.get(i).map(String::as_str);
    match (arg(1), arg(2), arg(3)) {
        (Some("write"), Some(dir), None) => write(dir)?,
        (Some("read"), Some(dir), Some(out)) => {
            let graph = load(dir)?;
            let observation = observe(&graph, &MockEncoder::new(DIM));
            std::fs::write(out, serde_json::to_string(&observation)?)?;
        }
        (Some("mismatch"), Some(dir), None) => {
            return Ok(match load(dir) {
                Err(EmbeddingError::GraphEmbeddingMismatch(_)) => 0,
                Err(other) => {
                    eprintln!("expected GraphEmbeddingMismatch, got: {other}");
                    1
                }
                Ok(_) => {
                    eprintln!("expected GraphEmbeddingMismatch, but the load succeeded");
                    1
                }
            });
        }
        _ => {
            eprintln!(
                "usage: cross_port_graph write <dir> | read <dir> <out.json> | mismatch <dir>"
            );
            return Ok(2);
        }
    }
    Ok(0)
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
