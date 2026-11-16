//! Graph digest: a content hash of a graph's nodes and edges.
//!
//! Recorded in the embeddings sidecar next to a saved graph so `load` can tell
//! that the two files are from the same save. It has to be byte-for-byte what
//! the Python port computes (`graph_digest`), because a graph saved by one port
//! is loaded and verified by another. The digest hashes a canonical JSON
//! rendering (`GRAPH_DIGEST_BASIS`):
//!
//! ```text
//! {"directed":true,"edges":[<edge>,...],"nodes":[<node>,...]}
//! node = [[type,"i"|"s",id-text],{properties, keys sorted}]
//! edge = [rel_type,[type,"i"|"s",id-text],[type,"i"|"s",id-text],{properties}]
//! ```
//!
//! with each list sorted, compact separators, every non-ASCII character written
//! as a `\uXXXX` escape and floats written the way Python's `repr` writes them.
//!
//! Node ids are strings in this crate, but a graph file written by another port
//! can carry integer ids (`1` unquoted) next to string ones (`"1"` quoted) and
//! the digest tags the two differently. `graph_digest` therefore treats every id
//! as a string, which is right for anything built in Rust, and
//! `graph_digest_with_int_ids` takes the set of ids that are integers.
//! Saving and loading learn that set from the file text itself (see
//! `graph_digest_of_text`).

use crate::EmbeddingError;
use ison_graph_rs::{Edge, ISONGraph, NodeId, Value};
use ison_rs::{Block, Document, Value as IsonValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

/// Name of the serialization `graph_digest` hashes. Written next to the digest
/// so a reader that computes a different digest (or none) can tell it is not
/// comparable and skip the check instead of reporting a false mismatch.
pub const GRAPH_DIGEST_BASIS: &str = "nodes-edges-json-v1";

/// Content digest of a graph, `"sha256:<hex>"`, treating every node id as a
/// string. See the module docs.
pub fn graph_digest(graph: &ISONGraph) -> String {
    graph_digest_with_int_ids(graph, &HashSet::new())
}

/// `graph_digest`, with `int_ids` naming the nodes whose id is an integer
/// rather than a string (the digest tags the two differently). Edge endpoints
/// take the kind of the node they point at.
pub fn graph_digest_with_int_ids(graph: &ISONGraph, int_ids: &HashSet<NodeId>) -> String {
    digest_impl(graph, int_ids, false)
}

/// The digest as the Python port computes it for a file *it* wrote.
///
/// Python writes floats with `repr`, so a value such as `1e-05` or `1.5e+20`
/// lands in the file in exponent form, and Python's own ISON parser reads that
/// token back as a *string*. The digest it records is taken of that read-back
/// graph, so those values are hashed as strings there. This crate's parser
/// reads the same token as a float. When the plain digest does not match, `load`
/// tries this reading before calling the sidecar a mismatch.
pub(crate) fn graph_digest_python_view(graph: &ISONGraph, int_ids: &HashSet<NodeId>) -> String {
    digest_impl(graph, int_ids, true)
}

/// Whether any property of the graph is a float that `repr` writes in
/// exponent form (below 1e-4 or from 1e16 up, other than zero).
pub(crate) fn has_exponent_floats(graph: &ISONGraph) -> bool {
    let exponent =
        |value: &Value| matches!(value, Value::Float(f) if python_json_float(*f).contains('e'));
    graph
        .nodes()
        .any(|node| node.properties.values().any(exponent))
        || graph.edge_types().into_iter().any(|rel| {
            graph
                .edges_of_type(rel)
                .any(|edge| edge.properties.values().any(exponent))
        })
}

fn digest_impl(
    graph: &ISONGraph,
    int_ids: &HashSet<NodeId>,
    exponent_floats_as_text: bool,
) -> String {
    let kind = |node_type: &str, id: &str| -> char {
        if graph.has_node(node_type, id) {
            if int_ids.contains(&NodeId::new(node_type, id)) {
                'i'
            } else {
                's'
            }
        } else if !id.is_empty() && id.chars().all(|c| c.is_ascii_digit()) {
            // Python's fallback for a reference to a node that is not there.
            'i'
        } else {
            's'
        }
    };

    let mut nodes: Vec<String> = graph
        .nodes()
        .map(|node| {
            let mut out = String::from("[");
            write_ref(
                &mut out,
                &node.node_type,
                kind(&node.node_type, &node.id),
                &node.id,
            );
            out.push(',');
            write_properties(&mut out, &node.properties, exponent_floats_as_text);
            out.push(']');
            out
        })
        .collect();
    nodes.sort();

    // Every stored edge counts, as in the Python port: an undirected graph
    // there also stores the reverse edge, exactly as ISONGraph does here.
    let mut edge_types: Vec<&String> = graph.edge_types();
    edge_types.sort();
    let mut edges: Vec<String> = Vec::new();
    for rel_type in edge_types {
        for edge in graph.edges_of_type(rel_type) {
            edges.push(edge_json(edge, &kind, exponent_floats_as_text));
        }
    }
    edges.sort();

    let body = format!(
        "{{\"directed\":{},\"edges\":[{}],\"nodes\":[{}]}}",
        if graph.directed { "true" } else { "false" },
        edges.join(","),
        nodes.join(",")
    );
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

/// Parse graph text (`isonl` or ISON) into a graph plus the set of node ids
/// that the text writes as integers.
pub(crate) fn parse_graph_text(
    text: &str,
    isonl: bool,
    name: &str,
) -> Result<(ISONGraph, HashSet<NodeId>), EmbeddingError> {
    let graph = if isonl {
        ISONGraph::from_isonl(text, Some(name))
    } else {
        ISONGraph::from_ison(text, Some(name))
    }
    .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;
    Ok((graph, int_ids_of_text(text, isonl)?))
}

/// The node ids a piece of graph text writes as integers (unquoted) rather
/// than strings (quoted); ISONGraph itself keeps every id as text.
pub(crate) fn int_ids_of_text(text: &str, isonl: bool) -> Result<HashSet<NodeId>, EmbeddingError> {
    let doc: Document = if isonl {
        ison_rs::loads_isonl(text)
    } else {
        ison_rs::loads(text)
    }
    .map_err(|e| EmbeddingError::GraphError(e.to_string()))?;
    Ok(int_ids_of(&doc))
}

fn int_ids_of(doc: &Document) -> HashSet<NodeId> {
    let mut ids = HashSet::new();
    for block in &doc.blocks {
        let Block {
            kind, name, rows, ..
        } = block;
        if kind != "nodes" {
            continue;
        }
        for row in rows {
            if let Some(IsonValue::Int(i)) = row.get("id") {
                ids.insert(NodeId::new(name.as_str(), i.to_string()));
            }
        }
    }
    ids
}

/// Digest of the graph a piece of ISON (or ISONL) text reads back as, with
/// integer ids told from string ids by the text itself. This is what `save`
/// records and `load` compares, so the value is the same whichever port wrote
/// the file.
pub fn graph_digest_of_text(text: &str, isonl: bool) -> Result<String, EmbeddingError> {
    let (graph, int_ids) = parse_graph_text(text, isonl, "graph")?;
    Ok(graph_digest_with_int_ids(&graph, &int_ids))
}

// -----------------------------------------------------------------------------
// Canonical JSON, as Python's json.dumps(sort_keys=True, separators=(',', ':'),
// ensure_ascii=True) writes it
// -----------------------------------------------------------------------------

fn write_ref(out: &mut String, node_type: &str, kind: char, id: &str) {
    out.push('[');
    write_json_string(out, node_type);
    out.push(',');
    write_json_string(out, &kind.to_string());
    out.push(',');
    write_json_string(out, id);
    out.push(']');
}

fn edge_json(
    edge: &Edge,
    kind: &dyn Fn(&str, &str) -> char,
    exponent_floats_as_text: bool,
) -> String {
    let mut out = String::from("[");
    write_json_string(&mut out, &edge.rel_type);
    out.push(',');
    write_ref(
        &mut out,
        &edge.source.node_type,
        kind(&edge.source.node_type, &edge.source.id),
        &edge.source.id,
    );
    out.push(',');
    write_ref(
        &mut out,
        &edge.target.node_type,
        kind(&edge.target.node_type, &edge.target.id),
        &edge.target.id,
    );
    out.push(',');
    write_properties(&mut out, &edge.properties, exponent_floats_as_text);
    out.push(']');
    out
}

fn write_properties(
    out: &mut String,
    properties: &HashMap<String, Value>,
    exponent_floats_as_text: bool,
) {
    let sorted: BTreeMap<&String, &Value> = properties.iter().collect();
    out.push('{');
    for (i, (key, value)) in sorted.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_string(out, key);
        out.push(':');
        write_value(out, value, exponent_floats_as_text);
    }
    out.push('}');
}

fn write_value(out: &mut String, value: &Value, exponent_floats_as_text: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => {
            let text = python_json_float(*f);
            if exponent_floats_as_text && text.contains('e') {
                write_json_string(out, &text);
            } else {
                out.push_str(&text);
            }
        }
        Value::String(s) => write_json_string(out, s),
        // Python hashes `str(reference)` here (json.dumps(default=str)), and
        // its Reference dataclass renders as "Reference(type:id)".
        Value::Reference(r) => {
            let text = match r.ref_type.as_deref() {
                Some(t) if !t.is_empty() => format!("Reference({}:{})", t, r.id),
                _ => format!("Reference({})", r.id),
            };
            write_json_string(out, &text);
        }
    }
}

/// A JSON string with every character outside printable ASCII written as
/// `\uXXXX` (a surrogate pair above the BMP), the way `ensure_ascii=True` does.
pub(crate) fn write_json_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let code = c as u32;
                if code < 0x10000 {
                    let _ = write!(out, "\\u{code:04x}");
                } else {
                    let v = code - 0x10000;
                    let _ = write!(
                        out,
                        "\\u{:04x}\\u{:04x}",
                        0xd800 + (v >> 10),
                        0xdc00 + (v & 0x3ff)
                    );
                }
            }
        }
    }
    out.push('"');
}

/// A float as `json.dumps` writes it: `repr(float)` for finite values (the
/// shortest digits that round-trip, positional between 1e-4 and 1e16 and
/// exponent form with a signed two-digit exponent outside it), and the bare
/// words `NaN` / `Infinity` / `-Infinity` otherwise.
pub(crate) fn python_json_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }

    // Rust's `{:e}` gives the shortest round-trip digits, as repr does.
    let sci = format!("{value:e}");
    let (negative, sci) = match sci.strip_prefix('-') {
        Some(rest) => (true, rest.to_string()),
        None => (false, sci),
    };
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let ndigits = digits.len() as i32;
    // value = 0.<digits> * 10^decpt
    let decpt = exponent + 1;

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if !(-3..=16).contains(&decpt) {
        out.push_str(&digits[..1]);
        if ndigits > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    } else if decpt <= 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-decpt) as usize));
        out.push_str(&digits);
    } else if decpt >= ndigits {
        out.push_str(&digits);
        out.push_str(&"0".repeat((decpt - ndigits) as usize));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_render_like_python_repr() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.5, "0.5"),
            (-2.5, "-2.5"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.000123, "0.000123"),
            (0.0000123, "1.23e-05"),
            (123456789012345.0, "123456789012345.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (9007199254740993.0, "9007199254740992.0"),
            (1e22, "1e+22"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (1.0 / 3.0, "0.3333333333333333"),
            (2.0 / 3.0, "0.6666666666666666"),
            (0.30000000000000004, "0.30000000000000004"),
            (12345.678, "12345.678"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
        ];
        for (value, want) in cases {
            assert_eq!(python_json_float(*value), *want, "{value:?}");
        }
    }

    #[test]
    fn strings_are_ascii_escaped_like_python() {
        let mut out = String::new();
        write_json_string(
            &mut out,
            "r\u{e9}sum\u{e9} \u{65e5}\u{672c} \"q\" \\ \n\t\u{1}\u{7f}\u{1f600}",
        );
        assert_eq!(
            out,
            "\"r\\u00e9sum\\u00e9 \\u65e5\\u672c \\\"q\\\" \\\\ \\n\\t\\u0001\\u007f\\ud83d\\ude00\""
        );
    }
}
