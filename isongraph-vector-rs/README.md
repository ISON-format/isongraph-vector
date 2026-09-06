# isongraph-vector-rs

[ISONGraph](https://github.com/ISON-format/isongraph) can walk a graph. It can
tell you who Alice knows, and who they know in turn. What it cannot tell you is
which node in the graph is *about* the thing you just asked for.

`isongraph-vector-rs` closes that gap. It embeds nodes as vectors, finds the
ones closest in meaning to a query, and then traverses the graph from there —
so a question like *"who works on machine learning?"* starts at the nodes that
actually mean that, rather than at a node you had to know the id of.

This is the Rust port. See the
[repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the cross-port picture.

## Installation

```toml
[dependencies]
isongraph-vector-rs = "1.0"

# or, with the SQLite-backed store and sqlite-vec:
isongraph-vector-rs = { version = "1.0", features = ["sqlite"] }
```

Built on [`ison-graph`](https://crates.io/crates/ison-graph) 1.4. The `sqlite`
feature pulls in `rusqlite` with its bundled SQLite, so it needs nothing
installed on the system.

## The idea, in one example

```rust
use isongraph_vector_rs::{MockEncoder, MultiHopOptions, PropertyValue, SemanticGraph};

let mut graph = SemanticGraph::new("social", Box::new(MockEncoder::new(384)));

for (id, name, role) in [
    ("1", "Alice", "software engineer"),
    ("2", "Bob", "data scientist"),
    ("3", "Carol", "product manager"),
] {
    graph.add_node(
        "person",
        id,
        vec![
            ("name", PropertyValue::String(name.into())),
            ("description", PropertyValue::String(role.into())),
        ],
        None,
    )?;
}
graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])?;
graph.graph_mut().add_edge("KNOWS", ("person", "2"), ("person", "3"), vec![])?;

let results = graph.semantic_multi_hop_with(
    "Alice software engineer",
    &MultiHopOptions {
        rel_type: Some("KNOWS"),
        max_hops: 2,
        threshold: -1.0,
        ..Default::default()
    },
)?;

for r in results {
    println!("{}:{}  score={:.4}  hops={}", r.node_id.node_type, r.node_id.id, r.score, r.hop_count);
}
```

```text
person:1  score=1.0000  hops=0
person:2  score=0.8000  hops=1
person:3  score=0.6400  hops=2
```

Alice matches the query directly, so she seeds the search at hop 0. Bob and
Carol are reached by traversal, and their relevance decays by `decay ** hops` —
0.8 and 0.64 with the default decay of 0.8. That decay is the whole point: a
node three hops from anything relevant should not outrank a direct match.

Three details in that snippet are worth knowing.

`threshold: -1.0` is there because `MockEncoder` is deterministic and
dependency-free but its vectors carry no meaning — they are hash-derived noise
for testing plumbing. Being near-orthogonal, and cosine similarity being
signed, mock scores straddle zero, so the default threshold of 0.3 would reject
every seed. Implement `EmbeddingEncoder` over a real model and the default is
sensible again.

Property values are typed. Since ISONGraph 1.4.0 a property can be a string, a
number, a bool or null, which is what `PropertyValue` is (a re-export of
`ison_rs::Value`, so you do not need a direct dependency on `ison-graph` to
build a property list). If all your values are strings,
`add_node_with_embed` takes plain `&str` and lifts them for you.

`ISONGraph` is wrapped by composition rather than inherited from, so the
underlying graph stays available through `graph()` and `graph_mut()` — which is
where the edges above are added.

## Beyond a single search

**Blending in a node's own relevance.** By default a node's score comes purely
from its seed and its distance. Sometimes a node two hops out is itself highly
relevant and deserves to outrank a closer but unrelated neighbour. `blend`
mixes the two:

```rust
let results = graph.semantic_multi_hop_with(
    query,
    &MultiHopOptions { blend: 0.5, ..Default::default() },
)?;
// score = 0.5 * (seed_score * decay^hops) + 0.5 * own_similarity
```

At `blend: 0.0` (the default) scoring is pure seed-and-decay. Seeds score the
same either way, so raising it only ever reorders the nodes you reached.

**Pulling out the slice that matters.** `semantic_subgraph` runs the same
search and hands back a real `SemanticGraph` — the matching nodes, every edge
induced between them, and their embeddings. This is what makes an ISON graph
useful as LLM context: rather than serializing a whole knowledge graph into a
prompt, you serialize the part the question is actually about.

**More like this.** `similar_to_node` searches with a node's own stored vector
— no query text, no encoder call — and drops the node itself from the results.

Alongside those: `semantic_path` to reach a specific target from a
semantically-relevant start, and `embed_all_nodes` to backfill an existing
graph in one batched encoder call. `semantic_multi_hop` keeps the original
positional form for callers who prefer it over the options struct.

## Storing the vectors

The default store is in memory. With the `sqlite` feature, embeddings persist:

```rust
use isongraph_vector_rs::{MockEncoder, SqliteEmbeddingStore};
use ison_graph_rs::NodeId;

let mut store = SqliteEmbeddingStore::open(
    "embeddings.db",
    Some(Box::new(MockEncoder::new(384))),
    false,          // sqlite_vec
)?;
store.add(NodeId::new("person", "1"), "Alice is a software engineer", None)?;
store.add_batch(&[                       // one encoder call, one transaction
    (NodeId::new("person", "2"), "Bob analyses ML models"),
    (NodeId::new("person", "3"), "Carol plans roadmaps"),
])?;
```

Pass `true` for that last argument and top-k searches are answered by a
sqlite-vec `vec0` virtual table instead of a scan. It does not change what you
get back: `vec0` runs an exact brute-force KNN in C, not an approximate index —
measured identical ordering and scores within 2e-07 of the scan. Measured over 20,000 vectors of 384 dimensions, top-10, median of five
runs: **21.0 ms** for the SQLite scan, **8.2 ms** through the index, and
**11.8 ms** for the in-memory store. Recall of the index against the exact
scan is **1.000** — the same answers, faster. Below a few thousand vectors
the scan wins outright, which is why the index is opt-in. Filtering by node type stays exact too: the type is a `vec0`
metadata column, so it narrows the search rather than filtering its results.

## One format, six languages

The SQLite schema and the portable JSON payload are shared by every port —
Python, C#, TypeScript, JavaScript, Rust and C++ — so a database or payload
written by one can be read by another. `MockEncoder` is byte-identical across
all six, which is what makes that guarantee testable rather than aspirational;
this suite proves it by importing a payload captured from the Python port and
checking the vectors match what Rust encodes for the same text.

```rust
let json = store.to_json()?;
store.from_json(&json, false)?;
```

## Things that will bite you

- **Search is brute force**, sqlite-vec included: linear in the number of
  nodes. This is built for an application's own knowledge graph, not as a
  vector-database replacement.
- **One embedding per node**, from one text field or a concatenation. No
  multi-vector nodes, no edge embeddings.
- **`MockEncoder` has no semantic structure.** Use it to test plumbing, never
  to judge relevance quality. This port ships no real encoder — implement
  `EmbeddingEncoder` over the model of your choice.
- **Seed order among equal scores is not deterministic.** The store is a
  `HashMap`, so ties can be broken differently between runs. It only shows up
  with hand-crafted identical vectors; distinct texts give distinct scores.
- **Auto-embed renders typed properties.** Nulls are skipped and booleans
  render as `true`/`false`, so the same graph produces the same text in every
  port.

## Tests

```bash
cargo test                      # 33 unit tests + 1 doctest
cargo test --features sqlite    # + 14 SQLite tests
```

## Links

- Repository: <https://github.com/ISON-format/isongraph-vector>
- Documentation: <https://graph.ison.dev/docs/isongraph-vector>

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
