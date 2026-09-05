# isongraph-vector-rs

`isongraph-vector` extends [ISONGraph](https://github.com/ISON-format/isongraph) with embedding-based
similarity search and semantic multi-hop traversal: find the nodes closest in
meaning to a query, then walk the graph from there with the relevance score
decaying per hop.

This is the Rust port. See the [repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the full API, the cross-port feature matrix, and the design notes.

## Installation

```toml
[dependencies]
isongraph-vector-rs = "1.0"
```

## Usage

```rust
use isongraph_vector_rs::{SemanticGraph, MockEncoder, MultiHopOptions};

let mut graph = SemanticGraph::new("social", Box::new(MockEncoder::new(384)));
graph.add_node("person", "1", vec![("name", "Alice")], Some("software engineer"))?;
graph.add_node("person", "2", vec![("name", "Bob")], Some("data scientist"))?;
graph.graph_mut().add_edge("KNOWS", ("person", "1"), ("person", "2"), vec![])?;

let results = graph.semantic_multi_hop_with("engineering", &MultiHopOptions {
    rel_type: Some("KNOWS"),
    max_hops: 2,
    ..Default::default()
})?;
```

## What you get

- `EmbeddingStore` - in-memory vector storage, cosine similarity search, dimension-checked
- `SemanticGraph` - wraps `ISONGraph` (reachable via `graph()` / `graph_mut()`); `add_node` auto-embeds from node properties
- `semantic_multi_hop` - similarity seeds plus traversal, with optional `blend`
  to mix in each node's own relevance
- `semantic_path`, `similar_to_node`, `embed_all_nodes`
- `semantic_subgraph` - the on-topic slice of a graph as a new graph, ready to
  serialize into an LLM context
- A portable embedding payload every language port reads and writes, so a store
  built by one can be searched by another

## Tests

```bash
cargo test
```

## Links

- Repository: https://github.com/ISON-format/isongraph-vector
- Documentation: https://graph.ison.dev/docs/isongraph-vector

## License

MIT - see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
