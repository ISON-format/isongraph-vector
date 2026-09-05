# isongraph-vector

`isongraph-vector` extends [ISONGraph](https://github.com/ISON-format/isongraph) with embedding-based
similarity search and semantic multi-hop traversal: find the nodes closest in
meaning to a query, then walk the graph from there with the relevance score
decaying per hop.

This is the Python port. See the [repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the full API, the cross-port feature matrix, and the design notes.

## Installation

```bash
pip install isongraph-vector          # core
pip install isongraph-vector[fast]    # + numpy, vectorized search
pip install isongraph-vector[vec]     # + sqlite-vec, KNN inside SQLite
```

## Usage

```python
from isongraph_vector import SemanticGraph

graph = SemanticGraph(name="social")
graph.add_node('person', 1, name='Alice', _embed_text='software engineer')
graph.add_node('person', 2, name='Bob', _embed_text='data scientist')
graph.add_edge('KNOWS', ('person', 1), ('person', 2))

results = graph.semantic_multi_hop("engineering", rel_type='KNOWS', max_hops=2)

graph.save("social.isong")   # graph state and embeddings together
```

## What you get

- `EmbeddingStore` - SQLite-backed, thread-safe vector storage, cosine
  similarity search, dimension-checked. Three interchangeable backends: a pure
  Python scan, a numpy matrix scan (`[fast]`), or KNN inside SQLite through
  sqlite-vec (`sqlite_vec=True`, `[vec]`) - all returning identical results
- `SemanticGraph` - a real `ISONGraph` subclass; `add_node` auto-embeds from node properties
- `semantic_multi_hop` - similarity seeds plus traversal, with optional `blend`
  to mix in each node's own relevance
- `semantic_path`, `similar_to_node`, `embed_all_nodes`
- `semantic_subgraph` - the on-topic slice of a graph as a new graph, ready to
  serialize into an LLM context
- A portable embedding payload every language port reads and writes, so a store
  built by one can be searched by another

## Tests

Run from the repository root:

```bash
pytest tests/
```

## Links

- Repository: https://github.com/ISON-format/isongraph-vector
- Documentation: https://graph.ison.dev/docs/isongraph-vector

## License

MIT - see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
