# isongraph-vector-js

`isongraph-vector` extends [ISONGraph](https://github.com/ISON-format/isongraph) with embedding-based
similarity search and semantic multi-hop traversal: find the nodes closest in
meaning to a query, then walk the graph from there with the relevance score
decaying per hop.

This is the JavaScript port. See the [repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the full API, the cross-port feature matrix, and the design notes.

## Installation

```bash
npm install isongraph-vector-js
```

## Usage

```javascript
import { SemanticGraph, MockEncoder } from 'isongraph-vector-js';

const graph = new SemanticGraph({ encoder: new MockEncoder() });
graph.addNode('person', 1, { name: 'Alice', description: 'software engineer' });
graph.addNode('person', 2, { name: 'Bob', description: 'data scientist' });
graph.addEdge('KNOWS', ['person', 1], ['person', 2]);

const results = graph.semanticMultiHop('engineering', {
  relType: 'KNOWS',
  maxHops: 2,
});
```

## What you get

- `EmbeddingStore` - in-memory vector storage, cosine similarity search, dimension-checked
- `SemanticGraph` - extends `ISONGraph`; `addNode` auto-embeds from node properties
- `semanticMultiHop` - similarity seeds plus traversal, with optional `blend`
  to mix in each node's own relevance
- `semanticPath`, `similarToNode`, `embedAllNodes`
- `semanticSubgraph` - the on-topic slice of a graph as a new graph, ready to
  serialize into an LLM context
- A portable embedding payload every language port reads and writes, so a store
  built by one can be searched by another

## Tests

```bash
npm install
npm test
```

## Links

- Repository: https://github.com/ISON-format/isongraph-vector
- Documentation: https://graph.ison.dev/docs/isongraph-vector

## License

MIT - see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
