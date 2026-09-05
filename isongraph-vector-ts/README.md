# isongraph-vector-ts

[ISONGraph](https://github.com/ISON-format/isongraph) can walk a graph. It can
tell you who Alice knows, and who they know in turn. What it cannot tell you is
which node in the graph is *about* the thing you just asked for.

`isongraph-vector-ts` closes that gap. It embeds nodes as vectors, finds the
ones closest in meaning to a query, and then traverses the graph from there —
so a question like *"who works on machine learning?"* starts at the nodes that
actually mean that, rather than at a node you had to know the id of.

This is the TypeScript port. See the
[repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the cross-port picture.

## Installation

```bash
npm install isongraph-vector-ts
npm install sqlite-vec          # optional, for the indexed SQLite store
```

Built on [`ison-graph-ts`](https://www.npmjs.com/package/ison-graph-ts) 1.4.0.
The main entry point has no Node-specific imports, so it runs in a browser as
well as in Node.

## The idea, in one example

```typescript
import { SemanticGraph, MockEncoder } from 'isongraph-vector-ts';

const graph = new SemanticGraph({ name: 'social', encoder: new MockEncoder(384) });

graph.addNode('person', 1, { name: 'Alice', description: 'software engineer' });
graph.addNode('person', 2, { name: 'Bob', description: 'data scientist' });
graph.addNode('person', 3, { name: 'Carol', description: 'product manager' });
graph.addEdge('KNOWS', ['person', 1], ['person', 2]);
graph.addEdge('KNOWS', ['person', 2], ['person', 3]);

const results = graph.semanticMultiHop('Alice software engineer', {
  relType: 'KNOWS',
  maxHops: 2,
  threshold: -1.0,
});

for (const r of results) {
  console.log(`${r.nodeRef.join(':')}  score=${r.score.toFixed(4)}  hops=${r.hopCount}`);
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

That example passes `threshold: -1.0`, which deserves an explanation.
`MockEncoder` is deterministic and dependency-free, but its vectors carry no
meaning — they are hash-derived noise for testing plumbing. Because they are
near-orthogonal and cosine similarity is signed, mock scores straddle zero, and
the default threshold of 0.3 would reject every seed. Supply your own
`EmbeddingEncoder` over a real model and the default is sensible again.

`addNode` auto-embeds from the node's properties, using the `embedFields` list
(`name`, `description`, `content`, `text` by default). Pass an explicit text
with `addNodeWithEmbed` when you want to control the wording.

## Beyond a single search

**Blending in a node's own relevance.** By default a node's score comes purely
from its seed and its distance. Sometimes a node two hops out is itself highly
relevant and deserves to outrank a closer but unrelated neighbour. `blend`
mixes the two:

```typescript
const results = graph.semanticMultiHop(query, { blend: 0.5 });
// score = 0.5 * (seedScore * decay^hops) + 0.5 * ownSimilarity
```

At `blend: 0` (the default) scoring is pure seed-and-decay. Seeds score the
same either way, so raising it only ever reorders the nodes you reached.

**Pulling out the slice that matters.** `semanticSubgraph` runs the same search
and hands back a real `SemanticGraph` — the matching nodes, every edge induced
between them, and their embeddings. This is what makes an ISON graph useful as
LLM context: rather than serializing a whole knowledge graph into a prompt, you
serialize the part the question is actually about.

**More like this.** `similarToNode(['person', 1])` searches with a node's own
stored vector — no query text, no encoder call — and drops the node itself from
the results.

Alongside those: `semanticPath` to reach a specific target from a
semantically-relevant start, and `embedAllNodes` to backfill an existing graph
in one batched encoder call.

Both the options-object and the original positional form of `semanticMultiHop`
are accepted, so older call sites keep working.

## Storing the vectors

The default store is in memory. For persistence, import the SQLite store from
the `/sqlite` entry point:

```typescript
import { SqliteEmbeddingStore } from 'isongraph-vector-ts/sqlite';
import { MockEncoder } from 'isongraph-vector-ts';

const store = new SqliteEmbeddingStore('embeddings.db', new MockEncoder(384));
store.add(['person', '1'], 'Alice is a software engineer');
store.addBatch([                          // one encoder call, one transaction
  [['person', '2'], 'Bob analyses ML models'],
  [['person', '3'], 'Carol plans roadmaps'],
]);
store.close();
```

It is a separate entry point on purpose: it imports `node:sqlite`, so pulling
it into the main module would break browser bundles. Node's built-in SQLite
means no native module to install.

Pass `true` as the third argument and top-k searches are answered by a
sqlite-vec `vec0` virtual table instead of a scan:

```typescript
const store = new SqliteEmbeddingStore('embeddings.db', encoder, true);
```

It does not change what you get back. `vec0` runs an exact brute-force KNN in
C, not an approximate index — measured identical ordering and scores within
2e-07 of the scan. It is worth roughly 4x over 20,000 vectors and is *slower*
below a few thousand, which is why it is opt-in. Filtering by node type stays
exact too: the type is a `vec0` metadata column, so it narrows the search
rather than filtering its results.

## One format, six languages

The SQLite schema and the portable JSON payload are shared by every port —
Python, C#, TypeScript, JavaScript, Rust and C++ — so a database or payload
written by one can be read by another. `MockEncoder` is byte-identical across
all six, which is what makes that guarantee testable rather than aspirational;
this suite proves it by importing a payload captured from the Python port and
checking the vectors match what TypeScript encodes for the same text.

```typescript
const payload = store.exportEmbeddings();
store.importEmbeddings(payload);
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
- **Auto-embed renders typed properties.** Since ISONGraph 1.4.0 a property can
  be a number, a bool or null. Nulls are skipped and booleans render as `true`
  and `false`, so the same graph produces the same text in every port.
- **The SQLite store is Node-only.** The in-memory store, and everything built
  on it, runs anywhere.

## Tests

```bash
npm test           # 66 tests
npm run typecheck  # tsc --noEmit, tests included
npm run build      # emits dist/, library only
```

## Links

- Repository: <https://github.com/ISON-format/isongraph-vector>
- Documentation: <https://graph.ison.dev/docs/isongraph-vector>

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
