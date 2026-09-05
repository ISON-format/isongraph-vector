# isongraph-vector

[ISONGraph](https://github.com/ISON-format/isongraph) can walk a graph. It can
tell you who Alice knows, and who they know in turn. What it cannot tell you is
which node in the graph is *about* the thing you just asked for.

`isongraph-vector` closes that gap. It embeds nodes as vectors, finds the ones
closest in meaning to a query, and then traverses the graph from there — so a
question like *"who works on machine learning?"* starts at the nodes that
actually mean that, rather than at a node you had to know the id of.

This is the Python port, and the reference implementation: it is the only one
with a real transformer encoder, graph persistence, and a numpy-accelerated
search. See the
[repository README](https://github.com/ISON-format/isongraph-vector#readme)
for the cross-port picture.

## Installation

```bash
pip install isongraph-vector          # core
pip install isongraph-vector[fast]    # + numpy, ~100x faster search
pip install isongraph-vector[vec]     # + sqlite-vec, KNN inside SQLite
```

## The idea, in one example

```python
from isongraph_vector import SemanticGraph, MockEncoder

graph = SemanticGraph(name="social", encoder=MockEncoder(384))

graph.add_node('person', 1, name='Alice', description='software engineer')
graph.add_node('person', 2, name='Bob', description='data scientist')
graph.add_node('person', 3, name='Carol', description='product manager')
graph.add_edge('KNOWS', ('person', 1), ('person', 2))
graph.add_edge('KNOWS', ('person', 2), ('person', 3))

for r in graph.semantic_multi_hop('Alice software engineer',
                                  rel_type='KNOWS', max_hops=2, threshold=-1.0):
    print(f"{r.node_ref}  score={r.score:.4f}  hops={r.hop_count}")
```

```text
('person', 1)  score=1.0000  hops=0
('person', 2)  score=0.8000  hops=1
('person', 3)  score=0.6400  hops=2
```

Alice matches the query directly, so she seeds the search at hop 0. Bob and
Carol are reached by traversal, and their relevance decays by `decay ** hops` —
0.8 and 0.64 with the default decay of 0.8. That decay is the whole point: a
node three hops from anything relevant should not outrank a direct match.

Two details in that snippet are worth knowing up front.

`MockEncoder` is a real encoder in the sense that it is deterministic and
dependency-free, but its vectors carry no meaning — they are hash-derived
noise, meant for testing plumbing. Because they are near-orthogonal and cosine
similarity is signed, mock scores straddle zero, which is why the example
passes `threshold=-1.0`. With the real encoder you can leave the default 0.3
alone. Swap it in by simply not passing one:

```python
graph = SemanticGraph(name="social")   # builds SentenceTransformerEncoder lazily
```

The first call that needs it downloads `all-MiniLM-L6-v2`. Pass
`SentenceTransformerEncoder("some-other-model")` to use a different one.

## Beyond a single search

`semantic_multi_hop` is the headline, but it is rarely the whole job.

**Blending in a node's own relevance.** By default a node's score comes purely
from its seed and its distance. Sometimes a node two hops out is itself highly
relevant and deserves to outrank a closer but unrelated neighbour. `blend`
mixes the two:

```python
results = graph.semantic_multi_hop(query, blend=0.5)
# score = 0.5 * (seed_score * decay**hops) + 0.5 * own_similarity
```

At `blend=0` (the default) scoring is pure seed-and-decay. Seeds score the same
either way, so raising it only ever reorders the nodes you reached.

**Pulling out the slice that matters.** `semantic_subgraph` runs the same
search and hands back a real `SemanticGraph` — the matching nodes, every edge
induced between them, and their embeddings:

```python
sub = graph.semantic_subgraph("who works on machine learning?", max_hops=2)
sub.save("context.isong")
```

This is what makes an ISON graph useful as LLM context. Rather than serializing
a whole knowledge graph into a prompt, you serialize the part the question is
actually about.

**More like this.** `similar_to_node(('person', 1))` searches with a node's own
stored vector — no query text, no encoder call — and drops the node itself from
the results.

## Storing the vectors

The store is SQLite-backed, and the default is `:memory:`. Give it a path and
the embeddings persist:

```python
from isongraph_vector import EmbeddingStore, SentenceTransformerEncoder

store = EmbeddingStore(db_path="embeddings.db", encoder=SentenceTransformerEncoder())
store.add(('person', 1), 'Alice is a software engineer')
store.add_batch([                      # one encoder call for the whole batch
    (('person', 2), 'Bob analyses ML models'),
    (('person', 3), 'Carol plans roadmaps'),
])
```

Search runs three ways, in increasing order of speed and dependencies:

| Backend | Requires | How it searches |
| --- | --- | --- |
| Pure Python | nothing | cosine loop over every vector |
| numpy | `[fast]` | one matrix-vector product against a cached matrix |
| sqlite-vec | `[vec]` | `vec0` virtual table, KNN in C |

numpy alone takes a query over 5,000 nodes from 246 ms to 2.5 ms, and is used
automatically whenever it is importable. sqlite-vec goes further, but only at
scale — and it is worth being precise about when:

| Vectors | numpy scan | sqlite-vec | |
| ---: | ---: | ---: | --- |
| 1,000 | 0.47 ms | 0.59 ms | slower — don't bother |
| 5,000 | 4.01 ms | 1.75 ms | 2.3x |
| 20,000 | 27.63 ms | 6.80 ms | 4.1x |

It also makes writes about 2.7x more expensive, which is why it is opt-in:

```python
store = EmbeddingStore(db_path="embeddings.db", encoder=encoder, sqlite_vec=True)
```

Turning it on does not change what you get back. `vec0` runs an exact
brute-force KNN in C, not an approximate index — measured identical top-10
ordering over 50 queries against 20,000 vectors, with scores within 2e-07 of
the scan. It is a speed change, not an accuracy trade.

## Saving a graph

The ISON graph format carries nodes and edges, not vectors, so `save` writes
both halves:

```python
graph.save("social.isong")        # + social.isong.embeddings.json
graph = SemanticGraph.load("social.isong", encoder=my_encoder)
```

Pass the `encoder=` you saved with. Without it the first text search on a
loaded graph constructs a `SentenceTransformerEncoder` and downloads a model,
which is rarely what you want in a test.

## One format, six languages

The embedding sidecar and the SQLite schema are shared by every port — Python,
C#, TypeScript, JavaScript, Rust and C++ — so a store built by one can be read
by another. `MockEncoder` is byte-identical across all six, which is what makes
that guarantee testable rather than aspirational.

```python
payload = store.export_embeddings()   # portable dict, JSON-serializable
store.import_embeddings(payload)
```

## Things that will bite you

- **Search is brute force in every backend**, sqlite-vec included. All three
  are linear in the number of nodes. This is built for an application's own
  knowledge graph, not as a vector-database replacement.
- **One embedding per node**, from one text field or a concatenation. No
  multi-vector nodes, no edge embeddings.
- **`MockEncoder` has no semantic structure.** Use it to test plumbing, never
  to judge relevance quality.
- **Auto-embed renders typed properties.** Since ISONGraph 1.4.0 a property can
  be a number, a bool or `None`. Nulls are skipped and booleans render as ISON
  spells them (`true`/`false`, not Python's `True`), so the same graph produces
  the same text in every port. Floats render in the shortest round-trip form,
  so an integral float is `"1"` and not Python's usual `"1.0"` — the other five
  ports would otherwise disagree with this one, and a different text means a
  different vector.

## Tests

```bash
pytest tests/                     # 70 tests, MockEncoder, no downloads
USE_REAL_ENCODER=1 pytest tests/  # against real sentence-transformers
```

Run from the repository root. `conftest.py` binds the import name
`isongraph_vector` to the `isongraph_vector-py/` directory, which is named to
line up with the other ports and so cannot be imported directly.

## Links

- Repository: <https://github.com/ISON-format/isongraph-vector>
- Documentation: <https://graph.ison.dev/docs/isongraph-vector>

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 ISON - Mahesh Vaikri.
