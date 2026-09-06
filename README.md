# isongraph-vector

[![PyPI](https://img.shields.io/pypi/v/isongraph-vector.svg)](https://pypi.org/project/isongraph-vector/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Semantic search over an ISON graph — find the nodes that mean what you asked,
then walk the graph from there.**

## The problem this solves

[ISONGraph](https://github.com/ISON-format/isongraph) is a good graph. It stores
nodes and edges compactly, walks relationships, finds shortest paths. What it
cannot do is answer *"which node is about machine learning?"* — traversal needs
a starting node, and you have to already know which one.

Vector search has the opposite shape. It will find the three nodes closest in
meaning to a query, and then stop. It knows nothing about who those nodes are
connected to.

`isongraph-vector` puts the two together. A query finds its seeds by meaning,
traversal takes over from there, and relevance decays with each hop — so the
answer is "these nodes are about your question, and these others are one or two
relationships away from them, in that order."

```text
person:1  score=1.0000  hops=0     ← matched the query directly
person:2  score=0.8000  hops=1     ← reached via KNOWS, decayed once
person:3  score=0.6400  hops=2     ← two hops out
```

## Six languages, one library

Each directory is a self-contained, separately published package with its own
README. The behaviour is the same in all six; the idioms are native to each.

| Port | Directory | Package | Base library | Tests |
| --- | --- | --- | --- | ---: |
| Python | [`isongraph_vector-py/`](isongraph_vector-py/) | `isongraph-vector` | `ison-graph` 1.4.0 | 70 |
| C++ | [`isongraph-vector-cpp/`](isongraph-vector-cpp/) | headers | vendored `ison-graph-cpp` 1.4.0 | 69 |
| TypeScript | [`isongraph-vector-ts/`](isongraph-vector-ts/) | `isongraph-vector-ts` | `ison-graph-ts` 1.4.0 | 67 |
| JavaScript | [`isongraph-vector-js/`](isongraph-vector-js/) | `isongraph-vector-js` | `ison-graph-js` 1.4.0 | 65 |
| C# | [`isongraph-vector-csharp/`](isongraph-vector-csharp/) | `IsonGraph.Vector` | `IsonGraph` 1.4.0 | 59 |
| Rust | [`isongraph-vector-rs/`](isongraph-vector-rs/) | `isongraph-vector-rs` | `ison-graph` 1.4 | 48 |

Parity is deliberate but not total:

| Capability | Python | C# | TS | JS | Rust | C++ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Semantic multi-hop, path, subgraph | yes | yes | yes | yes | yes | yes |
| `MockEncoder`, byte-identical everywhere | yes | yes | yes | yes | yes | yes |
| Portable embedding JSON | yes | yes | yes | yes | yes | yes |
| SQLite store, shared schema | yes | yes | yes | yes | yes | yes |
| sqlite-vec search backend | yes | yes | yes | yes | yes | yes |
| Real encoder (sentence-transformers) | yes | - | - | - | - | - |
| Graph save/load with embeddings | yes | - | - | - | - | - |
| Extends the base graph | subclass | composition | subclass | subclass | composition | composition |

Python is the reference implementation: the only port with a real transformer
encoder and graph persistence. The other five are `MockEncoder`-only and expect
you to supply an encoder over whatever model you already run.

C#, Rust and C++ wrap the base graph rather than subclassing it — `ISONGraph`
is `sealed` in C#, and Rust and C++ have no inheritance to use here. The
underlying graph stays reachable through `Graph`, `graph()` and `graph_mut()`.

## Installation

```bash
pip install isongraph-vector             # Python
pip install isongraph-vector[fast]       # + numpy, ~100x faster search
pip install isongraph-vector[vec]        # + sqlite-vec

dotnet add package IsonGraph.Vector      # C#
npm install isongraph-vector-ts          # TypeScript
npm install isongraph-vector-js          # JavaScript
cargo add isongraph-vector-rs            # Rust
```

C++ is header-only: add `isongraph-vector-cpp/include` and its vendored
`vendor/ison-graph-cpp` to your include path, or use the provided
`CMakeLists.txt`.

## Quick start

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

Alice seeds the search at hop 0 because she matches the query. Bob and Carol
are reached by traversal, decayed by `decay ** hops`.

That `threshold=-1.0` is a `MockEncoder` concession worth understanding early.
The mock encoder is deterministic and dependency-free, but its vectors are
hash-derived noise with no meaning. Being near-orthogonal, and cosine
similarity being signed, mock scores straddle zero — so the sensible default of
0.3 rejects every seed. With a real encoder, leave the default alone.

The same program in each language is in that port's README, with the same
output.

## How the search works

`semantic_multi_hop` does three things:

1. **Seed.** Score every embedded node against the query; keep the top *k*
   above `threshold`.
2. **Traverse.** Breadth-first out from each seed along `rel_type`, up to
   `max_hops`.
3. **Score.** `max(seed_similarity, 0) * decay ** hop_count`, keeping the best
   score per node.

A node that is itself a seed always keeps its own, direct similarity at hop 0.
It is never demoted to a decayed score because a better seed happened to reach
it first — a bug that lived in all five original ports and is now pinned by a
regression test in each.

The clamp at zero matters too. Cosine similarity is signed, and multiplying a
*negative* seed score by 0.8 moves it toward zero — so without the clamp,
everything a badly-matching seed touched would outrank the seed itself, and
relevance would grow with distance.

### Blending in a node's own relevance

By default a node's score comes purely from its seed and its distance. Raise
`blend` and each node's own similarity to the query is mixed in:

```python
results = graph.semantic_multi_hop(query, blend=0.5)
# score = (1 - blend) * seed_score * decay**hops + blend * own_similarity
```

Use it when a node two hops out is itself highly relevant and should outrank a
closer but unrelated neighbour. Seeds score the same either way, so raising it
only reorders what you reached.

### Extracting the slice that matters

`semantic_subgraph` runs the same search and returns a real `SemanticGraph` —
the matching nodes, every edge induced between them, and their embeddings:

```python
sub = graph.semantic_subgraph("who works on machine learning?", max_hops=2)
sub.save("context.isong")
```

This is what makes an ISON graph useful as LLM context. Instead of serializing
a whole knowledge graph into a prompt, you serialize the part the question is
about — the same argument ISON makes about token efficiency, one level up.

### The rest

- `similarity_search(query, top_k, node_type, threshold)` — plain vector search.
- `similar_to_node(ref)` — "more like this", using a node's own stored vector.
  No query text, no encoder call.
- `semantic_path(query, target)` — reach a specific node from a
  semantically-relevant start.
- `embed_all_nodes()` — backfill an existing graph, batched.

## Storing the vectors

Every port ships an in-memory store and a SQLite-backed one, and the SQLite
store can index vectors with sqlite-vec:

| Port | SQLite store | Enable it |
| --- | --- | --- |
| Python | built in | `EmbeddingStore(db_path=..., sqlite_vec=True)`, `[vec]` extra |
| C# | `IsonGraph.Vector.Sqlite` | `new SqliteEmbeddingStore(path, encoder, sqliteVec: true)` |
| TypeScript | `isongraph-vector-ts/sqlite` | `new SqliteEmbeddingStore(path, encoder, true)` |
| JavaScript | `isongraph-vector-js/sqlite` | `new SqliteEmbeddingStore(path, encoder, true)` |
| Rust | `sqlite` feature | `SqliteEmbeddingStore::open(path, encoder, true)` |
| C++ | `-DISONGRAPH_VECTOR_SQLITE=ON` | `SqliteEmbeddingStore(path, encoder, true)` |

Each is optional and off by default, so the core packages stay light: Python
falls back to a numpy or pure-Python scan, the Node ports keep working in a
browser (their SQLite store is a separate entry point importing `node:sqlite`),
Rust needs a feature flag, and the C++ header stays dependency-free until you
ask.

In Python the three backends line up like this:

| Backend | Requires | How it searches |
| --- | --- | --- |
| Pure Python | nothing | cosine loop over every vector |
| numpy (used automatically) | `[fast]` | one matrix-vector product against a cached matrix |
| sqlite-vec | `[vec]` | `vec0` virtual table, KNN in C |

numpy alone takes a query over 5,000 nodes from 246 ms to 2.5 ms. Indexing
costs writes — 20,000 rows take 0.16 s to insert without it and 0.43 s with —
which is why it stays opt-in.

**Turning it on does not change your results.** `vec0` runs an exact
brute-force KNN in C, not an approximate index. Two paths keep scanning
regardless — `score_map()` and blended multi-hop, which need a score for every
node rather than a top-k. Type filtering stays exact because the node type is a
`vec0` metadata column, narrowing the search rather than filtering its output.

### Measured retrieval

All six ports, 20,000 vectors of 384 dimensions, top-10, `threshold=-1.0` so
every candidate is scored. Median of five runs on one Windows machine, using
`MockEncoder` so every port scores byte-identical vectors:

| Port | In-process scan | SQLite scan | sqlite-vec | Recall |
| --- | ---: | ---: | ---: | ---: |
| C# | 7.7 ms | 42 ms | 7.6 ms | 1.000 |
| C++ | 11.6 ms | 18.8 ms | 7.1 ms | 1.000 |
| Rust | 11.8 ms | 21.0 ms | 8.2 ms | 1.000 |
| Python (numpy) | 14.7 ms | — | 7.2 ms | 1.000 |
| TypeScript / JavaScript | 14.9 ms | 61 ms | 7.4 ms | 1.000 |

**Recall is 1.000 everywhere**, which is the number that matters most: `vec0`
is an exact brute-force KNN, so the index returns precisely what the scan
returns. It was measured as the overlap of the top-10 from the index against
the top-10 from the exact scan, over 20 queries per port. Not an approximation
with a good hit rate — the same answers.

The timings say something less flattering about the SQLite scan than about the
index. Reading 20,000 blobs back out of SQLite and scoring them costs far more
than scoring them in memory, in every port. If you turn the SQLite store on and
leave `sqlite_vec` off, expect the scan column. Python has no separate SQLite
scan row because its store is SQLite-backed either way; its numpy path caches a
decoded matrix, so it behaves like the in-process column.

Treat the absolute numbers as one machine's shape, not a benchmark. The ratios
are the durable part: the index is worth roughly 2x over an in-process scan and
3-8x over a SQLite scan at this size, and *costs* you below a few thousand
vectors, where the scan wins outright.

## One format, six languages

`MockEncoder` produces **byte-identical vectors in all six ports**: a 32-bit
FNV-1a hash of the UTF-8 bytes seeds an xorshift32 generator, and the top 24
bits of each state become a value in [-1, 1). The same golden vectors are
asserted in every suite.

That is what makes the interchange formats meaningful rather than aspirational.
All six write the same embedding payload:

```json
{"format": "ison-embeddings", "version": 1, "dimension": 384, "count": 1,
 "embeddings": [{"type": "person", "id": 1, "id_type": "int",
                 "text": "Alice is a software engineer", "vector": [0.83, -0.82]}]}
```

…and the same SQLite schema:

```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,              -- md5("type:id")[:16]
    node_type TEXT, node_id TEXT, node_id_type TEXT,
    vector BLOB,                      -- little-endian float32, 4 bytes/dim
    text TEXT, model TEXT, created_at TIMESTAMP,
    UNIQUE(node_type, node_id));

CREATE VIRTUAL TABLE vec_embeddings USING vec0(   -- only when indexed
    node_type text, embedding float[N] distance_metric=cosine);
```

So a database written by the Rust port opens in Python, and the other way
round. This is verified, not assumed: Python opens databases written by the
Rust and Node ports and reads byte-identical vectors, and the C#, TypeScript,
JavaScript, Rust and C++ suites each import a payload captured from Python.

Opening a database written without the index, with the index on, backfills it.
Writes use an UPSERT rather than `INSERT OR REPLACE`, because REPLACE assigns a
new rowid and would orphan the matching `vec0` entry.

## Persistence

The ISON graph format carries nodes and edges, not vectors, so the Python port
writes both halves:

```python
graph.save("social.isong")        # + social.isong.embeddings.json
graph = SemanticGraph.load("social.isong", encoder=my_encoder)
```

Pass the `encoder=` you saved with. Without it, the first text search on a
loaded graph constructs a `SentenceTransformerEncoder` and downloads a model.

The other five ports persist embeddings to SQLite and exchange the portable
payload, but the graph itself stays in memory there.

## Typed properties

Since ISONGraph 1.4.0 a property value can be a string, a number, a bool or
null. Auto-embedding has to turn those into text, and the ports agree on how:
nulls are skipped, and booleans render as ISON spells them — `true`/`false`,
not Python's `True` or .NET's `True`. A golden test in all six asserts that the
same property bag yields the same string, because different text means a
different vector and would quietly break the cross-port guarantee.

Floats agree too, which took fixing in two places: Python alone wrote an
integral float as `"1.0"` where the others write `"1"`, and C++ used the default
six significant figures, rendering `1/3` as `"0.333333"`. All six now produce
the shortest round-trip form — `1`, `1.5`, `0.1`, `0.3333333333333333` — and
every suite asserts that set.

## Honest limits

- **Search is brute force in every backend**, sqlite-vec included — all of them
  linear in the number of nodes. This is built for an application's own
  knowledge graph, not as a vector-database replacement. For vectors on both
  nodes *and* edges, ANN indexing and link prediction, RudraDB is the right
  tool; these are not really competitors despite the surface similarity.
- **One embedding per node**, from one text field or a concatenation. No
  multi-vector nodes, no edge embeddings.
- **`MockEncoder` has no semantic structure.** It exists so tests and demos
  need no model download. Never use it to judge relevance quality.
- **`SemanticGraph.load()` reaches into ISONGraph's private attributes** to
  copy loaded state across. Verified working on 1.4.0, but it is a private-API
  dependency.
- **The C++ base library is vendored, not versioned.** Nothing signals when
  ISONGraph moves on.

## API reference

### EmbeddingStore

| Method | Description |
| --- | --- |
| `add(node_ref, text, vector=None)` | Add one embedding |
| `add_batch(entries)` | Add many, encoding in one batched call |
| `get(node_ref)` / `remove(node_ref)` | Fetch or delete one record |
| `similarity_search(query, top_k, node_type, threshold)` | Find similar nodes (`top_k=None` returns all) |
| `score_map(query, node_type=None)` | Score every node, keyed by ref |
| `export_embeddings()` / `import_embeddings(payload)` | Portable cross-language payload |
| `save_embeddings(path)` / `load_embeddings(path)` | The same payload as a file |
| `dimension` / `sqlite_vec` | Vector width; whether the index is in use |
| `count()` / `clear()` / `close()` | Size, reset, release |

### SemanticGraph

| Method | Description |
| --- | --- |
| `add_node(..., _embed_text=)` | Add a node, embedding it |
| `embed_node(ref, text)` / `embed_nodes(items)` | Embed one or many |
| `get_embedding(ref)` | Fetch a node's record |
| `similarity_search(query, ...)` | Find similar nodes |
| `similar_to_node(ref, ...)` | More like this |
| `semantic_multi_hop(query, ..., blend=0.0)` | Semantic traversal |
| `semantic_path(query, target, ...)` | Semantic path finding |
| `semantic_subgraph(query, ...)` | On-topic slice as a new graph |
| `embed_all_nodes(text_fn=None, skip_existing=False)` | Backfill, batched |
| `save(path, embeddings=True)` / `load(path, encoder=None, ...)` | Persist both halves |
| `encoder` / `set_encoder(encoder)` | Inspect or install the encoder |
| `close()` | Release the embedding database |

### Errors

`EmbeddingError` is the base; `DimensionMismatchError` and `NoEncoderError`
derive from it (and from `ValueError` in Python, so existing `except
ValueError` handlers keep working).

Vectors are validated: a store pins its dimension to the first vector it sees
and rejects anything else. Cosine similarity over a mismatched pair returns a
plausible-looking number rather than an error, so silence there would mean
silently wrong scores.

## Testing

```bash
pytest tests/                                            # Python, 70
cd isongraph-vector-csharp && dotnet test                # C#, 59
cd isongraph-vector-ts && npm test                       # TypeScript, 67
cd isongraph-vector-js && npm test                       # JavaScript, 65
cd isongraph-vector-rs && cargo test --features sqlite   # Rust, 48

cd isongraph-vector-cpp
cmake -S . -B build -DISONGRAPH_VECTOR_SQLITE=ON && cmake --build build
./build/test_isongraph_vector && ./build/test_isongraph_vector_sqlite   # C++, 69
```

378 tests in total. `pytest` works straight from a checkout:
[`conftest.py`](conftest.py) binds the import name `isongraph_vector` to the
`isongraph_vector-py/` directory, which is named to line up with the other
ports and so cannot be imported directly. An installed copy takes precedence.

`USE_REAL_ENCODER=1 pytest tests/` runs the Python suite against real
sentence-transformers instead of the mock.

## Dependencies

```text
ison-graph>=1.4.0
sentence-transformers>=2.2.0
numpy>=1.21.0            # optional, via the "fast" extra
sqlite-vec>=0.1.6        # optional, via the "vec" extra
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Links

- Repository: <https://github.com/ISON-format/isongraph-vector>
- Documentation: <https://graph.ison.dev/docs/isongraph-vector>
- Issues: <https://github.com/ISON-format/isongraph-vector/issues>

## Author

Mahesh Vaikri

## License

MIT License — see [LICENSE](LICENSE) for details.
Copyright (c) 2026 ISON - Mahesh Vaikri.
