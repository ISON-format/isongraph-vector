# isongraph-vector

[![PyPI](https://img.shields.io/pypi/v/isongraph-vector.svg)](https://pypi.org/project/isongraph-vector/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Semantic Graph Extension** - Combines ISONGraph with embedding-based similarity search for semantic multi-hop traversal.

## Features

- **EmbeddingStore**: vector storage with cosine similarity search (SQLite-backed in Python, in-memory elsewhere)
- **SemanticGraph**: ISONGraph extended with embedding capabilities
- **Semantic Multi-Hop**: combines embedding similarity with graph traversal
- **Semantic Subgraph**: extract the on-topic slice of a graph, ready to serialize into an LLM context
- **Portable embeddings**: one JSON format read and written by all five language ports
- **sentence-transformers**: `all-MiniLM-L6-v2` by default, or any model you name

## Language ports

The same library, implemented six times. Each directory is a self-contained,
separately published package with its own README.

| Port | Directory | Package | Base library | Tests |
| --- | --- | --- | --- | ---: |
| Python | [`isongraph_vector-py/`](isongraph_vector-py/) | `isongraph-vector` | `ison-graph` 1.4.0 | 68 |
| C# | [`isongraph-vector-csharp/`](isongraph-vector-csharp/) | `IsonGraph.Vector` | `IsonGraph` 1.4.0 | 59 |
| TypeScript | [`isongraph-vector-ts/`](isongraph-vector-ts/) | `isongraph-vector-ts` | `ison-graph-ts` 1.4.0 | 66 |
| JavaScript | [`isongraph-vector-js/`](isongraph-vector-js/) | `isongraph-vector-js` | `ison-graph-js` 1.4.0 | 64 |
| Rust | [`isongraph-vector-rs/`](isongraph-vector-rs/) | `isongraph-vector-rs` | `ison-graph` 1.4 | 47 |
| C++ (header-only) | [`isongraph-vector-cpp/`](isongraph-vector-cpp/) | headers | vendored `ison-graph-cpp` 1.4.0 | 67 |

Feature parity is deliberate but not total:

| Capability | Python | C# | TS | JS | Rust | C++ |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Persistent store (SQLite) | yes | yes | yes | yes | yes | yes |
| sqlite-vec search backend | yes | yes | yes | yes | yes | yes |
| Real encoder (sentence-transformers) | yes | - | - | - | - | - |
| `MockEncoder` (identical vectors in every port) | yes | yes | yes | yes | yes | yes |
| Semantic multi-hop, path, subgraph | yes | yes | yes | yes | yes | yes |
| Portable embedding JSON | yes | yes | yes | yes | yes | yes |
| Thread-safe store | yes | yes | - | - | - | - |
| Graph save/load with embeddings | yes | - | - | - | - | - |
| Extends the base graph | subclass | composition | subclass | subclass | composition | composition |

The Python port is the reference implementation; the others are
mock-encoder-only, and their in-memory store remains the default. `MockEncoder`
produces byte-identical vectors in all six, so a store built by one really can
be searched by another - the C#, TypeScript, JavaScript, Rust and C++ suites
each prove it by importing an embedding payload captured from the Python port.

Every port can also persist to SQLite, and they all write the **same schema**,
so a database file written by one opens directly in another. Verified, not
assumed: Python opens databases written by the Rust and Node ports and reads
byte-identical vectors out of them.

C#, Rust and C++ wrap the base graph rather than subclassing it: `ISONGraph` is
`sealed` in C#, and Rust and C++ have no inheritance to use here. The underlying
graph stays reachable (`Graph`, `graph()`, `graph_mut()`).

## Installation

```bash
pip install isongraph-vector          # Python
pip install isongraph-vector[fast]    # + numpy, vectorized search
pip install isongraph-vector[vec]     # + sqlite-vec, KNN inside SQLite
```

```bash
dotnet add package IsonGraph.Vector   # C#
npm install isongraph-vector-ts       # TypeScript
npm install isongraph-vector-js       # JavaScript
cargo add isongraph-vector-rs         # Rust
```

C++ is header-only: add `isongraph-vector-cpp/include` (and the vendored
`vendor/ison-graph-cpp`) to your include path, or use the provided
`CMakeLists.txt`.

## Quick Start

```python
from isongraph_vector import SemanticGraph

# Create semantic graph
graph = SemanticGraph(name="social")

# Add nodes with auto-embedding
graph.add_node('person', 1, name='Alice',
               _embed_text='Alice is a software engineer who builds APIs')
graph.add_node('person', 2, name='Bob',
               _embed_text='Bob is a data scientist analyzing ML models')
graph.add_node('person', 3, name='Charlie',
               _embed_text='Charlie is a product manager planning roadmaps')

# Add relationships
graph.add_edge('KNOWS', ('person', 1), ('person', 2))
graph.add_edge('KNOWS', ('person', 2), ('person', 3))

# Similarity search
results = graph.similarity_search("machine learning", top_k=3)
for r in results:
    print(f"{r.node_ref}: {r.score:.4f}")

# Semantic multi-hop search
results = graph.semantic_multi_hop(
    query="software engineer",
    rel_type='KNOWS',
    max_hops=2
)
for r in results:
    print(f"{r.node_ref}: score={r.score:.4f}, hops={r.hop_count}")
```

## Core Components

### EmbeddingStore

Embedding storage with similarity search:

```python
from isongraph_vector import EmbeddingStore, SentenceTransformerEncoder

encoder = SentenceTransformerEncoder()  # all-MiniLM-L6-v2
store = EmbeddingStore(db_path="embeddings.db", encoder=encoder)

store.add(('person', 1), 'Alice is a software engineer')
store.add(('person', 2), 'Bob is a data scientist')

# One batched encoder call instead of one per text
store.add_batch([
    (('person', 3), 'Charlie plans roadmaps'),
    (('person', 4), 'Diana designs interfaces'),
])

results = store.similarity_search("engineer", top_k=5)
results = store.similarity_search("tech role", node_type='person')
```

The store is safe to share across threads: every operation is serialized on a
lock, and the SQLite connection is opened with `check_same_thread=False` so a
request handled on a different thread than the one that built the store works.

Vectors are validated. The store pins its dimension to the first vector it
sees and raises `DimensionMismatchError` on anything else - cosine similarity
over a mismatched pair returns a plausible-looking number rather than an error,
so silence here means silently wrong scores.

### Storage and search backends

Every port ships an in-memory store and a SQLite-backed one. The SQLite store
can additionally index vectors with sqlite-vec, moving top-k search into a
`vec0` virtual table:

| Port | SQLite store | Enable it |
| --- | --- | --- |
| Python | built in | `EmbeddingStore(db_path=..., sqlite_vec=True)`, `[vec]` extra |
| C# | `IsonGraph.Vector.Sqlite` package | `new SqliteEmbeddingStore(path, encoder, sqliteVec: true)` |
| TypeScript | `isongraph-vector-ts/sqlite` entry | `new SqliteEmbeddingStore(path, encoder, true)` |
| JavaScript | `isongraph-vector-js/sqlite` entry | `new SqliteEmbeddingStore(path, encoder, true)` |
| Rust | `sqlite` feature | `SqliteEmbeddingStore::open(path, encoder, true)` |
| C++ | `-DISONGRAPH_VECTOR_SQLITE=ON` | `SqliteEmbeddingStore(path, encoder, true)` |

Each is optional and off by default, so the core packages stay light: Python
falls back to a numpy or pure-Python scan, the Node ports keep working in a
browser (the SQLite store is a separate entry point that imports `node:sqlite`),
Rust needs a feature flag, and the C++ header stays dependency-free until you
ask for the store.

In Python the three backends line up like this:

| Backend | Requires | How it searches |
| --- | --- | --- |
| Pure Python | nothing | cosine loop over every vector |
| numpy (default when installed) | `numpy` | one matrix-vector product against a cached matrix |
| sqlite-vec (`sqlite_vec=True`) | `sqlite-vec` | `vec0` virtual table, KNN in C |

numpy alone takes a query over 5,000 nodes from 246 ms to 2.5 ms. Turning on
sqlite-vec moves the top-k search into SQLite:

```python
store = EmbeddingStore(db_path="embeddings.db", encoder=encoder, sqlite_vec=True)
graph = SemanticGraph(embedding_db="embeddings.db", sqlite_vec=True)
```

```bash
pip install 'isongraph-vector[vec]'          # Python
dotnet add package IsonGraph.Vector.Sqlite   # C#
npm install sqlite-vec                       # TypeScript / JavaScript
cargo add isongraph-vector-rs --features sqlite
cmake -S . -B build -DISONGRAPH_VECTOR_SQLITE=ON   # C++
```

#### The shared database format

All six ports write the same two tables, so the files are interchangeable:

```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,              -- md5("type:id")[:16]
    node_type TEXT, node_id TEXT, node_id_type TEXT,
    vector BLOB,                      -- little-endian float32, 4 bytes/dim
    text TEXT, model TEXT, created_at TIMESTAMP,
    UNIQUE(node_type, node_id));

CREATE VIRTUAL TABLE vec_embeddings USING vec0(   -- only when the index is on
    node_type text, embedding float[N] distance_metric=cosine);
```

Opening a database that was written without the index, with the index on,
backfills it. Writes are an UPSERT rather than `INSERT OR REPLACE`, because
REPLACE assigns a new rowid and would orphan the matching `vec0` entry.

**Results do not change.** `vec0` runs an exact brute-force KNN in C, so this
is a speed change, not an accuracy trade: measured identical top-10 ordering
over 50 queries against 20,000 vectors, with scores within 2e-07 of the scan.
Whether it pays depends on scale:

| Vectors | Scan (numpy) | sqlite-vec | |
| ---: | ---: | ---: | --- |
| 1,000 | 0.47 ms | 0.59 ms | slower - do not bother |
| 5,000 | 4.01 ms | 1.75 ms | 2.3x faster |
| 20,000 | 27.63 ms | 6.80 ms | 4.1x faster |

Indexing costs writes: 20,000 rows take 0.16 s to insert without it and 0.43 s
with it. Below a few thousand vectors the scan wins outright, which is why the
default is off.

Two things keep using the exact scan even when it is on: `score_map()`, and
`semantic_multi_hop(..., blend > 0)`, which needs a score for every node rather
than a top-k. `node_type` filtering stays exact too - the type is a `vec0`
metadata column, so it narrows the search rather than filtering its output.

### SemanticGraph

ISONGraph extended with embeddings:

```python
from isongraph_vector import SemanticGraph

# Auto-embed from fields
graph = SemanticGraph(
    auto_embed=True,
    embed_fields=['name', 'description']
)

graph.add_node('article', 1,
    name='Python Tutorial',
    description='Learn Python programming basics'
)
# Automatically embeds: "Python Tutorial Learn Python programming basics"

# Manual embedding
graph.embed_node(('article', 1), 'Custom embedding text')

# Batch embed all nodes (one encoder call per batch)
count = graph.embed_all_nodes()
count = graph.embed_all_nodes(skip_existing=True)   # only what is missing
```

In the TypeScript, JavaScript, Rust and C++ ports the plain `addNode` /
`add_node` auto-embeds too, so the same code behaves the same way everywhere.

Since ISONGraph 1.4.0 property values are typed, so auto-embedding renders them
to text: nulls are skipped, and booleans use the ISON spelling (`true`/`false`)
rather than each language's own. That keeps the same property bag producing the
same embedding text - and so the same vector - in all six ports, which a golden
test in each suite pins down. Whole-number floats are the exception: Python
renders `1.0` as `"1.0"` where the others render `"1"`. Pass an explicit
`_embed_text` when the exact wording matters.

### Semantic Multi-Hop Search

Combines embedding similarity with graph traversal:

```python
results = graph.semantic_multi_hop(
    query="machine learning engineer",
    rel_type='KNOWS',           # Relationship to follow
    max_hops=3,                 # Maximum traversal depth
    top_k_seeds=5,              # Similar nodes to start from
    top_k_results=10,           # Results to return
    decay=0.8,                  # Score decay per hop
    threshold=0.3,              # Minimum similarity for seeds
    blend=0.0                   # Mix in each node's own relevance (0-1)
)

for result in results:
    print(result.node_ref, result.score, result.hop_count, result.path)
```

Scoring is `max(seed_similarity, 0) * decay ** hop_count`. A node that is
itself a seed always keeps its own, direct similarity at hop 0 - it is never
demoted to a decayed score just because a better seed happened to reach it
first.

`blend` mixes each node's own similarity to the query into its traversal
score:

```python
score = (1 - blend) * seed_score * decay**hops + blend * own_similarity
```

At `blend=0` (the default) scoring is pure seed-and-decay. Raise it when a
node found two hops out but highly relevant in its own right should not be
buried under a closer, unrelated neighbour. Seeds score the same either way.

### More Like This

```python
similar = graph.similar_to_node(('person', 1), top_k=5)
```

Searches with the node's own stored vector - no query text, no encoder call -
and excludes the node itself.

### Semantic Subgraph

Extract the slice of a graph a question is actually about:

```python
sub = graph.semantic_subgraph(
    query="who works on machine learning?",
    rel_type='KNOWS',
    max_hops=2,
    top_k_results=25,
)
sub.save("context.isong")     # small, on-topic graph to inject into a prompt
```

The result is a full `SemanticGraph`: the matching nodes, every edge induced
between them, and their embeddings. It can be traversed, searched or
serialized on its own.

### Semantic Path Finding

```python
result = graph.semantic_path(
    query="Python developer",
    target_ref=('person', 5),
    rel_type='KNOWS',
    max_hops=4
)

if result:
    print(f"Found path with score {result.score:.4f}")
    print(f"Path: {result.path}")
```

### Persistence

```python
graph.save("social.isong")
# writes social.isong AND social.isong.embeddings.json

graph = SemanticGraph.load("social.isong", encoder=my_encoder)
```

The ISON graph format carries nodes and edges only, so embeddings go to a JSON
sidecar that `load` picks up automatically. Pass the `encoder=` you saved
with: without it, the first text search on a loaded graph builds a
`SentenceTransformerEncoder` and downloads a model.

The sidecar is the portable cross-language format - `EmbeddingStore` in every
port reads and writes it:

```python
payload = store.export_embeddings()   # dict, JSON-serializable
store.import_embeddings(payload)
store.save_embeddings("vectors.json")
store.load_embeddings("vectors.json")
```

```rust
let json = store.to_json()?;          // Rust
store.from_json(&json, false)?;
```

```cpp
std::string json = store.toJson();    // C++
store.fromJson(json);
```

## Architecture

```
+------------------------------------------------------------------+
|                    SemanticGraph                                  |
+------------------------------------------------------------------+
|                                                                  |
|  ISONGraph (base)          EmbeddingStore                        |
|  - Nodes & Edges           - SQLite backend (Python)             |
|  - Multi-hop traversal     - sentence-transformers               |
|  - Path finding            - Cosine similarity (numpy optional)  |
|                                                                  |
|                     Combined via:                                 |
|                                                                  |
|  semantic_multi_hop() = similarity_search() + graph.multi_hop()  |
|  semantic_path()      = similarity_search() + graph.shortest_path|
|  semantic_subgraph()  = semantic_multi_hop() + induced edges     |
|                                                                  |
+------------------------------------------------------------------+
```

## Comparison with RudraDB

| Feature | RudraDB | isongraph-vector |
|---------|---------|----------------------|
| Graph storage | Native | SQLite via ISONGraph |
| Embeddings | Built-in | sentence-transformers |
| Vector search | Native | Brute-force cosine (vectorized with numpy) |
| Multi-hop | Native | Python BFS/DFS |
| Auto-relationships | Trishul | Manual |
| Performance | Optimized | Good to a few hundred thousand vectors |
| License | Proprietary | MIT (Open Source) |

## API Reference

### EmbeddingStore

| Method | Description |
|--------|-------------|
| `add(node_ref, text, vector=None)` | Add embedding |
| `add_batch(entries)` | Add many, encoding in one batched call |
| `get(node_ref)` | Get embedding record |
| `remove(node_ref)` | Remove embedding |
| `similarity_search(query, top_k, node_type, threshold)` | Find similar nodes (`top_k=None` returns all) |
| `sqlite_vec` | Whether top-k search runs through the vec0 index |
| `score_map(query, node_type=None)` | Score every node, keyed by ref |
| `export_embeddings()` / `import_embeddings(payload)` | Portable cross-language payload |
| `save_embeddings(path)` / `load_embeddings(path)` | Same payload, as a JSON file |
| `dimension` | Vector width this store holds |
| `count()` | Count embeddings |
| `clear()` | Clear all embeddings |
| `close()` | Close the database connection |

### SemanticGraph

| Method | Description |
|--------|-------------|
| `add_node(..., _embed_text=)` | Add node with embedding |
| `embed_node(node_ref, text)` | Add/update embedding |
| `embed_nodes(items)` | Batch embed (node_ref, text) pairs |
| `get_embedding(node_ref)` | Get embedding record |
| `similarity_search(query, top_k, ...)` | Find similar nodes |
| `similar_to_node(node_ref, top_k, ...)` | More like this |
| `semantic_multi_hop(query, ..., blend=0.0)` | Semantic traversal |
| `semantic_path(query, target, ...)` | Semantic path finding |
| `semantic_subgraph(query, ...)` | On-topic slice as a new graph |
| `embed_all_nodes(text_fn=None, skip_existing=False)` | Batch embed nodes |
| `save(path, embeddings=True)` | Save graph and embedding sidecar |
| `load(path, encoder=None, ...)` | Load both back |
| `encoder` / `set_encoder(encoder)` | Inspect or install the encoder |
| `close()` | Release the embedding database |

### Errors

`EmbeddingError` is the base; `DimensionMismatchError` and `NoEncoderError`
derive from it (and from `ValueError` in Python, so existing `except
ValueError` handlers keep working).

## Testing

```bash
pytest tests/                              # Python, MockEncoder (no downloads)
USE_REAL_ENCODER=1 pytest tests/           # Python, real sentence-transformers

cd isongraph-vector-csharp && dotnet test   # C#
cd isongraph-vector-ts && npm test          # TypeScript (npm run typecheck for tsc)
cd isongraph-vector-js && npm test          # JavaScript
cd isongraph-vector-rs && cargo test        # Rust

cd isongraph-vector-cpp && cmake -S . -B build && cmake --build build
./build/test_isongraph_vector               # C++
```

`pytest` works straight from a checkout: [`conftest.py`](conftest.py) binds the
import name `isongraph_vector` to the `isongraph_vector-py/` directory, which
is named to line up with the other ports and so cannot be imported directly.
An installed copy takes precedence.

A note on thresholds in tests: `MockEncoder` produces near-orthogonal vectors,
and cosine similarity is signed, so mock scores straddle zero. Tests that want
every match pass `threshold=-1.0` rather than relying on the 0.0 default.

## Dependencies

```
ison-graph>=1.0.0
sentence-transformers>=2.2.0
numpy>=1.21.0            # optional, via the "fast" extra
sqlite-vec>=0.1.6        # optional, via the "vec" extra
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Links

- Repository: https://github.com/ISON-format/isongraph-vector
- Documentation: https://graph.ison.dev/docs/isongraph-vector
- Issues: https://github.com/ISON-format/isongraph-vector/issues

## Author

Mahesh Vaikri

## License

MIT License - see [LICENSE](LICENSE) for details.
Copyright (c) 2026 ISON - Mahesh Vaikri.
