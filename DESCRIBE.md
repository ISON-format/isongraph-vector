# isongraph-vector — Detailed Capabilities

## Is this useful?

Yes, for a specific, real gap: `ison-graph`'s own graph traversal has no
concept of semantic relevance — it can walk relationships, but it can't
find "the nodes semantically closest to this query" and then traverse
from there. `isongraph-vector` adds that: `SemanticGraph` is a real
subclass of `ISONGraph` (not a wrapper or a reimplementation) that adds
a SQLite-backed embedding store and combines vector similarity with
graph traversal in one call (`semantic_multi_hop`) — find the most
relevant seed nodes by meaning, then walk the graph from there with the
relevance score decaying per hop.

It's deliberately lightweight — exact cosine similarity (an opt-in
IVF index exists in Python only), one embedding per node. That's a real scoping choice, not an
oversight: this is meant to sit on top of `ison-graph` for
small-to-medium graphs where "traverse from what's semantically
relevant" matters more than sub-millisecond search over millions of
vectors. For that latter case, the ecosystem's other project (RudraDB —
vectors on both nodes *and* edges, ANN indexing, link prediction) is the
right tool, not this one; they're not really competitors despite the
surface similarity.

## Shape of the repository

Six implementations of the same library, plus a benchmark suite that
belongs to a different package:

| Port | Implementation | Tests |
| --- | ---: | ---: |
| Rust (`isongraph-vector-rs/`) | 1,850 | 50 |
| C++ (`isongraph-vector-cpp/`, header-only) | 1,823 | 71 |
| Python (`isongraph_vector-py/`) | 1,768 | 73 |
| C# (`isongraph-vector-csharp/`) | 1,447 | 61 |
| TypeScript (`isongraph-vector-ts/`) | 1,441 | 69 |
| JavaScript (`isongraph-vector-js/`) | 1,108 | 67 |

Line counts are implementation only - the inline test module at the foot
of the Rust crate is excluded, as are every port's test files. Rust and
C++ lead because each carries its own SQLite backend in the same source;
C++ also brings its own MD5 and a hand-written JSON reader rather than
pulling in dependencies.

`LLMContext_Benchmark/` (3,269 lines) measures ISON *serialization* token
efficiency against NetworkX/igraph/SQLite/JSON. It contains zero
references to embeddings — it belongs to `ison-graph`, not here, and is
kept only because it still runs and its committed results are real.

The Python port is the reference implementation: the real
`sentence-transformers` encoder and graph save/load. The other five are
`MockEncoder`-only but share the traversal semantics, the scoring model,
the portable embedding format and the SQLite schema exactly. Every port
can persist to SQLite and index with sqlite-vec; all six write the same
two tables, so a database file written by one opens in another.

## Maturity

Real, working, tested — 391 tests across six languages, all passing,
all six verified by actually building and running them rather than by
reading the code and assuming it worked.

That method is the point. Every defect below was found by running
something, and several had been sitting in the code for months across
the five ports that existed at the time — the C# port was written later,
against the fixed behaviour:

- **The package could not be built or installed at all.**
  `pyproject.toml` declared `packages.find` with `where = ["src"]` and
  there has never been a `src/` directory: `pip install .` failed with
  `error in 'egg_base' option: 'src' does not exist`, and `pytest` could
  not even collect the suite. The package directory is named
  `isongraph_vector-py` to line up with the other ports, which is
  not a legal Python module name, so the import name is now mapped onto
  it explicitly via `[tool.setuptools.package-dir]`, and a root
  `conftest.py` does the same for a checkout. Wheel verified.

- **A directly-matching node lost its own score — in every port then written.**
  In `semantic_multi_hop`, a seed was inserted only `if node_ref not in
  results`, while the neighbour branch right below it compared scores
  before overwriting. So a node that was itself a strong direct match,
  but had already been reached by an earlier seed's traversal, was
  reported at the decayed score, at the wrong hop count, down a path it
  never needed: a 0.9 direct match came back as 0.8 at hop 1. This is
  the package's headline feature. Fixed in Python, TypeScript,
  JavaScript, Rust and C++, each with a regression test that asserts the
  actual score. The old tests missed it because they only asserted that
  results were non-empty and sorted.

- **`check_same_thread=False` had been added without a lock.** The
  comment on it says it exists so `ison-cache`'s HTTP API can reach the
  store from request threads — but a single `sqlite3.Connection` was
  shared with no serialization. Measured at 8 threads:
  `sqlite3.InterfaceError`, `None` rows returned mid-query, and ~0.5% of
  writes silently lost, reproducibly, on both `:memory:` and file
  databases. Every operation now holds a re-entrant lock; the same
  stress test passes with zero errors and zero lost writes, and it is
  part of the suite.

- **`MockEncoder` collapsed distinct texts onto identical vectors.** It
  emitted the *low* 16 bits of an LCG, and an LCG's low bits evolve
  independently of its high bits — so every 384-dim vector was
  determined by `seed mod 65536`. In the exact-arithmetic ports
  (Rust/C++) that was 280 outright collisions in 5,000 texts, with
  colliding texts scoring a perfect 1.0 against each other; JavaScript
  escaped it only because `hash * 1103515245` overflows float64, which
  also meant its vectors did not match the other ports at all. Replaced
  everywhere with FNV-1a seeding an xorshift32, taking the top 24 bits.
  All six ports now produce **byte-identical vectors for the same
  text**, asserted against the same golden values in all six suites.

- **Nothing validated vector dimensions.** A 2-dim vector added to a
  384-dim store scored a plausible-looking 0.06 against a 384-dim query
  instead of failing; an empty vector scored 0.0. A store now pins its
  dimension to the first vector it sees and raises
  `DimensionMismatchError` on anything else, queries included.

- **A save/load round trip silently lost every embedding.**
  `SemanticGraph.load()` defaulted `embedding_db=":memory:"`, so a
  reloaded graph searched an empty store and returned `[]` with no
  error, and the existing test passed because it only compared node and
  edge counts. `save()` now writes an embeddings sidecar next to the
  graph and `load()` restores it. `load()` also takes an `encoder=`
  argument: without one, the first search on a loaded graph quietly
  built a `SentenceTransformerEncoder` and downloaded a model, even for
  code that had used `MockEncoder` throughout.

- **The decay model was only correct for non-negative scores.** Cosine
  similarity is signed, and `seed_score * decay ** hops` *raises* a
  negative score toward zero — so every node a negatively-matching seed
  reached outranked the seed itself, and relevance grew with distance.
  The traversal base is now clamped at zero. This one surfaced because
  fixing `MockEncoder` made its vectors genuinely near-orthogonal, which
  broke a test that had been passing on the old encoder's accidental
  positivity.

- **`autoEmbed` did not apply to the plain `addNode` outside Python.**
  Python overrode `add_node`; the other four ports left the base method
  inherited and put the logic in a separate `addNodeWithEmbed`, so with
  `autoEmbed: true` — the default — `graph.addNode(...)` silently
  produced an unembedded, unsearchable node. All four now auto-embed
  through the method callers actually reach for.

## What it can actually do

**`EmbeddingStore`** — SQLite-backed (or in-memory) vector storage:
`add`, `add_batch` (one batched encoder call), `get`, `remove`,
`similarity_search(query, top_k, node_type, threshold)` where `top_k=None`
returns everything, `score_map` (score every node in one scan), `count`,
`clear`, `close`, and `export_embeddings` / `import_embeddings` /
`save_embeddings` / `load_embeddings` for the portable format. Thread-safe.
Three search backends, all returning the same results. Pure Python
scores every vector in a loop. With numpy installed (the `fast` extra,
the default when present) the scan becomes one matrix-vector product
against a cached matrix: 2.6 ms rather than 182 ms per query over 5,000
nodes. With `sqlite_vec=True` (the `vec` extra) top-k queries are
answered by a sqlite-vec `vec0` virtual table instead — `vec0` runs an
exact brute-force KNN in C, so this buys speed without trading accuracy:
identical top-10 ordering over 50 queries against 20,000 vectors, scores
within 2e-07. It earns its keep only at scale — 0.6x at 1,000 vectors,
1.3x at 5,000, 1.4x at 20,000 — and costs writes (20,000 rows in 0.43 s
rather than 0.16 s), which is why it is opt-in. `score_map()` and
blended multi-hop keep scanning, because they need a score for every
node rather than a top-k; `node_type` filtering stays exact either way,
since the type is a `vec0` metadata column. A bounded `top_k` is
answered by `heapq.nlargest` rather than by sorting every candidate —
the other five ports do the same with `std::partial_sort`,
`select_nth_unstable_by` or a bounded insertion, and every suite asserts
the selection returns what a full sort would.

**`SemanticGraph(ISONGraph)`** — every real `ISONGraph` method still
works (`add_edge`, `neighbors`, `shortest_path`, `save`/`load`, ...)
plus:

- `add_node(type, id, _embed_text=..., **properties)` — auto-embeds from
  `_embed_text`, or from `embed_fields` (default `name`/`description`/
  `content`/`text`) found in `properties` if `auto_embed=True`.
- `similarity_search(query, top_k, node_type=None, threshold=0.0)` —
  takes text or a pre-computed vector.
- `similar_to_node(node_ref, top_k, ...)` — "more like this", searching
  with the node's own stored vector and excluding itself. No encoder call.
- `semantic_multi_hop(query, rel_type=None, max_hops=2, top_k_seeds=5, top_k_results=10, decay=0.8, threshold=0.3, blend=0.0)`
  — the headline method: seed from similarity search, traverse, decay
  score by `decay ** hop_count`, keep the best score per node. `blend`
  mixes each node's own relevance to the query into its traversal score,
  so a node found two hops out but highly relevant in its own right is not
  buried under a closer, unrelated neighbour.
- `semantic_path(query, target_ref, rel_type=None, max_hops=5, top_k_seeds=3, decay=0.9, threshold=0.0)`
  — finds a path from a semantically-relevant seed to a specific target.
- `semantic_subgraph(query, ...)` — returns a **new `SemanticGraph`**
  holding the matching nodes, every edge induced between them, and their
  embeddings. This is what makes an ISON graph useful as LLM context:
  `graph.semantic_subgraph(question).save(path)` yields a small, on-topic
  graph to inject rather than the whole store.
- `embed_nodes(items)` / `embed_all_nodes(text_fn=None, skip_existing=False, batch_size=256)`
  — batch-embed an existing graph in one encoder call per batch.
- `save(path, embeddings=True)` / `load(path, encoder=None, embeddings=True)`
  — graph state and embeddings together.
- Context manager (`with SemanticGraph(...) as graph:`) and `close()`.

**Encoders** — `SentenceTransformerEncoder` (default `all-MiniLM-L6-v2`,
real model, requires `sentence-transformers`) or `MockEncoder`
(deterministic, zero dependencies, identical across all six ports — has
no real semantic structure, don't use it to validate relevance quality,
only plumbing).

**Portable embedding format** — one JSON shape
(`format: "ison-embeddings"`, version 1) that every port reads and
writes, so a store built in Python can be loaded by the Rust or C++ one.
Verified, not assumed: the C#, TypeScript, JavaScript, Rust and C++
suites each import a payload captured verbatim from the Python port and
check that the vectors match what they encode themselves for the same text.

**Already wired into `ison-cache`** as `SemanticGraphCache`, exposed
over that project's HTTP, RESP, and MCP interfaces — this package is the
thing that capability is actually built on. (Not re-verified in this
pass; that lives in a different repository.)

## Example: semantic multi-hop over a small social graph

```python
from isongraph_vector import SemanticGraph

graph = SemanticGraph(name="social")

graph.add_node('person', 1, name='Alice',
               _embed_text='Alice is a software engineer who builds APIs')
graph.add_node('person', 2, name='Bob',
               _embed_text='Bob is a data scientist analyzing ML models')
graph.add_node('person', 3, name='Charlie',
               _embed_text='Charlie is a product manager planning roadmaps')

graph.add_edge('KNOWS', ('person', 1), ('person', 2))
graph.add_edge('KNOWS', ('person', 2), ('person', 3))

# Who's semantically closest to "machine learning", and who do they know?
results = graph.semantic_multi_hop(
    query="machine learning",
    rel_type='KNOWS',
    max_hops=2,
)
for r in results:
    print(f"{r.node_ref}: score={r.score:.4f}, hops={r.hop_count}, path={r.path}")

# Keep the whole thing, embeddings included
graph.save("social.isong")
graph = SemanticGraph.load("social.isong", encoder=graph.encoder)
```

With the real `all-MiniLM-L6-v2` encoder that prints:

```text
('person', 2): score=0.4585, hops=0, path=[('person', 2)]
('person', 3): score=0.3668, hops=1, path=[('person', 2), ('person', 3)]
```

Bob is the data scientist, so he matches "machine learning" directly and
seeds the search at hop 0. Charlie is one `KNOWS` hop out and arrives
decayed by `decay ** hop_count`.

Alice is the instructive absence. She is in the graph, and she is one hop
from Bob — but `KNOWS` edges are directed and traversal follows
`Direction.OUT`, so from Bob the search reaches who *Bob* knows, not who
knows Bob. Pass `direction=Direction.BOTH` to walk the relationship in
either direction. Her own similarity to "machine learning" is also below
the default `threshold` of 0.3, so she never seeds the search either.

## Limitations

- Similarity search is exact (brute force) in every backend, including
  sqlite-vec: `vec0` is a fast exact KNN, not an approximate index. Cost is
  linear in the number of nodes. In Python the numpy path is vectorised and
  patches its cache on writes, which moved the point where a search passes
  50 ms from roughly 90,000 vectors to (extrapolated) around a million;
  see `benchmarks/`. Python has an opt-in IVF approximate index
  (`ann="ivf"`) that pays off from about a million vectors; the other ports are
  exact only. This stays an application's
  own knowledge graph rather than a vector-database replacement.
- One embedding per node, generated from a single text field/concat -
  no multi-vector-per-node, no edge embeddings (that's RudraDB's domain,
  not this package's).
- Only Python ships a model (`SentenceTransformerEncoder`). Every port has an
  `HttpEncoder` for an OpenAI- or Ollama-style endpoint (texts are sent to it);
  otherwise bring your own `EmbeddingEncoder` over a local model.
- Graph persistence (`save` / `load` of the graph file plus the
  `<path>.embeddings.json` sidecar) exists in every port. The sidecar carries a
  digest of the graph, verified on load, so a graph and embeddings from
  different saves are caught instead of silently mixed. The ISON file cannot
  store the graph name or directedness: after `load` the name comes from the
  file name and the graph is directed.
- `update_node` re-embeds an edited node, and the underlying graph is
  read-only in C#, C++ and Rust, so node edits cannot skip it. In C# the view
  returns copies, so editing one does nothing.
- Subgraph extraction keeps the top-k nodes and induced edges; pass
  `connect=True` to add the connector nodes needed to join them. The
  connection is greedy within a hop budget, not a minimum Steiner tree, and it
  can exceed `top_k_results`.
- Traversal direction matters more than decay or blend. The default follows
  edges outward, which answers "what does this depend on" but not "what
  depends on this"; for the latter pass `Direction.IN` or `BOTH`
  (`benchmarks/README.md` has the numbers).
- Vectors from different encoders are not comparable. Each stored vector
  records its model, and a text query against vectors from another model warns
  (or errors). Use `stale_embeddings()` and `reembed_stale()` to refresh them.
  An encoder `version` is recorded in the sidecar and warns on a change, but a
  version-only change is not treated as stale.
- Distributed or multi-tenant retrieval, access control and any server layer
  are out of scope for an embedded library.
- C++ remains the one port where the base graph library is vendored rather
  than declared as a dependency, because there is no package registry to
  declare it in.
- `MockEncoder` vectors are near-orthogonal noise by design. Cosine
  similarity is signed, so mock scores straddle zero and a `threshold` of
  0.0 will reject perfectly good matches. Tests that want everything pass
  `threshold=-1.0`.
- ISONGraph 1.4.0 made property values typed, so auto-embedding has to
  turn a value into text. Nulls are skipped, booleans render as ISON spells
  them (`true`/`false`) and a whole-number float renders without a trailing
  `.0`, in every port; a golden test in all six asserts the same property bag
  yields the same string.
