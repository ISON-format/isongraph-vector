# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-01

First public release, across six language ports: Python, C#, TypeScript,
JavaScript, Rust and C++.

### Added

- `EmbeddingStore` - vector storage with cosine similarity search. SQLite-backed
  and thread-safe in Python, in-memory in the other ports. Batched writes
  (`add_batch`), `score_map` for scoring every node in a single scan, and
  dimension validation on both stored vectors and queries.
- `SemanticGraph` - `ISONGraph` extended with embeddings. `add_node` auto-embeds
  from node properties in every port.
- `semantic_multi_hop` - similarity-seeded traversal scored by
  `max(seed_similarity, 0) * decay ** hop_count`, with an optional `blend`
  parameter that mixes each node's own relevance to the query into its
  traversal score.
- `semantic_path` - a path from a semantically relevant seed to a target node.
- `semantic_subgraph` - the on-topic slice of a graph returned as a new
  `SemanticGraph`, carrying the matching nodes, every edge induced between
  them, and their embeddings. Intended for building LLM context from a graph.
- `similar_to_node` - "more like this", searching with a node's own stored
  vector and no encoder call.
- `embed_all_nodes` / `embed_nodes` - batch embedding, with `skip_existing`.
- Graph persistence in Python: `save()` writes an embeddings sidecar next to
  the graph file and `load()` restores it. `load()` takes an `encoder=`
  argument so a reloaded graph does not silently download a model.
- A portable embedding payload (`ison-embeddings`, version 1) that all six
  ports read and write, so a store built by one can be searched by another.
  Exposed as `export_embeddings` / `import_embeddings` (`to_json` / `from_json`
  in Rust, `toJson` / `fromJson` in C++).
- `MockEncoder` - deterministic, dependency-free, and byte-identical across all
  six ports, so cross-port stores and tests line up exactly.
- Optional numpy fast path for Python similarity search, roughly two orders of
  magnitude faster than the pure-Python fallback, via the `fast` extra.
- A SQLite-backed store in every port, optionally indexed with sqlite-vec.
  Off by default everywhere, so the core packages stay light: Python's `vec`
  extra, the `IsonGraph.Vector.Sqlite` NuGet package, the
  `isongraph-vector-{ts,js}/sqlite` entry points (built on Node's own
  `node:sqlite`, so the main entry stays browser-safe), the Rust `sqlite`
  feature, and the C++ `-DISONGRAPH_VECTOR_SQLITE=ON` option.
- All six ports write the same SQLite schema, so a database written by one
  opens in another - verified by opening Rust- and Node-written databases from
  Python and reading byte-identical vectors.
- sqlite-vec answers top-k queries from a `vec0` virtual table rather than
  scanning. Exact, not approximate - identical ordering and scores within
  2e-07 of the scan - and about 4x faster over 20,000 vectors, though slower
  below a few thousand, which is why it is opt-in. `node_type` filtering is a
  `vec0` metadata column, so it narrows the search rather than filtering its
  results.
- Typed errors: `EmbeddingError`, `DimensionMismatchError`, `NoEncoderError`.
- 371 tests across the six ports, including cross-port checks that import an
  embedding payload produced by the Python port, checks in every port that the
  sqlite-vec backend returns exactly what the scan returns, and a golden
  embedding-text assertion shared by all six.

- Every port builds on ISONGraph 1.4.0 (`ison-graph` on PyPI and crates.io,
  `IsonGraph` on NuGet, `ison-graph-{ts,js}` on npm). The C++ port vendors the
  canonical `ison-graph-cpp` 1.4.0 header plus the `ison-cpp` 1.2.0 parser it
  requires, replacing a stale 1.0.0 backup copy.
- 1.4.0 makes graph property values typed rather than string-only, so
  auto-embedding renders them: nulls are skipped and booleans use the ISON
  spelling (`true`/`false`) in every port, keeping one property bag mapped to
  one embedding text - and so one vector - across all six. A golden test in
  each suite asserts it. Whole-number floats still differ (`"1.0"` from
  Python, `"1"` elsewhere); pass an explicit embed text when that matters.
- Rust `add_node` takes `Vec<(&str, PropertyValue)>`; `add_node_with_embed`
  keeps its string-valued signature and lifts values for you. The C# port
  takes `IDictionary<string, object?>`.

### Notes

- The Python import name is `isongraph_vector`; the other ports are published as
  `IsonGraph.Vector` (NuGet), `isongraph-vector-ts`, `isongraph-vector-js` and
  `isongraph-vector-rs`, with the C++ port consumed as headers.
- Only the Python port persists graphs. The others are in-memory and use
  `MockEncoder`; they can still exchange embeddings through the portable
  payload.
- Similarity search is brute force. There is no ANN index: this is built for an
  application's own knowledge graph, not as a vector-database replacement.
