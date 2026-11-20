# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [1.0.0] - 2026-11-15

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
  2e-07 of the scan. Over a SQLite scan it is worth 2x (C++) to 8x (Node) at
  20,000 vectors; over an in-process scan the margin narrows to 1.3-1.4x, and
  for C# it inverts, its in-memory scan at 5.9 ms beating the 7.1 ms round
  trip through `vec0`. Below a few thousand vectors the scan wins outright
  everywhere, which is why it is opt-in. `node_type` filtering is a `vec0`
  metadata column, so it narrows the search rather than filtering its
  results.
- Top-k results come out of a partial selection rather than a full sort in
  every port - `heapq.nlargest` in Python, `std::partial_sort` in C++,
  `select_nth_unstable_by` in Rust, a bounded insertion in C#, TypeScript and
  JavaScript. Ordering 20,000 candidates to return ten was costing about a
  third of each query. Every suite asserts the selection returns what a full
  sort would.
- Typed errors: `EmbeddingError`, `DimensionMismatchError`, `NoEncoderError`.
- A tag-driven release workflow. Pushing `v<major>.<minor>.<patch>` checks
  the tag against all nine version declarations, reruns CI on that ref,
  builds every artifact once, publishes to PyPI, NuGet, npm and crates.io in
  parallel through OpenID Connect trusted publishing, and cuts a GitHub
  release carrying the CHANGELOG section, every artifact, a C++ header
  tarball and `SHA256SUMS`. `RELEASING.md` documents it.
- Continuous integration across all six ports, plus a cross-port interchange
  job that has one port write a SQLite database and a JSON payload and another
  read both back, comparing float32 vectors exactly. Python is also run once
  with numpy and sqlite-vec uninstalled, so the pure-Python cosine loop and the
  table-scan fallback are exercised rather than assumed. Rust, C++ and C# build
  on Linux, Windows and macOS.
- 391 tests across the six ports, including cross-port checks that import an
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
  each suite asserts it, floats included: all six render the shortest
  round-trip form, so an integral float is `"1"` rather than Python's usual
  `"1.0"`, and `1/3` keeps full precision rather than C++'s default six
  significant figures.
- Rust `add_node` takes `Vec<(&str, PropertyValue)>`; `add_node_with_embed`
  keeps its string-valued signature and lifts values for you. The C# port
  takes a concrete `Dictionary<string, object?>` so a collection initializer
  (`new() { ["name"] = "Alice" }`) target-types against it.
- Rust re-exports `NodeId` alongside `PropertyValue`. It appears in every store
  signature, so without the re-export a caller who never touches `ISONGraph`
  directly still needed `ison-graph` in their `Cargo.toml` to name one.
- The `/sqlite` entry point of the TypeScript and JavaScript ports needs Node
  24, because it imports `node:sqlite`. The main entry point stays
  runtime-agnostic and runs on Node 18 and up, or in a browser; `npm run
  test:core` covers that half alone.
- **Node filtering** - `node_filter` / `node_types` (and `strict_filter` for
  traversal) on `similarity_search`, `semantic_multi_hop`, `semantic_path` and
  `semantic_subgraph`. Seeds and results must pass; by default a node that
  fails may still be traversed, with `strict_filter` it is never entered.
- **Edge weighting** - `weight_field` on `semantic_multi_hop` and
  `semantic_subgraph`: an edge's numeric property (clamped to 0..1) scales that
  hop's decay. Unset keeps scores bit-identical.
- **Connected subgraphs** - `connect` / `connect_max_hops` on
  `semantic_subgraph` add the shortest connector nodes needed to join the
  selected nodes, so a subgraph no longer silently splits when a bridging
  node scored below the cut. Greedy, not a minimum Steiner tree.
- **Encoder and model lifecycle** - encoders expose `model_name` and
  `version`; every stored vector records its model. New `models()`, `stale()`,
  `stale_embeddings()` and `reembed_stale()` find and refresh vectors from a
  different model or dimension, and a text query against vectors from another
  model warns (or errors, per `on_model_mismatch`).
- **Graph/vector consistency** - `save` writes the graph file and the
  embeddings sidecar atomically (temp file, flush, rename) and records a
  SHA-256 digest of the graph in the sidecar; `load` verifies it and, on a
  mismatch, errors, warns or re-embeds (`on_mismatch`). The digest is defined
  once (`nodes-edges-json-v1`) and checked in every port against a shared
  golden fixture, `tests/fixtures/graph_digest_golden.json`. Sidecars without
  a digest, and unknown extra fields, load as before.
- **Graph save/load in every port** - previously Python only. Rust, C#, C++,
  and TypeScript / JavaScript (under the new Node-only `/node` entry, so the
  browser-safe core stays free of `fs`) read and write the same graph file and
  `<path>.embeddings.json` sidecar as Python, in both directions.
- `benchmarks/` - a retrieval-quality benchmark (vector-only versus
  graph-aware, against structural ground truth) and a scale benchmark
  (latency, ingestion, memory, concurrency), with results and findings.
- `ROADMAP.md`.
- `update_node` re-embeds a node when one of `embed_fields` changes (and drops
  the vector if no text is left), so an edited node is never searched by its
  old text. Graphs with `auto_embed=False` are left alone.
- `semantic_path` compares every candidate seed and returns the best-scoring
  path, clamping a negative seed score to zero as `semantic_multi_hop` does.
- Python: the exact numpy search is vectorised and writes patch the cached
  matrix instead of discarding it (at 100k vectors a search takes about 5.6 ms
  and the search right after a write about the same).
- **Approximate search (Python, opt-in)** - `ann="ivf"` on `EmbeddingStore` /
  `SemanticGraph` builds a numpy IVF-flat index lazily, keeps it current under
  writes and deletes, applies `node_type` / `threshold` / `node_filter` to the
  probed rows (widening, then falling back to exact, when too few survive), and
  offers `recall_check()` to measure recall against the exact scan. Off by
  default; `exact=True` forces the full scan. At 1M clustered vectors it is
  20-160x faster than exact search; on hard, diffuse data only 2-3x, and it
  does not pay at or below 100k (see `benchmarks/`).
- **`HttpEncoder`** - embeds through an OpenAI-style (`data[].embedding`) or
  Ollama (`/api/embed`) endpoint in Python, JavaScript, TypeScript, C#, C++
  and Rust (Rust behind the `http-encoder` feature, C++ with a pluggable
  transport and an optional libcurl one). API keys are redacted from errors,
  redirects are refused, and every embedded text is sent to the configured
  endpoint. The JavaScript ports serve the synchronous encoder interface from
  a `prefetch` cache.
- **Model lifecycle for the standalone `SqliteEmbeddingStore`** in
  TypeScript, JavaScript, Rust and C++, matching the in-memory store. Existing
  databases gain a `model` column on open, and databases stay readable across
  ports.
- **Node edits cannot bypass re-embedding** in C#, C++ and Rust: C# `Graph` is
  a read-only view, C++ `graph()` is const and Rust has no `graph_mut()`;
  `AddEdge` / `RemoveEdge` live on `SemanticGraph`. C++ `addNode` and
  `updateNode` also undo themselves if embedding throws.
- A cross-port CI check (`scripts/cross_port/graph_files.py`) of graph files,
  sidecars and the graph digest: every port writes, every port reads, and a
  tampered file must raise each reader's mismatch error (72 checks).
- **Approximate search in every port (opt-in)** - the same lazily built,
  never-persisted IVF-flat index (seeded k-means, nlist about 4*sqrt(N), nprobe
  about 5%, minimum size 50,000, exact override, `recall_check`) in Rust
  (`set_ann`), C++ (`enableAnn`), C# (`AnnMode.Ivf`) and TypeScript /
  JavaScript (`ann: 'ivf'`). Graph files, sidecars and the digest are
  unchanged. Seeds do not reproduce across ports (each uses its own generator),
  so indexes differ while recall is comparable. The standalone SQLite stores
  have no index. See `benchmarks/competitive/` for measured speed and recall.
- **Rust: HNSW, int8 scan, binary snapshot** (all opt-in; design in
  `isongraph-vector-rs/DESIGN.md` sections 10-12, measured in
  `benchmarks/rust-stage-c/`). `AnnConfig::hnsw()` adds an HNSW index that
  holds links only (about 1.2x raw memory at 1M x 384); at recall@10 0.95 its
  single-thread speed is level with faiss HNSW and hnswlib and ahead of
  usearch. It builds about 8-20x slower than IVF (about 150 s at 1M) and is
  not persisted. `ScanPrecision::Int8` adds an int8 first pass with exact
  rerank: 3-3.6x faster single-query scans at 1M, recall 1.000 against the
  f32 scan from rerank factor 2 on real embeddings, no gain for batches, +0.25x
  memory, and not used by HNSW graph search. `save_with_snapshot` writes a
  binary `.snap` beside the JSON sidecar: 3x smaller and about 6x faster to
  load at 1M (JSON stays the portable default; graph file and digest bytes are
  unchanged; a stale or corrupt snapshot falls back to JSON).
- **Rust engine rework** (design in `isongraph-vector-rs/DESIGN.md`, measured
  in `benchmarks/rust-stage-a/`): exact search uses AVX2/FMA kernels and
  threads; searches take `&self` on a persistent pool, with batch search
  (`similarity_search_vector_batch` and friends) that returns bit-for-bit the
  same answers as one query at a time; `SharedEmbeddingStore` for threads that
  read and write together; each vector is held once in a chunked matrix
  (1.12x raw at 1M x 384, was 2.41x); pending rows are folded into the index
  in groups of 256 at write time, so the first query after 30k writes into 1M
  takes about 8 ms (was about 600 ms). Equal scores are ordered by arrival.
  **Breaking:** `EmbeddingStore::get` and `SemanticGraph::get_embedding` now
  return an owned `Option<EmbeddingRecord>`; use `get_vector` to borrow.
  Costs: single-thread exact scan is 0-13% slower, ingest is slower, and
  updates at 1M are 2-3x slower because index upkeep moved from the first
  search to the write.
- **C++ and C#: opt-in HNSW** (`DESIGN.md` section 12, ported from the Rust
  index; links only, about 145 B/node, never persisted, graph file / sidecar /
  digest / snapshot bytes unchanged). C++: `AnnOptions::makeHnsw()`,
  `setAnnEfSearch`, `similaritySearchVectorEf`. C#: `AnnMode.Hnsw`,
  `HnswOptions`, `AnnEfSearch`, `SimilaritySearchVectorEf`. Built
  deterministically (the graph does not depend on the thread count); batch
  equals single-query bit for bit. Checked against Rust on shared integer data:
  C# ids and order identical in 6 of 6 parameter sets, C++ in 4 of 5 (one query
  in 200 differed by one id), and identical in score bits too once C++ used
  Rust's summation order, so the spec reproduces the Rust graph; scores are
  bit-identical to the exact scan within a port, not across ports. Measured at
  100k x 384 on a shared Windows laptop, synthetic clustered vectors (not real
  text): the whole store is 1.21-1.22x raw; build 10-14 s (C++ g++) /
  28-35 s (C++ MSVC) / about 24 s (C#) against 1-3 s for IVF; inserts 1.1-2.6
  ms. Single-query speed over IVF at equal recall ranged from 1.2x (C++, recall
  0.74, hard noise) to about 10x (C#, recall 0.95); on Rust's real-text data it
  is 4x at 100k, so treat the C++ figure as unexplained and run your own data.
  Side by side on one machine C# is about 1.6x slower to build and 0.6-0.7x the
  QPS of Rust, because its scalar kernel stands in for AVX2/FMA. Not run on
  Linux/GCC 15, macOS or real text embeddings. **Breaking (binary only, C#):**
  new trailing optional `annHnsw` parameter on four public signatures;
  `AnnMode` gained a case. C++: none at source level (`enableAnn` now rejects
  `hnsw.m` outside 2..64).
- **Python, C#, TypeScript, JavaScript and C++: one copy of every vector,
  batch search, binary snapshot** (the Rust design of `DESIGN.md` sections 1, 5
  and 11 carried over; measured per port below). Each store now keeps every
  vector once in a chunked (or contiguous) float32 table and records refer to
  its rows, so the second copy that the index or the scan cache used to hold is
  gone, and pending rows fold into the IVF index at write time (JavaScript /
  TypeScript keep their search-time fold: a fold inside a write would block it
  for seconds). Memory at 100k x 384 (raw 146 MB, steady RSS above baseline,
  3 runs, shared laptop): Python 437 -> 211 MB, TypeScript / JavaScript 435 ->
  197 MB (590 -> 203 MB with the index), C# IVF store 409 -> 215 MB, C++ IVF
  store 329 -> 167 MB; at 1M x 384 (one run each) the new figures are 1.12-1.25x
  raw (Python 1.25x, TypeScript 1.21x, C# 1.15x exact / 1.18x with IVF, C++
  1.12x with IVF; JavaScript was not measured at 1M);
  the old 3.3-7.0 GB figures are in `benchmarks/competitive/`. New:
  `similarity_search_batch` (Python: one matrix-matrix product; 8x one-at-a-time
  at 100k), `SimilaritySearchBatch` (C#), `similaritySearchBatch` (C++),
  parallel across queries and bit-identical to single-query search in C# and
  C++; searches are safe to run concurrently in C# and C++. The optional
  `.embeddings.snap` file (`save(snapshot=True)` / `Save(..., snapshot: true)` /
  `saveGraph(..., {snapshot: true})`, byte layout in Rust `DESIGN.md` section
  11) is written and read by all six ports, is 3-5x smaller and roughly
  13-45x faster to save and load (C#, C++, Python) than the JSON sidecar at 100k x 384, and is the only
  way to persist a 100k-row store in JavaScript (the JSON export exceeds V8's
  string limit). A snapshot is used only when it provably belongs to the graph
  and sidecar on disk, otherwise the JSON is read; graph files, sidecar bytes
  and digests are unchanged. Cross-port: a snapshot written by any port loads
  in every other port (`scripts/cross_port/graph_files.py`).
  **Breaking, per port:** `get` / `get_embedding` return a copy of the vector
  in C#, C++ (`std::optional<EmbeddingRecord>`), TypeScript and JavaScript;
  `AnnStats.vectorBytes` / `cacheBytes` are 0 (the index owns no vectors);
  pending rows fold below 256 at write time (Python, C#, C++);
  TypeScript / JavaScript store vectors as float32 (inputs that are not
  float32-exact read back rounded: `0.1` -> `0.10000000149011612`; scores
  changed by at most 1.3e-8 in a 1,500-vector check); C# `Add` copies the
  caller's array; Python in-memory stores no longer have a SQLite table; equal
  scores order by arrival in C# and C++. **Python exact top-k scores are now
  canonical:** BLAS narrows the field and the rows near the k-th score are
  rescored with one kernel, so batch equals single bit for bit; scores can
  differ from the previous release in the last float32 bit and near-ties can
  reorder (`store.canonical_scores = False` restores the old behaviour).
  Costs: single-query IVF is within noise in Python and C++, about 9% slower in
  TypeScript / JavaScript and 8-25% slower in C# (its scalar kernel keeps C#'s
  own summation order, so C# scores can still differ from Rust's in the last
  bits, as before); C++ IVF writes cost about 5 us more because of the
  fold.
- **Async encoders in TypeScript / JavaScript** - `AsyncEncoder` and
  `...Async` methods (`addNodeAsync`, `updateNodeAsync`, `embedAllNodesAsync`,
  `reembedStaleAsync`, `similaritySearchAsync`, `semanticMultiHopAsync`, ...)
  let `HttpEncoder` auto-embed without a prefetch. Failures are atomic. The
  synchronous interface and `prefetch` still work.
- **C++ libcurl transport is built and tested against real libcurl** (8.11.1,
  loopback server, no internet). Two bugs fixed: `Expect: 100-continue`
  stalling large bodies, and a misleading error for over-cap responses. A new
  `cpp-curl.yml` workflow runs it on Linux and macOS.
- **`isongraph-vector-server` (alpha)** - a separate package that serves the
  library over HTTP: per-tenant API keys (salted hashes), isolated graphs,
  quotas and rate limits, search / multi-hop / path / subgraph / similar,
  atomic checksummed snapshots with digest verification, a tenant-sharding
  router and digest-verified read replicas. Not provided: TLS, failover,
  resharding, a write-ahead log, cross-tenant queries.
- **Real-data benchmarks** - `benchmarks/multihop/` (HotpotQA, 2Wiki,
  MuSiQue; two encoders; graphs built without the gold annotations; negative
  controls; BM25, hybrid, cross-encoder and personalised-PageRank baselines)
  and `benchmarks/competitive/` (faiss, hnswlib, usearch, chromadb, lancedb,
  sqlite-vec, numpy on text, GloVe and SIFT vectors at 100k and 1M).

### Known limits

- `C#` copies: nodes and edges read from the C# read-only graph view are
  copies, so editing one's `Properties` silently changes nothing.
- The synchronous JavaScript / TypeScript encoder interface still only serves
  prefetched texts; use the `...Async` methods to embed through a network
  encoder. `loadGraph({onMismatch: 'reembed'})` has no async form.
- Approximate search is IVF-flat in every port; Rust, C++ and C# also have opt-in HNSW. Recall on real embeddings was
  measured at 100k and 1M (`benchmarks/competitive/`); the index is not
  persisted and is rebuilt after load. JavaScript builds are slow (about 8
  minutes at 1M vectors).
- **Not a vector-database replacement.** Against faiss, hnswlib and usearch
  the ports are 2-4x slower per query at 100k and about 14x slower at 1M text
  vectors at equal recall; only Rust, C# and C++ have concurrent search and Python has batch search (TypeScript and JavaScript are single-threaded); the ANN index is
  never persisted and the binary snapshot is opt-in (JSON is the portable default); and memory is 1.12-1.25x raw at 1M in the four ports measured, against 1.6-1.9 GB for faiss / hnswlib at 1M x 384 (re-run `benchmarks/competitive/` before quoting the
  ratios: they predate the rework). Graph-aware retrieval
  beating plain vector search on multi-hop questions is the supported claim,
  and it depends on the graph carrying real structure
  (`benchmarks/multihop/README.md`).
- Fast filtered search was only measured for node-type filters, and only Rust
  is fast there (Python's filtered search is no faster than its unfiltered
  scan). The C++, C# and JS exact scans are scalar and still slower than numpy
  (about 2x faster than before this change in C++ and C#, 1.6x in JS).
- The digest cannot tell `1` from `1.0` in JavaScript / TypeScript, and
  integers beyond 64 bits are not supported in C# / C++ / Rust. Where a graph
  contains such values the loader warns instead of raising a false mismatch.
- C# cannot load an undirected graph file written by Python: the published
  IsonGraph 1.4.0 package rejects its `meta.graph` block.
- Only Python records the encoder version beyond the sidecar; a version-only
  change warns on load but is not treated as stale.

### Notes

- Python 3.10 or later. That is the floor `requires-python` declares and the
  floor CI tests; nothing below it is claimed.
- The Python import name is `isongraph_vector`; the other ports are published as
  `IsonGraph.Vector` (NuGet), `isongraph-vector-ts`, `isongraph-vector-js` and
  `isongraph-vector-rs`, with the C++ port consumed as headers.
- Only the Python port persists whole graphs to `.isong` with an embeddings
  sidecar. Every port persists embeddings, through its SQLite store or the
  portable payload.
- Only the Python port ships a real encoder (`SentenceTransformerEncoder`).
  The other five ship `MockEncoder` and the encoder interface; bring your own
  model.
- `IsonGraph.Vector.Sqlite` takes its vec0 natives from
  `HiraokaHyperTools.sqlite-vec` 0.1.9. asg017's own NuGet package has never
  left prerelease, and NuGet refuses to pack a stable package with a
  prerelease dependency (NU5104), so a stable 1.0.0 was not possible against
  it. The redistribution carries the same upstream C at the version the Rust,
  Node and C++ ports already use.
- Similarity search is exact (brute force) by default, with an opt-in IVF
  index; this is built for an application's own knowledge graph, not as a
  vector-database replacement.
- Payload vectors are float32 in every port, but the JSON text differs: Rust
  writes the shortest form that round-trips as f32 (`0.33590567`), Python
  widens to float64 first (`0.33590567111968994`). Same number, and importing
  narrows back, so a store built from either payload is identical - but two
  ports' payload files will not diff clean.

[Unreleased]: https://github.com/ISON-format/isongraph-vector/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ISON-format/isongraph-vector/releases/tag/v1.0.0
