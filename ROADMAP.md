# Roadmap

What is deliberately not done yet, in rough priority order. Items marked
(measure first) wait on evidence rather than on effort.

## Next

1. **Cut 1.0.0.** Nothing is tagged or published. Follow `RELEASING.md`.
2. **Run the new CI jobs once on real runners.** The cross-port graph-file
   check and `cpp-curl.yml` have run on Linux only through the first PR; the
   server package's tests and its Dockerfile have only run on Windows (the
   Dockerfile has never been built).
3. **Make Rust the reference engine.** The competitive benchmark
   (`benchmarks/competitive/`) found concrete defects: no concurrent or
   batched search, vectors stored twice (2-4x the memory of faiss/hnswlib), a
   slow first query after heavy writes, and JSON-only persistence. Fix them in
   Rust, write the design down, then port what proves out to the other ports.
4. **HNSW as an optional index** (measure first). IVF is 2-4x slower per query
   than HNSW libraries at 100k and about 14x slower at 1M; HNSW is the only
   route to closing that gap.
5. **Fix the slow ports.** The C++ exact scan has no scan cache and is scalar
   (6x slower than numpy); JavaScript k-means takes about 8 minutes at 1M.
6. **Per-record encoder version.** Today a version-only change warns on
   load but cannot be fixed by `reembed_stale`.

## Later

- **Binary persistence** for vectors and the index, as an addition to the
  portable JSON sidecar (which stays the default and keeps the digest).
- **Approximate search for the standalone SQLite stores.**
- **Relationship-type weights** (per type rather than a per-edge property),
  and richer node predicates than a callback.
- **Multi-vector and edge embeddings.**
- **C# loading Python's undirected graph files** - blocked on the IsonGraph
  NuGet package accepting the `meta.graph` block.
- **Server package:** TLS, a write-ahead log, failover and resharding, only if
  the service is to be run beyond a trusted network.

## Out of scope

- Distributed indexing and retrieval inside the library, and multi-node
  availability. The server package offers per-tenant sharding and
  bounded-staleness snapshot replicas only.
- Competing with dedicated vector databases on raw query throughput. The
  supported claim is graph-aware retrieval over up to about a million vectors.
