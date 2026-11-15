# Roadmap

What is deliberately not done yet, in rough priority order. Items marked
(measure first) wait on evidence rather than on effort.

## Next

1. **Cut 1.0.0.** Nothing is tagged or published. Follow `RELEASING.md`; the
   changelog's Unreleased section holds everything since the version bump.
2. **Cross-port CI for graph files.** CI cross-checks the embeddings payload
   and SQLite databases between ports, but not graph files plus sidecars or
   the graph digest. Add writers and readers for each port to
   `scripts/cross_port` and run them in the `cross-port` job.
3. **Retrieval-quality benchmark on real data.** `benchmarks/` uses a
   synthetic graph. Run it against a real knowledge graph and a real
   encoder, and add answer-groundedness to recall and precision.
4. **Model lifecycle for the standalone `SqliteEmbeddingStore`** (TypeScript,
   JavaScript, Rust, C++): `models`, `stale`, `version` and sidecar metadata.
5. **Per-record encoder version.** Today a version-only change warns on
   load but cannot be fixed by `reembed_stale`.

## Later

- **Real encoders outside Python.** Optional adapters (an embeddings HTTP
  endpoint, an ONNX / transformers.js model) that add no hard dependency.
- **Approximate nearest-neighbour index** (measure first). Re-run
  `benchmarks/scale.py --max-n 1000000` against the vectorised exact path;
  only if a search is still too slow at the sizes people use, evaluate an ANN
  index behind an optional flag.
- **Closing the `update_node` bypass** in C#, C++ and Rust, which needs the
  underlying graph object to stop being public (a major-version change).
- **Relationship-type weights** (per type rather than a per-edge property),
  and richer node predicates than a callback.
- **Multi-vector and edge embeddings.**
- **C# loading Python's undirected graph files** - blocked on the IsonGraph
  NuGet package accepting the `meta.graph` block.

## Out of scope

- Distributed indexing and retrieval, replication, multi-node availability.
- Multi-tenant access control and a service layer. Put those in front of the
  library.
