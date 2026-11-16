# Roadmap

What is deliberately not done yet, in rough priority order. Items marked
(measure first) wait on evidence rather than on effort.

## Next

1. **Cut 1.0.0.** Nothing is tagged or published. Follow `RELEASING.md`.
2. **Run the new CI step once on real runners.** `graph_files.py` passes
   locally on Windows only; the Linux paths (g++, TS symlink, dotnet) are
   unproven. Also build the C++ libcurl transport once against real libcurl.
3. **Retrieval-quality benchmark on real data.** `benchmarks/` uses a
   synthetic graph and synthetic vectors. Run it against a real knowledge
   graph and a real encoder, add answer-groundedness, and measure ANN recall on
   real embeddings.
4. **Per-record encoder version.** Today a version-only change warns on
   load but cannot be fixed by `reembed_stale`.

## Later

- **Approximate search beyond Python**, and HNSW as an optional extra if IVF's
  recall/latency trade-off proves too weak on real data.
- **Async encoders in TypeScript / JavaScript**, so an `HttpEncoder` works with
  auto-embedding without a `prefetch`.
- **Relationship-type weights** (per type rather than a per-edge property),
  and richer node predicates than a callback.
- **Multi-vector and edge embeddings.**
- **C# loading Python's undirected graph files** - blocked on the IsonGraph
  NuGet package accepting the `meta.graph` block.

## Out of scope

- Distributed indexing and retrieval, replication, multi-node availability.
- Multi-tenant access control and a service layer. Put those in front of the
  library.
