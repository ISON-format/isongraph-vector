# Scale benchmark - sample output

Python 3.12.7 | numpy 2.4.2 | sqlite-vec 0.1.9 | 32 logical CPUs | 47.7 GB RAM | Windows-11-10.0.26300-SP0

384-d random normalised float32 vectors, in-memory SQLite, medians of 15 runs (graph queries: 5 runs), top_k=10. Memory = process RSS growth (psutil).


## Latency (ms, warm)

| backend | N | similarity_search | p95 | semantic_multi_hop | multi_hop blend=0.3 | semantic_subgraph | search, GC off | multi_hop, GC off |
|---|---|---|---|---|---|---|---|---|
| numpy | 1,000 | 0.27 | 0.28 | 0.52 | 0.48 | 1.20 | 0.26 | 0.45 |
| numpy | 10,000 | 3.39 | 3.95 | 4.78 | 5.15 | 7.65 | 2.91 | 4.08 |
| numpy | 100,000 | 56.24 | 175.03 | 183.22 | 80.35 | 201.60 | 45.48 | 62.41 |
| sqlite_vec | 1,000 | 0.63 | 0.64 | 0.72 | 0.49 | 1.41 | 0.61 | 0.68 |
| sqlite_vec | 10,000 | 3.79 | 3.98 | 3.72 | 7.24 | 6.71 | 3.79 | 3.72 |
| sqlite_vec | 100,000 | 34.15 | 37.14 | 32.90 | 93.08 | 57.48 | 33.09 | 32.66 |

Reference, not part of the library: bare `matrix @ q` + `argpartition` over the same vectors (no SQLite, no Python loop): N=1,000: 0.03 ms, N=10,000: 0.22 ms, N=100,000: 4.02 ms


## Ingestion, snapshot build and write-after-read

| backend | N | add_batch vec/s | first query (snapshot build) ms | write 1 vector ms | search right after write ms | penalty vs warm ms | add_batch(100) ms | search after batch ms |
|---|---|---|---|---|---|---|---|---|
| numpy | 1,000 | 120,003 | 3.3 | 0.12 | 3.1 | 2.8 | 0.9 | 3.5 |
| numpy | 10,000 | 109,775 | 40.6 | 0.86 | 32.7 | 29.3 | 1.7 | 34.1 |
| numpy | 100,000 | 104,283 | 494.4 | 9.47 | 515.1 | 458.8 | 11.1 | 491.6 |
| sqlite_vec | 1,000 | 44,402 | 0.7 | 0.10 | 0.6 | 0.0 | 2.5 | 0.7 |
| sqlite_vec | 10,000 | 45,498 | 4.1 | 0.17 | 3.7 | 0.0 | 2.9 | 3.9 |
| sqlite_vec | 100,000 | 39,475 | 37.9 | 0.25 | 35.7 | 1.6 | 4.2 | 33.2 |

## Memory (MB growth over a pre-build baseline)

| backend | N | raw vectors | after graph build | after ingest | after first query | peak |
|---|---|---|---|---|---|---|
| numpy | 1,000 | 1 | 2 | 7 | 9 | 18 |
| numpy | 10,000 | 15 | 16 | 44 | 62 | 175 |
| numpy | 100,000 | 146 | 189 | 438 | 588 | 901 |
| sqlite_vec | 1,000 | 1 | 1 | 5 | 5 | 6 |
| sqlite_vec | 10,000 | 15 | 3 | 63 | 63 | 155 |
| sqlite_vec | 100,000 | 146 | 172 | 596 | 596 | 738 |

## Concurrency (N=20,000, 4.08 s per scenario)

`store_ops`: threads share one EmbeddingStore (documented thread-safe). `graph_ops`: readers run semantic_multi_hop while a writer adds nodes and edges to the SemanticGraph (the graph itself is not documented as thread-safe).

| backend | scenario | readers/writers | reads/s | writes/s | read p50 ms | read p95 ms | errors | lost writes |
|---|---|---|---|---|---|---|---|---|
| numpy | store_ops | 4/0 | 86 | 0 | 49.9 | 66.5 | 0 | 0 |
| numpy | store_ops | 4/1 | 35 | 10 | 112.0 | 135.5 | 0 | 0 |
| numpy | store_ops | 2/4 | 14 | 30 | 153.2 | 169.6 | 0 | 0 |
| numpy | graph_ops | 3/1 | 24 | 11 | 124.9 | 159.8 | 0 | 0 |
| sqlite_vec | store_ops | 4/0 | 143 | 0 | 7.1 | 127.3 | 0 | 0 |
| sqlite_vec | store_ops | 4/1 | 143 | 16 | 7.1 | 106.2 | 0 | 0 |
| sqlite_vec | store_ops | 2/4 | 137 | 122 | 7.2 | 46.5 | 0 | 0 |
| sqlite_vec | graph_ops | 3/1 | 134 | 51 | 22.1 | 27.8 | 0 | 0 |

## Where does the linear scan cross the latency budget?

Least-squares fit of median latency against N over the measured sizes of at least 10,000 vectors (all sizes if fewer than two).

| series | ms per 1k vectors | intercept ms | R^2 | N at 50 ms | N at 100 ms |
|---|---|---|---|---|---|
| numpy/similarity_search | 0.587 | -2.48 | 1.000 | 89,372 (interpolated) | 174,524 (extrapolated) |
| numpy/semantic_multi_hop | 1.983 | -15.05 | 1.000 | 32,808 (interpolated) | 58,027 (interpolated) |
| numpy/semantic_multi_hop_blend0.3 | 0.835 | -3.20 | 1.000 | 63,679 (interpolated) | 123,525 (extrapolated) |
| numpy/semantic_subgraph | 2.155 | -13.90 | 1.000 | 29,650 (interpolated) | 52,852 (interpolated) |
| numpy/similarity_search_gc_off | 0.473 | -1.82 | 1.000 | 109,566 (extrapolated) | 215,286 (extrapolated) |
| sqlite_vec/similarity_search | 0.337 | 0.41 | 1.000 | 146,987 (extrapolated) | 295,194 (extrapolated) |
| sqlite_vec/semantic_multi_hop | 0.324 | 0.48 | 1.000 | 152,746 (extrapolated) | 306,964 (extrapolated) |
| sqlite_vec/semantic_multi_hop_blend0.3 | 0.954 | -2.29 | 1.000 | 54,830 (interpolated) | 107,256 (extrapolated) |
| sqlite_vec/semantic_subgraph | 0.564 | 1.06 | 1.000 | 86,736 (interpolated) | 175,359 (extrapolated) |
| sqlite_vec/similarity_search_gc_off | 0.326 | 0.54 | 1.000 | 151,933 (extrapolated) | 305,511 (extrapolated) |
| raw_numpy_floor | 0.042 | -0.21 | 1.000 | 1,188,140 (extrapolated) | 2,371,417 (extrapolated) |

## Recommendation (auto-generated from the numbers above)

* The numpy exact scan reaches 50 ms at about **89,372 vectors** (interpolated between measured sizes). Below roughly 17,874 vectors (one fifth of that) an ANN index buys nothing you would notice.
* A bare matrix product over the same vectors reaches 50 ms at about 1,188,140: part of the library's cost is Python-side per-row work in `_score_all` (a tuple per stored vector on every query), which a leaner exact path could avoid without any ANN index.
* Past ~89,372 vectors, or if you need many concurrent queries per second, either an ANN index or a leaner exact path is warranted; exact search stays correct but scales linearly.
* sqlite-vec (exact, in C) fits 0.337 ms per 1k vectors vs 0.587 for numpy: it is still a linear scan, so it moves the crossover but does not remove it.
* With the cyclic GC disabled during the call the same numpy scan reaches 50 ms at about 109,566: in a process holding a large graph, part of the tail latency is Python garbage collection triggered by the per-query allocations, not search work (see the GC-off columns).
* On the numpy path any write invalidates the snapshot: at N=100,000 the next search costs ~515 ms instead of 56 ms. Write-heavy workloads should batch writes, or use sqlite_vec=True, which has no snapshot to rebuild.
