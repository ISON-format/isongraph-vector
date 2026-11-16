# Scale benchmark - sample output

Python 3.12.7 | numpy 2.4.2 | sqlite-vec 0.1.9 | 32 logical CPUs | 47.7 GB RAM | Windows-11-10.0.26300-SP0

384-d random normalised float32 vectors, in-memory SQLite, medians of 15 runs (graph queries: 5 runs), top_k=10. Memory = process RSS growth (psutil).


## Latency (ms, warm)

| backend | N | similarity_search | p95 | semantic_multi_hop | multi_hop blend=0.3 | semantic_subgraph | search, GC off | multi_hop, GC off |
|---|---|---|---|---|---|---|---|---|
| numpy | 10,000 | 0.42 | 0.57 | 0.52 | 4.04 | 3.42 | 0.34 | 0.42 |
| numpy | 100,000 | 5.51 | 11.44 | 4.17 | 57.09 | 33.67 | 5.58 | 5.00 |
| numpy | 1,000,000 | 51.98 | 55.29 | 44.27 | 707.13 | 346.22 | 51.59 | 38.62 |
| sqlite_vec | 10,000 | 4.83 | 7.10 | 4.07 | 4.49 | 6.62 | 3.54 | 3.73 |
| sqlite_vec | 100,000 | 32.52 | 36.99 | 35.34 | 77.47 | 66.97 | 32.98 | 33.92 |
| sqlite_vec | 1,000,000 | 347.66 | 392.61 | 357.42 | 797.40 | 691.71 | 361.02 | 364.33 |

Reference, not part of the library: bare `matrix @ q` + `argpartition` over the same vectors (no SQLite, no Python loop): N=10,000: 0.35 ms, N=100,000: 4.21 ms, N=1,000,000: 38.55 ms


## Ingestion, snapshot build and write-after-read

| backend | N | add_batch vec/s | first query (snapshot build) ms | write 1 vector ms | search right after write ms | penalty vs warm ms | add_batch(100) ms | search after batch ms |
|---|---|---|---|---|---|---|---|---|
| numpy | 10,000 | 111,145 | 35.3 | 0.06 | 0.4 | 0.0 | 1.7 | 0.6 |
| numpy | 100,000 | 96,706 | 351.9 | 0.15 | 5.7 | 0.2 | 2.9 | 5.6 |
| numpy | 1,000,000 | 84,062 | 3845.9 | 0.10 | 49.9 | 0.0 | 1.8 | 52.6 |
| sqlite_vec | 10,000 | 35,116 | 6.4 | 0.19 | 3.7 | 0.0 | 3.6 | 3.7 |
| sqlite_vec | 100,000 | 39,205 | 34.4 | 0.23 | 33.3 | 0.7 | 5.0 | 34.7 |
| sqlite_vec | 1,000,000 | 35,752 | 330.3 | 0.23 | 352.9 | 5.2 | 4.1 | 374.0 |

## Memory (MB growth over a pre-build baseline)

| backend | N | raw vectors | after graph build | after ingest | after first query | peak |
|---|---|---|---|---|---|---|
| numpy | 10,000 | 15 | 21 | 47 | 64 | 180 |
| numpy | 100,000 | 146 | 187 | 438 | 592 | 627 |
| numpy | 1,000,000 | 1465 | 1996 | 4344 | 5976 | 5996 |
| sqlite_vec | 10,000 | 15 | 2 | 61 | 61 | 139 |
| sqlite_vec | 100,000 | 146 | 159 | 583 | 583 | 762 |
| sqlite_vec | 1,000,000 | 1465 | 1968 | 6041 | 6041 | 7678 |

## Where does the linear scan cross the latency budget?

Least-squares fit of median latency against N over the measured sizes of at least 10,000 vectors (all sizes if fewer than two).

| series | ms per 1k vectors | intercept ms | R^2 | N at 50 ms | N at 100 ms |
|---|---|---|---|---|---|
| numpy/similarity_search | 0.052 | 0.10 | 1.000 | 961,685 (interpolated) | 1,924,989 (extrapolated) |
| numpy/semantic_multi_hop | 0.044 | -0.09 | 1.000 | 1,129,476 (extrapolated) | 2,256,999 (extrapolated) |
| numpy/semantic_multi_hop_blend0.3 | 0.715 | -8.50 | 1.000 | 87,979 (interpolated) | 159,416 (interpolated) |
| numpy/semantic_subgraph | 0.347 | -0.50 | 1.000 | 147,020 (interpolated) | 290,999 (interpolated) |
| numpy/similarity_search_gc_off | 0.052 | 0.11 | 1.000 | 968,837 (interpolated) | 1,939,166 (extrapolated) |
| sqlite_vec/similarity_search | 0.348 | -0.37 | 1.000 | 149,909 (interpolated) | 292,705 (interpolated) |
| sqlite_vec/semantic_multi_hop | 0.357 | 0.08 | 1.000 | 140,961 (interpolated) | 280,680 (interpolated) |
| sqlite_vec/semantic_multi_hop_blend0.3 | 0.801 | -3.07 | 1.000 | 66,120 (interpolated) | 128,160 (interpolated) |
| sqlite_vec/semantic_subgraph | 0.693 | -1.27 | 1.000 | 74,693 (interpolated) | 147,584 (interpolated) |
| sqlite_vec/similarity_search_gc_off | 0.362 | -1.60 | 1.000 | 146,705 (interpolated) | 283,880 (interpolated) |
| raw_numpy_floor | 0.038 | 0.16 | 1.000 | 1,297,615 (extrapolated) | 2,599,425 (extrapolated) |

## Recommendation (auto-generated from the numbers above)

* The numpy exact scan reaches 50 ms at about **961,685 vectors** (interpolated between measured sizes). Below roughly 192,337 vectors (one fifth of that) an ANN index buys nothing you would notice.
* A bare matrix product over the same vectors reaches 50 ms at about 1,297,615: that is the ceiling for any exact scan here, so the gap between the two lines is the library's own per-query overhead (the vectorised exact path sits close to this floor).
* Past ~961,685 vectors, or if you need many concurrent queries per second, an ANN index (or a different vector store) becomes worth evaluating; exact search stays correct but scales linearly.
* sqlite-vec (exact, in C) fits 0.348 ms per 1k vectors vs 0.052 for numpy: it is still a linear scan, so it moves the crossover but does not remove it.
* With the cyclic GC disabled during the call the same numpy scan reaches 50 ms at about 968,837: in a process holding a large graph, part of the tail latency is Python garbage collection triggered by the per-query allocations, not search work (see the GC-off columns).
* On the numpy path a write patches the cached matrix in place: at N=1,000,000 the search right after a write costs ~50 ms against 52 ms warm. A large gap here would mean the cache is being rebuilt instead of patched.
