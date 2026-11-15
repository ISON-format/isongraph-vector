# Scale benchmark - sample output

Python 3.12.7 | numpy 2.4.2 | sqlite-vec 0.1.9 | 32 logical CPUs | 47.7 GB RAM | Windows-11-10.0.26300-SP0

384-d random normalised float32 vectors, in-memory SQLite, medians of 9 runs (graph queries: 3 runs), top_k=10. Memory = process RSS growth (psutil).


## Latency (ms, warm)

| backend | N | similarity_search | p95 | semantic_multi_hop | multi_hop blend=0.3 | semantic_subgraph | search, GC off | multi_hop, GC off |
|---|---|---|---|---|---|---|---|---|
| numpy | 10,000 | 3.28 | 9.37 | 5.40 | 5.16 | 8.66 | 3.00 | 4.58 |
| numpy | 100,000 | 55.67 | 200.90 | 190.30 | 83.40 | 219.26 | 46.28 | 70.65 |
| numpy | 1,000,000 | 664.37 | 2142.85 | 2128.11 | 994.39 | 2319.53 | 499.62 | 678.51 |
| sqlite_vec | 10,000 | 3.31 | 3.40 | 3.66 | 15.06 | 6.29 | 3.41 | 3.54 |
| sqlite_vec | 100,000 | 32.77 | 35.79 | 32.83 | 77.24 | 60.62 | 32.74 | 32.35 |
| sqlite_vec | 1,000,000 | 892.91 | 934.85 | 882.48 | 2490.49 | 1622.85 | 887.74 | 883.14 |

Reference, not part of the library: bare `matrix @ q` + `argpartition` over the same vectors (no SQLite, no Python loop): N=10,000: 0.72 ms, N=100,000: 5.29 ms, N=1,000,000: 48.62 ms


## Ingestion, snapshot build and write-after-read

| backend | N | add_batch vec/s | first query (snapshot build) ms | write 1 vector ms | search right after write ms | penalty vs warm ms | add_batch(100) ms | search after batch ms |
|---|---|---|---|---|---|---|---|---|
| numpy | 10,000 | 111,392 | 36.0 | 0.86 | 38.8 | 35.5 | 2.3 | 38.9 |
| numpy | 100,000 | 101,754 | 529.1 | 8.79 | 494.6 | 438.9 | 10.3 | 478.7 |
| numpy | 1,000,000 | 85,764 | 5825.5 | 81.99 | 5251.3 | 4587.0 | 167.8 | 5866.8 |
| sqlite_vec | 10,000 | 43,648 | 4.6 | 0.15 | 3.4 | 0.1 | 3.0 | 3.7 |
| sqlite_vec | 100,000 | 40,846 | 34.1 | 0.22 | 33.9 | 1.1 | 4.1 | 32.8 |
| sqlite_vec | 1,000,000 | 22,954 | 842.7 | 0.28 | 777.0 | 0.0 | 10.1 | 839.5 |

## Memory (MB growth over a pre-build baseline)

| backend | N | raw vectors | after graph build | after ingest | after first query | peak |
|---|---|---|---|---|---|---|
| numpy | 10,000 | 15 | 21 | 50 | 65 | 180 |
| numpy | 100,000 | 146 | 193 | 442 | 589 | 909 |
| numpy | 1,000,000 | 1465 | 1991 | 4346 | 6557 | 9075 |
| sqlite_vec | 10,000 | 15 | 7 | 64 | 64 | 122 |
| sqlite_vec | 100,000 | 146 | 146 | 565 | 565 | 716 |
| sqlite_vec | 1,000,000 | 1465 | 1950 | 6020 | 6020 | 7499 |

## Where does the linear scan cross the latency budget?

Least-squares fit of median latency against N over the measured sizes of at least 10,000 vectors (all sizes if fewer than two).

| series | ms per 1k vectors | intercept ms | R^2 | N at 50 ms | N at 100 ms |
|---|---|---|---|---|---|
| numpy/similarity_search | 0.671 | -7.25 | 1.000 | 90,260 (interpolated) | 165,545 (interpolated) |
| numpy/semantic_multi_hop | 2.148 | -20.08 | 1.000 | 31,707 (interpolated) | 56,044 (interpolated) |
| numpy/semantic_multi_hop_blend0.3 | 1.004 | -10.68 | 1.000 | 61,580 (interpolated) | 116,398 (interpolated) |
| numpy/semantic_subgraph | 2.334 | -14.42 | 1.000 | 27,665 (interpolated) | 49,033 (interpolated) |
| numpy/similarity_search_gc_off | 0.502 | -2.95 | 1.000 | 107,382 (interpolated) | 206,645 (interpolated) |
| sqlite_vec/similarity_search | 0.922 | -31.38 | 0.997 | 118,030 (interpolated) | 170,347 (interpolated) |
| sqlite_vec/semantic_multi_hop | 0.911 | -30.58 | 0.997 | 118,185 (interpolated) | 171,148 (interpolated) |
| sqlite_vec/semantic_multi_hop_blend0.3 | 2.574 | -91.38 | 0.996 | 60,576 (interpolated) | 108,488 (interpolated) |
| sqlite_vec/semantic_subgraph | 1.675 | -56.35 | 0.997 | 82,404 (interpolated) | 122,684 (interpolated) |
| sqlite_vec/similarity_search_gc_off | 0.916 | -31.05 | 0.997 | 118,165 (interpolated) | 170,797 (interpolated) |
| raw_numpy_floor | 0.048 | 0.34 | 1.000 | 1,028,320 (extrapolated) | 2,063,772 (extrapolated) |

## Recommendation (auto-generated from the numbers above)

* The numpy exact scan reaches 50 ms at about **90,260 vectors** (interpolated between measured sizes). Below roughly 18,052 vectors (one fifth of that) an ANN index buys nothing you would notice.
* A bare matrix product over the same vectors reaches 50 ms at about 1,028,320: part of the library's cost is Python-side per-row work in `_score_all` (a tuple per stored vector on every query), which a leaner exact path could avoid without any ANN index.
* Past ~90,260 vectors, or if you need many concurrent queries per second, either an ANN index or a leaner exact path is warranted; exact search stays correct but scales linearly.
* sqlite-vec (exact, in C) fits 0.922 ms per 1k vectors vs 0.671 for numpy: it is still a linear scan, so it moves the crossover but does not remove it.
* With the cyclic GC disabled during the call the same numpy scan reaches 50 ms at about 107,382: in a process holding a large graph, part of the tail latency is Python garbage collection triggered by the per-query allocations, not search work (see the GC-off columns).
* On the numpy path any write invalidates the snapshot: at N=1,000,000 the next search costs ~5251 ms instead of 664 ms. Write-heavy workloads should batch writes, or use sqlite_vec=True, which has no snapshot to rebuild.
