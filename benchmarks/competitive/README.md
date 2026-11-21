# Competitive benchmark: ISONGraph Vector against the established vector-search libraries

> **Update, 2026-11-21: superseded by `README_post_rework.md`.** Every ratio
> on this page predates the rework. A full re-run against the current branch,
> on one machine in one session (faiss, hnswlib, numpy and all six ports at
> text384 100k and 1M), found: Rust HNSW is 1.28x faiss HNSW and 1.33x hnswlib
> single-thread at recall 0.95 (100k; 1.20x / 1.39x at 1M), batch throughput is
> level (1.05x faiss); C++ HNSW 1.11x / 1.15x; C# HNSW 0.76x / 0.79x; JS/TS HNSW
> 0.36x with a 4-minute build; Rust exact batch search is 3.1x faiss Flat; Rust
> and C++ HNSW use within 8% of faiss's memory (C# 1.4-1.85x, JS/TS 2.5x). The
> IVF indexes are still 0.1-0.3x of the HNSW libraries. The old "faiss Flat
> batch is ~35x ours" and "Rust exact beats numpy / faiss" lines are both
> obsolete. hnswlib still builds faster than any ISONGraph port at 100k.

> **Update, 2026-11-16.** Everything below was measured before the Rust engine
> rework. For Rust, the "no batch or concurrent search" loss, the 2-4x memory
> loss and the slow first query after writes are addressed; see
> `benchmarks/rust-stage-a/RESULTS.md` for old-versus-new numbers. The Rust
> single-thread exact scan is now 0-13% slower than the kernel measured here.
> `benchmarks/rust-stage-c/README.md` re-ran numpy and faiss against the
> current build: the "Rust exact beats numpy / faiss Flat by 6-33%" win is
> **refuted** (parity: 0.91-1.2x faiss Flat, 1.0-1.45x numpy), while Rust now
> has an opt-in HNSW index that matches faiss HNSW and hnswlib single-thread,
> int8 scanning, and a binary snapshot. The other ports are unchanged.

**One-paragraph answer.** On this machine, with identical vectors and an exact float64 ground truth, ISONGraph Vector is **not** the fastest vector search and the evidence does **not** support calling it the "best vector store". Its approximate index is a plain IVF-flat. At equal recall@10 its IVF is 1.4-2.3x slower than faiss IVFFlat at the same nlist and **1.9x (GloVe 100k) to 4.3x (text 100k) and 14-15x (text 1M) slower than the HNSW libraries** (hnswlib, faiss HNSW) in single-thread QPS; it has no batch or concurrent-query path, so its all-core throughput is 10-50x below faiss/hnswlib; and its only persistence is a JSON file (15-120x slower to save and load than faiss/hnswlib index files, 3-5x larger on disk). What it does well: the Rust exact scan is the fastest single-thread exact scan measured (6-33% ahead of numpy and faiss `IndexFlatIP` at 100k and 1M on text and GloVe, equal within noise at 10k); index build is 8-25x faster than HNSW builds at 100k-1M; the IVF index is maintained incrementally through insert/update/delete (faiss HNSW cannot update or delete; hnswlib/usearch can); and a node-type filter on a 1% subset is among the fastest filtered searches measured (see the filter table). Losses are listed as plainly as wins below.

Everything here was measured on one laptop (i9-14900HX, 24 cores / 32 threads, 47.7 GB, Windows 11), see `results/environment.json`. Nothing in the libraries was changed. All numbers are in `results/raw/*.json`; the tables are generated from those files by `report.py`.

## Read this first: what we can and cannot claim

| claim | supported? | evidence |
|---|---|---|
| "ISONGraph is the best / fastest vector store" | **No** | HNSW libraries are 1.9-15x faster in single-thread QPS at equal recall; faiss IVFFlat is ~1.4-2.3x faster than our IVF at the same nlist; batch/all-core throughput is 10-50x behind. |
| "ISONGraph's Rust exact search is competitive with faiss Flat / numpy" | **Yes (single thread)** | text384: Rust 154 QPS vs numpy 144, faiss Flat 131 at 100k; 15.1 vs 14.2 / 13.4 at 1M; glove 100k 700 vs 528 / 548. Not true for batched/all-core search (faiss Flat batch is ~35x ours at 100k, 540 vs 15 QPS at 1M) or for the C++/C#/JS ports. |
| "The opt-in IVF index gives a large speed-up over our exact scan with a recall you can choose" | **Yes** | e.g. Rust text384 100k: 154 -> 828 QPS at recall 0.95 (5x); at 1M 15 -> 103 (7x). Python 131 -> 530. |
| "ISONGraph Python is competitive with Rust" | partly | After pinning, Python IVF (numpy/BLAS) is within 1.5-2x of the Rust IVF, and Python exact is within 15% of Rust at 100k text. |
| "We are competitive with other embedded stores (lancedb, chromadb, sqlite-vec, qdrant local mode)" | **Yes, for exact and IVF; not vs chroma's HNSW** | sqlite-vec, lancedb flat and qdrant local mode are 10-100x slower than our exact scans; lancedb IVF_FLAT is 2.5-4x slower than our IVF; chromadb (HNSW) is faster than our IVF at 100k text (906 vs 828 QPS at 0.95) and 5x faster at 1M. |
| "Incremental updates are a strength" | **Yes versus HNSW, no edge versus faiss IVF** | The IVF index stays consistent through insert/update/delete (Rust 1M text: 35k inserts/s, 430k updates/s, 590k deletes/s, recall 0.951 -> 0.952) versus 3-5k inserts/s and updates/s for hnswlib/usearch at 1M; faiss HNSW cannot update or delete at all; faiss IVFFlat does the same job at 72-80k/s inserts. The first search after 30k writes at 1M costs 0.1-0.7 s (Rust 0.6 s) and 5.7 s in JS. |
| "Filtered search is a strength" | **Only for node-type equality** | Our only filter is `node_type`; arbitrary metadata predicates do not exist. On a 1% subset Rust gave 15,288 QPS at recall 1.0 (100k text) against hnswlib 259 (recall 0.98) and chromadb 21; see the filter table for faiss with a bitmap selector. Python/C++ filtered search is 25-120x slower than Rust (widening to an exact scan). |
| "Persistence is competitive" | **No** | JSON sidecar only: 100k x 384 takes 1.9-16 s to save and 1.3-11 s to load per port (450-810 MB), and the IVF index is rebuilt on the first query after load (Python: 5.1 s more). Rust at 1M: 4.5 GB, 21 s save, 20 s to ready. faiss/hnswlib/usearch: 0.06-0.2 s save, 150-180 MB. |
| "Memory efficiency" | **No** | 1M x 384: faiss HNSW 1.6 GB, hnswlib 1.8 GB, faiss IVF 1.9 GB vs Rust 3.5 GB, C++ 3.3 GB, C# 6.7 GB, JS 7.0 GB, Python 4.2 GB (records plus a second contiguous copy). |

## Headline results (single thread, QPS at recall@10; higher is better)

QPS at recall@10 = 0.95, copied from `results/tables_cells.md` (interpolated between measured sweep points; `*` = conservative measured point because the bracket was wide).

| system | text384 100k | text384 1M | glove 100k | glove 1M | sift 100k |
|---|---|---|---|---|---|
| hnswlib | 3,490 | 1,486 | 3,762 | not reached (ef<=640) | 22,439 |
| faiss HNSWFlat | 3,542 | 1,572 | 2,152 | not reached | 17,392 |
| usearch | 2,230 | 706 | 1,947 | not reached | 13,215 |
| chromadb | 906 | 605 | 933 | not reached | 1,163* |
| faiss IVFFlat | 1,193 | 190 | 2,893 | 43.8* | 8,764* |
| **ISONGraph Rust IVF** | 828 | 103 | 2,029 | 181 | 4,873 |
| **ISONGraph C++ IVF** (default SSE2 / AVX2 build) | 681 / 957 | 94 / 128 | 1,919 / 2,167 | 175 / 185 | 4,054* / 4,693* |
| **ISONGraph C# IVF** | 615 | not reached | 1,268 | not reached | 2,805* |
| **ISONGraph Python IVF** | 530 | 69 | 813 | 156 | 2,622 |
| **ISONGraph JS IVF** | 201 | not reached | 331 | not run | 795 |
| lancedb IVF_FLAT | 222 | 63 | 238 | 109 | 247 |
| *exact:* **ISONGraph Rust** | 154 | 15.1 | 700 | 43.5 | 344 |
| *exact:* numpy | 144 | 14.2 | 528 | 34.4 | 282 |
| *exact:* faiss Flat | 131 | 13.4 | 548 | 40.7 | 283 |
| *exact:* **ISONGraph Python** | 131 | 12.7 | 471 | 26.5 | 318 |
| *exact:* **ISONGraph C#** | 39 | 3.7 | 108 | 5.8 | 55 |
| *exact:* **ISONGraph C++** | 22.6 | 2.1 | 40.1 | 2.5 | 22.8 |
| *exact:* **ISONGraph JS** | 19.7 | 1.7 | 48.1 | not run | 29 |
| *exact:* sqlite-vec / lancedb / qdrant-local | 10 / 7 / 7 | 1.05 / 0.7 / - | 40 / 34 / - | - | 22 / 15 / - |

SIFT 1M (reduced set, QPS @0.95, 1T): hnswlib 7,311, faiss HNSW 5,603, faiss IVFFlat 1,270*, **Rust IVF 546***, **C++ IVF 562***; exact: **Rust 33.4**, numpy 26.9, **C++ 2.2**.

"not reached": the sweep (HNSW ef up to 640, IVF nprobe up to 256 at 1M) never got to recall 0.95 at M=16 / nlist=4000. HNSW with larger M would; we held M=16 for everyone.

Build time and memory (text384; ready = ingest + index + first query):

| system | 100k ready s | 1M ready s | 1M steady RSS (MB over baseline) |
|---|---|---|---|
| ISONGraph Rust IVF | 0.62 | 7.6 | 3,532 |
| ISONGraph C++ IVF (default / AVX2) | 0.97 / 0.61 | 12.4 / 7.1 | 3,279 |
| ISONGraph Python IVF | 3.6 | 42 | 4,251 |
| ISONGraph JS IVF | 57 | 522 | 7,002 |
| faiss IVFFlat | 1.7 | 18.6 | 1,885 |
| faiss HNSWFlat | 8.9 | 221 | 1,640 |
| hnswlib | 6.2 | 173 | 1,830 |
| usearch | 5.9 | 174 | 1,715 |
| chromadb | 14.7 | 345 | 4,023 |

## Wins and losses, ranked

Wins (measured):
1. Rust exact scan: 1-thread QPS 6-33% above numpy and faiss Flat in every 100k/1M cell measured (text, glove, sift), equal within noise at 10k. Single thread only; ours has no batch path (faiss Flat batch is 35x faster at text 100k).
2. Build time versus HNSW: 8-25x faster (text 1M: Rust 7.6 s vs 173-221 s for HNSW; 100k: 0.6 s vs 6-9 s); versus faiss IVFFlat 2.4x faster at text 1M (7.6 s vs 18.6 s) but not at 100k (0.6 vs 1.7 s, i.e. 2.8x).
3. Index maintenance: 7-100x faster writes than the HNSW libraries (Rust 1M text: 35k inserts/s, 430k updates/s, 590k deletes/s vs hnswlib 4.8k/3.2k/-), recall unchanged (0.951 -> 0.952).
4. Filtered search on a node-type subset, Rust only: 1% of 100k text 15,288 QPS at recall 1.0 vs faiss Flat (bitmap selector) 2,491, faiss IVF 1,630, hnswlib 259 @0.98; 10%: 3,094 vs 464 / 388 / 1,779 @0.91; at text 1M 10%: 650 QPS @0.94 vs faiss Flat 40, hnswlib 406 @0.955; 1%: 1,944 @0.906 vs 172 / 135 @0.945. Python and C++ filtered search is 25-120x slower than Rust (widening to an exact scan).
5. Python IVF on numpy/BLAS is within 1.5-2x of the native ports and is the only port besides Rust/C++/C# with realistic speed at 1M.

Losses (measured), worst first:
1. All-core / batched throughput: no batch API and one lock per store. text 100k @0.95: faiss HNSW 19,900 QPS, hnswlib 19,900, faiss Flat batch 5,437 vs ours <=1,518 (Rust auto-threads loop, 1.0x its own 1T loop). The "8 concurrent clients" probe confirms no scaling (`concurrent_clients` in the JSON).
2. Persistence: JSON sidecar, 3-5x larger and 15-120x slower than faiss/hnswlib index files; IVF rebuilt on load. Python 100k: save 15.5 s, load 11 s, then 5.1 s for the first query (index rebuild).
3. QPS at equal recall vs HNSW: 4-6x slower at 100k, 12-15x slower at 1M (text384).
4. QPS vs faiss IVFFlat at the same nlist: Rust 0.69x (text 100k), 0.54x (text 1M), 0.70x (glove 100k); C++ default build 0.57x; Python 0.44x.
5. Memory: 1.8-4x the HNSW/IVF libraries at 1M (the store keeps two copies of every vector).
6. C++ exact scan: 22.6 QPS vs numpy 144 at text 100k (6.4x slower), JS 7x slower, C# 3.7x slower (see library issues).
7. JS IVF build: 522 s at 1M (30x slower than Rust).
8. First search after heavy writes on a 1M index costs 0.1-0.7 s in Rust/C++/C#/Python and 5.7 s in JS (lazy fold / rebuild), versus about 1 ms for hnswlib/usearch.

## Cross-process variation: why the 1-thread loops are pinned

A first full run of the whole matrix (kept in `results/archive_unpinned/`) used unpinned single-thread loops. A second run of 12 cells hours later differed by **-7% to +227%** per system (for example faiss HNSW 1,083 vs 3,259 QPS at the same recall, Rust IVF 229 vs 748; `archive_unpinned/tables_repeat.md`), while the three timed runs inside each process agreed to 5%. On this hybrid CPU (8 P-cores + 16 E-cores) an unpinned thread may land on an E-core (measured 2.0x slower) and the machine as a whole spent some hours in a slower state. **All numbers in this README come from a second, complete run in which every single-thread loop is pinned to one P-core (logical CPU 8) at HIGH priority** (`bench_py.py: pinned`, `competitive_bench.{rs,cpp}: Pin`, C# `Point`, node pinned to P-cores 8,10,12,14 by `run_all.py`). Three timed runs per point agree to 1-5% (min-max in the full sweep tables). A second independent pinned process repeat of the final data set was not possible in the time available; cross-process stability of the pinned numbers is established only by a Rust spot-check (775 / 761 / 779 QPS in three separate processes at nprobe=128, text384 100k) and by the agreement of the three in-process runs. Treat differences below ~10% between systems as noise.



## How this was measured

### Data (identical normalised float32 vectors and queries for every system)

| dataset | vectors | dim | source | queries |
|---|---|---|---|---|
| `text384` | 1,060,000 passages (first 100k / 1M used) | 384 | **real text**: seeded random sample of the BeIR/hotpotqa Wikipedia-abstract corpus (5.2M passages) embedded with `sentence-transformers/all-MiniLM-L6-v2` (fp16 on GPU, `max_seq_length` 128, L2-normalised) | 2,000 real HotpotQA questions |
| `glove` | 1,183,514 (first 100k / 1M used) | 100 | ann-benchmarks `glove-100-angular` | first 2,000 of its test queries |
| `sift` | 1,000,000 (100k / 1M used) | 128 | ann-benchmarks `sift-128-euclidean`, **L2-normalised, scored as cosine** | first 2,000 of its test queries |

* The HDF5 files' own neighbour lists are **not** used. Ground truth is the exact **float64 cosine** top-100 over the same normalised float32 vectors (`prep_gt.py`), so one definition of "right" applies to all systems. Recall@10 = |returned ∩ true top-10| / 10, averaged over the queries. (On vectors with exact score ties the id sets can differ trivially; none of the datasets have duplicate vectors in the first 100k.)
* Because the vectors are unit length, cosine = inner product, so every system is configured with the metric its documentation recommends for normalised vectors (`ip`/`dot`/`cosine`) and returns the same ranking.
* Streaming-mutation and filter tests use deterministic plans written once by `prep_gt.py` (seeded): 10k inserted ids, 10k updated ids (a different vector is stored under an existing id), 10k deleted ids, all disjoint; a `grp` column splitting the ids into a random **10%** subset, a random **1%** subset and the remaining 89%. Ground truth after mutation / inside each subset is recomputed exactly.

### What is measured, and how

* **Recall versus queries/second** (`bench_py.py`, `competitive_bench.{rs,cpp}`, `...CompetitiveBench`, `bench/competitive.mjs`): every approximate system is swept over its accuracy knob (`nprobe` for IVF, `efSearch`/`ef`/`expansion_search`/`ef_search` for HNSW). At every knob value: warm-up (up to 50 queries or 3 s), then **three timed runs** of a strictly **single-threaded, one-query-at-a-time loop** (each query individually timed with a monotonic clock: median/p95/p99 latency from the pooled samples; QPS = queries / loop time; the min-max of the three runs is the run-to-run variation), then, where the system supports it, **three timed runs of a batched call using all 32 logical CPUs**. Up to 2,000 queries per run; systems slower than ~10 s per run are given fewer queries (the count is recorded as `nq`, never below 30).
* **Thread counts**: single-thread loops pin every library to one thread (`faiss.omp_set_num_threads(1)`, `hnswlib num_threads=1`, `usearch threads=1`, BLAS limited with `threadpoolctl`, ISONGraph Rust `set_scan_threads(1)`). The "all-core" column uses 32 threads (24 cores / 32 logical CPUs, hybrid P/E). The ISONGraph ports have **no batched search API and serialise searches on one lock**, so their all-core column is the *same one-query-at-a-time loop* with the library's automatic scan-thread choice (marked †), and a separate "N concurrent clients" probe shows the lock.
* **Build time**: ingest (all vectors added through the library's own add path from pre-parsed arrays) + index build/training + first query (some systems build lazily on the first query). At 100k the build is repeated three times in fresh instances.
* **Memory**: process working set (RSS, `psutil`/`GetProcessMemoryInfo`) minus the baseline measured after imports and before any data is touched. The harness streams vectors from disk in 20k-row chunks and keeps none, so the number is the system's own footprint. "Peak" is the peak working set after the build. File-backed engines (lancedb, chromadb, sqlite-vec) memory-map or cache data and are flattered by RSS; see the caveats.
* **Streaming mutation** (on the already-built index at the recall >= 0.95 operating point): insert 10k new vectors, update 10k existing ids, delete 10k ids, in batches of 1,000 through each system's natural API; recall@10 is re-measured against the exact post-mutation ground truth with the same knob value. `unsupported` means the library offers no such operation on that index type (it would need a rebuild).
* **Filtered search**: a separate build that carries the metadata column (ISONGraph: the node *type*, its only filter; others: an integer payload/metadata column), then restricted search over the 10% and 1% subsets at five accuracy settings. Reported: best single-thread QPS with recall >= 0.90 and the best recall reached.
* **Save/load**: each system's native persistence (faiss/hnswlib/usearch index files; lancedb/chroma/sqlite-vec are on-disk stores, so "save" is ~0 and "load" is reopening; ISONGraph's persistence is the JSON embeddings sidecar and the IVF index is rebuilt after load, which is why its time-to-ready is long).
* **Exact search** (`exact` tables): ISONGraph exact scans (Python numpy, Rust scan cache, C++ hash-map scan, C#, JS) against `faiss.IndexFlatIP`, plain numpy and the other exact engines, at 10k/100k/1M.

### Parameters that are held equal

| | value |
|---|---|
| HNSW (faiss, hnswlib, usearch, chromadb) | M (connectivity) = 16, efConstruction (expansion_add) = 200 |
| IVF (ISONGraph, faiss, lancedb) | nlist = round(4·sqrt(N)) (4,000 at 1M: ISONGraph's automatic choice, imposed on the others); k-means on a sample of ~40 points per cell for faiss, ISONGraph uses its own sample/8 iterations; lancedb its default sampling |
| k | 10 |
| Everything else | library defaults |

### Versions and hardware

See `results/environment.json` (written by `collect_env.py`).


## Results tables (generated by `report.py` from `results/raw/*.json`)


Plots (recall vs QPS, single thread and batched): `results/plots/<dataset>_<N>_{1t,batch}.png`.


### Recall versus QPS, build time and memory, per dataset and size

### text384, N = 100,000

Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.

| system | QPS 1T @0.90 | @0.95 | @0.99 | all-core QPS @0.90 | @0.95 | 1T latency ms median / p95 / p99 |
|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 23.0 | 23.0 | 23.0 | - | - | 43.5 / 45.4 / 47.6 (recall 1.000) |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 1697* | 957 | 114 | - | - | 1.10 / 1.30 / 1.48 (recall 0.954) |
| **ISONGraph C++ (exact)** | 22.6 | 22.6 | 22.6 | - | - | 44.2 / 47.2 / 57.7 (recall 1.000) |
| **ISONGraph C++ (IVF)** | 1232* | 681 | 93.3 | - | - | 1.55 / 1.81 / 2.00 (recall 0.954) |
| **ISONGraph C# (exact)** | 39.1 | 39.1 | 39.1 | - | - | 23.6 / 26.7 / 78.4 (recall 1.000) |
| **ISONGraph C# (IVF)** | 1134* | 615 | 112 | - | - | 1.66 / 2.00 / 2.22 (recall 0.952) |
| **ISONGraph Python (exact)** | 131 | 131 | 131 | - | - | 7.37 / 8.92 / 14.3 (recall 1.000) |
| **ISONGraph Python (IVF)** | 937* | 530 | 190 | - | - | 1.93 / 2.34 / 2.83 (recall 0.954) |
| **ISONGraph JS (exact)** | 19.7 | 19.7 | 19.7 | - | - | 50.8 / 55.8 / 58.1 (recall 1.000) |
| **ISONGraph JS (IVF)** | 358* | 201 | 51.3 | - | - | 5.11 / 6.06 / 6.71 (recall 0.953) |
| **ISONGraph Rust (exact)** | 154 | 154 | 154 | - | - | 6.41 / 6.99 / 7.86 (recall 1.000) |
| **ISONGraph Rust (IVF)** | 1562* | 828 | 253 | 1667* † | 1518 † | 1.25 / 1.51 / 1.63 (recall 0.953) |
| **ISONGraph Rust, target-cpu=native (exact)** | 163 | 163 | 163 | - | - | 6.06 / 6.79 / 7.39 (recall 1.000) |
| **ISONGraph Rust, target-cpu=native (IVF)** | 1506* | 802 | 254 | 1676* † | 1484 † | 1.30 / 1.57 / 1.70 (recall 0.953) |
| chromadb (HNSW) | 1029* | 906 | 602 | 4130* | 2890 | 1.14 / 1.34 / 1.54 (recall 0.965) |
| faiss IndexFlatIP | 131 | 131 | 131 | 5437 | 5437 | 7.29 / 8.84 / 15.0 (recall 1.000) |
| faiss HNSWFlat | 5913* | 3542 | 1329 | 30747* | 19930 | 0.296 / 0.377 / 0.553 (recall 0.957) |
| faiss IVFFlat | 2078* | 1193 | 317 | 11708* | 6189 | 0.866 / 1.07 / 1.26 (recall 0.955) |
| hnswlib | 5400 | 3490 | 1275 | 29766 | 19875 | 0.354 / 0.440 / 0.587 (recall 0.964) |
| lancedb (no index) | 7.19 | 7.19 | 7.19 | - | - | 137 / 151 / 173 (recall 1.000) |
| lancedb IVF_FLAT | 256* | 222 | 120 | - | - | 4.52 / 5.16 / 6.06 (recall 0.956) |
| numpy brute force | 144 | 144 | 144 | 1553 | 1553 | 6.75 / 7.66 / 10.1 (recall 1.000) |
| qdrant-client local mode (exact numpy) | 7.24 | 7.24 | 7.24 | - | - | 136 / 152 / 158 (recall 1.000) |
| sqlite-vec (vec0, exact) | 10.2 | 10.2 | 10.2 | - | - | 95.5 / 115 / 122 (recall 1.000) |
| usearch (HNSW) | 3624 | 2230 | 743 | 29559 | 19546 | 0.540 / 0.681 / 0.848 (recall 0.964) |

Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).

| system | ingest s | index s | total build s | 3 builds: median (min-max) s | ready s (incl. first query) | steady RSS +MB | peak RSS +MB (build) |
|---|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 0.081 | 0.000 | 0.081 | - | 0.125 | 192 | 206 |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 0.081 | 0.525 | 0.607 | - | 0.607 | 342 | 417 |
| **ISONGraph C++ (exact)** | 0.077 | 0.000 | 0.077 | - | 0.122 | 192 | 206 |
| **ISONGraph C++ (IVF)** | 0.077 | 0.891 | 0.968 | - | 0.968 | 342 | 417 |
| **ISONGraph C# (exact)** | 0.018 | 0.000 | 0.018 | - | 0.065 | 291 | 291 |
| **ISONGraph C# (IVF)** | 0.016 | 1.27 | 1.28 | - | 1.28 | 788 | 788 |
| **ISONGraph Python (exact)** | 2.01 | 0.000 | 2.01 | 1.96 (1.90-2.01) | 2.31 | 454 | 571 |
| **ISONGraph Python (IVF)** | 1.97 | 1.60 | 3.56 | 3.56 (3.48-3.57) | 3.57 | 481 | 693 |
| **ISONGraph JS (exact)** | 0.030 | 0.000 | 0.030 | - | 0.095 | 518 | 518 |
| **ISONGraph JS (IVF)** | 0.027 | 56.7 | 56.7 | - | 56.7 | 1,216 | 1,216 |
| **ISONGraph Rust (exact)** | 0.056 | 0.000 | 0.056 | - | 0.136 | 339 | 339 |
| **ISONGraph Rust (IVF)** | 0.056 | 0.480 | 0.616 | - | 0.616 | 342 | 420 |
| **ISONGraph Rust, target-cpu=native (exact)** | 0.058 | 0.000 | 0.058 | - | 0.137 | 339 | 339 |
| **ISONGraph Rust, target-cpu=native (IVF)** | 0.058 | 0.471 | 0.609 | - | 0.609 | 342 | 420 |
| chromadb (HNSW) | 14.6 | 0.000 | 14.6 | 14.8 (14.6-15.0) | 14.7 | 486 | 385 |
| faiss IndexFlatIP | 0.108 | 0.000 | 0.108 | 0.108 (0.108-0.111) | 0.117 | 172 | 317 |
| faiss HNSWFlat | 8.90 | 0.000 | 8.90 | 8.67 (8.41-8.90) | 8.90 | 184 | 328 |
| faiss IVFFlat | 0.000 | 1.72 | 1.72 | 1.83 (1.72-1.85) | 1.74 | 238 | 459 |
| hnswlib | 6.21 | 0.000 | 6.21 | 6.21 (6.09-6.21) | 6.21 | 194 | 226 |
| lancedb (no index) | 0.219 | 0.000 | 0.219 | 0.212 (0.211-0.219) | 0.284 | 149 | 224 |
| lancedb IVF_FLAT | 0.234 | 1.66 | 1.89 | 1.80 (1.77-1.89) | 1.98 | 339 | 487 |
| numpy brute force | 0.030 | 0.000 | 0.030 | 0.030 (0.030-0.030) | 0.037 | 159 | 191 |
| qdrant-client local mode (exact numpy) | 284 | 0.000 | 284 | - | 284 | 399 | 567 |
| sqlite-vec (vec0, exact) | 0.967 | 0.000 | 0.967 | 0.964 (0.952-0.967) | 1.07 | 4 | 62 |
| usearch (HNSW) | 5.86 | 0.000 | 5.86 | 5.86 (5.73-6.01) | 5.87 | 182 | 213 |

<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>

**ISONGraph C++, /arch:AVX2 (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 23.0 (23.0-23.1) | 43.5 | 45.4 | 47.6 | - | 68 |

**ISONGraph C++, /arch:AVX2 (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2808 | 17264 (17032-17852) | 0.056 | 0.069 | 0.080 | - | 2000 |
| 2 | 0.4022 | 14310 (14254-14358) | 0.068 | 0.086 | 0.102 | - | 2000 |
| 4 | 0.5291 | 11488 (11327-11627) | 0.085 | 0.110 | 0.132 | - | 2000 |
| 8 | 0.6479 | 8130 (7995-8407) | 0.118 | 0.158 | 0.212 | - | 2000 |
| 16 | 0.7548 | 5301 (5296-5315) | 0.183 | 0.243 | 0.305 | - | 2000 |
| 32 | 0.8427 | 3053 (3021-3054) | 0.319 | 0.414 | 0.486 | - | 2000 |
| 64 | 0.9091 | 1697 (1694-1723) | 0.576 | 0.717 | 0.869 | - | 2000 |
| 128 | 0.9543 | 902 (900-906) | 1.10 | 1.30 | 1.48 | - | 2000 |
| 256 | 0.9813 | 455 (452-462) | 2.15 | 2.59 | 2.98 | - | 985 |
| 1265 | 1.0000 | 23.0 (23.0-23.2) | 43.4 | 45.5 | 46.5 | - | 65 |

**ISONGraph C++ (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 22.6 (22.5-23.3) | 44.2 | 47.2 | 57.7 | - | 66 |

**ISONGraph C++ (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2808 | 16276 (16014-17282) | 0.059 | 0.077 | 0.090 | - | 2000 |
| 2 | 0.4022 | 14590 (14424-14658) | 0.067 | 0.089 | 0.110 | - | 2000 |
| 4 | 0.5291 | 10815 (10091-10998) | 0.091 | 0.127 | 0.151 | - | 2000 |
| 8 | 0.6479 | 6399 (6091-6902) | 0.152 | 0.206 | 0.247 | - | 2000 |
| 16 | 0.7548 | 3908 (3831-3928) | 0.250 | 0.336 | 0.409 | - | 2000 |
| 32 | 0.8427 | 2255 (2200-2287) | 0.436 | 0.570 | 0.649 | - | 2000 |
| 64 | 0.9091 | 1232 (1225-1234) | 0.806 | 0.993 | 1.10 | - | 2000 |
| 128 | 0.9542 | 641 (639-645) | 1.55 | 1.81 | 2.00 | - | 1869 |
| 256 | 0.9811 | 331 (330-332) | 3.01 | 3.38 | 3.66 | - | 1016 |
| 1265 | 1.0000 | 22.5 (22.4-22.6) | 44.5 | 46.8 | 48.1 | - | 67 |

**ISONGraph C# (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 39.1 (38.6-39.9) | 23.6 | 26.7 | 78.4 | - | 128 |

**ISONGraph C# (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2776 | 9706 (5430-9894) | 0.104 | 0.212 | 0.231 | - | 2000 |
| 2 | 0.3992 | 8524 (8519-8604) | 0.113 | 0.136 | 0.157 | - | 2000 |
| 4 | 0.5201 | 6881 (6817-6932) | 0.143 | 0.176 | 0.200 | - | 2000 |
| 8 | 0.6376 | 4982 (4893-5001) | 0.197 | 0.248 | 0.277 | - | 2000 |
| 16 | 0.7452 | 3315 (3315-3349) | 0.296 | 0.372 | 0.411 | - | 2000 |
| 32 | 0.8359 | 1975 (1971-2019) | 0.494 | 0.620 | 0.690 | - | 2000 |
| 64 | 0.9060 | 1134 (1127-1135) | 0.878 | 1.07 | 1.16 | - | 2000 |
| 128 | 0.9521 | 598 (591-600) | 1.66 | 2.00 | 2.22 | - | 1848 |
| 256 | 0.9805 | 306 (305-306) | 3.27 | 3.69 | 3.90 | - | 897 |
| 1265 | 1.0000 | 38.6 (37.4-39.4) | 23.7 | 25.1 | 95.1 | - | 122 |

**ISONGraph Python (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 0.9999 | 131 (129-131) | 7.37 | 8.92 | 14.3 | - | 411 |

**ISONGraph Python (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2833 | 8545 (8514-8573) | 0.114 | 0.139 | 0.175 | - | 2000 |
| 2 | 0.4083 | 7156 (6794-7226) | 0.135 | 0.175 | 0.216 | - | 2000 |
| 4 | 0.5305 | 5400 (5019-5451) | 0.180 | 0.230 | 0.324 | - | 2000 |
| 8 | 0.6501 | 3877 (3854-4006) | 0.245 | 0.302 | 0.351 | - | 2000 |
| 16 | 0.7542 | 2640 (2575-2669) | 0.369 | 0.458 | 0.593 | - | 2000 |
| 32 | 0.8428 | 1581 (1579-1603) | 0.591 | 0.741 | 1.08 | - | 2000 |
| 64 | 0.9091 | 937 (927-947) | 1.03 | 1.26 | 1.68 | - | 2000 |
| 128 | 0.9540 | 501 (500-506) | 1.93 | 2.34 | 2.83 | - | 1546 |
| 256 | 0.9809 | 268 (268-271) | 3.59 | 4.14 | 5.35 | - | 814 |
| 1265 | 0.9999 | 131 (130-132) | 7.36 | 8.47 | 14.2 | - | 422 |

**ISONGraph JS (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 19.7 (19.1-20.3) | 50.8 | 55.8 | 58.1 | - | 56 |

**ISONGraph JS (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2654 | 2125 (2089-2145) | 0.467 | 0.532 | 0.590 | - | 2000 |
| 2 | 0.3867 | 1927 (1927-1931) | 0.513 | 0.597 | 0.674 | - | 2000 |
| 4 | 0.5155 | 1637 (1632-1653) | 0.602 | 0.718 | 0.794 | - | 2000 |
| 8 | 0.6369 | 1304 (1294-1307) | 0.763 | 0.897 | 0.968 | - | 2000 |
| 16 | 0.7420 | 933 (929-937) | 1.07 | 1.27 | 1.40 | - | 2000 |
| 32 | 0.8353 | 607 (605-609) | 1.65 | 1.95 | 2.11 | - | 1779 |
| 64 | 0.9045 | 358 (357-359) | 2.77 | 3.29 | 3.55 | - | 1172 |
| 128 | 0.9527 | 194 (193-195) | 5.11 | 6.06 | 6.71 | - | 607 |
| 256 | 0.9828 | 103 (100-103) | 9.70 | 11.3 | 12.3 | - | 319 |
| 1265 | 1.0000 | 19.5 (19.4-20.0) | 50.4 | 54.3 | 59.1 | - | 59 |

**ISONGraph Rust (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 154 (154-154) | 6.41 | 6.99 | 7.86 | - | 476 |

**ISONGraph Rust (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2730 | 24266 (23139-24374) | 0.041 | 0.053 | 0.061 | 22169 † | 2000 |
| 2 | 0.3918 | 17640 (17404-17743) | 0.056 | 0.071 | 0.079 | 17989 † | 2000 |
| 4 | 0.5168 | 12425 (12047-12885) | 0.079 | 0.102 | 0.116 | 12368 † | 2000 |
| 8 | 0.6402 | 7664 (7441-8108) | 0.126 | 0.169 | 0.201 | 8336 † | 2000 |
| 16 | 0.7450 | 5069 (4964-5091) | 0.196 | 0.251 | 0.285 | 5069 † | 2000 |
| 32 | 0.8346 | 2840 (2774-2848) | 0.349 | 0.452 | 0.509 | 2891 † | 2000 |
| 64 | 0.9071 | 1562 (1537-1569) | 0.634 | 0.800 | 0.870 | 1667 † | 2000 |
| 128 | 0.9529 | 793 (791-796) | 1.25 | 1.51 | 1.63 | 1509 † | 2000 |
| 256 | 0.9807 | 401 (393-402) | 2.50 | 2.88 | 3.20 | 1084 † | 1148 |
| 1265 | 1.0000 | 154 (153-157) | 6.32 | 7.30 | 8.65 | 300 † | 478 |

**ISONGraph Rust, target-cpu=native (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 163 (161-164) | 6.06 | 6.79 | 7.39 | - | 501 |

**ISONGraph Rust, target-cpu=native (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2730 | 24669 (23997-24957) | 0.040 | 0.051 | 0.060 | 24453 † | 2000 |
| 2 | 0.3918 | 17732 (17481-17842) | 0.056 | 0.071 | 0.081 | 18452 † | 2000 |
| 4 | 0.5168 | 13134 (12835-13139) | 0.075 | 0.099 | 0.111 | 12737 † | 2000 |
| 8 | 0.6402 | 8489 (8363-8498) | 0.116 | 0.153 | 0.177 | 8381 † | 2000 |
| 16 | 0.7450 | 4821 (4660-4993) | 0.203 | 0.269 | 0.310 | 5055 † | 2000 |
| 32 | 0.8346 | 2789 (2748-2792) | 0.355 | 0.458 | 0.508 | 2838 † | 2000 |
| 64 | 0.9071 | 1506 (1480-1529) | 0.656 | 0.830 | 0.914 | 1676 † | 2000 |
| 128 | 0.9529 | 769 (761-774) | 1.30 | 1.57 | 1.70 | 1472 † | 2000 |
| 256 | 0.9807 | 387 (385-389) | 2.58 | 2.94 | 3.20 | 1095 † | 1127 |
| 1265 | 1.0000 | 161 (161-163) | 6.05 | 6.99 | 8.32 | 310 † | 470 |

**chromadb (HNSW)** (ef_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.6149 | 1361 (1337-1385) | 0.702 | 0.850 | 0.974 | 10714 | 2000 |
| 16 | 0.7140 | 1254 (1229-1255) | 0.787 | 0.910 | 1.07 | 8882 | 2000 |
| 32 | 0.8313 | 1170 (1131-1187) | 0.848 | 0.988 | 1.28 | 6354 | 2000 |
| 64 | 0.9152 | 1029 (1013-1034) | 0.964 | 1.12 | 1.39 | 4130 | 2000 |
| 128 | 0.9646 | 858 (851-898) | 1.14 | 1.34 | 1.54 | 2489 | 2000 |
| 256 | 0.9870 | 680 (676-703) | 1.44 | 1.66 | 1.96 | 1445 | 2000 |
| 512 | 0.9955 | 483 (466-490) | 2.06 | 2.41 | 2.86 | 821 | 1479 |

**faiss IndexFlatIP** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 0.9999 | 131 (130-135) | 7.29 | 8.84 | 15.0 | 5437 | 410 |

**faiss HNSWFlat** (efSearch)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.6361 | 15713 (15200-17057) | 0.058 | 0.088 | 0.118 | 96809 | 2000 |
| 16 | 0.7315 | 12783 (12779-12881) | 0.073 | 0.105 | 0.136 | 70072 | 2000 |
| 24 | 0.8014 | 9414 (9092-9648) | 0.097 | 0.135 | 0.217 | 50335 | 2000 |
| 32 | 0.8492 | 7879 (7578-8243) | 0.121 | 0.164 | 0.210 | 44864 | 2000 |
| 48 | 0.9004 | 5913 (5763-5919) | 0.165 | 0.209 | 0.261 | 30747 | 2000 |
| 64 | 0.9279 | 4554 (3996-4704) | 0.217 | 0.305 | 0.395 | 24862 | 2000 |
| 96 | 0.9569 | 3273 (3103-3312) | 0.296 | 0.377 | 0.553 | 18591 | 2000 |
| 128 | 0.9714 | 2370 (2277-2432) | 0.386 | 0.519 | 0.929 | 14274 | 2000 |
| 192 | 0.9842 | 1701 (1642-1724) | 0.558 | 0.691 | 1.04 | 10206 | 2000 |
| 256 | 0.9909 | 1282 (1249-1318) | 0.742 | 0.967 | 1.59 | 7844 | 2000 |
| 400 | 0.9958 | 823 (781-839) | 1.17 | 1.50 | 2.42 | 5285 | 2000 |
| 640 | 0.9983 | 508 (492-515) | 1.89 | 2.44 | 4.01 | 3847 | 1585 |

**faiss IVFFlat** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2834 | 18763 (18006-18898) | 0.052 | 0.065 | 0.082 | 192,036 | 2000 |
| 2 | 0.4099 | 16033 (16022-16181) | 0.060 | 0.076 | 0.095 | 172,377 | 2000 |
| 4 | 0.5287 | 12664 (11616-12949) | 0.077 | 0.101 | 0.141 | 116,179 | 2000 |
| 8 | 0.6503 | 9036 (8998-9414) | 0.107 | 0.134 | 0.171 | 84704 | 2000 |
| 16 | 0.7551 | 6072 (5829-6281) | 0.157 | 0.199 | 0.249 | 38405 | 2000 |
| 32 | 0.8439 | 3761 (3701-3838) | 0.255 | 0.324 | 0.385 | 23327 | 2000 |
| 64 | 0.9096 | 2078 (2072-2081) | 0.458 | 0.571 | 0.689 | 11708 | 2000 |
| 128 | 0.9549 | 1116 (1103-1154) | 0.866 | 1.07 | 1.26 | 5728 | 2000 |
| 256 | 0.9841 | 567 (566-570) | 1.72 | 2.00 | 2.43 | 2721 | 1702 |
| 1265 | 0.9999 | 120 (118-121) | 8.10 | 9.46 | 12.5 | 505 | 377 |

**hnswlib** (ef)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.6185 | 16855 (16407-17226) | 0.055 | 0.082 | 0.104 | 93453 | 2000 |
| 16 | 0.7160 | 13172 (13035-13312) | 0.072 | 0.105 | 0.134 | 75207 | 2000 |
| 24 | 0.7870 | 9860 (9650-9987) | 0.096 | 0.132 | 0.176 | 52454 | 2000 |
| 32 | 0.8309 | 7740 (7450-8230) | 0.116 | 0.161 | 0.244 | 46850 | 2000 |
| 48 | 0.8855 | 6085 (5802-6165) | 0.157 | 0.206 | 0.266 | 33393 | 2000 |
| 64 | 0.9162 | 4726 (4337-4839) | 0.202 | 0.273 | 0.377 | 26177 | 2000 |
| 96 | 0.9488 | 3563 (3414-3576) | 0.280 | 0.344 | 0.428 | 20267 | 2000 |
| 128 | 0.9645 | 2715 (2704-2728) | 0.354 | 0.440 | 0.587 | 15705 | 2000 |
| 192 | 0.9809 | 1959 (1930-1962) | 0.504 | 0.615 | 0.777 | 10961 | 2000 |
| 256 | 0.9869 | 1558 (1553-1565) | 0.634 | 0.759 | 0.862 | 8428 | 2000 |
| 400 | 0.9933 | 1026 (1017-1037) | 0.949 | 1.17 | 1.46 | 6018 | 2000 |
| 640 | 0.9970 | 696 (690-697) | 1.41 | 1.74 | 2.27 | 3946 | 1874 |

**lancedb (no index)** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 7.19 (7.16-7.22) | 137 | 151 | 173 | - | 30 |

**lancedb IVF_FLAT** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2415 | 243 (198-288) | 3.28 | 4.56 | 12.3 | - | 924 |
| 2 | 0.3584 | 300 (298-301) | 3.24 | 4.06 | 5.79 | - | 912 |
| 4 | 0.4898 | 285 (282-300) | 3.30 | 4.06 | 6.32 | - | 933 |
| 8 | 0.6160 | 294 (286-299) | 3.33 | 3.94 | 5.10 | - | 843 |
| 16 | 0.7339 | 286 (278-288) | 3.41 | 4.03 | 5.67 | - | 903 |
| 32 | 0.8351 | 266 (265-278) | 3.59 | 4.31 | 5.36 | - | 786 |
| 64 | 0.9090 | 256 (251-257) | 3.86 | 4.43 | 5.06 | - | 790 |
| 128 | 0.9556 | 218 (212-220) | 4.52 | 5.16 | 6.06 | - | 652 |
| 256 | 0.9828 | 177 (168-178) | 5.61 | 6.48 | 7.70 | - | 500 |
| 1265 | 0.9999 | 70.3 (69.0-71.1) | 14.0 | 15.6 | 17.4 | - | 210 |

**numpy brute force** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 0.9999 | 144 (143-146) | 6.75 | 7.66 | 10.1 | 1553 | 448 |

**qdrant-client local mode (exact numpy)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 7.24 (7.14-7.25) | 136 | 152 | 158 | - | 30 |

**sqlite-vec (vec0, exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 10.2 (10.2-10.3) | 95.5 | 115 | 122 | - | 30 |

**usearch (HNSW)** (expansion_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.6119 | 11865 (10700-12258) | 0.080 | 0.127 | 0.172 | 97939 | 2000 |
| 16 | 0.7104 | 8530 (7906-8812) | 0.109 | 0.166 | 0.218 | 79790 | 2000 |
| 24 | 0.7848 | 6731 (6269-6756) | 0.140 | 0.198 | 0.268 | 58748 | 2000 |
| 32 | 0.8294 | 5372 (5330-5488) | 0.172 | 0.229 | 0.280 | 48489 | 2000 |
| 48 | 0.8844 | 4092 (4055-4107) | 0.235 | 0.302 | 0.368 | 33537 | 2000 |
| 64 | 0.9149 | 3228 (3164-3248) | 0.299 | 0.378 | 0.453 | 26211 | 2000 |
| 96 | 0.9468 | 2334 (2290-2353) | 0.418 | 0.520 | 0.618 | 20803 | 2000 |
| 128 | 0.9643 | 1820 (1775-1833) | 0.540 | 0.681 | 0.848 | 14806 | 2000 |
| 192 | 0.9790 | 1297 (1266-1307) | 0.765 | 0.927 | 1.06 | 11370 | 2000 |
| 256 | 0.9854 | 1020 (1001-1029) | 0.974 | 1.19 | 1.38 | 8865 | 2000 |
| 400 | 0.9913 | 679 (656-692) | 1.46 | 1.81 | 2.26 | 5836 | 1759 |
| 640 | 0.9951 | 451 (443-460) | 2.20 | 2.72 | 3.28 | 3704 | 1355 |

</details>

### glove, N = 100,000

Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.

| system | QPS 1T @0.90 | @0.95 | @0.99 | all-core QPS @0.90 | @0.95 | 1T latency ms median / p95 / p99 |
|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 40.2 | 40.2 | 40.2 | - | - | 24.4 / 28.5 / 31.7 (recall 1.000) |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 2655* | 2167 | 162 | - | - | 0.708 / 0.882 / 1.03 (recall 0.974) |
| **ISONGraph C++ (exact)** | 40.1 | 40.1 | 40.1 | - | - | 24.5 / 28.4 / 29.7 (recall 1.000) |
| **ISONGraph C++ (IVF)** | 2361* | 1919 | 150 | - | - | 0.781 / 1.00 / 1.20 (recall 0.974) |
| **ISONGraph C# (exact)** | 108 | 108 | 108 | - | - | 7.79 / 26.4 / 35.6 (recall 1.000) |
| **ISONGraph C# (IVF)** | 1677* | 1268 | 220 | - | - | 1.14 / 1.43 / 1.62 (recall 0.973) |
| **ISONGraph Python (exact)** | 471 | 471 | 471 | - | - | 1.98 / 2.72 / 4.89 (recall 1.000) |
| **ISONGraph Python (IVF)** | 1035* | 813 | 477 | - | - | 1.73 / 2.31 / 3.53 (recall 0.973) |
| **ISONGraph JS (exact)** | 48.1 | 48.1 | 48.1 | - | - | 20.1 / 24.6 / 27.1 (recall 1.000) |
| **ISONGraph JS (IVF)** | 432* | 331 | 91.8 | - | - | 4.37 / 5.46 / 6.29 (recall 0.975) |
| **ISONGraph Rust (exact)** | 700 | 700 | 700 | - | - | 1.42 / 1.79 / 2.14 (recall 1.000) |
| **ISONGraph Rust (IVF)** | 2595* | 2029 | 888 | 2638* † | 2381 † | 0.733 / 0.912 / 1.01 (recall 0.974) |
| **ISONGraph Rust, target-cpu=native (exact)** | 621 | 621 | 621 | - | - | 1.57 / 1.94 / 2.25 (recall 1.000) |
| **ISONGraph Rust, target-cpu=native (IVF)** | 2584* | 2028 | 851 | 2610* † | 2334 † | 0.725 / 0.909 / 1.02 (recall 0.974) |
| chromadb (HNSW) | 956* | 933 | - | 3684* | 3468 | 1.29 / 1.50 / 1.92 (recall 0.977) |
| faiss IndexFlatIP | 548 | 548 | 548 | 16430 | 16430 | 1.72 / 2.23 / 3.01 (recall 1.000) |
| faiss HNSWFlat | 4756 | 2152 | - | 94888 | 51124 | 0.679 / 0.908 / 1.71 (recall 0.970) |
| faiss IVFFlat | 3719* | 2893 | 785 | 53383* | 40104 | 0.457 / 0.603 / 0.990 (recall 0.974) |
| hnswlib | 7183 | 3762 | - | 70103 | 48987 | 0.366 / 0.452 / 0.573 (recall 0.969) |
| lancedb (no index) | 34.3 | 34.3 | 34.3 | - | - | 29.5 / 34.2 / 37.4 (recall 1.000) |
| lancedb IVF_FLAT | 253* | 238 | 145 | - | - | 4.45 / 5.51 / 7.19 (recall 0.974) |
| numpy brute force | 528 | 528 | 528 | 1733 | 1733 | 1.82 / 2.53 / 3.70 (recall 1.000) |
| sqlite-vec (vec0, exact) | 40.1 | 40.1 | 40.1 | - | - | 24.3 / 27.4 / 34.3 (recall 1.000) |
| usearch (HNSW) | 3763 | 1947 | - | 58683 | 27876 | 0.720 / 0.949 / 1.18 (recall 0.967) |

Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).

| system | ingest s | index s | total build s | 3 builds: median (min-max) s | ready s (incl. first query) | steady RSS +MB | peak RSS +MB (build) |
|---|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 0.042 | 0.000 | 0.042 | - | 0.069 | 72 | 84 |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 0.042 | 0.253 | 0.295 | - | 0.295 | 111 | 131 |
| **ISONGraph C++ (exact)** | 0.038 | 0.000 | 0.038 | - | 0.064 | 72 | 84 |
| **ISONGraph C++ (IVF)** | 0.038 | 0.290 | 0.328 | - | 0.328 | 112 | 132 |
| **ISONGraph C# (exact)** | 0.017 | 0.000 | 0.017 | - | 0.035 | 87 | 87 |
| **ISONGraph C# (IVF)** | 0.011 | 0.602 | 0.613 | - | 0.613 | 249 | 249 |
| **ISONGraph Python (exact)** | 0.934 | 0.000 | 0.934 | 0.931 (0.914-0.934) | 1.11 | 159 | 164 |
| **ISONGraph Python (IVF)** | 0.929 | 0.848 | 1.78 | 1.78 (1.73-1.78) | 1.78 | 176 | 278 |
| **ISONGraph JS (exact)** | 0.029 | 0.000 | 0.029 | - | 0.053 | 308 | 308 |
| **ISONGraph JS (IVF)** | 0.027 | 20.6 | 20.7 | - | 20.7 | 642 | 645 |
| **ISONGraph Rust (exact)** | 0.054 | 0.000 | 0.054 | - | 0.094 | 121 | 121 |
| **ISONGraph Rust (IVF)** | 0.054 | 0.212 | 0.307 | - | 0.307 | 125 | 144 |
| **ISONGraph Rust, target-cpu=native (exact)** | 0.053 | 0.000 | 0.053 | - | 0.095 | 121 | 121 |
| **ISONGraph Rust, target-cpu=native (IVF)** | 0.053 | 0.176 | 0.270 | - | 0.270 | 125 | 144 |
| chromadb (HNSW) | 9.46 | 0.000 | 9.46 | 9.40 (9.34-9.46) | 9.55 | 264 | 202 |
| faiss IndexFlatIP | 0.031 | 0.000 | 0.031 | 0.035 (0.031-0.047) | 0.033 | 63 | 101 |
| faiss HNSWFlat | 2.25 | 0.000 | 2.25 | 2.28 (2.25-2.41) | 2.26 | 76 | 112 |
| faiss IVFFlat | 0.000 | 0.973 | 0.973 | 1.08 (0.973-1.09) | 0.976 | 91 | 155 |
| hnswlib | 1.33 | 0.000 | 1.33 | 1.31 (1.28-1.33) | 1.34 | 88 | 96 |
| lancedb (no index) | 0.083 | 0.000 | 0.083 | 0.080 (0.077-0.083) | 0.105 | 100 | 122 |
| lancedb IVF_FLAT | 0.085 | 0.643 | 0.729 | 0.697 (0.693-0.729) | 0.797 | 173 | 201 |
| numpy brute force | 0.012 | 0.000 | 0.012 | 0.012 (0.012-0.012) | 0.014 | 51 | 61 |
| sqlite-vec (vec0, exact) | 0.529 | 0.000 | 0.529 | 0.521 (0.493-0.529) | 0.554 | 4 | 19 |
| usearch (HNSW) | 1.72 | 0.000 | 1.72 | 1.72 (1.62-1.72) | 1.73 | 75 | 84 |

<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>

**ISONGraph C++, /arch:AVX2 (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 40.2 (39.7-41.2) | 24.4 | 28.5 | 31.7 | - | 126 |

**ISONGraph C++, /arch:AVX2 (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2848 | 35648 (33711-35885) | 0.028 | 0.034 | 0.039 | - | 2000 |
| 2 | 0.4074 | 30432 (29216-30692) | 0.033 | 0.041 | 0.046 | - | 2000 |
| 4 | 0.5255 | 25325 (24998-25645) | 0.039 | 0.049 | 0.055 | - | 2000 |
| 8 | 0.6348 | 19291 (19117-19436) | 0.051 | 0.066 | 0.075 | - | 2000 |
| 16 | 0.7357 | 13080 (13064-13094) | 0.075 | 0.098 | 0.120 | - | 2000 |
| 32 | 0.8185 | 8058 (7722-8187) | 0.123 | 0.161 | 0.185 | - | 2000 |
| 64 | 0.8845 | 4787 (4719-4842) | 0.205 | 0.265 | 0.309 | - | 2000 |
| 128 | 0.9386 | 2655 (2628-2693) | 0.371 | 0.464 | 0.524 | - | 2000 |
| 256 | 0.9741 | 1408 (1345-1414) | 0.708 | 0.882 | 1.03 | - | 2000 |
| 1265 | 1.0000 | 41.6 (41.5-41.7) | 23.5 | 26.8 | 27.9 | - | 125 |

**ISONGraph C++ (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 40.1 (39.7-40.1) | 24.5 | 28.4 | 29.7 | - | 123 |

**ISONGraph C++ (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2848 | 31630 (24601-32671) | 0.032 | 0.047 | 0.058 | - | 2000 |
| 2 | 0.4074 | 27775 (27261-27919) | 0.035 | 0.044 | 0.060 | - | 2000 |
| 4 | 0.5255 | 22908 (22737-23047) | 0.042 | 0.055 | 0.074 | - | 2000 |
| 8 | 0.6348 | 16744 (16490-17301) | 0.057 | 0.077 | 0.099 | - | 2000 |
| 16 | 0.7357 | 11815 (11676-12044) | 0.082 | 0.110 | 0.136 | - | 2000 |
| 32 | 0.8185 | 7116 (6877-7441) | 0.136 | 0.187 | 0.224 | - | 2000 |
| 64 | 0.8845 | 4281 (4218-4369) | 0.227 | 0.304 | 0.353 | - | 2000 |
| 128 | 0.9386 | 2361 (2314-2397) | 0.411 | 0.547 | 0.648 | - | 2000 |
| 256 | 0.9741 | 1235 (1234-1247) | 0.781 | 1.00 | 1.20 | - | 2000 |
| 1265 | 1.0000 | 39.7 (39.7-40.3) | 24.6 | 27.8 | 29.0 | - | 116 |

**ISONGraph C# (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 108 (107-108) | 7.79 | 26.4 | 35.6 | - | 387 |

**ISONGraph C# (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2868 | 14066 (6337-14394) | 0.069 | 0.212 | 0.232 | - | 2000 |
| 2 | 0.4092 | 13476 (13083-13665) | 0.073 | 0.084 | 0.091 | - | 2000 |
| 4 | 0.5273 | 12150 (11932-12195) | 0.081 | 0.096 | 0.106 | - | 2000 |
| 8 | 0.6336 | 9682 (9672-9773) | 0.100 | 0.123 | 0.139 | - | 2000 |
| 16 | 0.7249 | 7268 (7001-7362) | 0.136 | 0.175 | 0.195 | - | 2000 |
| 32 | 0.8096 | 4612 (4436-4695) | 0.215 | 0.276 | 0.309 | - | 2000 |
| 64 | 0.8780 | 2983 (2918-2984) | 0.334 | 0.423 | 0.482 | - | 2000 |
| 128 | 0.9323 | 1677 (1632-1700) | 0.594 | 0.748 | 0.829 | - | 2000 |
| 256 | 0.9727 | 886 (826-886) | 1.14 | 1.43 | 1.62 | - | 2000 |
| 1265 | 1.0000 | 98.7 (95.5-101) | 7.90 | 38.7 | 49.3 | - | 329 |

**ISONGraph Python (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 471 (468-487) | 1.98 | 2.72 | 4.89 | - | 1462 |

**ISONGraph Python (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2853 | 14708 (14670-16768) | 0.059 | 0.083 | 0.117 | - | 2000 |
| 2 | 0.4022 | 12857 (12777-14101) | 0.069 | 0.092 | 0.130 | - | 2000 |
| 4 | 0.5192 | 9844 (9812-10234) | 0.094 | 0.127 | 0.166 | - | 2000 |
| 8 | 0.6309 | 6339 (6092-6998) | 0.133 | 0.184 | 0.248 | - | 2000 |
| 16 | 0.7282 | 4918 (4775-4947) | 0.198 | 0.261 | 0.303 | - | 2000 |
| 32 | 0.8134 | 3161 (3152-3178) | 0.313 | 0.396 | 0.454 | - | 2000 |
| 64 | 0.8796 | 1876 (1867-1876) | 0.521 | 0.658 | 0.771 | - | 2000 |
| 128 | 0.9362 | 1035 (1011-1071) | 0.930 | 1.18 | 1.70 | - | 2000 |
| 256 | 0.9728 | 547 (535-558) | 1.73 | 2.31 | 3.53 | - | 1803 |
| 1265 | 1.0000 | 440 (435-446) | 2.09 | 3.11 | 5.46 | - | 1414 |

**ISONGraph JS (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 48.1 (47.8-48.9) | 20.1 | 24.6 | 27.1 | - | 167 |

**ISONGraph JS (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2795 | 6775 (6501-6778) | 0.145 | 0.184 | 0.233 | - | 2000 |
| 2 | 0.3983 | 5710 (5550-5862) | 0.171 | 0.224 | 0.250 | - | 2000 |
| 4 | 0.5165 | 4523 (4349-4604) | 0.218 | 0.290 | 0.328 | - | 2000 |
| 8 | 0.6258 | 3533 (3512-3553) | 0.279 | 0.371 | 0.415 | - | 2000 |
| 16 | 0.7241 | 2301 (2293-2328) | 0.426 | 0.580 | 0.676 | - | 2000 |
| 32 | 0.8079 | 1401 (1350-1412) | 0.709 | 0.968 | 1.13 | - | 2000 |
| 64 | 0.8763 | 779 (764-799) | 1.27 | 1.69 | 1.90 | - | 2000 |
| 128 | 0.9321 | 432 (431-447) | 2.28 | 2.96 | 3.26 | - | 1366 |
| 256 | 0.9745 | 230 (221-232) | 4.37 | 5.46 | 6.29 | - | 651 |
| 1265 | 1.0000 | 50.8 (49.5-51.3) | 19.1 | 23.1 | 24.9 | - | 154 |

**ISONGraph Rust (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 700 (660-705) | 1.42 | 1.79 | 2.14 | - | 2000 |

**ISONGraph Rust (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2851 | 52956 (49100-55464) | 0.018 | 0.025 | 0.037 | 53680 † | 2000 |
| 2 | 0.4016 | 40791 (40768-41483) | 0.024 | 0.031 | 0.038 | 42250 † | 2000 |
| 4 | 0.5209 | 32168 (32032-32233) | 0.030 | 0.041 | 0.050 | 32288 † | 2000 |
| 8 | 0.6351 | 21635 (21002-22195) | 0.045 | 0.062 | 0.076 | 21857 † | 2000 |
| 16 | 0.7308 | 13906 (13495-14167) | 0.070 | 0.097 | 0.120 | 14291 † | 2000 |
| 32 | 0.8134 | 8542 (8508-8573) | 0.115 | 0.154 | 0.180 | 8303 † | 2000 |
| 64 | 0.8817 | 4864 (4744-4875) | 0.205 | 0.267 | 0.301 | 4573 † | 2000 |
| 128 | 0.9355 | 2595 (2484-2619) | 0.384 | 0.507 | 0.570 | 2638 † | 2000 |
| 256 | 0.9740 | 1350 (1347-1359) | 0.733 | 0.912 | 1.01 | 2010 † | 2000 |
| 1265 | 1.0000 | 684 (673-693) | 1.43 | 1.72 | 2.13 | 1703 † | 1978 |

**ISONGraph Rust, target-cpu=native (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 621 (616-627) | 1.57 | 1.94 | 2.25 | - | 1749 |

**ISONGraph Rust, target-cpu=native (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2851 | 52724 (50519-53187) | 0.019 | 0.025 | 0.029 | 52173 † | 2000 |
| 2 | 0.4016 | 40953 (40918-43267) | 0.024 | 0.030 | 0.033 | 42748 † | 2000 |
| 4 | 0.5209 | 31286 (27148-32777) | 0.032 | 0.045 | 0.055 | 32611 † | 2000 |
| 8 | 0.6351 | 21679 (21020-22277) | 0.045 | 0.059 | 0.068 | 20838 † | 2000 |
| 16 | 0.7308 | 13851 (13535-13940) | 0.072 | 0.095 | 0.108 | 14078 † | 2000 |
| 32 | 0.8134 | 8218 (8199-8559) | 0.118 | 0.154 | 0.171 | 7957 † | 2000 |
| 64 | 0.8817 | 4859 (4614-4861) | 0.207 | 0.267 | 0.297 | 4645 † | 2000 |
| 128 | 0.9355 | 2584 (2419-2716) | 0.382 | 0.512 | 0.589 | 2610 † | 2000 |
| 256 | 0.9740 | 1358 (1352-1383) | 0.725 | 0.909 | 1.02 | 1940 † | 2000 |
| 1265 | 1.0000 | 636 (627-640) | 1.55 | 1.84 | 2.04 | 1625 † | 2000 |

**chromadb (HNSW)** (ef_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5396 | 1412 (1390-1486) | 0.717 | 0.801 | 0.950 | 23630 | 2000 |
| 16 | 0.6245 | 1404 (1357-1450) | 0.714 | 0.815 | 0.942 | 20762 | 2000 |
| 32 | 0.7422 | 1367 (1291-1381) | 0.753 | 0.860 | 0.953 | 15763 | 2000 |
| 64 | 0.8288 | 1153 (1132-1160) | 0.861 | 0.966 | 1.12 | 9913 | 2000 |
| 128 | 0.8957 | 1042 (1034-1051) | 0.940 | 1.09 | 1.35 | 6356 | 2000 |
| 256 | 0.9470 | 956 (933-981) | 1.03 | 1.21 | 1.43 | 3684 | 2000 |
| 512 | 0.9767 | 753 (749-776) | 1.29 | 1.50 | 1.92 | 2029 | 2000 |

**faiss IndexFlatIP** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 548 (543-570) | 1.72 | 2.23 | 3.01 | 16430 | 1809 |

**faiss HNSWFlat** (efSearch)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5377 | 32429 (29418-34917) | 0.028 | 0.043 | 0.063 | 823,215 | 2000 |
| 16 | 0.6264 | 25691 (25351-27102) | 0.036 | 0.053 | 0.068 | 502,740 | 2000 |
| 24 | 0.6956 | 20186 (19218-20931) | 0.047 | 0.068 | 0.088 | 499,650 | 2000 |
| 32 | 0.7423 | 17280 (15500-17482) | 0.057 | 0.079 | 0.108 | 381,767 | 2000 |
| 48 | 0.7966 | 11435 (11370-12845) | 0.078 | 0.108 | 0.177 | 257,689 | 2000 |
| 64 | 0.8334 | 9621 (8971-9751) | 0.100 | 0.137 | 0.240 | 211,013 | 2000 |
| 96 | 0.8744 | 6390 (4882-6712) | 0.149 | 0.208 | 0.355 | 120,766 | 2000 |
| 128 | 0.9023 | 4629 (4219-4853) | 0.200 | 0.274 | 0.430 | 92811 | 2000 |
| 192 | 0.9305 | 3206 (3076-3243) | 0.296 | 0.401 | 0.833 | 66248 | 2000 |
| 256 | 0.9483 | 2237 (2230-2250) | 0.408 | 0.539 | 1.09 | 54004 | 2000 |
| 400 | 0.9696 | 1381 (1379-1435) | 0.679 | 0.908 | 1.71 | 27177 | 2000 |
| 640 | 0.9842 | 785 (780-789) | 1.23 | 1.58 | 2.59 | 17016 | 2000 |

**faiss IVFFlat** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2827 | 51821 (50764-51839) | 0.019 | 0.022 | 0.029 | 310,318 | 2000 |
| 2 | 0.4049 | 44673 (41408-45531) | 0.021 | 0.026 | 0.042 | 382,168 | 2000 |
| 4 | 0.5251 | 35842 (34299-39054) | 0.026 | 0.034 | 0.042 | 275,217 | 2000 |
| 8 | 0.6316 | 27988 (27219-28595) | 0.034 | 0.043 | 0.056 | 246,831 | 2000 |
| 16 | 0.7287 | 19426 (11660-19775) | 0.049 | 0.066 | 0.157 | 134,086 | 2000 |
| 32 | 0.8119 | 10764 (10512-12410) | 0.083 | 0.118 | 0.165 | 153,105 | 2000 |
| 64 | 0.8796 | 6348 (5918-6871) | 0.149 | 0.203 | 0.298 | 93354 | 2000 |
| 128 | 0.9332 | 3719 (3637-3728) | 0.256 | 0.334 | 0.498 | 53383 | 2000 |
| 256 | 0.9738 | 2031 (1979-2051) | 0.457 | 0.603 | 0.990 | 26799 | 2000 |
| 1265 | 1.0000 | 438 (428-438) | 2.18 | 2.75 | 4.53 | 6400 | 1387 |

**hnswlib** (ef)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5421 | 44724 (41330-46904) | 0.022 | 0.031 | 0.040 | 373,518 | 2000 |
| 16 | 0.6244 | 36871 (34289-36877) | 0.027 | 0.038 | 0.044 | 328,116 | 2000 |
| 24 | 0.6949 | 27706 (27313-29071) | 0.035 | 0.048 | 0.058 | 212,972 | 2000 |
| 32 | 0.7392 | 23477 (23317-23597) | 0.042 | 0.055 | 0.065 | 171,278 | 2000 |
| 48 | 0.7937 | 17284 (15501-17469) | 0.059 | 0.078 | 0.090 | 132,860 | 2000 |
| 64 | 0.8287 | 12918 (12807-14084) | 0.073 | 0.095 | 0.112 | 150,099 | 2000 |
| 96 | 0.8706 | 9092 (7599-9613) | 0.103 | 0.134 | 0.216 | 103,974 | 2000 |
| 128 | 0.8950 | 7641 (7197-7770) | 0.131 | 0.164 | 0.195 | 71297 | 2000 |
| 192 | 0.9278 | 5112 (5038-5168) | 0.190 | 0.238 | 0.320 | 63883 | 2000 |
| 256 | 0.9466 | 3983 (3913-4050) | 0.243 | 0.310 | 0.412 | 51910 | 2000 |
| 400 | 0.9692 | 2709 (2670-2731) | 0.366 | 0.452 | 0.573 | 35142 | 2000 |
| 640 | 0.9843 | 1769 (1750-1809) | 0.559 | 0.693 | 0.853 | 23069 | 2000 |

**lancedb (no index)** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 34.3 (31.2-34.5) | 29.5 | 34.2 | 37.4 | - | 86 |

**lancedb IVF_FLAT** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2556 | 308 (307-310) | 3.20 | 3.67 | 4.23 | - | 950 |
| 2 | 0.3718 | 309 (304-314) | 3.18 | 3.74 | 4.39 | - | 922 |
| 4 | 0.4942 | 309 (307-309) | 3.20 | 3.71 | 4.47 | - | 987 |
| 8 | 0.6105 | 302 (297-304) | 3.25 | 3.85 | 4.72 | - | 959 |
| 16 | 0.7110 | 305 (304-310) | 3.24 | 3.66 | 4.37 | - | 922 |
| 32 | 0.8030 | 298 (297-300) | 3.33 | 3.74 | 4.41 | - | 887 |
| 64 | 0.8755 | 283 (278-284) | 3.50 | 3.90 | 4.58 | - | 880 |
| 128 | 0.9348 | 253 (247-256) | 3.84 | 4.68 | 5.99 | - | 786 |
| 256 | 0.9740 | 217 (207-219) | 4.45 | 5.51 | 7.19 | - | 670 |
| 1265 | 1.0000 | 113 (112-114) | 8.54 | 10.2 | 16.3 | - | 352 |

**numpy brute force** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 528 (479-537) | 1.82 | 2.53 | 3.70 | 1733 | 1607 |

**sqlite-vec (vec0, exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 40.1 (40.0-40.4) | 24.3 | 27.4 | 34.3 | - | 125 |

**usearch (HNSW)** (expansion_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5376 | 27475 (26481-27781) | 0.035 | 0.050 | 0.059 | 293,591 | 2000 |
| 16 | 0.6238 | 21039 (19802-21251) | 0.046 | 0.067 | 0.095 | 282,602 | 2000 |
| 24 | 0.6930 | 15794 (15665-16627) | 0.060 | 0.086 | 0.108 | 227,663 | 2000 |
| 32 | 0.7346 | 12833 (11838-13645) | 0.073 | 0.102 | 0.130 | 190,170 | 2000 |
| 48 | 0.7900 | 9549 (9154-10111) | 0.101 | 0.141 | 0.171 | 121,533 | 2000 |
| 64 | 0.8295 | 7238 (7210-7919) | 0.130 | 0.181 | 0.227 | 108,436 | 2000 |
| 96 | 0.8690 | 5186 (5094-5409) | 0.181 | 0.241 | 0.329 | 80683 | 2000 |
| 128 | 0.8950 | 3966 (3948-4152) | 0.240 | 0.316 | 0.449 | 60907 | 2000 |
| 192 | 0.9264 | 2848 (2796-2901) | 0.345 | 0.447 | 0.549 | 48221 | 2000 |
| 256 | 0.9456 | 2135 (1926-2190) | 0.471 | 0.637 | 0.736 | 29190 | 2000 |
| 400 | 0.9667 | 1375 (1372-1376) | 0.720 | 0.949 | 1.18 | 23417 | 2000 |
| 640 | 0.9819 | 876 (853-896) | 1.14 | 1.47 | 1.80 | 15709 | 2000 |

</details>

### sift, N = 100,000

Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.

| system | QPS 1T @0.90 | @0.95 | @0.99 | all-core QPS @0.90 | @0.95 | 1T latency ms median / p95 / p99 |
|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 23.4 | 23.4 | 23.4 | - | - | 42.3 / 45.4 / 47.4 (recall 1.000) |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 4693* | 4693* | 2451 | - | - | 0.211 / 0.293 / 0.336 (recall 0.952) |
| **ISONGraph C++ (exact)** | 22.8 | 22.8 | 22.8 | - | - | 44.2 / 47.5 / 49.5 (recall 1.000) |
| **ISONGraph C++ (IVF)** | 4054* | 4054* | 2083 | - | - | 0.241 / 0.336 / 0.393 (recall 0.952) |
| **ISONGraph C# (exact)** | 55.0 | 55.0 | 55.0 | - | - | 14.6 / 42.8 / 80.8 (recall 1.000) |
| **ISONGraph C# (IVF)** | 2805* | 2805* | 1635 | - | - | 0.362 / 0.482 / 0.555 (recall 0.953) |
| **ISONGraph Python (exact)** | 318 | 318 | 318 | - | - | 3.22 / 4.42 / 5.68 (recall 1.000) |
| **ISONGraph Python (IVF)** | 2722* | 2622 | 1395 | - | - | 0.570 / 0.778 / 1.08 (recall 0.987) |
| **ISONGraph JS (exact)** | 29.0 | 29.0 | 29.0 | - | - | 33.7 / 41.6 / 48.7 (recall 1.000) |
| **ISONGraph JS (IVF)** | 810* | 795 | 416 | - | - | 2.10 / 2.73 / 3.12 (recall 0.988) |
| **ISONGraph Rust (exact)** | 344 | 344 | 344 | - | - | 2.86 / 3.36 / 3.84 (recall 1.000) |
| **ISONGraph Rust (IVF)** | 4980* | 4873 | 2247 | 2648* † | 2592 † | 0.365 / 0.508 / 0.626 (recall 0.986) |
| **ISONGraph Rust, target-cpu=native (exact)** | 364 | 364 | 364 | - | - | 2.62 / 3.14 / 3.42 (recall 1.000) |
| **ISONGraph Rust, target-cpu=native (IVF)** | 4741* | 4650 | 2335 | 2523* † | 2470 † | 0.357 / 0.476 / 0.545 (recall 0.986) |
| chromadb (HNSW) | 1163* | 1163* | 1107 | 12847* | 12847* | 0.818 / 1.11 / 1.83 (recall 0.952) |
| faiss IndexFlatIP | 283 | 283 | 283 | 4537 | 4537 | 3.30 / 4.26 / 6.04 (recall 1.000) |
| faiss HNSWFlat | 21197* | 17392 | 8549 | 354,906* | 331,856 | 0.056 / 0.078 / 0.121 (recall 0.953) |
| faiss IVFFlat | 8764* | 8764* | 4401 | 112,537* | 112,537* | 0.105 / 0.160 / 0.248 (recall 0.950) |
| hnswlib | 26532* | 22439 | 11479 | 281,005* | 193,015 | 0.044 / 0.061 / 0.079 (recall 0.950) |
| lancedb (no index) | 15.1 | 15.1 | 15.1 | - | - | 66.9 / 78.5 / 82.0 (recall 1.000) |
| lancedb IVF_FLAT | 247* | 247 | 231 | - | - | 4.06 / 5.08 / 6.78 (recall 0.987) |
| numpy brute force | 282 | 282 | 282 | 1543 | 1543 | 3.34 / 4.87 / 6.69 (recall 1.000) |
| sqlite-vec (vec0, exact) | 21.6 | 21.6 | 21.6 | - | - | 47.0 / 52.6 / 57.7 (recall 1.000) |
| usearch (HNSW) | 16246* | 13215 | 5873 | 160,195* | 106,020 | 0.072 / 0.101 / 0.133 (recall 0.952) |

Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).

| system | ingest s | index s | total build s | 3 builds: median (min-max) s | ready s (incl. first query) | steady RSS +MB | peak RSS +MB (build) |
|---|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 0.049 | 0.000 | 0.049 | - | 0.079 | 84 | 102 |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 0.049 | 0.638 | 0.687 | - | 0.687 | 135 | 160 |
| **ISONGraph C++ (exact)** | 0.050 | 0.000 | 0.050 | - | 0.081 | 84 | 102 |
| **ISONGraph C++ (IVF)** | 0.050 | 0.970 | 1.02 | - | 1.02 | 134 | 160 |
| **ISONGraph C# (exact)** | 0.018 | 0.000 | 0.018 | - | 0.041 | 108 | 108 |
| **ISONGraph C# (IVF)** | 0.028 | 1.48 | 1.51 | - | 1.51 | 311 | 311 |
| **ISONGraph Python (exact)** | 1.26 | 0.000 | 1.26 | 4.28 (1.26-4.49) | 1.48 | 170 | 195 |
| **ISONGraph Python (IVF)** | 1.11 | 0.965 | 2.08 | 2.08 (2.02-2.08) | 2.08 | 192 | 303 |
| **ISONGraph JS (exact)** | 0.035 | 0.000 | 0.035 | - | 0.063 | 317 | 317 |
| **ISONGraph JS (IVF)** | 0.044 | 35.4 | 35.5 | - | 35.5 | 651 | 651 |
| **ISONGraph Rust (exact)** | 0.055 | 0.000 | 0.055 | - | 0.106 | 143 | 143 |
| **ISONGraph Rust (IVF)** | 0.055 | 0.565 | 0.671 | - | 0.671 | 146 | 172 |
| **ISONGraph Rust, target-cpu=native (exact)** | 0.057 | 0.000 | 0.057 | - | 0.115 | 143 | 143 |
| **ISONGraph Rust, target-cpu=native (IVF)** | 0.057 | 0.575 | 0.690 | - | 0.690 | 146 | 172 |
| chromadb (HNSW) | 9.06 | 0.000 | 9.06 | 9.15 (9.06-9.23) | 9.16 | 289 | 219 |
| faiss IndexFlatIP | 0.051 | 0.000 | 0.051 | 0.082 (0.051-0.088) | 0.054 | 74 | 122 |
| faiss HNSWFlat | 1.98 | 0.000 | 1.98 | 2.00 (1.98-2.09) | 1.98 | 85 | 132 |
| faiss IVFFlat | 0.000 | 1.29 | 1.29 | 1.26 (1.25-1.29) | 1.30 | 105 | 181 |
| hnswlib | 1.13 | 0.000 | 1.13 | 1.13 (1.09-1.20) | 1.13 | 97 | 108 |
| lancedb (no index) | 0.103 | 0.000 | 0.103 | 0.223 (0.103-0.247) | 0.137 | 107 | 125 |
| lancedb IVF_FLAT | 0.113 | 0.858 | 0.971 | 0.869 (0.830-0.971) | 1.06 | 189 | 230 |
| numpy brute force | 0.015 | 0.000 | 0.015 | 0.015 (0.015-0.018) | 0.018 | 62 | 74 |
| sqlite-vec (vec0, exact) | 0.556 | 0.000 | 0.556 | 1.99 (0.556-2.16) | 0.588 | 4 | 23 |
| usearch (HNSW) | 1.57 | 0.000 | 1.57 | 1.56 (1.34-1.57) | 1.57 | 85 | 96 |

<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>

**ISONGraph C++, /arch:AVX2 (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 23.4 (23.3-27.3) | 42.3 | 45.4 | 47.4 | - | 97 |

**ISONGraph C++, /arch:AVX2 (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2796 | 23360 (21573-23763) | 0.045 | 0.056 | 0.066 | - | 2000 |
| 2 | 0.4219 | 19563 (18794-20324) | 0.051 | 0.066 | 0.077 | - | 2000 |
| 4 | 0.5793 | 16494 (16250-17277) | 0.059 | 0.078 | 0.091 | - | 2000 |
| 8 | 0.7401 | 11995 (11642-12084) | 0.082 | 0.112 | 0.127 | - | 2000 |
| 16 | 0.8696 | 7631 (7570-7635) | 0.128 | 0.177 | 0.203 | - | 2000 |
| 32 | 0.9522 | 4693 (4524-4768) | 0.211 | 0.293 | 0.336 | - | 2000 |
| 64 | 0.9882 | 2719 (2711-2761) | 0.361 | 0.484 | 0.553 | - | 2000 |
| 128 | 0.9984 | 1488 (1403-1494) | 0.675 | 0.885 | 1.02 | - | 2000 |
| 256 | 0.9999 | 787 (783-833) | 1.24 | 1.52 | 1.67 | - | 2000 |
| 1265 | 1.0000 | 23.0 (22.9-23.3) | 43.3 | 46.3 | 48.4 | - | 69 |

**ISONGraph C++ (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 22.8 (22.4-25.2) | 44.2 | 47.5 | 49.5 | - | 92 |

**ISONGraph C++ (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2796 | 19362 (18833-19708) | 0.052 | 0.065 | 0.072 | - | 2000 |
| 2 | 0.4219 | 16480 (16363-16797) | 0.059 | 0.075 | 0.086 | - | 2000 |
| 4 | 0.5793 | 13533 (13443-13864) | 0.071 | 0.097 | 0.118 | - | 2000 |
| 8 | 0.7401 | 10563 (10319-10609) | 0.092 | 0.129 | 0.154 | - | 2000 |
| 16 | 0.8696 | 6672 (6639-6802) | 0.144 | 0.204 | 0.242 | - | 2000 |
| 32 | 0.9522 | 4054 (3966-4060) | 0.241 | 0.336 | 0.393 | - | 2000 |
| 64 | 0.9882 | 2290 (2202-2307) | 0.430 | 0.605 | 0.739 | - | 2000 |
| 128 | 0.9984 | 1321 (1305-1339) | 0.737 | 1.02 | 1.22 | - | 2000 |
| 256 | 0.9999 | 679 (677-688) | 1.45 | 1.84 | 2.06 | - | 2000 |
| 1265 | 1.0000 | 22.3 (22.3-23.5) | 44.8 | 47.5 | 49.3 | - | 66 |

**ISONGraph C# (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 55.0 (55.0-72.9) | 14.6 | 42.8 | 80.8 | - | 314 |

**ISONGraph C# (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2850 | 8528 (5455-9395) | 0.116 | 0.295 | 0.324 | - | 2000 |
| 2 | 0.4301 | 7905 (7740-8001) | 0.124 | 0.142 | 0.156 | - | 2000 |
| 4 | 0.5908 | 6901 (6828-7039) | 0.143 | 0.170 | 0.187 | - | 2000 |
| 8 | 0.7513 | 5823 (5639-5949) | 0.172 | 0.212 | 0.236 | - | 2000 |
| 16 | 0.8749 | 4284 (4243-4379) | 0.231 | 0.294 | 0.331 | - | 2000 |
| 32 | 0.9526 | 2805 (2528-2838) | 0.362 | 0.482 | 0.555 | - | 2000 |
| 64 | 0.9889 | 1727 (1719-1855) | 0.556 | 0.748 | 0.870 | - | 2000 |
| 128 | 0.9984 | 1056 (1029-1103) | 0.915 | 1.28 | 1.76 | - | 2000 |
| 256 | 0.9999 | 622 (597-652) | 1.58 | 2.12 | 2.44 | - | 1547 |
| 1265 | 1.0000 | 53.6 (53.0-53.7) | 15.4 | 40.9 | 87.7 | - | 198 |

**ISONGraph Python (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 318 (265-323) | 3.22 | 4.42 | 5.68 | - | 1127 |

**ISONGraph Python (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2856 | 15392 (15349-15420) | 0.061 | 0.084 | 0.121 | - | 2000 |
| 2 | 0.4253 | 13796 (13604-13991) | 0.069 | 0.091 | 0.119 | - | 2000 |
| 4 | 0.5855 | 10215 (10168-10222) | 0.093 | 0.130 | 0.163 | - | 2000 |
| 8 | 0.7427 | 7098 (6537-7152) | 0.136 | 0.186 | 0.230 | - | 2000 |
| 16 | 0.8690 | 4348 (4293-4513) | 0.212 | 0.298 | 0.458 | - | 2000 |
| 32 | 0.9468 | 2722 (2610-2761) | 0.339 | 0.516 | 0.929 | - | 2000 |
| 64 | 0.9869 | 1688 (1663-1711) | 0.570 | 0.778 | 1.08 | - | 2000 |
| 128 | 0.9976 | 880 (876-894) | 1.07 | 1.58 | 2.25 | - | 2000 |
| 256 | 0.9998 | 499 (496-509) | 1.90 | 2.53 | 3.37 | - | 1542 |
| 1265 | 1.0000 | 338 (337-354) | 2.79 | 3.71 | 4.72 | - | 1068 |

**ISONGraph JS (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 29.0 (27.6-32.5) | 33.7 | 41.6 | 48.7 | - | 106 |

**ISONGraph JS (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2886 | 3499 (3416-3525) | 0.289 | 0.350 | 0.424 | - | 2000 |
| 2 | 0.4338 | 3127 (3108-3164) | 0.317 | 0.380 | 0.426 | - | 2000 |
| 4 | 0.5894 | 2439 (2407-2486) | 0.399 | 0.524 | 0.617 | - | 2000 |
| 8 | 0.7425 | 1900 (1884-1909) | 0.516 | 0.683 | 0.797 | - | 2000 |
| 16 | 0.8685 | 1319 (1308-1363) | 0.739 | 1.02 | 1.19 | - | 2000 |
| 32 | 0.9486 | 810 (763-820) | 1.24 | 1.70 | 1.93 | - | 2000 |
| 64 | 0.9877 | 480 (464-482) | 2.10 | 2.73 | 3.12 | - | 1328 |
| 128 | 0.9984 | 248 (244-256) | 3.99 | 4.99 | 5.65 | - | 761 |
| 256 | 1.0000 | 142 (141-145) | 7.06 | 8.09 | 8.66 | - | 424 |
| 1265 | 1.0000 | 27.8 (26.8-28.0) | 35.2 | 42.4 | 45.9 | - | 79 |

**ISONGraph Rust (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 344 (333-398) | 2.86 | 3.36 | 3.84 | - | 1231 |

**ISONGraph Rust (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2912 | 33756 (33679-34626) | 0.029 | 0.038 | 0.045 | 16984 † | 2000 |
| 2 | 0.4356 | 25889 (25153-26284) | 0.038 | 0.051 | 0.060 | 14003 † | 2000 |
| 4 | 0.5955 | 20026 (19055-22804) | 0.048 | 0.066 | 0.079 | 10165 † | 2000 |
| 8 | 0.7474 | 13525 (13509-13581) | 0.072 | 0.100 | 0.117 | 6768 † | 2000 |
| 16 | 0.8710 | 8413 (8216-8451) | 0.116 | 0.162 | 0.191 | 4405 † | 2000 |
| 32 | 0.9486 | 4980 (4685-5131) | 0.199 | 0.276 | 0.332 | 2648 † | 2000 |
| 64 | 0.9860 | 2725 (2580-2747) | 0.365 | 0.508 | 0.626 | 1464 † | 2000 |
| 128 | 0.9984 | 1490 (1475-1517) | 0.666 | 0.856 | 0.989 | 738 † | 2000 |
| 256 | 0.9998 | 827 (800-837) | 1.20 | 1.50 | 1.66 | 468 † | 2000 |
| 1265 | 1.0000 | 335 (333-339) | 2.96 | 3.43 | 3.81 | 378 † | 1030 |

**ISONGraph Rust, target-cpu=native (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 364 (362-465) | 2.62 | 3.14 | 3.42 | - | 1261 |

**ISONGraph Rust, target-cpu=native (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2912 | 36185 (33982-42399) | 0.027 | 0.035 | 0.042 | 16215 † | 2000 |
| 2 | 0.4356 | 27078 (25967-27602) | 0.036 | 0.049 | 0.056 | 14669 † | 2000 |
| 4 | 0.5955 | 20258 (20173-22787) | 0.047 | 0.063 | 0.073 | 10146 † | 2000 |
| 8 | 0.7474 | 13568 (13551-14109) | 0.071 | 0.098 | 0.114 | 7176 † | 2000 |
| 16 | 0.8710 | 8586 (8391-9422) | 0.112 | 0.157 | 0.179 | 4303 † | 2000 |
| 32 | 0.9486 | 4741 (4733-4973) | 0.204 | 0.276 | 0.317 | 2523 † | 2000 |
| 64 | 0.9860 | 2780 (2727-2801) | 0.357 | 0.476 | 0.545 | 1399 † | 2000 |
| 128 | 0.9984 | 1612 (1463-1643) | 0.628 | 0.829 | 0.976 | 744 † | 2000 |
| 256 | 0.9998 | 839 (799-840) | 1.19 | 1.55 | 1.90 | 486 † | 1908 |
| 1265 | 1.0000 | 360 (352-360) | 2.79 | 3.26 | 3.65 | 316 † | 1103 |

**chromadb (HNSW)** (ef_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.7916 | 1279 (1260-1281) | 0.751 | 0.990 | 1.55 | 21363 | 2000 |
| 16 | 0.8714 | 1258 (1240-1307) | 0.766 | 0.934 | 1.69 | 18336 | 2000 |
| 32 | 0.9524 | 1163 (1116-1185) | 0.818 | 1.11 | 1.83 | 12847 | 2000 |
| 64 | 0.9888 | 1129 (1112-1154) | 0.841 | 1.09 | 1.71 | 8413 | 2000 |
| 128 | 0.9978 | 982 (965-1008) | 0.952 | 1.35 | 2.21 | 5786 | 2000 |
| 256 | 0.9998 | 812 (779-818) | 1.19 | 1.66 | 2.62 | 3652 | 2000 |
| 512 | 0.9999 | 764 (757-783) | 1.24 | 1.72 | 2.46 | 2105 | 2000 |

**faiss IndexFlatIP** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 283 (280-332) | 3.30 | 4.26 | 6.04 | 4537 | 920 |

**faiss HNSWFlat** (efSearch)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.7926 | 35472 (34557-35531) | 0.027 | 0.038 | 0.053 | 579,257 | 2000 |
| 16 | 0.8737 | 28109 (27242-28189) | 0.034 | 0.048 | 0.061 | 407,672 | 2000 |
| 24 | 0.9257 | 21197 (21007-21337) | 0.045 | 0.063 | 0.084 | 354,906 | 2000 |
| 32 | 0.9533 | 16937 (16706-17368) | 0.056 | 0.078 | 0.121 | 328,882 | 2000 |
| 48 | 0.9778 | 12453 (12093-12677) | 0.078 | 0.109 | 0.149 | 191,734 | 2000 |
| 64 | 0.9889 | 9172 (7963-9527) | 0.101 | 0.148 | 0.263 | 140,578 | 2000 |
| 96 | 0.9958 | 6013 (5876-6248) | 0.150 | 0.228 | 0.409 | 99528 | 2000 |
| 128 | 0.9981 | 4997 (4910-5057) | 0.191 | 0.264 | 0.397 | 90958 | 2000 |
| 192 | 0.9992 | 3318 (3279-3386) | 0.289 | 0.393 | 0.566 | 70793 | 2000 |
| 256 | 0.9997 | 2528 (2456-2663) | 0.380 | 0.508 | 0.801 | 48638 | 2000 |
| 400 | 0.9999 | 1537 (1491-1552) | 0.627 | 0.836 | 1.33 | 29966 | 2000 |

**faiss IVFFlat** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2837 | 46449 (45779-47166) | 0.021 | 0.025 | 0.031 | 260,845 | 2000 |
| 2 | 0.4257 | 39480 (39424-40291) | 0.024 | 0.031 | 0.041 | 250,401 | 2000 |
| 4 | 0.5849 | 30978 (28532-32812) | 0.030 | 0.042 | 0.062 | 221,614 | 2000 |
| 8 | 0.7447 | 22575 (21736-24142) | 0.041 | 0.058 | 0.083 | 207,689 | 2000 |
| 16 | 0.8686 | 14664 (14010-15618) | 0.064 | 0.095 | 0.132 | 144,067 | 2000 |
| 32 | 0.9505 | 8764 (8547-8823) | 0.105 | 0.160 | 0.248 | 112,537 | 2000 |
| 64 | 0.9876 | 4991 (4987-5213) | 0.181 | 0.279 | 0.610 | 51491 | 2000 |
| 128 | 0.9986 | 2804 (2624-2813) | 0.340 | 0.506 | 0.854 | 29165 | 2000 |
| 256 | 1.0000 | 1438 (1423-1475) | 0.654 | 1.01 | 1.51 | 15785 | 2000 |
| 1265 | 1.0000 | 308 (300-311) | 3.13 | 3.91 | 4.82 | 3462 | 988 |

**hnswlib** (ef)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.7877 | 44466 (42336-45863) | 0.021 | 0.031 | 0.043 | 392,472 | 2000 |
| 16 | 0.8690 | 35361 (34241-36359) | 0.028 | 0.038 | 0.049 | 310,299 | 2000 |
| 24 | 0.9228 | 26532 (26390-27774) | 0.036 | 0.051 | 0.068 | 281,005 | 2000 |
| 32 | 0.9503 | 22404 (21713-22520) | 0.044 | 0.061 | 0.079 | 192,350 | 2000 |
| 48 | 0.9762 | 16412 (15860-16465) | 0.060 | 0.081 | 0.103 | 175,231 | 2000 |
| 64 | 0.9879 | 12748 (12557-13235) | 0.075 | 0.105 | 0.137 | 130,878 | 2000 |
| 96 | 0.9954 | 8801 (8414-8850) | 0.107 | 0.157 | 0.246 | 99854 | 2000 |
| 128 | 0.9978 | 6789 (6288-7469) | 0.137 | 0.210 | 0.373 | 80265 | 2000 |
| 192 | 0.9990 | 4893 (4496-4938) | 0.193 | 0.308 | 0.601 | 53647 | 2000 |
| 256 | 0.9997 | 4274 (4261-4377) | 0.230 | 0.315 | 0.412 | 47855 | 2000 |
| 400 | 1.0000 | 2940 (2766-2973) | 0.329 | 0.474 | 0.840 | 36822 | 2000 |

**lancedb (no index)** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 15.1 (13.7-16.0) | 66.9 | 78.5 | 82.0 | - | 43 |

**lancedb IVF_FLAT** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2785 | 277 (270-279) | 3.55 | 4.48 | 5.87 | - | 835 |
| 2 | 0.4209 | 287 (268-289) | 3.47 | 4.27 | 5.94 | - | 780 |
| 4 | 0.5785 | 276 (270-283) | 3.53 | 4.43 | 5.90 | - | 832 |
| 8 | 0.7388 | 269 (264-281) | 3.59 | 4.54 | 6.01 | - | 825 |
| 16 | 0.8673 | 263 (263-270) | 3.68 | 4.55 | 6.20 | - | 800 |
| 32 | 0.9486 | 247 (247-251) | 3.90 | 4.89 | 7.06 | - | 766 |
| 64 | 0.9868 | 238 (234-244) | 4.06 | 5.08 | 6.78 | - | 731 |
| 128 | 0.9977 | 213 (203-215) | 4.59 | 5.78 | 8.62 | - | 624 |
| 256 | 0.9999 | 186 (185-189) | 5.21 | 6.41 | 8.26 | - | 550 |
| 1265 | 1.0000 | 92.3 (90.6-92.3) | 10.6 | 12.9 | 17.3 | - | 300 |

**numpy brute force** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 282 (267-297) | 3.34 | 4.87 | 6.69 | 1543 | 842 |

**sqlite-vec (vec0, exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 21.6 (21.0-22.4) | 47.0 | 52.6 | 57.7 | - | 93 |

**usearch (HNSW)** (expansion_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.7861 | 27069 (26424-27812) | 0.035 | 0.052 | 0.072 | 360,458 | 2000 |
| 16 | 0.8702 | 19975 (19615-21362) | 0.045 | 0.066 | 0.089 | 235,977 | 2000 |
| 24 | 0.9224 | 16246 (15886-16879) | 0.058 | 0.080 | 0.103 | 160,195 | 2000 |
| 32 | 0.9517 | 13048 (12862-13737) | 0.072 | 0.101 | 0.133 | 103,358 | 2000 |
| 48 | 0.9768 | 9040 (8229-9296) | 0.105 | 0.167 | 0.257 | 133,914 | 2000 |
| 64 | 0.9868 | 6439 (6418-7550) | 0.134 | 0.222 | 0.319 | 91078 | 2000 |
| 96 | 0.9939 | 5250 (5048-5360) | 0.181 | 0.274 | 0.393 | 77948 | 2000 |
| 128 | 0.9958 | 4190 (4062-4302) | 0.228 | 0.336 | 0.633 | 32827 | 2000 |
| 192 | 0.9970 | 2769 (2674-3016) | 0.336 | 0.519 | 0.885 | 47475 | 2000 |
| 256 | 0.9975 | 2330 (1993-2437) | 0.422 | 0.630 | 0.957 | 37724 | 2000 |
| 400 | 0.9976 | 1592 (1525-1656) | 0.607 | 0.875 | 1.32 | 25792 | 2000 |
| 640 | 0.9977 | 1064 (1030-1108) | 0.909 | 1.29 | 1.82 | 13865 | 2000 |

</details>

### text384, N = 1,000,000

Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.

| system | QPS 1T @0.90 | @0.95 | @0.99 | all-core QPS @0.90 | @0.95 | 1T latency ms median / p95 / p99 |
|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 2.13 | 2.13 | 2.13 | - | - | 466 / 491 / 493 (recall 1.000) |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 221* | 128 | - | - | - | 8.38 / 9.75 / 11.4 (recall 0.955) |
| **ISONGraph C++ (exact)** | 2.11 | 2.11 | 2.11 | - | - | 469 / 500 / 516 (recall 1.000) |
| **ISONGraph C++ (IVF)** | 173* | 94.0 | - | - | - | 11.1 / 12.9 / 14.1 (recall 0.953) |
| **ISONGraph C# (exact)** | 3.72 | 3.72 | 3.72 | - | - | 245 / 394 / 398 (recall 1.000) |
| **ISONGraph C# (IVF)** | 141* | - | - | - | - | - |
| **ISONGraph Python (exact)** | 12.7 | 12.7 | 12.7 | - | - | 77.2 / 86.1 / 93.0 (recall 1.000) |
| **ISONGraph Python (IVF)** | 125* | 69.2 | 16.7 | - | - | 14.6 / 18.1 / 23.7 (recall 0.953) |
| **ISONGraph JS (exact)** | 1.67 | 1.67 | 1.67 | - | - | 545 / 838 / 872 (recall 1.000) |
| **ISONGraph JS (IVF)** | 52.0* | - | - | - | - | - |
| **ISONGraph Rust (exact)** | 15.1 | 15.1 | 15.1 | - | - | 65.3 / 70.0 / 80.5 (recall 1.000) |
| **ISONGraph Rust (IVF)** | 192* | 103 | - | 579* † | 356 † | 9.89 / 11.4 / 12.3 (recall 0.951) |
| **ISONGraph Rust, target-cpu=native (exact)** | 15.8 | 15.8 | 15.8 | - | - | 62.7 / 68.9 / 71.3 (recall 1.000) |
| **ISONGraph Rust, target-cpu=native (IVF)** | 190* | 103 | - | 548* † | 342 † | 9.91 / 11.7 / 12.5 (recall 0.952) |
| chromadb (HNSW) | 687* | 605 | - | 1939* | 1464 | 1.83 / 2.15 / 2.93 (recall 0.967) |
| faiss IndexFlatIP | 13.4 | 13.4 | 13.4 | 540 | 540 | 73.4 / 83.8 / 87.6 (recall 1.000) |
| faiss HNSWFlat | 2880 | 1572 | 406 | 10413 | 5757 | 0.796 / 0.979 / 1.24 (recall 0.965) |
| faiss IVFFlat | 321* | 190 | 22.5 | 910* | 520 | 5.63 / 7.03 / 13.2 (recall 0.956) |
| hnswlib | 2663 | 1486 | 474 | 10424 | 5894 | 0.718 / 0.905 / 1.29 (recall 0.957) |
| lancedb (no index) | 0.725 | 0.725 | 0.725 | - | - | 1372 / 1438 / 1463 (recall 1.000) |
| lancedb IVF_FLAT | 95.4* | 11.4* | 11.4* | - | - | 87.4 / 90.4 / 92.8 (recall 1.000) |
| numpy brute force | 14.2 | 14.2 | 14.2 | 142 | 142 | 69.3 / 77.1 / 84.0 (recall 1.000) |
| sqlite-vec (vec0, exact) | 1.05 | 1.05 | 1.05 | - | - | 947 / 988 / 991 (recall 1.000) |
| usearch (HNSW) | 1274 | 706 | 214 | 10254 | 5827 | 1.52 / 1.93 / 2.44 (recall 0.957) |

Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).

| system | ingest s | index s | total build s | 3 builds: median (min-max) s | ready s (incl. first query) | steady RSS +MB | peak RSS +MB (build) |
|---|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 0.922 | 0.000 | 0.922 | - | 1.43 | 1,783 | 1,922 |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 0.922 | 6.20 | 7.12 | - | 7.12 | 3,279 | 3,514 |
| **ISONGraph C++ (exact)** | 0.917 | 0.000 | 0.917 | - | 1.43 | 1,783 | 1,922 |
| **ISONGraph C++ (IVF)** | 0.917 | 11.4 | 12.4 | - | 12.4 | 3,279 | 3,514 |
| **ISONGraph C# (exact)** | 0.217 | 0.000 | 0.217 | - | 0.515 | 2,447 | 2,447 |
| **ISONGraph C# (IVF)** | 0.177 | 10.7 | 10.9 | - | 10.9 | 6,656 | 6,664 |
| **ISONGraph Python (exact)** | 19.6 | 0.000 | 19.6 | - | 22.8 | 4,223 | 2,682 |
| **ISONGraph Python (IVF)** | 20.8 | 21.4 | 42.2 | - | 42.2 | 4,251 | 4,795 |
| **ISONGraph JS (exact)** | 0.470 | 0.000 | 0.470 | - | 1.03 | 3,620 | 3,620 |
| **ISONGraph JS (IVF)** | 0.691 | 521 | 522 | - | 522 | 7,002 | 7,002 |
| **ISONGraph Rust (exact)** | 0.812 | 0.000 | 0.812 | - | 1.84 | 3,517 | 3,524 |
| **ISONGraph Rust (IVF)** | 0.812 | 5.71 | 7.55 | - | 7.55 | 3,532 | 3,779 |
| **ISONGraph Rust, target-cpu=native (exact)** | 0.763 | 0.000 | 0.763 | - | 1.83 | 3,517 | 3,525 |
| **ISONGraph Rust, target-cpu=native (IVF)** | 0.763 | 5.56 | 7.40 | - | 7.40 | 3,532 | 3,779 |
| chromadb (HNSW) | 343 | 0.000 | 343 | - | 345 | 4,023 | 2,147 |
| faiss IndexFlatIP | 1.10 | 0.000 | 1.10 | - | 1.18 | 1,553 | 2,122 |
| faiss HNSWFlat | 221 | 0.000 | 221 | - | 221 | 1,640 | 2,168 |
| faiss IVFFlat | 0.000 | 18.5 | 18.5 | - | 18.6 | 1,885 | 3,592 |
| hnswlib | 173 | 0.000 | 173 | - | 173 | 1,830 | 1,860 |
| lancedb (no index) | 2.04 | 0.000 | 2.04 | - | 2.69 | 152 | 226 |
| lancedb IVF_FLAT | 2.06 | 10.7 | 12.8 | - | 13.4 | 1,736 | 3,081 |
| numpy brute force | 0.298 | 0.000 | 0.298 | - | 0.368 | 1,576 | 1,605 |
| sqlite-vec (vec0, exact) | 9.60 | 0.000 | 9.60 | - | 10.5 | 4 | 62 |
| usearch (HNSW) | 174 | 0.000 | 174 | - | 174 | 1,715 | 1,745 |

<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>

**ISONGraph C++, /arch:AVX2 (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 2.13 (2.13-2.13) | 466 | 491 | 493 | - | 30 |

**ISONGraph C++, /arch:AVX2 (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2666 | 4780 (4475-4797) | 0.206 | 0.281 | 0.373 | - | 2000 |
| 2 | 0.3725 | 3920 (3909-3942) | 0.246 | 0.330 | 0.381 | - | 2000 |
| 4 | 0.4895 | 3016 (2956-3039) | 0.323 | 0.432 | 0.499 | - | 2000 |
| 8 | 0.6020 | 1989 (1984-2021) | 0.489 | 0.658 | 0.772 | - | 2000 |
| 16 | 0.7007 | 1243 (1230-1245) | 0.795 | 1.02 | 1.15 | - | 2000 |
| 32 | 0.7890 | 713 (696-718) | 1.39 | 1.74 | 2.00 | - | 2000 |
| 64 | 0.8603 | 404 (397-406) | 2.45 | 3.04 | 3.41 | - | 1301 |
| 128 | 0.9141 | 221 (216-223) | 4.50 | 5.33 | 5.77 | - | 714 |
| 256 | 0.9547 | 119 (115-120) | 8.38 | 9.75 | 11.4 | - | 360 |

**ISONGraph C++ (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 2.11 (2.11-2.11) | 469 | 500 | 516 | - | 30 |

**ISONGraph C++ (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2669 | 4177 (3635-4194) | 0.239 | 0.334 | 0.550 | - | 2000 |
| 2 | 0.3726 | 3308 (3222-3337) | 0.297 | 0.399 | 0.457 | - | 2000 |
| 4 | 0.4897 | 2387 (2353-2421) | 0.409 | 0.556 | 0.668 | - | 2000 |
| 8 | 0.6026 | 1589 (1576-1602) | 0.613 | 0.819 | 0.954 | - | 2000 |
| 16 | 0.7010 | 961 (954-985) | 1.02 | 1.32 | 1.49 | - | 2000 |
| 32 | 0.7849 | 559 (554-559) | 1.78 | 2.21 | 2.50 | - | 1627 |
| 64 | 0.8613 | 311 (303-313) | 3.16 | 3.87 | 4.28 | - | 969 |
| 128 | 0.9152 | 173 (171-173) | 5.78 | 6.77 | 7.48 | - | 499 |
| 256 | 0.9528 | 89.5 (89.5-90.5) | 11.1 | 12.9 | 14.1 | - | 286 |

**ISONGraph C# (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 3.72 (3.72-3.72) | 245 | 394 | 398 | - | 30 |

**ISONGraph C# (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2741 | 2529 (2179-2620) | 0.384 | 0.630 | 0.733 | - | 2000 |
| 2 | 0.3823 | 2129 (2066-2179) | 0.455 | 0.612 | 0.754 | - | 2000 |
| 4 | 0.5001 | 1685 (1601-1685) | 0.591 | 0.783 | 0.916 | - | 2000 |
| 8 | 0.6082 | 1171 (1136-1175) | 0.840 | 1.11 | 1.26 | - | 2000 |
| 16 | 0.7054 | 747 (734-756) | 1.32 | 1.74 | 2.02 | - | 2000 |
| 32 | 0.7822 | 445 (438-446) | 2.23 | 2.83 | 3.18 | - | 1183 |
| 64 | 0.8550 | 253 (250-258) | 3.88 | 4.87 | 5.62 | - | 857 |
| 128 | 0.9096 | 141 (141-145) | 6.93 | 8.40 | 9.48 | - | 426 |
| 256 | 0.9462 | 74.5 (72.8-77.0) | 13.1 | 16.2 | 19.2 | - | 234 |

**ISONGraph Python (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 12.7 (12.7-12.8) | 77.2 | 86.1 | 93.0 | - | 36 |

**ISONGraph Python (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2745 | 3293 (3160-3317) | 0.290 | 0.423 | 0.519 | - | 2000 |
| 2 | 0.3764 | 2389 (2356-2438) | 0.393 | 0.569 | 0.712 | - | 2000 |
| 4 | 0.4940 | 1711 (1689-1742) | 0.557 | 0.786 | 1.03 | - | 2000 |
| 8 | 0.6007 | 1166 (1077-1175) | 0.857 | 1.13 | 1.33 | - | 2000 |
| 16 | 0.7006 | 668 (658-687) | 1.44 | 1.88 | 2.25 | - | 2000 |
| 32 | 0.7813 | 396 (385-399) | 2.50 | 3.08 | 3.62 | - | 1191 |
| 64 | 0.8540 | 217 (213-219) | 4.47 | 5.41 | 6.55 | - | 689 |
| 128 | 0.9119 | 125 (125-125) | 7.76 | 9.83 | 15.3 | - | 404 |
| 256 | 0.9532 | 65.8 (65.0-67.4) | 14.6 | 18.1 | 23.7 | - | 211 |
| 4000 | 1.0000 | 11.5 (11.4-12.1) | 83.0 | 99.3 | 139 | - | 37 |

**ISONGraph JS (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 1.67 (1.67-1.67) | 545 | 838 | 872 | - | 30 |

**ISONGraph JS (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2670 | 666 (660-668) | 1.47 | 1.74 | 1.99 | - | 1907 |
| 2 | 0.3726 | 594 (587-594) | 1.66 | 2.02 | 2.28 | - | 1634 |
| 4 | 0.4833 | 493 (488-493) | 2.00 | 2.55 | 2.80 | - | 1452 |
| 8 | 0.5923 | 368 (366-368) | 2.70 | 3.40 | 3.68 | - | 1071 |
| 16 | 0.6892 | 245 (244-249) | 4.03 | 5.05 | 5.61 | - | 647 |
| 32 | 0.7796 | 158 (158-161) | 6.23 | 7.67 | 8.31 | - | 465 |
| 64 | 0.8444 | 95.9 (94.8-96.2) | 10.3 | 12.4 | 13.5 | - | 252 |
| 128 | 0.9132 | 52.0 (50.9-54.3) | 18.7 | 23.3 | 25.6 | - | 167 |
| 256 | 0.9488 | 29.0 (28.3-29.4) | 34.3 | 40.4 | 42.1 | - | 84 |

**ISONGraph Rust (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 15.1 (15.1-15.1) | 65.3 | 70.0 | 80.5 | - | 46 |

**ISONGraph Rust (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2757 | 5323 (5238-5829) | 0.175 | 0.248 | 0.324 | 5692 † | 2000 |
| 2 | 0.3806 | 4242 (4037-4392) | 0.229 | 0.330 | 0.408 | 4320 † | 2000 |
| 4 | 0.4961 | 3080 (2980-3091) | 0.323 | 0.436 | 0.509 | 2997 † | 2000 |
| 8 | 0.6046 | 1936 (1889-1993) | 0.507 | 0.673 | 0.791 | 1970 † | 2000 |
| 16 | 0.7022 | 1147 (1140-1157) | 0.867 | 1.09 | 1.22 | 1233 † | 2000 |
| 32 | 0.7902 | 648 (647-652) | 1.53 | 1.88 | 2.05 | 1131 † | 1961 |
| 64 | 0.8606 | 356 (353-356) | 2.77 | 3.37 | 4.00 | 867 † | 1083 |
| 128 | 0.9149 | 192 (190-193) | 5.13 | 6.23 | 7.19 | 579 † | 598 |
| 256 | 0.9512 | 100 (100-100) | 9.89 | 11.4 | 12.3 | 350 † | 287 |

**ISONGraph Rust, target-cpu=native (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 15.8 (15.6-15.8) | 62.7 | 68.9 | 71.3 | - | 49 |

**ISONGraph Rust, target-cpu=native (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2757 | 5805 (5410-5872) | 0.171 | 0.243 | 0.283 | 6133 † | 2000 |
| 2 | 0.3806 | 4753 (4604-4781) | 0.206 | 0.294 | 0.342 | 4672 † | 2000 |
| 4 | 0.4961 | 3050 (2960-3122) | 0.315 | 0.444 | 0.555 | 3007 † | 2000 |
| 8 | 0.6046 | 1935 (1921-1950) | 0.510 | 0.678 | 0.774 | 1956 † | 2000 |
| 16 | 0.7022 | 1143 (1127-1148) | 0.869 | 1.12 | 1.26 | 1200 † | 2000 |
| 32 | 0.7908 | 647 (639-650) | 1.54 | 1.89 | 2.06 | 1086 † | 1986 |
| 64 | 0.8593 | 354 (352-361) | 2.79 | 3.36 | 3.61 | 872 † | 1149 |
| 128 | 0.9149 | 190 (188-191) | 5.21 | 6.19 | 6.68 | 548 † | 570 |
| 256 | 0.9521 | 99.8 (98.9-100) | 9.91 | 11.7 | 12.5 | 333 † | 292 |

**chromadb (HNSW)** (ef_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5578 | 1205 (1198-1215) | 0.821 | 0.960 | 1.14 | 8619 | 2000 |
| 16 | 0.6533 | 1110 (1108-1170) | 0.866 | 1.02 | 1.36 | 6744 | 2000 |
| 32 | 0.7740 | 992 (991-1007) | 0.968 | 1.15 | 1.61 | 5012 | 2000 |
| 64 | 0.8717 | 878 (851-887) | 1.09 | 1.29 | 1.89 | 3256 | 2000 |
| 128 | 0.9323 | 687 (646-713) | 1.42 | 1.70 | 2.43 | 1939 | 2000 |
| 256 | 0.9674 | 535 (535-540) | 1.83 | 2.15 | 2.93 | 1112 | 1640 |
| 512 | 0.9859 | 349 (348-365) | 2.77 | 3.50 | 4.82 | 614 | 1103 |

**faiss IndexFlatIP** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 13.4 (13.3-13.5) | 73.4 | 83.8 | 87.6 | 540 | 40 |

**faiss HNSWFlat** (efSearch)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5647 | 11088 (10192-11826) | 0.084 | 0.137 | 0.175 | 44533 | 2000 |
| 16 | 0.6714 | 8786 (8337-8862) | 0.110 | 0.168 | 0.203 | 31150 | 2000 |
| 24 | 0.7482 | 6566 (5968-6666) | 0.146 | 0.225 | 0.286 | 24367 | 2000 |
| 32 | 0.7935 | 5568 (5413-5569) | 0.173 | 0.247 | 0.321 | 19413 | 2000 |
| 48 | 0.8516 | 4032 (3892-4187) | 0.238 | 0.331 | 0.409 | 14764 | 2000 |
| 64 | 0.8875 | 3259 (3140-3290) | 0.299 | 0.401 | 0.510 | 11597 | 2000 |
| 96 | 0.9255 | 2238 (2124-2263) | 0.426 | 0.566 | 0.792 | 8359 | 2000 |
| 128 | 0.9445 | 1721 (1665-1743) | 0.558 | 0.717 | 0.982 | 6400 | 2000 |
| 192 | 0.9648 | 1233 (1191-1284) | 0.796 | 0.979 | 1.24 | 4330 | 2000 |
| 256 | 0.9742 | 913 (853-926) | 1.09 | 1.43 | 1.89 | 3621 | 2000 |
| 400 | 0.9843 | 594 (578-602) | 1.65 | 2.10 | 3.03 | 2364 | 1626 |
| 640 | 0.9913 | 371 (370-377) | 2.65 | 3.14 | 3.91 | 1590 | 1086 |

**faiss IVFFlat** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2711 | 5665 (5464-5733) | 0.168 | 0.219 | 0.344 | 48473 | 2000 |
| 2 | 0.3827 | 4593 (4539-4850) | 0.197 | 0.260 | 0.462 | 33894 | 2000 |
| 4 | 0.4944 | 3674 (3513-3767) | 0.255 | 0.350 | 0.654 | 17040 | 2000 |
| 8 | 0.6087 | 2710 (2700-2774) | 0.351 | 0.475 | 0.670 | 11517 | 2000 |
| 16 | 0.7099 | 1693 (1644-1827) | 0.537 | 0.750 | 1.51 | 6263 | 2000 |
| 32 | 0.7952 | 1079 (1069-1087) | 0.874 | 1.16 | 1.62 | 3435 | 2000 |
| 64 | 0.8620 | 579 (578-598) | 1.59 | 2.10 | 3.37 | 1784 | 1690 |
| 128 | 0.9183 | 321 (315-324) | 3.00 | 3.76 | 5.05 | 910 | 986 |
| 256 | 0.9561 | 171 (169-171) | 5.63 | 7.03 | 13.2 | 467 | 541 |
| 4000 | 1.0000 | 12.3 (12.2-12.6) | 78.6 | 89.4 | 100 | 30.4 | 38 |

**hnswlib** (ef)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5589 | 9990 (7940-11885) | 0.092 | 0.170 | 0.251 | 45715 | 2000 |
| 16 | 0.6512 | 8804 (8071-8872) | 0.109 | 0.164 | 0.208 | 36683 | 2000 |
| 24 | 0.7278 | 6992 (6565-7080) | 0.136 | 0.197 | 0.249 | 27873 | 2000 |
| 32 | 0.7763 | 5558 (5293-5665) | 0.173 | 0.247 | 0.312 | 22939 | 2000 |
| 48 | 0.8354 | 4276 (4196-4387) | 0.224 | 0.306 | 0.392 | 17165 | 2000 |
| 64 | 0.8727 | 3223 (3211-3270) | 0.296 | 0.406 | 0.557 | 13525 | 2000 |
| 96 | 0.9101 | 2482 (2352-2494) | 0.398 | 0.523 | 0.663 | 9472 | 2000 |
| 128 | 0.9315 | 1908 (1862-1987) | 0.510 | 0.654 | 0.816 | 7534 | 2000 |
| 192 | 0.9572 | 1348 (1332-1385) | 0.718 | 0.905 | 1.29 | 5355 | 2000 |
| 256 | 0.9685 | 1043 (1037-1047) | 0.944 | 1.20 | 1.53 | 4059 | 2000 |
| 400 | 0.9817 | 700 (685-721) | 1.41 | 1.75 | 2.37 | 2681 | 2000 |
| 640 | 0.9901 | 472 (461-472) | 2.15 | 2.57 | 2.99 | 1848 | 1224 |

**lancedb (no index)** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 0.725 (0.725-0.725) | 1372 | 1438 | 1463 | - | 30 |

**lancedb IVF_FLAT** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2147 | 228 (225-231) | 4.26 | 5.05 | 5.76 | - | 688 |
| 2 | 0.3104 | 229 (223-229) | 4.29 | 5.03 | 6.11 | - | 703 |
| 4 | 0.4113 | 240 (234-243) | 4.13 | 4.66 | 5.25 | - | 728 |
| 8 | 0.5227 | 225 (223-234) | 4.31 | 4.95 | 5.88 | - | 630 |
| 16 | 0.6346 | 211 (211-220) | 4.57 | 5.35 | 6.32 | - | 689 |
| 32 | 0.7313 | 185 (185-192) | 5.16 | 6.38 | 8.70 | - | 593 |
| 64 | 0.8150 | 162 (159-165) | 5.95 | 7.13 | 9.41 | - | 525 |
| 128 | 0.8876 | 134 (127-135) | 7.29 | 8.52 | 10.2 | - | 389 |
| 256 | 0.9378 | 95.4 (95.0-96.7) | 10.2 | 11.7 | 13.3 | - | 299 |
| 4000 | 1.0000 | 11.4 (11.3-11.5) | 87.4 | 90.4 | 92.8 | - | 33 |

**numpy brute force** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 14.2 (14.1-14.4) | 69.3 | 77.1 | 84.0 | 142 | 42 |

**sqlite-vec (vec0, exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 1.05 (1.05-1.05) | 947 | 988 | 991 | - | 30 |

**usearch (HNSW)** (expansion_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.5583 | 5894 (5335-6034) | 0.166 | 0.252 | 0.307 | 47685 | 2000 |
| 16 | 0.6489 | 4309 (4212-4677) | 0.217 | 0.329 | 0.396 | 33250 | 2000 |
| 24 | 0.7299 | 3584 (3020-3614) | 0.271 | 0.407 | 0.564 | 27185 | 2000 |
| 32 | 0.7786 | 2935 (2833-2957) | 0.328 | 0.473 | 0.566 | 22420 | 2000 |
| 48 | 0.8359 | 2082 (2039-2155) | 0.450 | 0.631 | 0.811 | 15681 | 2000 |
| 64 | 0.8685 | 1651 (1638-1655) | 0.574 | 0.780 | 1.15 | 13218 | 2000 |
| 96 | 0.9090 | 1183 (1141-1229) | 0.804 | 1.08 | 1.64 | 9541 | 2000 |
| 128 | 0.9307 | 920 (908-926) | 1.07 | 1.35 | 1.70 | 7204 | 2000 |
| 192 | 0.9569 | 642 (627-669) | 1.52 | 1.93 | 2.44 | 5398 | 2000 |
| 256 | 0.9673 | 522 (511-541) | 1.90 | 2.33 | 2.79 | 4125 | 1639 |
| 400 | 0.9822 | 335 (333-339) | 2.97 | 3.69 | 4.92 | 2616 | 999 |
| 640 | 0.9900 | 214 (208-220) | 4.61 | 5.83 | 11.1 | 1810 | 634 |

</details>

### glove, N = 1,000,000

Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.

| system | QPS 1T @0.90 | @0.95 | @0.99 | all-core QPS @0.90 | @0.95 | 1T latency ms median / p95 / p99 |
|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 2.58 | 2.58 | 2.58 | - | - | 387 / 405 / 407 (recall 1.000) |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 325* | 185 | - | - | - | 5.63 / 7.38 / 8.03 (recall 0.953) |
| **ISONGraph C++ (exact)** | 2.48 | 2.48 | 2.48 | - | - | 406 / 415 / 420 (recall 1.000) |
| **ISONGraph C++ (IVF)** | 322* | 175 | - | - | - | 5.80 / 7.43 / 8.02 (recall 0.951) |
| **ISONGraph C# (exact)** | 5.81 | 5.81 | 5.81 | - | - | 126 / 370 / 399 (recall 1.000) |
| **ISONGraph C# (IVF)** | 248* | - | - | - | - | - |
| **ISONGraph Python (exact)** | 26.5 | 26.5 | 26.5 | - | - | 37.1 / 42.5 / 47.8 (recall 1.000) |
| **ISONGraph Python (IVF)** | 255* | 156 | 47.9 | - | - | 6.95 / 9.14 / 14.9 (recall 0.957) |
| **ISONGraph Rust (exact)** | 43.5 | 43.5 | 43.5 | - | - | 22.5 / 24.6 / 25.9 (recall 1.000) |
| **ISONGraph Rust (IVF)** | 314* | 181 | - | 273* † | 176 † | 6.01 / 7.67 / 8.42 (recall 0.954) |
| **ISONGraph Rust, target-cpu=native (exact)** | 40.1 | 40.1 | 40.1 | - | - | 25.7 / 26.9 / 28.1 (recall 1.000) |
| **ISONGraph Rust, target-cpu=native (IVF)** | 325* | 171 | - | 292* † | 175 † | 6.36 / 8.01 / 8.54 (recall 0.954) |
| faiss IndexFlatIP | 40.7 | 40.7 | 40.7 | 639 | 639 | 24.2 / 27.1 / 33.2 (recall 1.000) |
| faiss HNSWFlat | 1405 | - | - | 11921 | - | - |
| faiss IVFFlat | 989* | 43.8* | 43.8* | 3993* | 159* | 22.2 / 28.9 / 31.4 (recall 1.000) |
| hnswlib | 1934 | - | - | 9075 | - | - |
| lancedb (no index) | 2.89 | 2.89 | 2.89 | - | - | 346 / 385 / 393 (recall 1.000) |
| lancedb IVF_FLAT | 133* | 109 | 29.3 | - | - | 9.32 / 10.9 / 11.4 (recall 0.951) |
| numpy brute force | 34.4 | 34.4 | 34.4 | 69.2 | 69.2 | 28.8 / 33.2 / 36.0 (recall 1.000) |
| usearch (HNSW) | 840 | - | - | 10396 | - | - |

Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).

| system | ingest s | index s | total build s | 3 builds: median (min-max) s | ready s (incl. first query) | steady RSS +MB | peak RSS +MB (build) |
|---|---|---|---|---|---|---|---|
| **ISONGraph C++, /arch:AVX2 (exact)** | 0.556 | 0.000 | 0.556 | - | 0.871 | 671 | 810 |
| **ISONGraph C++, /arch:AVX2 (IVF)** | 0.556 | 6.73 | 7.29 | - | 7.29 | 1,079 | 1,141 |
| **ISONGraph C++ (exact)** | 0.533 | 0.000 | 0.533 | - | 0.815 | 672 | 810 |
| **ISONGraph C++ (IVF)** | 0.533 | 8.72 | 9.25 | - | 9.25 | 1,080 | 1,141 |
| **ISONGraph C# (exact)** | 0.198 | 0.000 | 0.198 | - | 0.320 | 810 | 810 |
| **ISONGraph C# (IVF)** | 0.339 | 11.3 | 11.6 | - | 11.6 | 2,073 | 2,073 |
| **ISONGraph Python (exact)** | 27.9 | 0.000 | 27.9 | - | 32.8 | 1,424 | 764 |
| **ISONGraph Python (IVF)** | 10.1 | 11.8 | 21.9 | - | 22.0 | 1,444 | 1,634 |
| **ISONGraph Rust (exact)** | 0.786 | 0.000 | 0.786 | - | 1.45 | 1,345 | 1,353 |
| **ISONGraph Rust (IVF)** | 0.786 | 5.69 | 7.14 | - | 7.14 | 1,356 | 1,422 |
| **ISONGraph Rust, target-cpu=native (exact)** | 0.750 | 0.000 | 0.750 | - | 1.41 | 1,345 | 1,353 |
| **ISONGraph Rust, target-cpu=native (IVF)** | 0.750 | 5.69 | 7.10 | - | 7.10 | 1,356 | 1,422 |
| faiss IndexFlatIP | 0.375 | 0.000 | 0.375 | - | 0.404 | 470 | 605 |
| faiss HNSWFlat | 60.1 | 0.000 | 60.1 | - | 60.2 | 546 | 666 |
| faiss IVFFlat | 0.000 | 11.9 | 11.9 | - | 11.9 | 534 | 983 |
| hnswlib | 51.0 | 0.000 | 51.0 | - | 51.0 | 749 | 757 |
| lancedb (no index) | 0.801 | 0.000 | 0.801 | - | 0.958 | 107 | 124 |
| lancedb IVF_FLAT | 0.765 | 8.92 | 9.68 | - | 9.96 | 555 | 866 |
| numpy brute force | 0.122 | 0.000 | 0.122 | - | 0.149 | 493 | 500 |
| usearch (HNSW) | 105 | 0.000 | 105 | - | 105 | 631 | 639 |

<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>

**ISONGraph C++, /arch:AVX2 (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 2.58 (2.58-2.58) | 387 | 405 | 407 | - | 30 |

**ISONGraph C++, /arch:AVX2 (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2955 | 7425 (7276-7598) | 0.137 | 0.190 | 0.229 | - | 2000 |
| 2 | 0.4190 | 6023 (5907-6306) | 0.165 | 0.232 | 0.292 | - | 2000 |
| 4 | 0.5330 | 4520 (4502-4652) | 0.218 | 0.318 | 0.386 | - | 2000 |
| 8 | 0.6322 | 2864 (2804-3030) | 0.339 | 0.494 | 0.556 | - | 2000 |
| 16 | 0.7206 | 1825 (1797-1840) | 0.538 | 0.785 | 0.872 | - | 2000 |
| 32 | 0.7965 | 1111 (1061-1130) | 0.890 | 1.32 | 1.50 | - | 2000 |
| 64 | 0.8616 | 610 (601-633) | 1.60 | 2.33 | 2.64 | - | 1991 |
| 128 | 0.9203 | 325 (323-343) | 3.02 | 4.14 | 4.55 | - | 943 |
| 256 | 0.9527 | 176 (170-183) | 5.63 | 7.38 | 8.03 | - | 552 |

**ISONGraph C++ (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 2.48 (2.48-2.48) | 406 | 415 | 420 | - | 30 |

**ISONGraph C++ (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2955 | 6471 (6388-6694) | 0.153 | 0.198 | 0.223 | - | 2000 |
| 2 | 0.4190 | 5429 (5367-5468) | 0.183 | 0.237 | 0.260 | - | 2000 |
| 4 | 0.5330 | 4324 (4301-4344) | 0.228 | 0.309 | 0.346 | - | 2000 |
| 8 | 0.6322 | 2985 (2982-3000) | 0.327 | 0.459 | 0.511 | - | 2000 |
| 16 | 0.7206 | 1908 (1818-1972) | 0.514 | 0.745 | 0.848 | - | 2000 |
| 32 | 0.7965 | 1086 (1065-1086) | 0.905 | 1.29 | 1.43 | - | 2000 |
| 64 | 0.8636 | 589 (588-594) | 1.67 | 2.30 | 2.51 | - | 1797 |
| 128 | 0.9202 | 322 (313-326) | 3.10 | 4.07 | 4.39 | - | 924 |
| 256 | 0.9508 | 172 (167-175) | 5.80 | 7.43 | 8.02 | - | 494 |

**ISONGraph C# (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 5.81 (5.81-5.81) | 126 | 370 | 399 | - | 30 |

**ISONGraph C# (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2969 | 2483 (2139-2665) | 0.411 | 0.500 | 0.945 | - | 2000 |
| 2 | 0.4143 | 2367 (2353-2403) | 0.431 | 0.501 | 0.540 | - | 2000 |
| 4 | 0.5305 | 2036 (2021-2060) | 0.499 | 0.601 | 0.652 | - | 2000 |
| 8 | 0.6330 | 1558 (1547-1575) | 0.637 | 0.814 | 0.892 | - | 2000 |
| 16 | 0.7188 | 1108 (1092-1115) | 0.893 | 1.19 | 1.32 | - | 2000 |
| 32 | 0.7955 | 713 (707-719) | 1.38 | 1.88 | 2.04 | - | 2000 |
| 64 | 0.8603 | 423 (417-425) | 2.35 | 3.13 | 3.45 | - | 1483 |
| 128 | 0.9160 | 248 (246-273) | 3.86 | 5.20 | 5.63 | - | 674 |
| 256 | 0.9485 | 131 (129-145) | 7.37 | 9.61 | 10.5 | - | 336 |

**ISONGraph Python (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 26.5 (26.2-27.9) | 37.1 | 42.5 | 47.8 | - | 84 |

**ISONGraph Python (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.3014 | 7447 (7436-7674) | 0.127 | 0.183 | 0.227 | - | 2000 |
| 2 | 0.4195 | 5473 (5429-5495) | 0.170 | 0.246 | 0.320 | - | 2000 |
| 4 | 0.5359 | 3762 (3631-3937) | 0.248 | 0.360 | 0.445 | - | 2000 |
| 8 | 0.6390 | 2438 (2412-2542) | 0.383 | 0.570 | 0.689 | - | 2000 |
| 16 | 0.7264 | 1503 (1497-1512) | 0.636 | 0.942 | 1.10 | - | 2000 |
| 32 | 0.7996 | 881 (854-891) | 1.10 | 1.58 | 1.90 | - | 2000 |
| 64 | 0.8634 | 492 (492-498) | 1.98 | 2.70 | 3.07 | - | 1635 |
| 128 | 0.9207 | 255 (251-256) | 3.78 | 5.02 | 11.5 | - | 774 |
| 256 | 0.9574 | 138 (137-140) | 6.95 | 9.14 | 14.9 | - | 418 |
| 4000 | 1.0000 | 34.6 (32.8-35.7) | 28.0 | 35.0 | 53.5 | - | 111 |

**ISONGraph Rust (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 43.5 (42.8-48.6) | 22.5 | 24.6 | 25.9 | - | 170 |

**ISONGraph Rust (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.3088 | 10525 (10503-10566) | 0.092 | 0.134 | 0.167 | 5474 † | 2000 |
| 2 | 0.4230 | 7216 (6962-7478) | 0.135 | 0.197 | 0.241 | 4107 † | 2000 |
| 4 | 0.5377 | 5054 (5004-5225) | 0.192 | 0.282 | 0.345 | 2611 † | 2000 |
| 8 | 0.6346 | 3277 (3218-3329) | 0.297 | 0.441 | 0.522 | 1647 † | 2000 |
| 16 | 0.7284 | 1923 (1908-1938) | 0.508 | 0.751 | 0.857 | 1028 † | 2000 |
| 32 | 0.8013 | 1168 (1096-1174) | 0.854 | 1.21 | 1.39 | 564 † | 2000 |
| 64 | 0.8640 | 644 (588-644) | 1.58 | 2.19 | 2.47 | 363 † | 1587 |
| 128 | 0.9204 | 314 (312-321) | 3.19 | 4.24 | 4.63 | 273 † | 921 |
| 256 | 0.9537 | 169 (162-170) | 6.01 | 7.67 | 8.42 | 166 † | 475 |

**ISONGraph Rust, target-cpu=native (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 40.1 (38.9-40.2) | 25.7 | 26.9 | 28.1 | - | 159 |

**ISONGraph Rust, target-cpu=native (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.3088 | 9787 (9759-10042) | 0.099 | 0.140 | 0.165 | 5548 † | 2000 |
| 2 | 0.4230 | 7586 (7559-7781) | 0.129 | 0.178 | 0.211 | 3924 † | 2000 |
| 4 | 0.5377 | 5136 (5043-5309) | 0.190 | 0.270 | 0.322 | 2675 † | 2000 |
| 8 | 0.6346 | 3220 (3142-3377) | 0.300 | 0.441 | 0.520 | 1596 † | 2000 |
| 16 | 0.7284 | 1903 (1795-1965) | 0.519 | 0.765 | 0.875 | 1049 † | 2000 |
| 32 | 0.8013 | 1010 (1005-1016) | 0.971 | 1.39 | 1.54 | 634 † | 2000 |
| 64 | 0.8626 | 576 (554-581) | 1.73 | 2.39 | 2.64 | 336 † | 1848 |
| 128 | 0.9166 | 325 (308-344) | 3.05 | 4.08 | 4.53 | 292 † | 1032 |
| 256 | 0.9544 | 157 (156-158) | 6.36 | 8.01 | 8.54 | 164 † | 548 |

**faiss IndexFlatIP** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 40.7 (39.8-42.6) | 24.2 | 27.1 | 33.2 | 639 | 149 |

**faiss HNSWFlat** (efSearch)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.4765 | 22888 (18723-24051) | 0.041 | 0.066 | 0.086 | 166,477 | 2000 |
| 16 | 0.5605 | 17882 (14705-18039) | 0.052 | 0.083 | 0.117 | 134,983 | 2000 |
| 24 | 0.6299 | 13430 (13067-13770) | 0.071 | 0.108 | 0.134 | 103,151 | 2000 |
| 32 | 0.6724 | 10956 (10906-11267) | 0.086 | 0.129 | 0.157 | 73210 | 2000 |
| 48 | 0.7281 | 7966 (7703-8175) | 0.118 | 0.179 | 0.224 | 57418 | 2000 |
| 64 | 0.7658 | 6097 (6026-6304) | 0.151 | 0.224 | 0.295 | 48203 | 2000 |
| 96 | 0.8093 | 4504 (4414-4594) | 0.213 | 0.301 | 0.391 | 33683 | 2000 |
| 128 | 0.8340 | 3283 (3280-3489) | 0.282 | 0.392 | 0.503 | 27685 | 2000 |
| 192 | 0.8646 | 2297 (2103-2329) | 0.419 | 0.551 | 0.717 | 19009 | 2000 |
| 256 | 0.8850 | 1782 (1671-1796) | 0.553 | 0.701 | 0.980 | 14590 | 2000 |
| 400 | 0.9119 | 1164 (1152-1232) | 0.803 | 1.16 | 1.87 | 10161 | 2000 |
| 640 | 0.9347 | 653 (650-654) | 1.52 | 1.79 | 2.02 | 6535 | 1892 |

**faiss IVFFlat** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.3046 | 19282 (18048-19407) | 0.050 | 0.071 | 0.091 | 170,065 | 2000 |
| 2 | 0.4281 | 15642 (13255-16241) | 0.061 | 0.091 | 0.135 | 117,092 | 2000 |
| 4 | 0.5371 | 12355 (11603-12379) | 0.078 | 0.106 | 0.134 | 83273 | 2000 |
| 8 | 0.6348 | 8461 (8222-8862) | 0.112 | 0.163 | 0.216 | 49956 | 2000 |
| 16 | 0.7203 | 5748 (5576-5760) | 0.170 | 0.239 | 0.288 | 26400 | 2000 |
| 32 | 0.7933 | 3221 (3128-3365) | 0.286 | 0.406 | 0.527 | 14622 | 2000 |
| 64 | 0.8588 | 1796 (1793-1819) | 0.522 | 0.721 | 0.882 | 7627 | 2000 |
| 128 | 0.9123 | 989 (977-997) | 0.973 | 1.28 | 1.64 | 3993 | 2000 |
| 256 | 0.9488 | 525 (519-534) | 1.85 | 2.34 | 3.19 | 2064 | 1649 |
| 4000 | 1.0000 | 43.8 (43.0-43.8) | 22.2 | 28.9 | 31.4 | 159 | 136 |

**hnswlib** (ef)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.4829 | 27767 (24854-28305) | 0.035 | 0.057 | 0.075 | 141,005 | 2000 |
| 16 | 0.5657 | 22942 (21474-23347) | 0.042 | 0.065 | 0.079 | 105,029 | 2000 |
| 24 | 0.6309 | 17335 (15586-17788) | 0.054 | 0.080 | 0.097 | 82464 | 2000 |
| 32 | 0.6724 | 13832 (13632-14170) | 0.067 | 0.101 | 0.125 | 67729 | 2000 |
| 48 | 0.7266 | 10275 (10124-10618) | 0.090 | 0.127 | 0.158 | 54141 | 2000 |
| 64 | 0.7621 | 8681 (8081-8779) | 0.111 | 0.154 | 0.189 | 44182 | 2000 |
| 96 | 0.8022 | 6005 (5703-6227) | 0.158 | 0.224 | 0.449 | 31593 | 2000 |
| 128 | 0.8283 | 4941 (4752-5057) | 0.198 | 0.260 | 0.311 | 25814 | 2000 |
| 192 | 0.8605 | 3292 (2942-3346) | 0.289 | 0.384 | 0.822 | 17875 | 2000 |
| 256 | 0.8807 | 2481 (2466-2493) | 0.380 | 0.539 | 1.12 | 10936 | 2000 |
| 400 | 0.9119 | 1660 (1584-1692) | 0.585 | 0.832 | 1.32 | 8093 | 2000 |
| 640 | 0.9367 | 1086 (1071-1101) | 0.893 | 1.24 | 1.82 | 5013 | 2000 |

**lancedb (no index)** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 2.89 (2.89-2.89) | 346 | 385 | 393 | - | 30 |

**lancedb IVF_FLAT** (nprobes)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2826 | 232 (165-257) | 4.14 | 7.38 | 9.20 | - | 710 |
| 2 | 0.3981 | 212 (158-234) | 4.48 | 7.54 | 10.2 | - | 499 |
| 4 | 0.5174 | 260 (220-261) | 3.81 | 6.06 | 8.14 | - | 787 |
| 8 | 0.6209 | 166 (159-166) | 6.23 | 7.77 | 8.86 | - | 638 |
| 16 | 0.7151 | 168 (167-177) | 5.81 | 7.56 | 8.25 | - | 481 |
| 32 | 0.7928 | 166 (160-171) | 5.93 | 7.75 | 8.37 | - | 557 |
| 64 | 0.8552 | 150 (146-152) | 6.70 | 8.27 | 9.45 | - | 384 |
| 128 | 0.9126 | 133 (124-134) | 7.72 | 9.20 | 11.3 | - | 338 |
| 256 | 0.9513 | 109 (107-111) | 9.32 | 10.9 | 11.4 | - | 323 |
| 4000 | 1.0000 | 20.9 (20.2-21.2) | 48.0 | 52.1 | 54.9 | - | 59 |

**numpy brute force** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 34.4 (33.3-36.6) | 28.8 | 33.2 | 36.0 | 69.2 | 121 |

**usearch (HNSW)** (expansion_search)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.4733 | 10942 (10420-11107) | 0.087 | 0.135 | 0.163 | 60767 | 2000 |
| 16 | 0.5670 | 7685 (6804-8557) | 0.124 | 0.194 | 0.240 | 54147 | 2000 |
| 24 | 0.6345 | 6417 (6102-6460) | 0.150 | 0.230 | 0.281 | 45841 | 2000 |
| 32 | 0.6755 | 5292 (5219-5339) | 0.180 | 0.271 | 0.333 | 32467 | 2000 |
| 48 | 0.7307 | 3777 (3637-3788) | 0.248 | 0.369 | 0.470 | 22370 | 2000 |
| 64 | 0.7647 | 2879 (2823-2890) | 0.340 | 0.512 | 0.646 | 19236 | 2000 |
| 96 | 0.8055 | 2037 (1988-2059) | 0.484 | 0.710 | 0.853 | 13595 | 2000 |
| 128 | 0.8315 | 2194 (1636-2462) | 0.443 | 0.798 | 1.14 | 11891 | 2000 |
| 192 | 0.8649 | 1400 (1137-1461) | 0.712 | 1.15 | 1.61 | 16330 | 2000 |
| 256 | 0.8839 | 1041 (1019-1070) | 0.908 | 1.36 | 2.44 | 12443 | 2000 |
| 400 | 0.9121 | 716 (712-726) | 1.33 | 1.91 | 3.53 | 9088 | 2000 |
| 640 | 0.9394 | 437 (404-451) | 2.21 | 3.61 | 5.66 | 2503 | 1353 |

</details>

### sift, N = 1,000,000

Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.

| system | QPS 1T @0.90 | @0.95 | @0.99 | all-core QPS @0.90 | @0.95 | 1T latency ms median / p95 / p99 |
|---|---|---|---|---|---|---|
| **ISONGraph C++ (exact)** | 2.19 | 2.19 | 2.19 | - | - | 459 / 472 / 474 (recall 1.000) |
| **ISONGraph C++ (IVF)** | 979* | 562* | 314 | - | - | 1.75 / 2.54 / 2.83 (recall 0.970) |
| **ISONGraph Rust (exact)** | 33.4 | 33.4 | 33.4 | - | - | 30.3 / 31.9 / 32.3 (recall 0.999) |
| **ISONGraph Rust (IVF)** | 1078* | 546* | 314 | 600* † | 382* † | 1.76 / 2.62 / 2.89 (recall 0.970) |
| faiss HNSWFlat | 8350 | 5603 | 2175 | 56008 | 37075 | 0.190 / 0.255 / 0.324 (recall 0.957) |
| faiss IVFFlat | 2209* | 1270* | 714 | 7504* | 3254* | 0.761 / 1.14 / 1.30 (recall 0.970) |
| hnswlib | 10660 | 7311 | 3266 | 51989 | 37313 | 0.151 / 0.193 / 0.226 (recall 0.958) |
| numpy brute force | 26.9 | 26.9 | 26.9 | 67.4 | 67.4 | 37.4 / 41.8 / 46.8 (recall 0.998) |

Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).

| system | ingest s | index s | total build s | 3 builds: median (min-max) s | ready s (incl. first query) | steady RSS +MB | peak RSS +MB (build) |
|---|---|---|---|---|---|---|---|
| **ISONGraph C++ (exact)** | 0.615 | 0.000 | 0.615 | - | 0.944 | 783 | 922 |
| **ISONGraph C++ (IVF)** | 0.615 | 10.9 | 11.5 | - | 11.5 | 1,299 | 1,377 |
| **ISONGraph Rust (exact)** | 0.827 | 0.000 | 0.827 | - | 1.67 | 1,560 | 1,567 |
| **ISONGraph Rust (IVF)** | 0.827 | 6.37 | 8.04 | - | 8.04 | 1,570 | 1,654 |
| faiss HNSWFlat | 99.7 | 0.000 | 99.7 | - | 99.7 | 663 | 814 |
| faiss IVFFlat | 0.000 | 26.8 | 26.8 | - | 26.9 | 684 | 1,258 |
| hnswlib | 60.4 | 0.000 | 60.4 | - | 60.4 | 854 | 864 |
| numpy brute force | 0.137 | 0.000 | 0.137 | - | 0.166 | 599 | 609 |

<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>

**ISONGraph C++ (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 1.0000 | 2.19 (2.19-2.19) | 459 | 472 | 474 | - | 30 |

**ISONGraph C++ (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2549 | 5716 (5606-5905) | 0.174 | 0.222 | 0.277 | - | 2000 |
| 2 | 0.3890 | 4864 (4720-4877) | 0.204 | 0.259 | 0.301 | - | 2000 |
| 4 | 0.5391 | 3921 (3791-3942) | 0.251 | 0.344 | 0.417 | - | 2000 |
| 8 | 0.6881 | 2732 (2601-2735) | 0.356 | 0.526 | 0.640 | - | 2000 |
| 16 | 0.8212 | 1686 (1661-1687) | 0.566 | 0.879 | 1.05 | - | 2000 |
| 32 | 0.9147 | 979 (944-989) | 0.984 | 1.51 | 1.74 | - | 2000 |
| 64 | 0.9700 | 562 (525-579) | 1.75 | 2.54 | 2.83 | - | 1522 |
| 128 | 0.9918 | 297 (292-302) | 3.30 | 4.47 | 5.00 | - | 870 |
| 256 | 0.9978 | 155 (154-174) | 6.18 | 7.79 | 8.36 | - | 411 |

**ISONGraph Rust (exact)** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 0.9990 | 33.4 (32.8-33.9) | 30.3 | 31.9 | 32.3 | - | 105 |

**ISONGraph Rust (IVF)** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2633 | 10887 (10198-10960) | 0.092 | 0.122 | 0.149 | 6064 † | 2000 |
| 2 | 0.3933 | 7937 (7581-8142) | 0.123 | 0.168 | 0.205 | 4494 † | 2000 |
| 4 | 0.5365 | 5763 (5718-5832) | 0.165 | 0.245 | 0.308 | 3000 † | 2000 |
| 8 | 0.6876 | 3349 (3312-3668) | 0.277 | 0.442 | 0.545 | 1974 † | 2000 |
| 16 | 0.8216 | 2089 (2066-2111) | 0.453 | 0.734 | 0.862 | 1224 † | 2000 |
| 32 | 0.9163 | 1078 (1046-1088) | 0.891 | 1.41 | 1.60 | 600 † | 2000 |
| 64 | 0.9701 | 546 (541-565) | 1.76 | 2.62 | 2.89 | 382 † | 1518 |
| 128 | 0.9924 | 294 (294-295) | 3.34 | 4.57 | 5.02 | 290 † | 1014 |
| 256 | 0.9988 | 157 (155-162) | 6.33 | 7.87 | 8.50 | 239 † | 502 |

**faiss HNSWFlat** (efSearch)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.6877 | 17412 (15222-17554) | 0.055 | 0.092 | 0.165 | 135,453 | 2000 |
| 16 | 0.7802 | 13831 (12997-13850) | 0.070 | 0.104 | 0.201 | 92820 | 2000 |
| 24 | 0.8484 | 11228 (11083-12040) | 0.085 | 0.124 | 0.169 | 72882 | 2000 |
| 32 | 0.8891 | 9050 (8957-9248) | 0.107 | 0.153 | 0.248 | 60222 | 2000 |
| 48 | 0.9339 | 6490 (6449-7113) | 0.148 | 0.201 | 0.275 | 44633 | 2000 |
| 64 | 0.9573 | 5243 (5201-5347) | 0.190 | 0.255 | 0.324 | 34095 | 2000 |
| 96 | 0.9778 | 3716 (3678-4041) | 0.265 | 0.344 | 0.410 | 24544 | 2000 |
| 128 | 0.9866 | 2622 (2368-2680) | 0.390 | 0.512 | 0.609 | 16650 | 2000 |
| 192 | 0.9936 | 1790 (1707-1800) | 0.584 | 0.723 | 0.835 | 11858 | 2000 |
| 256 | 0.9962 | 1384 (1364-1415) | 0.742 | 0.918 | 1.03 | 8866 | 2000 |
| 400 | 0.9984 | 809 (803-836) | 1.27 | 1.50 | 1.65 | 5686 | 2000 |
| 640 | 0.9990 | 482 (482-488) | 2.14 | 2.51 | 2.66 | 3978 | 1572 |

**faiss IVFFlat** (nprobe)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 1 | 0.2635 | 9853 (9769-10714) | 0.097 | 0.138 | 0.178 | 47015 | 2000 |
| 2 | 0.3934 | 9352 (9319-9919) | 0.107 | 0.146 | 0.202 | 53015 | 2000 |
| 4 | 0.5409 | 7229 (6865-7675) | 0.135 | 0.197 | 0.253 | 38314 | 2000 |
| 8 | 0.6917 | 5382 (4759-5567) | 0.187 | 0.280 | 0.349 | 23059 | 2000 |
| 16 | 0.8229 | 3576 (3541-3728) | 0.267 | 0.409 | 0.483 | 13783 | 2000 |
| 32 | 0.9173 | 2209 (2168-2230) | 0.434 | 0.664 | 0.777 | 7504 | 2000 |
| 64 | 0.9700 | 1270 (1266-1270) | 0.761 | 1.14 | 1.30 | 3254 | 2000 |
| 128 | 0.9923 | 668 (663-678) | 1.46 | 2.02 | 2.26 | 2326 | 2000 |
| 256 | 0.9984 | 358 (346-363) | 2.80 | 3.63 | 4.04 | 1115 | 1035 |
| 4000 | 0.9988 | 27.4 (26.9-28.1) | 36.2 | 39.0 | 40.8 | 79.8 | 86 |

**hnswlib** (ef)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| 10 | 0.6795 | 22475 (21929-22477) | 0.044 | 0.062 | 0.073 | 80773 | 2000 |
| 16 | 0.7752 | 17160 (15591-17310) | 0.058 | 0.088 | 0.111 | 72843 | 2000 |
| 24 | 0.8472 | 13361 (13153-14935) | 0.073 | 0.098 | 0.115 | 52927 | 2000 |
| 32 | 0.8900 | 11346 (11208-11457) | 0.089 | 0.117 | 0.136 | 55624 | 2000 |
| 48 | 0.9339 | 8629 (8446-8866) | 0.117 | 0.155 | 0.180 | 41344 | 2000 |
| 64 | 0.9576 | 6765 (6690-6876) | 0.151 | 0.193 | 0.226 | 35561 | 2000 |
| 96 | 0.9768 | 4866 (4847-4880) | 0.210 | 0.267 | 0.312 | 25048 | 2000 |
| 128 | 0.9862 | 3957 (3729-4163) | 0.257 | 0.328 | 0.374 | 24450 | 2000 |
| 192 | 0.9937 | 2717 (2703-2840) | 0.374 | 0.468 | 0.540 | 14951 | 2000 |
| 256 | 0.9966 | 2159 (1950-2229) | 0.488 | 0.627 | 0.725 | 11572 | 2000 |
| 400 | 0.9987 | 1475 (1412-1490) | 0.705 | 0.896 | 0.994 | 8898 | 2000 |
| 640 | 0.9993 | 973 (950-976) | 1.06 | 1.35 | 1.50 | 5333 | 2000 |

**numpy brute force** (exact)

| knob | recall@10 | QPS 1T (min-max) | median ms | p95 ms | p99 ms | all-core QPS | queries/run |
|---|---|---|---|---|---|---|---|
| - | 0.9982 | 26.9 (26.7-28.1) | 37.4 | 41.8 | 46.8 | 67.4 | 110 |

</details>



### Exact search latency (10k / 100k / 1M)

`median ms (1T)` is the pinned single-thread loop; `all-threads single query` lets the library use every core for one query at a time; `batch QPS` is a batched call on 32 logical CPUs.

#### text384

| N | system | median ms (1T) | p95 | p99 | QPS 1T (min-max) | all-threads single query | batch QPS | recall | first query ms |
|---|---|---|---|---|---|---|---|---|---|
| 10,000 | **ISONGraph C++, /arch:AVX2 (exact)** | 3.08 | 3.48 | 3.84 | 328 (308-328) | - | - | 1.000 | 3.58 |
| 10,000 | **ISONGraph C++ (exact)** | 3.23 | 3.63 | 4.30 | 311 (289-311) | - | - | 1.000 | 4.08 |
| 10,000 | **ISONGraph C# (exact)** | 2.33 | 2.76 | 4.29 | 422 (382-423) | - | - | 1.000 | 24.2 |
| 10,000 | **ISONGraph Python (exact)** | 0.421 | 0.503 | 0.873 | 2206 (2202-2337) | 0.283 ms (3334 QPS) | - | 1.000 | 32.4 |
| 10,000 | **ISONGraph JS (exact)** | 5.00 | 5.64 | 6.04 | 198 (192-198) | - | - | 1.000 | 5.88 |
| 10,000 | **ISONGraph Rust (exact)** | 0.273 | 0.310 | 0.363 | 3728 (3617-3796) | 0.258 ms (3896 QPS) | - | 1.000 | 7.19 |
| 10,000 | **ISONGraph Rust, target-cpu=native (exact)** | 0.274 | 0.308 | 0.351 | 3617 (3469-3630) | 0.261 ms (3836 QPS) | - | 1.000 | 7.38 |
| 10,000 | faiss IndexFlatIP | 0.336 | 0.441 | 0.590 | 2845 (2753-2860) | 3.48 ms (293 QPS) | 55693 | 1.000 | 0.873 |
| 10,000 | lancedb (no index) | 16.0 | 18.9 | 23.2 | 60.9 (60.4-61.1) | 13.3 ms (74.6 QPS) | - | 1.000 | 16.3 |
| 10,000 | numpy brute force | 0.304 | 0.350 | 0.453 | 3364 (3126-3518) | 0.151 ms (6581 QPS) | 14690 | 1.000 | 0.842 |
| 10,000 | qdrant-client local mode (exact numpy) | 13.2 | 15.9 | 22.1 | 73.0 (72.4-73.4) | 12.5 ms (79.2 QPS) | - | 1.000 | 39.9 |
| 10,000 | sqlite-vec (vec0, exact) | 9.36 | 11.2 | 14.0 | 104 (103-105) | 9.57 ms (103 QPS) | - | 1.000 | 9.76 |
| 100,000 | **ISONGraph C++, /arch:AVX2 (exact)** | 43.5 | 45.4 | 47.6 | 23.0 (23.0-23.1) | - | - | 1.000 | 43.8 |
| 100,000 | **ISONGraph C++ (exact)** | 44.2 | 47.2 | 57.7 | 22.6 (22.5-23.3) | - | - | 1.000 | 45.0 |
| 100,000 | **ISONGraph C# (exact)** | 23.6 | 26.7 | 78.4 | 39.1 (38.6-39.9) | - | - | 1.000 | 46.9 |
| 100,000 | **ISONGraph Python (exact)** | 7.37 | 8.92 | 14.3 | 131 (129-131) | 5.30 ms (183 QPS) | - | 1.000 | 303 |
| 100,000 | **ISONGraph JS (exact)** | 50.8 | 55.8 | 58.1 | 19.7 (19.1-20.3) | - | - | 1.000 | 65.7 |
| 100,000 | **ISONGraph Rust (exact)** | 6.41 | 6.99 | 7.86 | 154 (154-154) | 3.21 ms (299 QPS) | - | 1.000 | 80.0 |
| 100,000 | **ISONGraph Rust, target-cpu=native (exact)** | 6.06 | 6.79 | 7.39 | 163 (161-164) | 3.24 ms (293 QPS) | - | 1.000 | 79.1 |
| 100,000 | faiss IndexFlatIP | 7.29 | 8.84 | 15.0 | 131 (130-135) | 4.97 ms (198 QPS) | 5437 | 1.000 | 8.38 |
| 100,000 | lancedb (no index) | 137 | 151 | 173 | 7.19 (7.16-7.22) | 72.1 ms (13.7 QPS) | - | 1.000 | 64.9 |
| 100,000 | numpy brute force | 6.75 | 7.66 | 10.1 | 144 (143-146) | 4.54 ms (217 QPS) | 1553 | 1.000 | 7.36 |
| 100,000 | qdrant-client local mode (exact numpy) | 136 | 152 | 158 | 7.24 (7.14-7.25) | 132 ms (7.55 QPS) | - | 1.000 | 127 |
| 100,000 | sqlite-vec (vec0, exact) | 95.5 | 115 | 122 | 10.2 (10.2-10.3) | 95.3 ms (10.4 QPS) | - | 1.000 | 98.5 |
| 1,000,000 | **ISONGraph C++, /arch:AVX2 (exact)** | 466 | 491 | 493 | 2.13 (2.13-2.13) | - | - | 1.000 | 510 |
| 1,000,000 | **ISONGraph C++ (exact)** | 469 | 500 | 516 | 2.11 (2.11-2.11) | - | - | 1.000 | 509 |
| 1,000,000 | **ISONGraph C# (exact)** | 245 | 394 | 398 | 3.72 (3.72-3.72) | - | - | 1.000 | 299 |
| 1,000,000 | **ISONGraph Python (exact)** | 77.2 | 86.1 | 93.0 | 12.7 (12.7-12.8) | 46.6 ms (21.3 QPS) | - | 1.000 | 3159 |
| 1,000,000 | **ISONGraph JS (exact)** | 545 | 838 | 872 | 1.67 (1.67-1.67) | - | - | 1.000 | 558 |
| 1,000,000 | **ISONGraph Rust (exact)** | 65.3 | 70.0 | 80.5 | 15.1 (15.1-15.1) | 28.7 ms (34.2 QPS) | - | 1.000 | 1030 |
| 1,000,000 | **ISONGraph Rust, target-cpu=native (exact)** | 62.7 | 68.9 | 71.3 | 15.8 (15.6-15.8) | 30.9 ms (31.9 QPS) | - | 1.000 | 1070 |
| 1,000,000 | faiss IndexFlatIP | 73.4 | 83.8 | 87.6 | 13.4 (13.3-13.5) | 43.1 ms (22.8 QPS) | 540 | 1.000 | 77.6 |
| 1,000,000 | lancedb (no index) | 1372 | 1438 | 1463 | 0.725 (0.725-0.725) | 576 ms (1.73 QPS) | - | 1.000 | 654 |
| 1,000,000 | numpy brute force | 69.3 | 77.1 | 84.0 | 14.2 (14.1-14.4) | 35.1 ms (28.0 QPS) | 142 | 1.000 | 70.3 |
| 1,000,000 | sqlite-vec (vec0, exact) | 947 | 988 | 991 | 1.05 (1.05-1.05) | 938 ms (1.06 QPS) | - | 1.000 | 943 |

#### glove

| N | system | median ms (1T) | p95 | p99 | QPS 1T (min-max) | all-threads single query | batch QPS | recall | first query ms |
|---|---|---|---|---|---|---|---|---|---|
| 100,000 | **ISONGraph C++, /arch:AVX2 (exact)** | 24.4 | 28.5 | 31.7 | 40.2 (39.7-41.2) | - | - | 1.000 | 27.1 |
| 100,000 | **ISONGraph C++ (exact)** | 24.5 | 28.4 | 29.7 | 40.1 (39.7-40.1) | - | - | 1.000 | 25.5 |
| 100,000 | **ISONGraph C# (exact)** | 7.79 | 26.4 | 35.6 | 108 (107-108) | - | - | 1.000 | 18.3 |
| 100,000 | **ISONGraph Python (exact)** | 1.98 | 2.72 | 4.89 | 471 (468-487) | 1.28 ms (734 QPS) | - | 1.000 | 179 |
| 100,000 | **ISONGraph JS (exact)** | 20.1 | 24.6 | 27.1 | 48.1 (47.8-48.9) | - | - | 1.000 | 24.7 |
| 100,000 | **ISONGraph Rust (exact)** | 1.42 | 1.79 | 2.14 | 700 (660-705) | 0.578 ms (1670 QPS) | - | 1.000 | 40.8 |
| 100,000 | **ISONGraph Rust, target-cpu=native (exact)** | 1.57 | 1.94 | 2.25 | 621 (616-627) | 0.611 ms (1594 QPS) | - | 1.000 | 41.8 |
| 100,000 | faiss IndexFlatIP | 1.72 | 2.23 | 3.01 | 548 (543-570) | 0.413 ms (2335 QPS) | 16430 | 1.000 | 2.19 |
| 100,000 | lancedb (no index) | 29.5 | 34.2 | 37.4 | 34.3 (31.2-34.5) | 11.4 ms (86.5 QPS) | - | 1.000 | 21.9 |
| 100,000 | numpy brute force | 1.82 | 2.53 | 3.70 | 528 (479-537) | 0.907 ms (1018 QPS) | 1733 | 1.000 | 2.37 |
| 100,000 | sqlite-vec (vec0, exact) | 24.3 | 27.4 | 34.3 | 40.1 (40.0-40.4) | 24.5 ms (40.5 QPS) | - | 1.000 | 25.3 |
| 1,000,000 | **ISONGraph C++, /arch:AVX2 (exact)** | 387 | 405 | 407 | 2.58 (2.58-2.58) | - | - | 1.000 | 316 |
| 1,000,000 | **ISONGraph C++ (exact)** | 406 | 415 | 420 | 2.48 (2.48-2.48) | - | - | 1.000 | 281 |
| 1,000,000 | **ISONGraph C# (exact)** | 126 | 370 | 399 | 5.81 (5.81-5.81) | - | - | 1.000 | 123 |
| 1,000,000 | **ISONGraph Python (exact)** | 37.1 | 42.5 | 47.8 | 26.5 (26.2-27.9) | 35.6 ms (27.7 QPS) | - | 1.000 | 4900 |
| 1,000,000 | **ISONGraph Rust (exact)** | 22.5 | 24.6 | 25.9 | 43.5 (42.8-48.6) | 13.3 ms (70.9 QPS) | - | 1.000 | 664 |
| 1,000,000 | **ISONGraph Rust, target-cpu=native (exact)** | 25.7 | 26.9 | 28.1 | 40.1 (38.9-40.2) | 12.8 ms (68.8 QPS) | - | 1.000 | 658 |
| 1,000,000 | faiss IndexFlatIP | 24.2 | 27.1 | 33.2 | 40.7 (39.8-42.6) | 20.4 ms (49.7 QPS) | 639 | 1.000 | 28.8 |
| 1,000,000 | lancedb (no index) | 346 | 385 | 393 | 2.89 (2.89-2.89) | 105 ms (9.45 QPS) | - | 1.000 | 157 |
| 1,000,000 | numpy brute force | 28.8 | 33.2 | 36.0 | 34.4 (33.3-36.6) | 24.7 ms (39.4 QPS) | 69.2 | 1.000 | 26.9 |

#### sift

| N | system | median ms (1T) | p95 | p99 | QPS 1T (min-max) | all-threads single query | batch QPS | recall | first query ms |
|---|---|---|---|---|---|---|---|---|---|
| 100,000 | **ISONGraph C++, /arch:AVX2 (exact)** | 42.3 | 45.4 | 47.4 | 23.4 (23.3-27.3) | - | - | 1.000 | 29.6 |
| 100,000 | **ISONGraph C++ (exact)** | 44.2 | 47.5 | 49.5 | 22.8 (22.4-25.2) | - | - | 1.000 | 31.6 |
| 100,000 | **ISONGraph C# (exact)** | 14.6 | 42.8 | 80.8 | 55.0 (55.0-72.9) | - | - | 1.000 | 22.9 |
| 100,000 | **ISONGraph Python (exact)** | 3.22 | 4.42 | 5.68 | 318 (265-323) | 3.99 ms (222 QPS) | - | 1.000 | 218 |
| 100,000 | **ISONGraph JS (exact)** | 33.7 | 41.6 | 48.7 | 29.0 (27.6-32.5) | - | - | 1.000 | 28.8 |
| 100,000 | **ISONGraph Rust (exact)** | 2.86 | 3.36 | 3.84 | 344 (333-398) | 2.86 ms (336 QPS) | - | 1.000 | 50.6 |
| 100,000 | **ISONGraph Rust, target-cpu=native (exact)** | 2.62 | 3.14 | 3.42 | 364 (362-465) | 2.88 ms (328 QPS) | - | 1.000 | 58.6 |
| 100,000 | faiss IndexFlatIP | 3.30 | 4.26 | 6.04 | 283 (280-332) | 3.44 ms (285 QPS) | 4537 | 1.000 | 3.51 |
| 100,000 | lancedb (no index) | 66.9 | 78.5 | 82.0 | 15.1 (13.7-16.0) | 46.0 ms (21.1 QPS) | - | 1.000 | 33.7 |
| 100,000 | numpy brute force | 3.34 | 4.87 | 6.69 | 282 (267-297) | 2.15 ms (428 QPS) | 1543 | 1.000 | 3.02 |
| 100,000 | sqlite-vec (vec0, exact) | 47.0 | 52.6 | 57.7 | 21.6 (21.0-22.4) | 89.9 ms (11.0 QPS) | - | 1.000 | 32.2 |
| 1,000,000 | **ISONGraph C++ (exact)** | 459 | 472 | 474 | 2.19 (2.19-2.19) | - | - | 1.000 | 330 |
| 1,000,000 | **ISONGraph Rust (exact)** | 30.3 | 31.9 | 32.3 | 33.4 (32.8-33.9) | 16.2 ms (61.3 QPS) | - | 0.999 | 841 |
| 1,000,000 | numpy brute force | 37.4 | 41.8 | 46.8 | 26.9 (26.7-28.1) | 32.0 ms (31.0 QPS) | 67.4 | 0.998 | 29.7 |



### Streaming mutation (insert 10k / update 10k / delete 10k on the built index)

Batches of 1,000 through each system's own API. ISONGraph rows include index maintenance; numpy/faiss Flat have no index to maintain. `recall before` is 0.000 where no sweep point reached 0.95 (the highest nprobe was used).

| dataset | N | system | insert 10k | update 10k | delete 10k | recall@10 before | after | first query after ms |
|---|---|---|---|---|---|---|---|---|
| glove | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 0.012 s (851,332/s) | 0.006 s (1,540,002/s) | 0.004 s (2,662,903/s) | 0.974 | 0.973 | 19.9 |
| glove | 100,000 | **ISONGraph C++ (IVF)** | 0.013 s (797,925/s) | 0.007 s (1,488,427/s) | 0.004 s (2,524,870/s) | 0.974 | 0.973 | 22.0 |
| glove | 100,000 | **ISONGraph C# (IVF)** | 0.049 s (202,219/s) | 0.008 s (1,294,080/s) | 0.003 s (2,900,316/s) | 0.973 | 0.972 | 33.1 |
| glove | 100,000 | **ISONGraph Python (exact)** | 0.130 s (77055/s) | 0.126 s (79264/s) | 0.116 s (86540/s) | 1.00 | 1.000 | 3.41 |
| glove | 100,000 | **ISONGraph Python (IVF)** | 0.137 s (73067/s) | 0.140 s (71216/s) | 0.118 s (84936/s) | 0.973 | 0.972 | 78.5 |
| glove | 100,000 | **ISONGraph JS (IVF)** | 0.018 s (550,973/s) | 0.009 s (1,170,412/s) | 0.006 s (1,739,615/s) | 0.975 | 0.974 | 1174 |
| glove | 100,000 | **ISONGraph Rust (IVF)** | 0.015 s (689,579/s) | 0.014 s (702,020/s) | 0.009 s (1,127,218/s) | 0.974 | 0.974 | 83.7 |
| glove | 100,000 | **ISONGraph Rust, target-cpu=native (IVF)** | 0.014 s (708,567/s) | 0.014 s (732,220/s) | 0.009 s (1,092,383/s) | 0.974 | 0.974 | 83.6 |
| glove | 100,000 | chromadb (HNSW) | 2.42 s (4125/s) | 3.46 s (2892/s) | 1.95 s (5121/s) | 0.977 | 0.976 | 89.0 |
| glove | 100,000 | faiss IndexFlatIP | 0.002 s (4,972,403/s) | 0.102 s (97716/s) | 0.099 s (101,166/s) | 1.00 | 1.000 | 2.72 |
| glove | 100,000 | faiss HNSWFlat | 0.989 s (10112/s) | unsupported | unsupported | 0.970 | n/a | - |
| glove | 100,000 | faiss IVFFlat | 0.015 s (668,717/s) | 0.017 s (600,017/s) | 0.002 s (5,319,149/s) | 0.974 | 0.974 | 1.32 |
| glove | 100,000 | hnswlib | 0.226 s (44217/s) | 0.934 s (10703/s) | 0.008 s (1,242,329/s) | 0.969 | 0.969 | 0.644 |
| glove | 100,000 | lancedb (no index) | 0.029 s (349,167/s) | 0.134 s (74457/s) | 0.108 s (92786/s) | 1.00 | 1.000 | 20.8 |
| glove | 100,000 | lancedb IVF_FLAT | 0.031 s (319,488/s) | 0.138 s (72668/s) | 0.103 s (97019/s) | 0.974 | 0.977 | 9.27 |
| glove | 100,000 | numpy brute force | 0.001 s (8,836,264/s) | 0.006 s (1,624,643/s) | 0.002 s (4,615,314/s) | 1.00 | 1.000 | 2.93 |
| glove | 100,000 | sqlite-vec (vec0, exact) | 0.080 s (124,613/s) | 2.47 s (4042/s) | 2.07 s (4838/s) | 1.00 | 1.000 | 28.8 |
| glove | 100,000 | usearch (HNSW) | 0.261 s (38339/s) | 0.281 s (35606/s) | 0.003 s (3,713,882/s) | 0.967 | 0.964 | 3.67 |
| glove | 1,000,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 0.159 s (63000/s) | 0.018 s (571,167/s) | 0.013 s (783,687/s) | 0.953 | 0.951 | 86.5 |
| glove | 1,000,000 | **ISONGraph C++ (IVF)** | 0.155 s (64395/s) | 0.016 s (620,409/s) | 0.012 s (803,006/s) | 0.951 | 0.951 | 92.5 |
| glove | 1,000,000 | **ISONGraph C# (IVF)** | 0.029 s (350,733/s) | 0.016 s (616,922/s) | 0.039 s (258,405/s) | 0.000 | 0.949 | 136 |
| glove | 1,000,000 | **ISONGraph Python (exact)** | 0.881 s (11357/s) | 0.602 s (16617/s) | 0.509 s (19659/s) | 1.00 | 1.000 | 100 |
| glove | 1,000,000 | **ISONGraph Python (IVF)** | 0.167 s (59736/s) | 0.156 s (64253/s) | 0.148 s (67731/s) | 0.957 | 0.957 | 294 |
| glove | 1,000,000 | **ISONGraph Rust (IVF)** | 0.156 s (64014/s) | 0.042 s (238,538/s) | 0.033 s (303,255/s) | 0.954 | 0.954 | 746 |
| glove | 1,000,000 | **ISONGraph Rust, target-cpu=native (IVF)** | 0.164 s (61106/s) | 0.039 s (253,556/s) | 0.034 s (297,217/s) | 0.954 | 0.953 | 750 |
| glove | 1,000,000 | faiss IndexFlatIP | 0.004 s (2,325,527/s) | 3.94 s (2536/s) | 4.19 s (2387/s) | 1.00 | 1.000 | 65.7 |
| glove | 1,000,000 | faiss HNSWFlat | 1.79 s (5594/s) | unsupported | unsupported | 0.935 | n/a | - |
| glove | 1,000,000 | faiss IVFFlat | 0.044 s (225,461/s) | 0.083 s (119,972/s) | 0.005 s (2,143,898/s) | 1.00 | 1.000 | 31.1 |
| glove | 1,000,000 | hnswlib | 1.07 s (9386/s) | 2.66 s (3766/s) | 0.015 s (671,768/s) | 0.937 | 0.935 | 2.27 |
| glove | 1,000,000 | lancedb (no index) | 0.070 s (141,898/s) | 1.50 s (6686/s) | 1.40 s (7153/s) | 1.00 | 1.000 | 273 |
| glove | 1,000,000 | lancedb IVF_FLAT | 0.105 s (95045/s) | 1.29 s (7723/s) | 1.27 s (7897/s) | 0.951 | 0.939 | 41.9 |
| glove | 1,000,000 | numpy brute force | 0.004 s (2,843,494/s) | 0.151 s (66231/s) | 0.024 s (419,856/s) | 1.00 | 1.000 | 68.9 |
| glove | 1,000,000 | usearch (HNSW) | 1.37 s (7313/s) | 1.49 s (6723/s) | 0.004 s (2,399,635/s) | 0.939 | 0.939 | 25.1 |
| sift | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 0.030 s (333,745/s) | 0.011 s (876,263/s) | 0.006 s (1,582,804/s) | 0.952 | 0.949 | 30.4 |
| sift | 100,000 | **ISONGraph C++ (IVF)** | 0.025 s (400,699/s) | 0.012 s (835,939/s) | 0.007 s (1,374,004/s) | 0.952 | 0.949 | 37.1 |
| sift | 100,000 | **ISONGraph C# (IVF)** | 0.030 s (332,383/s) | 0.015 s (655,476/s) | 0.009 s (1,169,946/s) | 0.953 | 0.949 | 51.8 |
| sift | 100,000 | **ISONGraph Python (exact)** | 0.810 s (12346/s) | 0.676 s (14783/s) | 0.426 s (23490/s) | 1.00 | 1.000 | 10.1 |
| sift | 100,000 | **ISONGraph Python (IVF)** | 0.151 s (66209/s) | 0.157 s (63624/s) | 0.135 s (74040/s) | 0.987 | 0.986 | 96.4 |
| sift | 100,000 | **ISONGraph JS (IVF)** | 0.044 s (225,056/s) | 0.012 s (845,101/s) | 0.008 s (1,271,407/s) | 0.988 | 0.987 | 2043 |
| sift | 100,000 | **ISONGraph Rust (IVF)** | 0.039 s (253,924/s) | 0.033 s (304,480/s) | 0.022 s (463,420/s) | 0.986 | 0.985 | 261 |
| sift | 100,000 | **ISONGraph Rust, target-cpu=native (IVF)** | 0.028 s (354,949/s) | 0.029 s (342,604/s) | 0.020 s (506,183/s) | 0.986 | 0.985 | 277 |
| sift | 100,000 | chromadb (HNSW) | 2.33 s (4284/s) | 3.34 s (2991/s) | 2.33 s (4296/s) | 0.952 | 0.942 | 111 |
| sift | 100,000 | faiss IndexFlatIP | 0.005 s (2,211,215/s) | 0.520 s (19235/s) | 0.361 s (27688/s) | 1.00 | 1.000 | 7.96 |
| sift | 100,000 | faiss HNSWFlat | 0.995 s (10053/s) | unsupported | unsupported | 0.953 | n/a | - |
| sift | 100,000 | faiss IVFFlat | 0.035 s (284,106/s) | 0.063 s (157,573/s) | 0.004 s (2,790,957/s) | 0.950 | 0.947 | 0.438 |
| sift | 100,000 | hnswlib | 0.238 s (42037/s) | 0.570 s (17557/s) | 0.008 s (1,250,750/s) | 0.950 | 0.941 | 0.101 |
| sift | 100,000 | lancedb (no index) | 0.120 s (83312/s) | 0.366 s (27354/s) | 0.249 s (40241/s) | 1.00 | 1.000 | 77.4 |
| sift | 100,000 | lancedb IVF_FLAT | 0.040 s (252,179/s) | 0.149 s (67089/s) | 0.120 s (83050/s) | 0.987 | 0.989 | 12.1 |
| sift | 100,000 | numpy brute force | 0.002 s (6,123,324/s) | 0.006 s (1,562,402/s) | 0.003 s (3,611,412/s) | 1.00 | 1.000 | 3.07 |
| sift | 100,000 | sqlite-vec (vec0, exact) | 0.307 s (32571/s) | 11.0 s (912/s) | 8.30 s (1205/s) | 1.00 | 1.000 | 109 |
| sift | 100,000 | usearch (HNSW) | 0.216 s (46402/s) | 0.258 s (38686/s) | 0.003 s (3,433,948/s) | 0.952 | 0.945 | 2.84 |
| text384 | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 0.038 s (262,907/s) | 0.014 s (708,602/s) | 0.005 s (2,130,243/s) | 0.954 | 0.951 | 43.8 |
| text384 | 100,000 | **ISONGraph C++ (IVF)** | 0.038 s (265,111/s) | 0.013 s (764,427/s) | 0.004 s (2,329,482/s) | 0.954 | 0.950 | 64.3 |
| text384 | 100,000 | **ISONGraph C# (IVF)** | 0.026 s (386,698/s) | 0.009 s (1,148,805/s) | 0.004 s (2,422,891/s) | 0.952 | 0.949 | 44.8 |
| text384 | 100,000 | **ISONGraph Python (exact)** | 0.206 s (48551/s) | 0.195 s (51276/s) | 0.128 s (78119/s) | 1.000 | 1.000 | 9.15 |
| text384 | 100,000 | **ISONGraph Python (IVF)** | 0.211 s (47387/s) | 0.204 s (49074/s) | 0.126 s (79668/s) | 0.954 | 0.948 | 203 |
| text384 | 100,000 | **ISONGraph JS (IVF)** | 0.063 s (158,336/s) | 0.014 s (729,001/s) | 0.008 s (1,216,382/s) | 0.953 | 0.952 | 3284 |
| text384 | 100,000 | **ISONGraph Rust (IVF)** | 0.044 s (228,710/s) | 0.017 s (579,979/s) | 0.010 s (1,037,086/s) | 0.953 | 0.949 | 192 |
| text384 | 100,000 | **ISONGraph Rust, target-cpu=native (IVF)** | 0.039 s (256,465/s) | 0.016 s (606,160/s) | 0.010 s (1,032,983/s) | 0.953 | 0.949 | 192 |
| text384 | 100,000 | chromadb (HNSW) | 3.49 s (2866/s) | 5.23 s (1912/s) | 1.98 s (5063/s) | 0.965 | 0.965 | 171 |
| text384 | 100,000 | faiss IndexFlatIP | 0.005 s (1,992,310/s) | 0.225 s (44475/s) | 0.196 s (51042/s) | 1.000 | 1.000 | 9.09 |
| text384 | 100,000 | faiss HNSWFlat | 2.84 s (3524/s) | unsupported | unsupported | 0.957 | n/a | - |
| text384 | 100,000 | faiss IVFFlat | 0.049 s (202,182/s) | 0.047 s (211,541/s) | 0.002 s (4,849,425/s) | 0.955 | 0.951 | 1.23 |
| text384 | 100,000 | hnswlib | 1.11 s (9049/s) | 2.97 s (3372/s) | 0.011 s (876,186/s) | 0.964 | 0.968 | 0.687 |
| text384 | 100,000 | lancedb (no index) | 0.053 s (190,268/s) | 0.165 s (60618/s) | 0.111 s (89958/s) | 1.00 | 1.000 | 103 |
| text384 | 100,000 | lancedb IVF_FLAT | 0.056 s (178,739/s) | 0.207 s (48245/s) | 0.135 s (74346/s) | 0.956 | 0.969 | 15.5 |
| text384 | 100,000 | numpy brute force | 0.003 s (3,145,346/s) | 0.007 s (1,341,526/s) | 0.002 s (4,841,208/s) | 1.000 | 1.000 | 7.10 |
| text384 | 100,000 | qdrant-client local mode (exact numpy) | 25.7 s (390/s) | 26.4 s (379/s) | 31.8 s (315/s) | 1.00 | 1.000 | 136 |
| text384 | 100,000 | sqlite-vec (vec0, exact) | 0.170 s (58976/s) | 9.85 s (1016/s) | 7.37 s (1356/s) | 1.00 | 1.000 | 110 |
| text384 | 100,000 | usearch (HNSW) | 1.18 s (8452/s) | 1.19 s (8434/s) | 0.003 s (3,381,920/s) | 0.964 | 0.958 | 3.58 |
| text384 | 1,000,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 0.288 s (34735/s) | 0.016 s (615,483/s) | 0.007 s (1,382,361/s) | 0.955 | 0.956 | 81.2 |
| text384 | 1,000,000 | **ISONGraph C++ (IVF)** | 0.292 s (34211/s) | 0.018 s (569,256/s) | 0.007 s (1,378,607/s) | 0.953 | 0.953 | 149 |
| text384 | 1,000,000 | **ISONGraph C# (IVF)** | 0.015 s (677,952/s) | 0.009 s (1,124,227/s) | 0.006 s (1,552,650/s) | 0.000 | 0.946 | 111 |
| text384 | 1,000,000 | **ISONGraph Python (exact)** | 0.284 s (35183/s) | 0.212 s (47127/s) | 0.148 s (67540/s) | 1.00 | 1.000 | 76.1 |
| text384 | 1,000,000 | **ISONGraph Python (IVF)** | 0.269 s (37230/s) | 0.228 s (43813/s) | 0.163 s (61533/s) | 0.953 | 0.954 | 665 |
| text384 | 1,000,000 | **ISONGraph JS (IVF)** | 0.017 s (589,240/s) | 0.014 s (703,408/s) | 0.011 s (935,559/s) | 0.000 | 0.946 | 5743 |
| text384 | 1,000,000 | **ISONGraph Rust (IVF)** | 0.283 s (35306/s) | 0.023 s (428,754/s) | 0.017 s (591,195/s) | 0.951 | 0.952 | 611 |
| text384 | 1,000,000 | **ISONGraph Rust, target-cpu=native (IVF)** | 0.277 s (36069/s) | 0.021 s (479,409/s) | 0.015 s (645,561/s) | 0.952 | 0.954 | 602 |
| text384 | 1,000,000 | chromadb (HNSW) | 12.7 s (789/s) | 14.4 s (694/s) | 10.7 s (937/s) | 0.967 | 0.968 | 2631 |
| text384 | 1,000,000 | faiss IndexFlatIP | 0.007 s (1,357,958/s) | 2.31 s (4331/s) | 2.25 s (4445/s) | 1.00 | 1.000 | 72.1 |
| text384 | 1,000,000 | faiss HNSWFlat | 4.32 s (2315/s) | unsupported | unsupported | 0.965 | n/a | - |
| text384 | 1,000,000 | faiss IVFFlat | 0.125 s (79926/s) | 0.139 s (71689/s) | 0.005 s (2,116,626/s) | 0.956 | 0.957 | 8.73 |
| text384 | 1,000,000 | hnswlib | 2.09 s (4774/s) | 3.09 s (3236/s) | 0.009 s (1,138,771/s) | 0.957 | 0.956 | 0.775 |
| text384 | 1,000,000 | lancedb (no index) | 0.056 s (179,892/s) | 0.635 s (15755/s) | 0.358 s (27946/s) | 1.00 | 1.000 | 940 |
| text384 | 1,000,000 | lancedb IVF_FLAT | 0.055 s (181,022/s) | 0.603 s (16597/s) | 0.356 s (28076/s) | 1.00 | 1.000 | 108 |
| text384 | 1,000,000 | numpy brute force | 0.004 s (2,584,647/s) | 0.087 s (115,119/s) | 0.015 s (670,628/s) | 1.00 | 1.000 | 73.8 |
| text384 | 1,000,000 | sqlite-vec (vec0, exact) | 0.194 s (51559/s) | 10.5 s (949/s) | 8.07 s (1240/s) | 1.00 | 1.000 | 958 |
| text384 | 1,000,000 | usearch (HNSW) | 2.25 s (4440/s) | 2.37 s (4225/s) | 0.005 s (2,170,940/s) | 0.957 | 0.956 | 23.9 |


### Filtered search (random 10% and 1% subsets)

#### glove, N = 100,000

| system | build with metadata s | 10% subset: QPS 1T @ recall>=0.90 | max recall | 1% subset: QPS 1T @ recall>=0.90 | max recall |
|---|---|---|---|---|---|
| **ISONGraph C++ (IVF)** | 0.354 | 152 @ 1.000 | 1.000 | 316 @ 1.000 | 1.000 |
| **ISONGraph Python (exact)** | 0.952 | 368 @ 1.000 | 1.000 | 409 @ 1.000 | 1.000 |
| **ISONGraph Python (IVF)** | 1.78 | 366 @ 1.000 | 1.000 | 400 @ 1.000 | 1.000 |
| **ISONGraph Rust (IVF)** | 2.13 | 8377 @ 1.000 | 1.000 | 18699 @ 1.000 | 1.000 |
| chromadb (HNSW) | 11.1 | 4.23 @ 0.912 | 1.000 | 28.0 @ 0.936 | 1.000 |
| faiss IndexFlatIP | 0.036 | 1813 @ 1.000 | 1.000 | 2913 @ 1.000 | 1.000 |
| faiss HNSWFlat | 3.85 | 534 @ 0.957 | 0.957 | - (max recall 0.835 at 541 QPS) | 0.835 |
| faiss IVFFlat | 1.03 | 1083 @ 1.000 | 1.000 | 1910 @ 1.000 | 1.000 |
| hnswlib | 1.31 | 1354 @ 0.935 | 1.000 | 449 @ 0.951 | 1.000 |
| lancedb (no index) | 0.081 | 33.8 @ 1.000 | 1.000 | 37.6 @ 1.000 | 1.000 |
| lancedb IVF_FLAT | 0.740 | 88.3 @ 1.000 | 1.000 | 99.0 @ 1.000 | 1.000 |
| numpy brute force | 0.012 | 552 @ 1.000 | 1.000 | 624 @ 1.000 | 1.000 |
| sqlite-vec (vec0, exact) | 0.579 | 46.1 @ 1.000 | 1.000 | 48.0 @ 1.000 | 1.000 |
| usearch (HNSW) | 1.57 | unsupported | - | unsupported | - |

#### text384, N = 100,000

| system | build with metadata s | 10% subset: QPS 1T @ recall>=0.90 | max recall | 1% subset: QPS 1T @ recall>=0.90 | max recall |
|---|---|---|---|---|---|
| **ISONGraph C++ (IVF)** | 0.986 | 116 @ 1.000 | 1.000 | 345 @ 1.000 | 1.000 |
| **ISONGraph Python (exact)** | 1.97 | 121 @ 1.000 | 1.000 | 131 @ 1.000 | 1.000 |
| **ISONGraph Python (IVF)** | 3.62 | 122 @ 1.000 | 1.000 | 130 @ 1.000 | 1.000 |
| **ISONGraph Rust (IVF)** | 4.77 | 3094 @ 1.000 | 1.000 | 15288 @ 1.000 | 1.000 |
| chromadb (HNSW) | 16.5 | 2.24 @ 0.953 | 1.000 | 20.9 @ 0.971 | 1.000 |
| faiss IndexFlatIP | 0.109 | 464 @ 1.000 | 1.000 | 2491 @ 1.000 | 1.000 |
| faiss HNSWFlat | 13.7 | 1279 @ 0.944 | 0.992 | 377 @ 0.928 | 0.928 |
| faiss IVFFlat | 1.69 | 388 @ 1.000 | 1.000 | 1630 @ 1.000 | 1.000 |
| hnswlib | 6.44 | 1779 @ 0.912 | 1.000 | 259 @ 0.981 | 1.000 |
| lancedb (no index) | 0.217 | 7.09 @ 1.000 | 1.000 | 7.28 @ 1.000 | 1.000 |
| lancedb IVF_FLAT | 1.94 | 73.1 @ 1.000 | 1.000 | 97.8 @ 1.000 | 1.000 |
| numpy brute force | 0.030 | 141 @ 1.000 | 1.000 | 146 @ 1.000 | 1.000 |
| sqlite-vec (vec0, exact) | 1.01 | 13.0 @ 1.000 | 1.000 | 13.3 @ 1.000 | 1.000 |
| usearch (HNSW) | 5.68 | unsupported | - | unsupported | - |

#### text384, N = 1,000,000

| system | build with metadata s | 10% subset: QPS 1T @ recall>=0.90 | max recall | 1% subset: QPS 1T @ recall>=0.90 | max recall |
|---|---|---|---|---|---|
| **ISONGraph C++ (IVF)** | 12.6 | 308 @ 0.943 | 1.000 | 506 @ 0.905 | 1.000 |
| **ISONGraph Python (IVF)** | 117 | 8.64 @ 1.000 | 1.000 | 9.52 @ 1.000 | 1.000 |
| **ISONGraph Rust (IVF)** | 69.5 | 650 @ 0.940 | 1.000 | 1944 @ 0.906 | 1.000 |
| faiss IndexFlatIP | 1.13 | 39.7 @ 1.000 | 1.000 | 172 @ 1.000 | 1.000 |
| faiss HNSWFlat | 268 | 267 @ 0.973 | 0.973 | - (max recall 0.854 at 275 QPS) | 0.854 |
| faiss IVFFlat | 47.0 | 38.1 @ 1.000 | 1.000 | 140 @ 1.000 | 1.000 |
| hnswlib | 151 | 406 @ 0.955 | 0.999 | 135 @ 0.945 | 1.000 |
| lancedb IVF_FLAT | 25.3 | 8.73 @ 1.000 | 1.000 | 13.1 @ 1.000 | 1.000 |
| numpy brute force | 0.300 | 12.9 @ 1.000 | 1.000 | 14.0 @ 1.000 | 1.000 |



### Save / load and on-disk size

| dataset | N | system | save s | on-disk MB | load s | first query after load ms | time to ready s |
|---|---|---|---|---|---|---|---|
| glove | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 4.26 | 133 | 2.02 | 236 | 2.25 |
| glove | 100,000 | **ISONGraph C++ (IVF)** | 4.40 | 133 | 2.03 | 302 | 2.33 |
| glove | 100,000 | **ISONGraph C# (IVF)** | 1.36 | 121 | 1.27 | 333 | 1.61 |
| glove | 100,000 | **ISONGraph Python (exact)** | 4.35 | 215 | 3.37 | 173 | 3.54 |
| glove | 100,000 | **ISONGraph Python (IVF)** | 4.42 | 215 | 3.41 | 2017 | 5.43 |
| glove | 100,000 | **ISONGraph Rust (IVF)** | 0.754 | 121 | 0.414 | 237 | 0.651 |
| glove | 100,000 | chromadb (HNSW) | 0.000 | 76.3 | 0.012 | 87.3 | 0.099 |
| glove | 100,000 | faiss IndexFlatIP | 0.045 | 38.9 | 0.025 | 2.13 | 0.027 |
| glove | 100,000 | faiss HNSWFlat | 0.052 | 57.1 | 0.029 | 1.18 | 0.031 |
| glove | 100,000 | faiss IVFFlat | 0.020 | 39.4 | 0.027 | 0.713 | 0.028 |
| glove | 100,000 | hnswlib | 0.036 | 57.5 | 0.142 | 0.708 | 0.142 |
| glove | 100,000 | lancedb (no index) | 0.000 | 46.8 | 0.002 | 16.9 | 0.018 |
| glove | 100,000 | lancedb IVF_FLAT | 0.000 | 86.1 | 0.002 | 26.8 | 0.028 |
| glove | 100,000 | numpy brute force | 0.025 | 42.8 | 0.017 | 2.38 | 0.020 |
| glove | 100,000 | sqlite-vec (vec0, exact) | 0.000 | 48.9 | 0.000 | 30.7 | 0.031 |
| glove | 100,000 | usearch (HNSW) | 0.097 | 57.5 | 0.052 | 1.56 | 0.053 |
| glove | 1,000,000 | **ISONGraph Rust (IVF)** | 23.1 | 1216 | 10.8 | 7592 | 18.3 |
| glove | 1,000,000 | faiss IndexFlatIP | 0.655 | 389 | 0.973 | 66.8 | 1.04 |
| glove | 1,000,000 | faiss HNSWFlat | 0.307 | 524 | 0.228 | 2.15 | 0.230 |
| glove | 1,000,000 | faiss IVFFlat | 0.149 | 391 | 0.246 | 23.8 | 0.270 |
| glove | 1,000,000 | hnswlib | 0.777 | 528 | 3.66 | 3.27 | 3.66 |
| glove | 1,000,000 | usearch (HNSW) | 0.585 | 528 | 0.499 | 3.37 | 0.503 |
| sift | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 12.2 | 134 | 5.14 | 636 | 5.78 |
| sift | 100,000 | **ISONGraph C++ (IVF)** | 12.4 | 134 | 4.97 | 826 | 5.80 |
| sift | 100,000 | **ISONGraph C# (IVF)** | 3.51 | 122 | 3.06 | 1133 | 4.20 |
| sift | 100,000 | **ISONGraph Python (exact)** | 11.0 | 222 | 9.70 | 528 | 10.2 |
| sift | 100,000 | **ISONGraph Python (IVF)** | 4.97 | 222 | 3.66 | 2380 | 6.04 |
| sift | 100,000 | **ISONGraph Rust (IVF)** | 1.67 | 128 | 0.904 | 853 | 1.76 |
| sift | 100,000 | chromadb (HNSW) | 0.000 | 88.5 | 0.015 | 108 | 0.123 |
| sift | 100,000 | faiss IndexFlatIP | 0.060 | 49.6 | 0.087 | 12.9 | 0.100 |
| sift | 100,000 | faiss HNSWFlat | 0.044 | 68.9 | 0.039 | 0.340 | 0.039 |
| sift | 100,000 | faiss IVFFlat | 0.025 | 50.2 | 0.035 | 0.216 | 0.036 |
| sift | 100,000 | hnswlib | 0.067 | 69.3 | 0.158 | 0.214 | 0.158 |
| sift | 100,000 | lancedb (no index) | 0.000 | 59.6 | 0.005 | 74.0 | 0.079 |
| sift | 100,000 | lancedb IVF_FLAT | 0.000 | 110 | 0.008 | 18.6 | 0.027 |
| sift | 100,000 | numpy brute force | 0.031 | 54.6 | 0.026 | 4.71 | 0.030 |
| sift | 100,000 | sqlite-vec (vec0, exact) | 0.001 | 61.8 | 0.001 | 105 | 0.106 |
| sift | 100,000 | usearch (HNSW) | 0.080 | 69.3 | 0.061 | 0.200 | 0.062 |
| sift | 1,000,000 | **ISONGraph Rust (IVF)** | 22.7 | 1285 | 11.2 | 8390 | 19.6 |
| sift | 1,000,000 | faiss HNSWFlat | 1.12 | 626 | 1.10 | 0.567 | 1.10 |
| sift | 1,000,000 | faiss IVFFlat | 0.353 | 498 | 0.723 | 1.63 | 0.725 |
| sift | 1,000,000 | hnswlib | 1.39 | 630 | 4.90 | 0.936 | 4.90 |
| text384 | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 16.4 | 500 | 6.95 | 531 | 7.48 |
| text384 | 100,000 | **ISONGraph C++ (IVF)** | 16.5 | 500 | 7.34 | 903 | 8.24 |
| text384 | 100,000 | **ISONGraph C# (IVF)** | 4.41 | 453 | 3.95 | 911 | 4.86 |
| text384 | 100,000 | **ISONGraph Python (exact)** | 15.5 | 807 | 10.8 | 291 | 11.1 |
| text384 | 100,000 | **ISONGraph Python (IVF)** | 16.0 | 807 | 10.9 | 5129 | 16.0 |
| text384 | 100,000 | **ISONGraph Rust (IVF)** | 1.94 | 453 | 1.25 | 546 | 1.80 |
| text384 | 100,000 | chromadb (HNSW) | 0.000 | 202 | 0.012 | 175 | 0.187 |
| text384 | 100,000 | faiss IndexFlatIP | 0.128 | 147 | 0.069 | 7.27 | 0.076 |
| text384 | 100,000 | faiss HNSWFlat | 0.134 | 176 | 0.074 | 0.496 | 0.075 |
| text384 | 100,000 | faiss IVFFlat | 0.055 | 149 | 0.091 | 1.29 | 0.092 |
| text384 | 100,000 | hnswlib | 0.147 | 177 | 0.225 | 0.615 | 0.225 |
| text384 | 100,000 | lancedb (no index) | 0.000 | 177 | 0.003 | 110 | 0.113 |
| text384 | 100,000 | lancedb IVF_FLAT | 0.000 | 326 | 0.009 | 23.8 | 0.033 |
| text384 | 100,000 | numpy brute force | 0.110 | 162 | 0.068 | 9.16 | 0.077 |
| text384 | 100,000 | sqlite-vec (vec0, exact) | 0.000 | 180 | 0.000 | 109 | 0.110 |
| text384 | 100,000 | usearch (HNSW) | 0.178 | 177 | 0.136 | 0.949 | 0.137 |
| text384 | 1,000,000 | **ISONGraph Rust (IVF)** | 21.5 | 4534 | 13.0 | 6601 | 19.6 |
| text384 | 1,000,000 | faiss IndexFlatIP | 0.968 | 1472 | 0.736 | 74.1 | 0.811 |
| text384 | 1,000,000 | faiss HNSWFlat | 0.994 | 1618 | 0.805 | 1.21 | 0.806 |
| text384 | 1,000,000 | faiss IVFFlat | 0.417 | 1478 | 0.670 | 6.22 | 0.676 |
| text384 | 1,000,000 | hnswlib | 1.07 | 1623 | 2.12 | 1.58 | 2.12 |
| text384 | 1,000,000 | usearch (HNSW) | 1.54 | 1623 | 1.30 | 1.72 | 1.30 |


### Jobs that failed or timed out

None in the final run.


## Library issues found (described, not fixed)

1. **C++ exact search has no scan cache and is scalar** (`isongraph-vector-cpp/include/isongraph_vector.hpp`, `similaritySearchVector`, the loop over `embeddings_` that calls `cosineSimilarity` at ~line 1246/1672). It iterates an `unordered_map` of records, and for each one runs a scalar loop with three dependent accumulators and recomputes both vector norms. Measured: 22.6 QPS at 100k x 384 (numpy 144, Rust 154), 2.1 QPS at 1M. `nprobe >= nlist` falls into the same path, which is why the last point of every C++ IVF sweep collapses to exact speed. Suggested fix: reuse the contiguous vector copy the IVF index already keeps (or build one lazily like the Rust `ScanCache`), cache inverse norms, and score with the existing `simd::dot` kernels.
2. **No batched or concurrent search in any port.** Rust `search_vector` (`isongraph-vector-rs/src/ann.rs` ~line 1722) holds the `scan` mutex for the whole scan; C++ takes `ann_.mutex`; C# `lock (_lock)`; the Python store holds an RLock. The `concurrent_clients` probe in the Rust/C++ JSON shows N client threads do not scale. When a large probe is spread over threads, Rust spawns fresh threads per query (`std::thread::scope` in `scan_probed`). Suggested fix: search against an immutable snapshot (`Arc<ScanCache>` / read lock) and add `search_batch(&[Query])` backed by a persistent pool; this alone is worth 10-50x on the all-core column.
3. **First search after writes pays a lazy rebuild/fold.** After 10k inserts + 10k updates + 10k deletes on a 1M text index the first query took 0.6 s (Rust), 0.15 s (C++), 0.11 s (C#), 0.67 s (Python), 5.7 s (JS) versus ~1 ms for hnswlib/usearch (`first_query_after_ms` in the mutation table). Suggested fix: fold pending rows incrementally at write time or on a background thread.
4. **JS IVF build is 522 s at 1M** (Rust 7.6 s, C++ 12 s default / 6 s AVX2, C# 11 s): single-threaded k-means over plain JS arrays of doubles; the JS port also needs ~7 GB for 1M x 384.
5. **Persistence is the JSON sidecar only; the IVF index is never persisted.** 100k x 384 = 0.45-0.8 GB, 1M = 4.5 GB; load then rebuilds the index. A binary vector file (and an index file) would close most of the 20-100x gap to faiss/hnswlib.
6. **Two copies of every vector** (record map + contiguous cache/index copy) give 2-4x the RSS of faiss/hnswlib at 1M.
7. Build flags matter for C++: the default bench build is SSE2; `/arch:AVX2` makes the IVF 1.1-1.4x faster (text 100k 681 -> 957 at 0.95). Rust `target-cpu=native` made no measurable difference to the IVF (828 vs 828) and +5% to the exact scan.

## What did not run, and why

* **Qdrant server (HNSW)**: not run. Docker Desktop's daemon was not running, and downloading and executing the official Windows binary was not permitted in this session. Only `qdrant-client` **local mode** ran, which is a pure-numpy brute-force scan (the client itself warns against more than 20,000 points). It was run at 10k and 100k only (100k ingest alone took 284 s) and is not a Qdrant performance number.
* **chromadb build repeats** after the first build failed ("Collection already exists": chromadb 1.5.9 shares one system object per process). The `3 builds` column for chromadb therefore has 1-2 samples at most; everything else for chromadb ran. chromadb ignores a changed `ef_search` until the index is reloaded, so the harness reloads the collection for every knob value (verified separately).
* **chromadb at 1M** ran for text384 only; **sqlite-vec at 1M** for text384 only (and 100k for all three datasets); **qdrant local** only text384 10k and 100k; **JS at 1M** text384 only (a glove/sift 1M JS run would add 30 min each).
* **SIFT 1M** ran for a reduced set only (ISONGraph Rust and C++, faiss HNSW, faiss IVFFlat, hnswlib, numpy brute force) because of the time budget; no streaming-mutation test is possible at SIFT 1M (the file has exactly 1,000,000 vectors, no spare rows for inserts). SIFT 100k ran for every system.
* **usearch filtered search**: the Python binding has no filter predicate: recorded as `unsupported`. hnswlib filtering is a Python callback per visited node, which is the only API it has.
* **faiss HNSW update/delete**: unsupported by that index type (recorded as `unsupported`; it would need a rebuild).
* **Streaming mutation for ISONGraph "exact" and numpy/faiss Flat** measures only cache patching, which is not an index; the like-for-like comparisons are the IVF/HNSW rows. hnswlib `delete` is `mark_deleted` (memory is not freed).
* **Save/load**: not run for ISONGraph at 1M in Python, C++, C#, JS (JSON of 4.5 GB at 1M; Python's `json.dumps` would need tens of GB). Rust did run it. JS has no save/load call for vectors.
* **Cross-process repeat of the pinned run** (second independent process per headline cell) was not done for the final data set because of the time budget; in-process 3 runs per point and the Rust spot-check are the only repeatability evidence (see above).
* **numpy BLAS** is the PyPI wheel (OpenBLAS, `MAX_THREADS=24`), so "all-core" numpy uses up to 24 threads, not 32.

## Unfairness that could not be removed

* **Language/runtime**: Python-callable libraries pay Python call overhead per query (20-100 us; invisible for ms-scale queries, material for the 0.3 ms HNSW queries); ISONGraph Python additionally builds result objects. Rust/C++ harnesses call the library directly.
* **SIMD**: faiss, numpy/OpenBLAS and usearch dispatch AVX2/AVX-512 at run time; hnswlib's wheel and our Rust/C++/C#/JS builds do not necessarily. We report both baseline and `target-cpu=native` (Rust) / `/arch:AVX2` (C++) builds. The 14900HX has no AVX-512.
* **Defaults vs tuning**: HNSW M=16, efConstruction=200 for all (M=32 would reach higher recall on GloVe, which is why HNSW rows are missing at glove 1M); IVF nlist = 4*sqrt(N) for all (not tuned per library). faiss IVF is trained on a 40-points-per-cell sample like ours but with its default 25 k-means iterations and non-spherical centroids; ISONGraph runs 8 spherical iterations. Nobody was tuned further than anyone else.
* **Knob grid**: nprobe grid 1,2,4,...,256 (plus nlist at 100k); the interpolation is log-linear between bracketing points, flagged `*` when the bracket is wider than 0.05 recall.
* **Hybrid CPU**: single-thread loops are pinned to one P-core, all-core runs use all 32 logical CPUs including E-cores; the process-wide affinity switch also affects any helper threads.
* **Memory** is process RSS: file-backed engines (lancedb, sqlite-vec, chromadb) look small because pages are cached by the OS or mapped, and chromadb/lancedb hold index data off-heap.
* **Filters**: ISONGraph supports node-type equality only; the others ran an integer metadata column. faiss was given an `IDSelectorBitmap` (the cheapest selector) after an earlier run with `IDSelectorBatch`; hnswlib uses a Python callback.
* **Ground truth ties**: SIFT has many near-tied float32 scores; faiss/numpy score 0.998-0.999 recall on exact search against the float64 ground truth for that reason.
* **Datasets**: text384 is a random sample of Wikipedia abstracts embedded with MiniLM (fp16, max 128 tokens) and 2,000 HotpotQA questions that are mostly not answerable from that sample, so query-to-neighbour similarity is lower than for in-corpus queries; GloVe/SIFT are classic ANN sets (SIFT is Euclidean data scored as cosine after normalisation).
* **Machine state**: other users' background processes (VS Code, Windows services) kept CPU at 3-17% during runs (`load_before_pct` in every JSON); the orchestrator waited for < 25%. The earlier unpinned run showed the machine can also change speed over hours; pinned numbers within one run are consistent to 1-5%, but absolute QPS from a different day may differ.

## How to reproduce

```powershell
# one-off setup (venv outside the repo, data outside the repo)
python -m venv D:\Goldmine\ISON\bench_env
D:\Goldmine\ISON\bench_env\Scripts\pip install numpy psutil h5py pyarrow faiss-cpu hnswlib usearch chromadb qdrant-client lancedb sqlite-vec threadpoolctl matplotlib
# data: ann-benchmarks files + MiniLM embeddings of BeIR/hotpotqa, then float64 ground truth, filter groups and mutation plans
curl -O http://ann-benchmarks.com/glove-100-angular.hdf5 ; curl -O http://ann-benchmarks.com/sift-128-euclidean.hdf5
python benchmarks/competitive/prep_ann.py glove-100-angular glove-100-angular.hdf5      # and sift-128-euclidean
python benchmarks/competitive/prep_text.py --out D:\Goldmine\ISON\bench_data\prep\text384   # needs sentence-transformers (GPU: ~6 min)
python benchmarks/competitive/prep_gt.py text384 10000,100000,1000000                    # and glove, sift
# native harnesses
cd isongraph-vector-rs  && cargo build --release --example competitive_bench                                  # + RUSTFLAGS="-C target-cpu=native" in a second target dir
cl /std:c++17 /O2 /EHsc /MD /DNDEBUG [/arch:AVX2] /I include /I vendor/ison-graph-cpp bench/competitive_bench.cpp /link psapi.lib  # from a vcvars64 prompt
dotnet build isongraph-vector-csharp/bench/IsonGraph.Vector.CompetitiveBench -c Release -o <out>
# (JS needs no build: node --max-old-space-size=24000 isongraph-vector-js/bench/competitive.mjs ...)
# run everything (resumable; waits for CPU load < 25% before each job), then the tables and plots
python benchmarks/competitive/run_all.py --out benchmarks/competitive/results/raw
python benchmarks/competitive/report.py && python benchmarks/competitive/collect_env.py
```
Paths for the native executables are the constants at the top of `run_all.py`. One cell by hand: `python bench_py.py --system faiss_hnsw --ds text384 --n 100000 --exp core,mutation,saveload --queries 2000 --budget 3 --out x.json`.

Files: `adapters.py` (one adapter per Python-callable system), `bench_py.py` (runner), `common.py` (data layout, float64 ground truth), `prep_*.py`, `run_all.py`, `report.py`, `compare_runs.py`, `collect_env.py`; native harnesses `isongraph-vector-rs/examples/competitive_bench.rs`, `isongraph-vector-cpp/bench/competitive_bench.cpp`, `isongraph-vector-csharp/bench/IsonGraph.Vector.CompetitiveBench/`, `isongraph-vector-js/bench/competitive.mjs`. Plots: `results/plots/`.
