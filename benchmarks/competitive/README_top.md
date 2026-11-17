# Competitive benchmark: ISONGraph Vector against the established vector-search libraries

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

