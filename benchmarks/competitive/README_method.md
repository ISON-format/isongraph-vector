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
