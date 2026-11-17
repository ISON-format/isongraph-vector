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
