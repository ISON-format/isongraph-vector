## How this was measured

**Data and ground truth** are the ones of `benchmarks/competitive/` (`D:\Goldmine\ISON\bench_data\prep`): `text384` (real
HotpotQA passages embedded with MiniLM-L6-v2, 384-d, 2,000 real questions as queries), `glove` (100-d) and `sift` (128-d,
L2-normalised, cosine). Ground truth is the exact float64 cosine top-100; recall@10 is against it. The first 1,000 queries are
used everywhere. The vectors fed to every system are byte-identical.

**What ran** (`run_stage_c.py` -> one process per cell, strictly one timed job at a time, never two at once):

* Rust: `isongraph-vector-rs/examples/stage_c_bench.rs` (new in this stage; derived from `competitive_bench.rs`), built with
  `cargo build --release` at HEAD 5767252 (default flags: AVX2/FMA kernels are chosen at run time; no `target-cpu=native`).
  One process per (dataset, n, mode, precision) with `--mode exact|ivf|hnsw` and `--int8 4` for the int8 variants, so every
  memory number is that mode alone.
* Libraries: `benchmarks/competitive/adapters.py` + `bench_py.py` through the wrapper `bench_c.py` (same code, same pinning,
  same loops; only the accuracy-knob list is shortened: HNSW libs ef 24-400, faiss IVF nprobe 16-512). Equal parameters as in
  the competitive benchmark: HNSW M=16, efConstruction=200; IVF nlist = 4*sqrt(N) (Rust and faiss).
* Single-thread numbers: process pinned to logical CPU 8 (a P-core) at high priority (`SetProcessAffinityMask` in Rust,
  `psutil.cpu_affinity` in Python), `set_scan_threads(1)` / `omp_set_num_threads(1)` / `num_threads=1` / `threadpoolctl`;
  each query timed individually; per sweep point: warm-up, then **three timed loops** (query count sized to a 3 s budget, at most
  1,000 queries), QPS = median of the three loops.
* Batch / all-core: one call with all 1,000 queries on 32 logical CPUs (24 cores, hybrid), warm-up call then 3 timed calls
  (Rust `similarity_search_vector[_exact]_batch`, faiss `search` with 1,000 queries, hnswlib/usearch `knn_query`/`search` with
  32 threads, numpy BLAS matmul).
* **Process-level repeats**: text384 100k (all systems) and glove / sift 100k (exact) were run **3 times each, interleaved** (Rust,
  numpy, faiss, ... then again), medians and min-max reported. The int8 study, the concurrent-client probe and the 10k-write test at
  100k ran in round 1 (the exact int8 study also in round 2 for text384). 1M cells ran **once**.
* Build time = ingest + index build + first query on each library's own path, all with 32 threads (Rust HNSW builds on the
  pool; hnswlib/usearch `add_items`/`add` with 32 threads; faiss `add` with OpenMP 32).
* Memory = process working set (`GetProcessMemoryInfo` / psutil) after the step minus the baseline (before data for Rust, after
  imports for Python). hnswlib is preallocated for N + 20k elements by the harness; numpy's matrix likewise.
* Interpolation to recall 0.95: log-linear in QPS between the two sweep points that bracket 0.95; nothing is extrapolated.

**Python libraries are driven from a Python loop, Rust from a Rust loop.** Every one-query-at-a-time number for hnswlib, faiss,
usearch and numpy therefore includes the Python call (about 10-50 microseconds per query, 3-15% of a 0.3 ms HNSW query),
while Rust has none. That is the same bias the competitive benchmark had; it favours Rust slightly in the HNSW rows. A C++ driver
for hnswlib would remove it; none was built.

## Machine state

See `environment.json` (collected before the runs by `collect_env_c.py`). i9-14900HX laptop (8 P-cores with SMT + 16 E-cores =
32 logical CPUs), 47.7 GB, Windows 11, **Balanced** power plan, plugged-in state not recorded. **The machine was not idle in the
strict sense**: the user's desktop session was open (Chrome, VS Code, ChatGPT, Task Manager, a security-suite host process) and
the agent host itself. Mean CPU load over 15 s before the runs: 11.5% of 32 logical CPUs; the pinned CPU 8 was at 0%. Before every
job `run_stage_c.py` sampled total load and the load on CPU 8 for 4 s and waited up to 45 s for total < 30% and CPU 8 < 15%
(stored in `raw/*.load.json`); a job that still saw CPU 8 above 15% is flagged `noisy` (appendix of `tables.md`). Brief spikes on CPU 8 from desktop processes (not from this harness) are frequent on this laptop, so 22 of the first 48 jobs at 100k were flagged. Every 100k job flagged that way was moved to `raw/discarded_noisy/` and **re-run once** (`rerun_noisy.py`); the re-runs ran later in the session, so the original round-robin interleaving of Rust/numpy/faiss is only approximate for the re-run cells. 9 of the 60 jobs that count are still flagged (appendix of the tables), all at 100k; the four flagged 1M jobs were re-run once as well. The min-max shown in the tables is the honest spread.
Python library versions here differ from `benchmarks/competitive/results/environment.json` (numpy 2.4.2 vs 2.5.5, faiss-cpu
1.9.0 vs 1.16.0, usearch 2.26.0 vs 2.26.6); numbers in this stage are comparable only with each other.
