# Competitive benchmark, re-run after the rework (text384, 100,000 and 1,000,000 vectors)

Every ratio and ranking in `README.md`, `README_top.md` and the CHANGELOG predates the rework (one copy of every vector, batch and
concurrent search, HNSW in Rust / C++ / C# / TypeScript / JavaScript, int8 scan in Rust, binary snapshots). This page re-measures the
ISONGraph ports at the current branch (`stage-d-part3`, see `results/post_rework/environment.json` for the exact commit) against faiss,
hnswlib and numpy, **on the same machine in the same few hours**, and records what changed. The old results and README are untouched.

All numbers below are copied from the tables in `results/post_rework/` (`tables_100000.md`, `tables_1000000.md`, `tables_jobs.md`), which
`report_post_rework.py` generates from the raw files in `results/post_rework/raw/*.json` (one JSON + one `.stdout.txt` per job).

## What was run, what was skipped

| | 100,000 | 1,000,000 |
|---|---|---|
| faiss Flat / IVFFlat / HNSW, hnswlib, numpy | run (3 in-process builds, 3 timed runs per point) | run (1 build) |
| ISONGraph Rust: exact, int8 exact, IVF, **HNSW**, JSON vs snapshot save/load; also `target-cpu=native` | run | default build only; no `native` |
| ISONGraph C++: exact, IVF, **HNSW**, JSON vs snapshot; also `/arch:AVX2` | run | default build only; no JSON/snapshot, no AVX2 |
| ISONGraph C#: exact, IVF, **HNSW**, JSON vs snapshot | run | exact, IVF, HNSW (no save/load) |
| ISONGraph TypeScript and JavaScript (Node 24): exact, IVF, **HNSW**, snapshot | run | JS exact + IVF only. **JS/TS HNSW skipped at 1M**: the 100k build alone took 251 s single-threaded (the other ports use all cores: 7-13 s), and the Rust build grew 25x from 100k to 1M (6.8 s to 170 s), so JS would need about 100 minutes. TS skipped at 1M altogether (it is the same code as JS: 100k results agree within 1%). |
| ISONGraph Python: exact, IVF, snapshot (no HNSW: not shipped) | run | exact + IVF, run |
| Mutation, filter, glove, sift | **not run** (outside the agreed scope) | not run |

Whole-process repeats (a second, independent run of every job) were done at 100k for faiss, hnswlib, numpy, Rust (default), C++ (default) and C#
(`__r2` files; `tables_100000.md`, "Whole-process repeats"). Rust `native`, C++ `AVX2`, JS, TS, Python and everything at 1M are single runs.

## Method (same as `README_method.md`, plus what changed)

* Data, ground truth (float64 exact), M = 16, efConstruction = 200 for every HNSW, nlist = round(4 sqrt N) for every IVF, k = 10, recall@10,
  2,000 queries: as in `README_method.md`.
* Single-thread QPS: one query at a time, pinned to logical CPU 8 (a P-core) at HIGH priority, warm-up, three timed runs per accuracy setting
  (median reported, min-max shown). Node is pinned to P-cores 8,10,12,14 as before. Batch QPS: one call over all 32 logical CPUs
  (`similarity_search_vector_batch` / `similaritySearchBatch` / `SimilaritySearchBatch` / `similarity_search_batch`; faiss `search` with 32 OpenMP threads;
  hnswlib `knn_query(num_threads=32)`). The JS/TS ports have no batch API.
* "QPS at recall 0.95" is log-linearly interpolated between the two sweep points that bracket 0.95 (the kit's own rule, `report.interp_at`). The raw first
  point at or above 0.95 is shown beside it; ISONGraph needs a higher ef than faiss for the same recall (ef 128 gives 0.965, faiss ef 96 gives 0.957), so
  the raw points are not at equal recall and the interpolation is the fairer comparison.
* HNSW builds use every core for faiss, hnswlib, Rust, C++ and C# (their builds are multi-threaded); JS/TS build on one thread. ISONGraph HNSW is built through
  the library's own path (`build_ann_index` after ingest; `AnnConfig::hnsw()`, `AnnOptions::makeHnsw`, `AnnMode.Hnsw`, `ann: 'hnsw'`).
* Memory = process working set above the baseline measured before any data is touched. The ISONGraph HNSW and IVF-plus-exact figures: HNSW runs in its own
  process (its RSS is vectors + graph); the "IVF" row is the exact store plus the IVF index, as in the original tables.
* Machine hygiene: nothing else of mine ran during timed jobs; `run_post_rework.py` waited for other-process CPU load below 25% before every job (the
  load before the jobs was 9-24%, background Windows/IDE processes that were not stopped; no job was flagged contaminated; `tables_jobs.md`).
  A first start launched two copies of the runner by mistake; they were killed within three minutes and **all of their output was deleted**; every file
  in `results/post_rework/raw` comes from the clean restart.
* Noise: whole-process repeats of the same job agree to 0.0-3.7% for ISONGraph and numpy and 5.6-6.6% for faiss HNSW and hnswlib (HNSW sweeps are
  the noisiest); treat differences under ~7% as noise.

## Headline: text384, N = 100,000

Single-thread QPS at recall@10 = 0.95 (interpolated), batch QPS at 0.95 on 32 CPUs, build, RSS above baseline after the build.

| system | QPS 1T @0.95 | batch QPS @0.95 | build s | RSS +MB |
|---|---|---|---|---|
| faiss HNSWFlat | 2671 | 17081 | 12.8 | 184 |
| hnswlib | 2575 | 16326 | 6.10 | 194 |
| faiss IVFFlat | 936 | 3938 | 1.60 | 237 |
| **ISONGraph Rust HNSW** | **3416** | 17979 | 6.90 | 200 |
| ISONGraph Rust HNSW, target-cpu=native | 3302 | 16711 | 6.62 | 198 |
| **ISONGraph C++ HNSW** | **2969** | 16543 | 9.24 | 193 |
| ISONGraph C++ HNSW, /arch:AVX2 | 4004 | 21099 | 7.38 | 196 |
| ISONGraph C# HNSW | 2026 | 14175 | 12.9 | 341 |
| ISONGraph JS HNSW | 969 | n/a | 251 | 476 |
| ISONGraph TS HNSW | 965 | n/a | 250 | 473 |
| ISONGraph Rust IVF | 742 | 24702 | 1.52 | 172 |
| ISONGraph C++ IVF | 535 | 4060 | 2.49 | 171 |
| ISONGraph C# IVF | 491 | 2872 | 2.89 | 511 |
| ISONGraph Python IVF | 346 | n/a | 4.25 | 224 |
| ISONGraph JS IVF | 118 | n/a | 93.0 | 462 |
| ISONGraph TS IVF | 113 | n/a | 93.0 | 462 |

Exact search (recall 1.000), 100k:

| system | QPS 1T | batch QPS | one query, library may use all threads | RSS +MB |
|---|---|---|---|---|
| numpy | 112 | 722 | 202 | 160 |
| faiss IndexFlatIP | 108 | 1775 | 118 | 172 |
| ISONGraph Python | 106 | 623 | 185 | 199 |
| **ISONGraph Rust** | 115 | **5454** | 252 | 164 |
| ISONGraph Rust, int8 first pass + f32 rerank | **372** | 4338 | 1216 | 202 |
| ISONGraph Rust, native | 111 | 5320 | 227 | 163 |
| ISONGraph C++ (AVX2 build: 37.1) | 35.9 | 299 | n/a | 166 |
| ISONGraph C# | 57.5 | 508 | n/a | 352 |
| ISONGraph JS / TS | 27.1 / 27.4 | n/a | n/a | 400 |

### Ratios at 100k (ISONGraph divided by the competitor; above 1 = ISONGraph faster)

| ISONGraph | 1T @0.95 vs faiss HNSW | vs hnswlib | vs faiss IVFFlat | batch @0.95 vs faiss HNSW | build time vs faiss HNSW / hnswlib | RSS vs faiss HNSW / hnswlib |
|---|---|---|---|---|---|---|
| Rust HNSW | 1.28x | 1.33x | 3.65x | 1.05x | 0.54x / 1.13x | 1.08x / 1.03x |
| C++ HNSW | 1.11x | 1.15x | 3.17x | 0.97x | 0.72x / 1.51x | 1.05x / 1.00x |
| C++ HNSW /arch:AVX2 | 1.50x | 1.56x | 4.28x | 1.24x | 0.58x / 1.21x | 1.06x / 1.01x |
| C# HNSW | 0.76x | 0.79x | 2.16x | 0.83x | 1.01x / 2.11x | 1.85x / 1.76x |
| JS / TS HNSW | 0.36x / 0.36x | 0.38x / 0.37x | 1.04x / 1.03x | no batch | 20x / 41x slower | 2.6x / 2.5x |
| Rust IVF | 0.28x | 0.29x | 0.79x | 1.45x vs faiss HNSW (6.3x vs faiss IVF batch) | 0.12x / 0.25x | 0.93x / 0.89x |
| C++ IVF | 0.20x | 0.21x | 0.57x | 0.24x | 0.20x / 0.41x | 0.93x / 0.88x |
| C# IVF | 0.18x | 0.19x | 0.52x | 0.17x | 0.23x / 0.47x | 2.8x / 2.6x |
| Python IVF | 0.13x | 0.13x | 0.37x | n/a | 0.33x / 0.70x | 1.21x / 1.15x |

Build time ratios compare the first build in a fresh process (faiss 12.8 s, hnswlib 6.10 s at 100k); the second and third in-process builds of faiss and hnswlib
were slower (13.9 s and 8.9 s medians), so the 3-build medians would flatter ISONGraph. Even so, **hnswlib builds faster than every ISONGraph port at 100k** (6.10 s vs Rust 6.90 s, C++ 9.24 s).

Exact, single thread: Rust 1.03x numpy and 1.06x faiss Flat (level); Python 0.95x / 0.98x (level); C# 0.52x / 0.53x; C++ 0.32x / 0.33x;
JS/TS 0.24x / 0.25x. Exact, batch: Rust 7.6x numpy and 3.1x faiss Flat; Python 0.86x numpy and 0.35x faiss Flat; C# 0.70x / 0.29x; C++ 0.41x / 0.17x.
Rust int8 exact (single thread): 3.3x numpy, 3.4x faiss Flat (recall vs float64 truth 1.000 on these queries).

## Headline: text384, N = 1,000,000

| system | QPS 1T @0.95 | batch QPS @0.95 | build s | RSS +MB |
|---|---|---|---|---|
| faiss HNSWFlat | 1207 | 5473 | 204 | 1,652 |
| hnswlib | 1040 | 4608 | 210 | 1,830 |
| faiss IVFFlat | 153 | 454 | 39.5 | 1,875 |
| **ISONGraph Rust HNSW** | **1446** | 5613 | 172 | 1,786 |
| **ISONGraph C++ HNSW** | **1308** | 5244 | 206 | 1,780 |
| ISONGraph C# HNSW | 926 | 4645 | 260 | 2,289 |
| ISONGraph Rust IVF | 101 | 4624 | 17.8 | 1,649 |
| ISONGraph C++ IVF | 74.4 | 384 | 29.0 | 1,650 |
| ISONGraph Python IVF | 54.6 | n/a | 101 | 1,915 |
| ISONGraph JS IVF | 16.7 | n/a | 875 | 2,131 |
| ISONGraph C# IVF | not reached (0.945 at nprobe 256, the widest setting measured) | | 30.0 | 4,140 |

Exact, 1M: numpy 11.2 QPS (batch 67.5), faiss Flat 10.6 (batch 199), ISONGraph Python 11.1 (49.1), **Rust 11.1 (batch 570)**, Rust int8 33.7 (542),
C# 5.60 (38.9), C++ 2.89 (30.2), JS 2.46.

Ratios at 1M: Rust HNSW 1.20x faiss HNSW, 1.39x hnswlib, 9.5x faiss IVFFlat (batch 1.03x faiss HNSW); C++ HNSW 1.08x / 1.26x / 8.6x (batch 0.96x);
C# HNSW 0.77x / 0.89x / 6.1x (batch 0.85x). Build time vs faiss HNSW / hnswlib: Rust 0.84x / 0.82x, C++ 1.01x / 0.98x, C# 1.27x / 1.24x.
RSS after build vs faiss HNSW / hnswlib: Rust 1.08x / 0.98x, C++ 1.08x / 0.97x, C# 1.39x / 1.25x. Exact single thread: Rust 0.99x numpy, 1.05x faiss Flat; Python 0.99x / 1.04x;
C# 0.50x / 0.53x; C++ 0.26x / 0.27x; JS 0.22x / 0.23x.

## Plain-language comparison

**Where ISONGraph is ahead or level**
* *HNSW, Rust.* Single-thread QPS at recall 0.95 is 1.28x faiss HNSW and 1.33x hnswlib at 100k, 1.20x / 1.39x at 1M. Batch QPS is level with faiss (1.05x, 1.03x). The
  build is faster than faiss at 100k (6.9 s vs 12.8 s, 0.54x) but slower than hnswlib (6.10 s, 1.13x); at 1M it is 0.84x faiss and 0.82x hnswlib (172 s vs 204 s / 210 s). Memory is within 8% of faiss and hnswlib.
* *HNSW, C++.* 1.11x faiss / 1.15x hnswlib at 100k (inside the 7% noise floor for faiss: its two process repeats differed by 6.6%), 1.08x / 1.26x at 1M.
  The `/arch:AVX2` build is 1.50x / 1.56x at 100k.
* *Exact scan, Rust and Python, one thread:* level with numpy and faiss IndexFlatIP (within 6% at 100k and 1M). Rust's batch exact search is 3.1x faiss Flat and
  7.6x numpy at 100k (2.9x and 8.4x at 1M). Rust's int8 scan is 3.3-3.5x faster than f32 on one thread.
* *IVF batch in Rust:* 24,702 QPS at 0.95 vs faiss IVFFlat 3,938 (6.3x), but faiss HNSW and the ISONGraph HNSW are the right comparison for recall 0.95 and
  are faster still per query on one thread.
* *Memory:* about half of the pre-rework figures (see below); Rust and C++ are within 8% of faiss/hnswlib at HNSW.

**Where ISONGraph is slower**
* *C# HNSW:* 0.76x faiss and 0.79x hnswlib at 100k (0.77x / 0.89x at 1M); 1.85x the RSS at 100k (1.39x at 1M); build 1.27x slower than faiss at 1M.
* *JS / TS HNSW:* 0.36x faiss HNSW (2.8x slower) on one thread; **the build takes 251 s at 100k (20x faiss, 41x hnswlib)** because it is single-threaded; RSS 2.5x.
  Still 8x faster than the same ports' IVF (969 vs 118 QPS at 0.95).
* *IVF in every port* is 4-8x slower per thread than HNSW of the same port at recall 0.95 (Rust IVF 742 vs Rust HNSW 3416; JS 118 vs 969) and 0.37-0.79x of faiss IVFFlat (Rust 0.79x, C++ 0.57x, C# 0.52x,
  Python 0.37x at 100k; Rust 0.66x at 1M). IVF is not competitive with faiss HNSW or hnswlib: 0.28x (Rust) down to 0.04x (JS) at 100k, 0.08x (Rust) at 1M.
* *Exact scan in C++, C#, JS/TS:* 0.32x, 0.52x and 0.25x of numpy/faiss Flat on one thread at 100k (0.26x, 0.50x, 0.22x at 1M); their batch search is 0.17-0.29x of faiss Flat.
  Python's numpy batch (`similarity_search_batch`) is 0.35x faiss Flat batch at 100k, 0.25x at 1M.
* *Build time for HNSW:* C# 260 s vs 204 s (faiss) at 1M; C++ 9.24 s and Rust 6.90 s vs hnswlib 6.10 s at 100k.
* *Persistence of an index:* faiss and hnswlib save and reload their index (0.13-0.55 s at 100k); ISONGraph persists vectors only, so time to ready after a restart includes
  rebuilding the index (HNSW: the full build time above; IVF: time to ready 4.05 s Rust, 11.4 s C#, 18.5 s C++, 45.6 s Python at 100k).

## Save / load at 100k (JSON sidecar vs the new binary snapshot)

| system | JSON save s | JSON load s | snapshot save s | snapshot load s | load speed-up | file MB (JSON / snapshot) |
|---|---|---|---|---|---|---|
| ISONGraph Python | 41.1 | 27.6 | 1.61 | 1.02 | 27x | 807 / 149 |
| ISONGraph Rust | 3.71 | 2.79 | 0.385 | 0.416 | 6.7x | 453 / 149 |
| ISONGraph C++ | 42.6 | 16.3 | 1.17 | 1.13 | 14x | 500 / 149 |
| ISONGraph C# | 8.11 | 8.80 | 0.428 | 0.585 | 15x | 453 / 149 |
| ISONGraph JS / TS | **fails: "Invalid string length"** | | 0.382 / 0.359 | 0.424 / 0.424 | n/a | n/a / 149 |
| faiss Flat / HNSW / IVF (native index file) | 0.27 / 0.20 / 0.12 | 0.14 / 0.13 / 0.18 | | | | 147-160 |
| hnswlib | 0.219 | 0.545 | | | | 161 |

At 1M the Rust snapshot loads in 4.47 s vs 28.7 s for JSON (6.4x; save 8.6 s vs 42.8 s). The snapshot is within 3x of faiss's reload time for the vectors alone, but it does not hold the index.

## What changed against the pre-rework table (`README.md` / `results/raw`)

The pre-rework run was taken hours earlier and the whole machine was slower during this one: the five reference systems whose code did not change (faiss Flat, IVF, HNSW, hnswlib, numpy)
are **0.77x** of their old QPS at both 100k and 1M (e.g. faiss HNSW 3542 to 2671, numpy 144 to 112). A raw before/after QPS comparison would therefore blame the rework for a 23% loss that is
not its own; `change / drift` in the tables divides it out (`tables_100000.md`, "Before / after").

| system, 100k | QPS before | QPS after | change / drift | RSS +MB before to after |
|---|---|---|---|---|
| Rust exact | 154 | 115 | 0.96x | 339 to 164 |
| Rust IVF @0.95 | 828 | 742 | 1.16x | 342 to 172 |
| C++ exact | 22.6 | 35.9 | 2.05x | 192 to 166 |
| C++ IVF | 681 | 535 | 1.01x | 342 to 171 |
| C# exact | 39.1 | 57.5 | 1.90x | 291 to 352 |
| C# IVF | 615 | 491 | 1.03x | 788 to 511 |
| JS exact | 19.7 | 27.1 | 1.78x | 518 to 400 |
| JS IVF | 201 | 118 | 0.76x | 1,216 to 462 |
| Python exact | 131 | 106 | 1.05x | 454 to 199 |
| Python IVF | 530 | 346 | 0.84x | 481 to 224 |

At 1M the same pattern: Rust 3,517 MB to 1,623 MB, Python 4,223 MB to 1,888 MB, C++ 3,279 to 1,650 (IVF); exact single-thread QPS C++ 1.77x, C# 1.96x, JS 1.92x (drift-corrected), Rust 0.96x, Python 1.13x.

Read-outs: the "one copy of the vectors" rework roughly halves memory in Rust, C++ and Python and shows a real speed-up for the C++, C# and JS exact scans; Rust and Python single-thread exact were already at the
numpy/faiss level and stay there; what is new is batch search (Rust exact 3x faiss Flat) and HNSW. The old claim "faiss Flat batch is ~35x ours" (README_top.md, 100k) is now **Rust exact batch 5454 QPS vs faiss Flat 1775 (3.1x ahead)**.

## Things that look wrong or unexplained (not fixed; library code was not touched)

1. **IVF index builds are 2-3x slower than in the pre-rework run in every native port, which is not machine drift** (faiss IVF build is unchanged, 1.72 s to 1.60 s). Rust `build_seconds` 0.480 s to
   1.42 s (second process: 1.27 s) and `index_bytes` 2,753,168 to 3,173,400 at 100k; C++ 0.97 s to 2.49 s; C# 1.28 s to 2.89 s; JS 56.7 s to 93.0 s; at 1M Rust 7.55 s to 17.8 s, C++ 12.4 s to 29.0 s. Looks like a change to IVF training or the
   index layout since the old measurement; worth a look.
2. **Python ingest at 1M is 3.1x slower than before (19.6 s to 59.9 s) while at 100k it is unchanged (2.01 s to 2.05 s)**: a super-linear cost somewhere in `add_batch` growth. Memory fell from 4.2 GB to 1.9 GB.
3. **The JS/TS JSON sidecar cannot be written at 100k x 384** (`RangeError: Invalid string length`: the whole payload is one JS string). `loadSnapshot`/`saveSnapshot` work (0.4 s). So the JSON sidecar is a hard limit for these ports on real 384-d data at this size (the limit is V8's maximum string length; I did not measure where it begins).
   (The failing first attempt is kept in `results/post_rework/js_saveload_first_attempt_json_error.json`; the driver was changed to measure the snapshot separately and re-run.)
4. **The library's automatic `ef_search` misses 0.95 on this data at 100k**: 96 gives recall 0.948 in all five ports (identical recall everywhere, which is a good cross-port consistency check); at 1M the automatic 192 gives 0.9565.
   Choosing ef 128 gives 0.965.
5. **ISONGraph's HNSW needs a higher ef than faiss for the same recall** (faiss 0.957 at ef 96, ISONGraph 0.948). Per-query it is still faster; the `QPS at 0.95` columns are interpolated so this is accounted for.
6. C# IVF reaches 0.945 at nprobe 256 at 1M (0.9462 before): there is no point at or above 0.95; the table says "not reached".
7. JS IVF (0.76x) and Python IVF (0.84x) are slower than before after drift correction; Rust, C++ and C# IVF are level with or ahead of before. Not investigated.
8. ISONGraph HNSW does not use more memory per vector than the others in Rust/C++ (graph 14.5 MB at 100k, 146 MB at 1M for M=16), but C# uses 1.4-1.85x and JS/TS 2.5x the RSS of faiss (cause not investigated).
9. faiss HNSW's first in-process build at 100k took 12.8 s (13.7 s in the repeat process) versus 8.9 s in the pre-rework run: the machine was slower, and in both faiss and hnswlib the 2nd and 3rd in-process builds were slower than the first (faiss 12.8, 14.2, 13.9 s; hnswlib 6.1, 8.9, 9.2 s).
   Build-time ratios against faiss/hnswlib therefore carry about +-25% and depend on which build is compared (first, as in the ratio table, is the conservative choice for ISONGraph).

## Driver and API changes made (benchmark code only)

* `isongraph-vector-rs/examples/competitive_bench.rs`: new experiments `hnsw` (HNSW in its own store, ef sweep, batch per point, library-default ef, concurrent clients), `int8` (exact with int8 first pass), snapshot save/load in `saveload`.
* `isongraph-vector-cpp/bench/competitive_bench.cpp`: `hnsw` experiment; batch search (`similaritySearchBatch`) for exact and every IVF/HNSW point; snapshot save/load. (It built unchanged against the current headers; `get()` returning `std::optional` is not used by this driver.)
* `isongraph-vector-csharp/bench/IsonGraph.Vector.CompetitiveBench/Program.cs`: `hnsw` experiment (`AnnMode.Hnsw`), batch search, `ScanThreads = 1` during the pinned single-thread loops (the scan otherwise fans out on one core). It built unchanged otherwise (the new trailing constructor parameter `annHnsw` is used by name).
* `isongraph-vector-js/bench/competitive.mjs`: `hnsw` and `saveload` experiments, `--entry` to run the compiled TypeScript build; the JSON and snapshot paths are measured in separate try-blocks.
* `benchmarks/competitive/adapters.py` / `bench_py.py`: ISONGraph Python exact adapter now has `searchN` (`similarity_search_batch`); snapshot save/load is recorded beside the JSON sidecar.
* New: `run_post_rework.py` (job list and runner against freshly built binaries), `report_post_rework.py` (tables). `run_all.py`, `report.py`, `README*.md` and `results/raw` etc. are unchanged.
* Binaries were rebuilt from the branch into `D:\Goldmine\ISON\bench_data\post_rework_build\{rs_default,rs_native,cpp,cs}`; `isongraph-vector-ts/dist` was rebuilt with `npm run build`. `hnswlib` 0.8.0 (real hnswlib) was already installed in `bench_env`.

## Files

`results/post_rework/`: `environment.json`, `run_log.txt` (job start/end, load before each job), `run_stdout*.txt`, `raw/*.json` and `raw/*.stdout.txt` (the evidence), `tables_100000.md`, `tables_1000000.md` (full sweeps inside),
`tables_jobs.md`, `js_saveload_first_attempt_json_error.json`.

Reproduce: build the binaries as above, then `python run_post_rework.py --cell 100000` and `--cell 1000000` (resumable), `python report_post_rework.py`.
