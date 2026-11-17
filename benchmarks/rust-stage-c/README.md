# Stage C: honest re-benchmark of the reworked Rust crate

Crate `isongraph-vector-rs` at HEAD 5767252 (branch `feature/ann-ports-and-server`), measured against numpy, faiss
(IndexFlatIP / IVFFlat / HNSWFlat), hnswlib and usearch on the data and ground truth of `benchmarks/competitive/`.
Everything below is regenerated from `raw/` by `summarize.py`; nothing was carried over from earlier READMEs.

## Verdict (one paragraph)

The reworked Rust crate is now a credible ANN engine, but the old "fastest exact scan" claim does not survive a same-session
re-measurement. **Exact search is at parity, not ahead**: single-thread Rust vs faiss IndexFlatIP is 0.91-1.2x depending on the
dataset and run (it is slower than faiss on GloVe and SIFT 100k, equal at text384 1M: 11.1 vs 11.4 QPS), and it beats numpy
only on GloVe (1.45x) and, within run noise, text384 100k (1.27x); at 1M and on SIFT it ties numpy. **HNSW is the big change**:
at recall@10 0.95 it matches faiss HNSWFlat and hnswlib single-thread (text384 100k: 3,893 vs 3,965 / 2,723 QPS; 1M: 1,813 vs
1,773 / 1,517; one run each, and the discarded first attempt of the Rust cell gave 1,531), beats usearch (2,070 / 612), builds in about the same time
(145-173 s vs 157-216 s at 1M) and uses about the same memory. IVF is still slower than
faiss IVFFlat at equal recall (0.71x at 100k, 0.76x at 1M), but IVF+int8 beats it (1.9x / 1.3x). The Rust batch API is a clear
win for IVF (8x faiss IVFFlat on 32 threads at 100k, 10-21x at 1M) and level with the HNSW libraries; int8 gives a 3.0-3.6x single-thread speedup with
recall 1.000 against the f32 scan from factor 2 on real embeddings, and does nothing useful for batches. The binary snapshot
loads 6.6x faster than the JSON sidecar and is 3x smaller, but an ANN index is still never persisted: time-to-ready is 21.6 s
(IVF) / about 2.5-3 minutes (HNSW) at 1M, against about 1 s for faiss and usearch. Single-call HNSW inserts are slow (1,393/s at 1M,
versus 3,789-6,005/s for hnswlib/usearch, which were driven in parallel batches of 1,000); HNSW updates are on par with hnswlib (1,440 vs 1,696/s). The machine was a laptop with the
user's desktop apps running, not a quiet bench: run-to-run spread is visible (20-25% on one-query QPS, up to 2x on batch, build and write rates between the first and the repeated attempt of the same 1M cell), so differences under about 15% are
reported as parity.

## Answers

| # | question | answer | evidence (details in the tables) |
|---|---|---|---|
| 1 | Rust 1T exact still faster than numpy and faiss Flat (text384 100k/1M, glove 100k, sift 100k)? | **Refuted** as stated. Parity with faiss; slight edge over numpy on some cells | faiss Flat/Rust: text 100k 117.8 / 142.6 (Rust runs 114-146, faiss 117-136: overlap); text 1M 11.4 / 11.1; glove 605 / 548 (ranges 604-622 vs 537-548: faiss wins by ~10%); sift 432 / 397 (ranges overlap). numpy: 113 / 11.1 / 377 / 386 vs Rust 143 / 11.1 / 548 / 397 |
| 2 | HNSW vs hnswlib / faiss HNSW / usearch at recall 0.95, 1T | **Confirmed: parity with the best (at 1M ahead of hnswlib by 19% in one run), well ahead of usearch.** IVF **refuted** (0.71-0.76x faiss IVFFlat); IVF+int8 **confirmed faster** than faiss IVFFlat (1.3-1.9x); HNSW+int8 is accepted but a **no-op that costs the int8 copy's memory (+370 MB at 1M)** | Q2 table; 100k 3 runs, 1M 1 run |
| 3 | Batch / all-core vs faiss Flat batch and HNSW libs at 100k | **Confirmed for IVF (8x faiss IVFFlat at 100k, 10-21x at 1M); inconclusive for exact (5,320 vs 5,085, run ranges overlap; 560 vs 298 at 1M, 1 run); parity for HNSW (14.0k vs 16.7k faiss, 12.7k hnswlib, 13.7k usearch)** | Q3 table; batch timings varied up to 2.5x between runs |
| 4 | int8 recall on real embeddings, factors 1/2/4/8, speedup | **Confirmed**: factor >= 2 gives recall 1.0000 vs f32 scan on text384 (100k, 1M), GloVe and SIFT; factor 1 gives 0.976-0.992 (worst query 0.80). 1T speedup 3.0-3.6x. **Batch: no benefit (0.4-1.0x (slower than f32 in every measured batch))** | Q4 table |
| 5 | Persistence | Snapshot vs JSON: **confirmed big win** (1M load 4.6 s vs 30 s, save 7.7 s vs 42.9 s, 1,489 vs 4,534 MB). Competitive with faiss/hnswlib for a ready-to-search ANN index: **refuted** (index not persisted) | Q5 table |
| 6 | Writes at 1M, IVF and HNSW vs hnswlib/usearch | IVF: **faster than faiss IVFFlat on inserts/updates** (138k / 133k per s vs 87k / 69k; first query after writes 8.0 vs 8.1 ms). HNSW single-call inserts **slower** than hnswlib/usearch batch inserts (1,393 vs 6,005 / 3,789 per s), updates on par (1,440 vs 1,696 / 3,414); first query after writes 0.6 ms and recall holds (0.9549 -> 0.9550) | Q6 table |
| 7 | Memory at 1M x 384 | exact 1.12x raw, IVF 1.12x, HNSW 1.22x (faiss HNSW 1.24x, hnswlib 1.28x, usearch 1.17x, faiss IVF 1.28x); int8 adds 371 MB. **Confirmed no worse than the libraries** | Q7 table |

## What we win and lose now (each line is exactly what the data supports)

**Win**
* Single-thread HNSW at recall@10 ~0.95 on text384 is level with faiss HNSWFlat and 19% above hnswlib at 1M (1,813 vs 1,773 / 1,517 QPS, one run each) and 43% above hnswlib, level with faiss, at 100k (3 runs; faiss 3,965, hnswlib 2,723). The one-query loop for the libraries runs through Python, Rust's does not: part of the 100k margin over hnswlib may be that overhead.
* IVF batch search on 32 threads: 24,360 QPS vs faiss IVFFlat 3,014 at 100k (recall ~0.955, 3 runs, ranges 21-25k vs 2.9-3.6k) and 10,068 vs 471 at 1M (1 run; 4,524 in the discarded noisy first attempt).
* IVF + int8 (rerank 4) single-thread at recall 0.955: 2,059 QPS vs faiss IVFFlat f32 1,065 at 100k (1 run), 235 vs 180 at 1M (1 run).
* int8 exact scan: 3.0-3.6x single-thread QPS with exact returned scores and recall 1.0000 vs the f32 scan at factor >= 2 on all four real datasets; the copy costs 25% of the vector bytes.
* Binary snapshot load: 6.6x faster than the JSON sidecar at 100k and 1M (0.19 vs 1.27 s; 4.6 vs 30 s) and 3x smaller.
* Index build: IVF 7.6 s vs faiss 35.3 s at 1M (1 run; 18.8 s in the discarded first attempt); HNSW is level: 173 s vs hnswlib 157 s, faiss 190 s, usearch 216 s at 1M (1 run; 152 s in the discarded attempt); 7.1 s vs 8.4 / 9.6 / 6.8 s at 100k (3 runs).
* IVF memory: 1,643 MB working set at 1M vs faiss IVFFlat 1,881 MB, and a 1,886 MB build peak vs faiss 3,588 MB.
* IVF writes at 1M: 138k inserts/s and 133k updates/s one call at a time vs faiss IVFFlat 87k / 69k (1 run each; 81k / 69k in the discarded first attempt); first query after 30k writes 8.0 ms, recall 0.9526 -> 0.9530.
* 8 concurrent clients scale (HNSW 1M: 700 -> 4,990 QPS from 1 to 8 clients); numpy/faiss searches are not compared here.

**Lose / parity**
* Exact single-thread scan: no longer ahead. Equal to faiss Flat and numpy at text384 1M (11.1 / 11.4 / 11.1); 9-10% slower than faiss Flat on GloVe 100k (548 vs 605); not distinguishable on text384 100k and SIFT 100k.
* IVF (f32) at equal recall: 0.71x faiss IVFFlat at 100k (759 vs 1,065), 0.76x at 1M (137 vs 180; 108 in the discarded first attempt).
* ANN indexes are not persisted: after a snapshot load, IVF is ready in 0.7 s (100k) / 21.6 s (1M) and HNSW in 6.1 s (100k) / about 2.5-3 minutes (1M: load 4.6 s + a 145-173 s build, not run as one measurement); faiss/usearch are ready in 0.1-1.3 s.
* HNSW insert rate: 944/s at 100k and 1,393/s at 1M one call at a time (650/s in the discarded 1M first attempt) against hnswlib 6,005/s and usearch 3,789/s in batches of 1,000 on 32 threads. Not like for like, but it is what a caller gets. HNSW updates (1,440/s at 1M) are on par with hnswlib (1,696/s) and below usearch (3,414/s).
* int8 does nothing for batches (0.4-1.0x (slower than f32 in every measured batch)) and does nothing for HNSW (setting accepted, search path unchanged, +370 MB at 1M).
* JSON save/load is still 43 s / 30 s at 1M (4.5 GB).
* Exact batch (5,320 vs faiss Flat 5,085 at 100k): not distinguishable; at 1M 560 vs 298 (1 run).

**Anything surprising / possible bug**: `set_scan_precision(Int8)` together with `AnnConfig::hnsw()` is silently accepted. Searches never use the int8 copy (identical recall at identical ef: 0.9626 / 0.9549; QPS equal within noise), yet 37 MB (100k) / 370 MB (1M) are allocated. Not a correctness bug, but neither the README nor DESIGN.md says it, and it wastes memory. Repro: `stage_c_bench --mode hnsw --int8 4`, compare `mem_after_build` with `--mode hnsw`. No wrong answers, crashes or recall regressions were found anywhere (every int8 score checked bit-exact; every snapshot/JSON load returned the same top-10 as the original store; recall after the 30k writes within 0.005 of before at every operating point).



# Tables

Generated by `summarize.py` from `raw/`. Q1-Q7 match the questions of the stage.

## Q1. Exact scan: ISONGraph Rust vs numpy vs faiss IndexFlatIP (single thread, one pinned P-core)

QPS, median over runs (min-max). `ratio` = Rust / other, > 1 means Rust is faster. Rounds were interleaved (Rust, numpy, faiss, Rust, ...) in one session.

| cell | runs | Rust exact | numpy | faiss Flat | Rust/numpy | Rust/faiss | Rust auto-thread loop | faiss Flat batch (32 thr) | numpy batch | Rust batch |
|---|---|---|---|---|---|---|---|---|---|---|
| text384 100,000 | 3/3/3 | 142.6 (114.0-145.7) | 112.7 (112.2-121.3) | 117.8 (116.5-135.5) | 1.27x | 1.21x | 269 (216-284) | 5,085 (3,070-6,552) | 673 (543-1,052) | 5,320 (5,206-12,743) |
| text384 1,000,000 | 1/1/1 | 11.1 (1 run) | 11.1 (1 run) | 11.4 (1 run) | 1.00x | 0.97x | 25 (1 run) | 298 (1 run) | 61 (1 run) | 560 (1 run) |
| glove 100,000 | 3/3/3 | 548.2 (537.1-548.4) | 377.0 (323.4-456.6) | 605.0 (603.5-621.6) | 1.45x | 0.91x | 1,568 (708-1,669) | 14,934 (5,444-15,184) | 857 (783-1,434) | 19,239 (10,552-25,285) |
| sift 100,000 | 3/3/3 | 396.9 (350.0-470.5) | 385.5 (313.9-389.3) | 431.7 (421.5-521.6) | 1.03x | 0.92x | 722 (645-1,546) | 5,898 (5,075-14,071) | 1,364 (786-1,512) | 11,457 (10,910-28,197) |

## Q2. Approximate search at recall@10 = 0.95, single thread (pinned), text384

QPS interpolated (log-linear) between the two sweep points that bracket recall 0.95; `-` = not bracketed by the sweep (recall 0.95 not reached, or the lowest point already above it: then the lowest measured point is shown in the `first point` column). Build = ingest + index build on the library's own path with 32 threads. Memory = working set after the build minus baseline.

### text384 100,000

| system | runs | QPS @0.95 (1T) | knob at first recall>=0.95 | recall at that knob | p99 ms there | build s | of which index s | RSS after build MB | peak MB | first point (knob: recall, QPS) | best recall reached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ISONGraph Rust HNSW | 3 | 3,893 (3,314-4,486) | 128 | 0.9626 | 0.47 | 7.1 (6.9-7.4) | 7.0 (6.8-7.3) | 199 (198-200) | 201 (201-201) | 24: 0.782, 8,823 | 0.9936 |
| ISONGraph Rust IVF | 3 | 759 (752-968) | 128 | 0.9559 | 1.81 | 0.6 (0.6-0.6) | 0.5 (0.5-0.5) | 166 (166-166) | 246 (246-246) | 16: 0.750, 5,848 | 0.9952 |
| ISONGraph Rust IVF + int8 x4 | 1 | 2,059 (1 run) | 128 | 0.9559 | 0.81 | 0.6 (1 run) | 0.5 (1 run) | 205 (1 run) | 282 (1 run) | 16: 0.750, 7,552 | 0.9953 |
| ISONGraph Rust HNSW + int8 x4 (setting on) | 1 | 4,571 (1 run) | 128 | 0.9626 | 0.47 | 5.7 (1 run) | 5.6 (1 run) | 238 (1 run) | 239 (1 run) | 64: 0.914, 7,034 | 0.9796 |
| hnswlib | 3 | 2,723 (2,480-2,751) | 128 | 0.9631 | 0.66 | 8.4 (8.0-8.7) | 0.0 (0.0-0.0) | 201 (201-201) | 231 (231-231) | 24: 0.785, 7,766 | 0.9930 |
| faiss HNSWFlat | 3 | 3,965 (3,400-4,117) | 96 | 0.9639 | 0.48 | 9.6 (7.6-12.2) | 0.0 (0.0-0.0) | 207 (206-207) | 348 (347-349) | 24: 0.810, 8,382 | 0.9971 |
| usearch | 3 | 2,070 (2,012-2,174) | 128 | 0.9634 | 0.89 | 6.8 (6.1-8.2) | 0.0 (0.0-0.0) | 182 (180-183) | 211 (210-213) | 24: 0.781, 6,608 | 0.9927 |
| faiss IVFFlat | 3 | 1,065 (1,028-1,096) | 128 | 0.9553 | 1.76 | 1.5 (1.4-1.6) | 1.5 (1.4-1.6) | 240 (240-242) | 461 (461-463) | 16: 0.751, 7,235 | 0.9958 |
| ISONGraph Rust exact (exact, recall 1.0) | 3 | 142.6 (114.0-145.7) | - | 1.000 | 10.58 | 0.10 (0.10-0.11) | 0 | - | - | - | 1.000 |
| faiss IndexFlatIP (exact, recall 1.0) | 3 | 117.8 (116.5-135.5) | - | 1.000 | 11.04 | 0.11 (0.10-0.13) | 0 | 174 (174-174) | - | - | 1.000 |

### text384 1,000,000

| system | runs | QPS @0.95 (1T) | knob at first recall>=0.95 | recall at that knob | p99 ms there | build s | of which index s | RSS after build MB | peak MB | first point (knob: recall, QPS) | best recall reached |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ISONGraph Rust HNSW | 1 | 1,813 (1 run) | 192 | 0.9549 | 0.90 | 172.7 (1 run) | 171.6 (1 run) | 1,786 (1 run) | 1,794 (1 run) | 48: 0.834, 4,599 | 0.9901 |
| ISONGraph Rust IVF | 1 | 137 (1 run) | 256 | 0.9526 | 9.32 | 7.6 (1 run) | 6.5 (1 run) | 1,643 (1 run) | 1,886 (1 run) | 16: 0.699, 1,412 | 0.9772 |
| ISONGraph Rust IVF + int8 x4 | 1 | 235 (1 run) | 256 | 0.9532 | 6.22 | 8.1 (1 run) | 6.9 (1 run) | 2,011 (1 run) | 2,255 (1 run) | 64: 0.858, 771 | 0.9780 |
| ISONGraph Rust HNSW + int8 x4 (setting on) | 1 | - | 192 | 0.9549 | 0.83 | 145.9 (1 run) | 144.7 (1 run) | 2,156 (1 run) | 2,164 (1 run) | 192: 0.955, 1,784 | 0.9685 |
| hnswlib | 1 | 1,517 (1 run) | 192 | 0.9552 | 0.99 | 157.2 (1 run) | 0.0 (1 run) | 1,876 (1 run) | 1,906 (1 run) | 24: 0.728, 6,005 | 0.9817 |
| faiss HNSWFlat | 1 | 1,773 (1 run) | 192 | 0.9668 | 1.03 | 189.5 (1 run) | 0.0 (1 run) | 1,815 (1 run) | 2,298 (1 run) | 24: 0.760, 6,168 | 0.9885 |
| usearch | 1 | 612 (1 run) | 192 | 0.9560 | 3.17 | 215.8 (1 run) | 0.0 (1 run) | 1,714 (1 run) | 1,744 (1 run) | 24: 0.726, 2,924 | 0.9818 |
| faiss IVFFlat | 1 | 180 (1 run) | 256 | 0.9561 | 15.10 | 35.3 (1 run) | 35.3 (1 run) | 1,881 (1 run) | 3,588 (1 run) | 16: 0.703, 1,561 | 0.9801 |
| ISONGraph Rust exact (exact, recall 1.0) | 1 | 11.1 (1 run) | - | 1.000 | 99.27 | 1.14 (1 run) | 0 | - | - | - | 1.000 |
| faiss IndexFlatIP (exact, recall 1.0) | 1 | 11.4 (1 run) | - | 1.000 | 95.71 | 1.05 (1 run) | 0 | 1,555 (1 run) | - | - | 1.000 |

### Sweeps (median over runs; recall / QPS 1T)

**text384 100,000**

* ISONGraph Rust HNSW (ef_search): 24: 0.782 / 10,824; 48: 0.885 / 7,752; 64: 0.914 / 4,799; 96: 0.948 / 3,967; 128: 0.963 / 3,526; 192: 0.980 / 2,502; 256: 0.986 / 1,521; 400: 0.994 / 1,051
* ISONGraph Rust IVF (nprobe): 16: 0.750 / 5,848; 32: 0.840 / 3,335; 64: 0.908 / 1,385; 128: 0.956 / 697; 256: 0.984 / 360; 512: 0.995 / 182
* ISONGraph Rust IVF + int8 x4 (nprobe): 16: 0.750 / 7,552; 32: 0.840 / 5,316; 64: 0.908 / 3,251; 128: 0.956 / 1,930; 256: 0.984 / 847; 512: 0.995 / 503
* ISONGraph Rust HNSW + int8 x4 (setting on) (ef_search): 64: 0.914 / 7,034; 96: 0.948 / 4,773; 128: 0.963 / 3,642; 192: 0.980 / 2,728
* hnswlib (ef): 24: 0.786 / 7,766; 32: 0.831 / 6,152; 48: 0.885 / 4,375; 64: 0.914 / 3,843; 96: 0.949 / 2,789; 128: 0.963 / 2,175; 192: 0.980 / 1,572; 256: 0.986 / 1,243; 400: 0.993 / 813
* faiss HNSWFlat (efSearch): 24: 0.811 / 8,382; 32: 0.857 / 8,922; 48: 0.907 / 6,351; 64: 0.936 / 4,365; 96: 0.964 / 3,579; 128: 0.975 / 2,808; 192: 0.987 / 1,945; 256: 0.992 / 1,518; 400: 0.997 / 965
* usearch (expansion_search): 24: 0.781 / 6,608; 32: 0.828 / 4,933; 48: 0.882 / 4,000; 64: 0.915 / 3,141; 96: 0.948 / 2,145; 128: 0.963 / 1,780; 192: 0.978 / 1,183; 256: 0.986 / 893; 400: 0.992 / 617
* faiss IVFFlat (nprobe): 16: 0.751 / 6,404; 32: 0.840 / 3,363; 64: 0.907 / 1,885; 128: 0.955 / 985; 256: 0.986 / 483; 512: 0.996 / 266

**text384 1,000,000**

* ISONGraph Rust HNSW (ef_search): 48: 0.834 / 4,599; 64: 0.868 / 4,306; 96: 0.906 / 2,898; 128: 0.929 / 2,312; 192: 0.955 / 1,714; 256: 0.969 / 1,357; 400: 0.983 / 961; 640: 0.990 / 638
* ISONGraph Rust IVF (nprobe): 16: 0.699 / 1,412; 32: 0.784 / 829; 64: 0.858 / 444; 128: 0.916 / 244; 256: 0.953 / 131; 512: 0.977 / 67
* ISONGraph Rust IVF + int8 x4 (nprobe): 64: 0.858 / 771; 128: 0.915 / 424; 256: 0.953 / 223; 512: 0.978 / 121
* ISONGraph Rust HNSW + int8 x4 (setting on) (ef_search): 192: 0.955 / 1,784; 256: 0.969 / 1,371
* hnswlib (ef): 24: 0.728 / 6,005; 32: 0.774 / 6,181; 48: 0.832 / 4,509; 64: 0.868 / 3,471; 96: 0.905 / 2,659; 128: 0.928 / 1,772; 192: 0.955 / 1,461; 256: 0.967 / 1,136; 400: 0.982 / 760
* faiss HNSWFlat (efSearch): 24: 0.760 / 6,168; 32: 0.802 / 5,394; 48: 0.859 / 4,129; 64: 0.892 / 3,336; 96: 0.929 / 2,386; 128: 0.947 / 1,872; 192: 0.967 / 1,357; 256: 0.977 / 1,117; 400: 0.989 / 709
* usearch (expansion_search): 24: 0.726 / 2,924; 32: 0.773 / 2,408; 48: 0.830 / 1,762; 64: 0.866 / 1,409; 96: 0.910 / 1,051; 128: 0.931 / 803; 192: 0.956 / 561; 256: 0.968 / 432; 400: 0.982 / 288
* faiss IVFFlat (nprobe): 16: 0.703 / 1,561; 32: 0.792 / 905; 64: 0.863 / 567; 128: 0.918 / 283; 256: 0.956 / 165; 512: 0.980 / 100

## Q3. Batch / all-core throughput, text384 100k (32 logical CPUs, 1,000 queries per batch, k=10)

Batch QPS is the median of 3 timed batches after a warm-up batch, at the sweep point nearest recall 0.95 (first knob with recall >= 0.95; its recall is shown). Exact rows are recall 1.0.

| system | runs | knob | recall there | batch QPS (32 thr) | 1T QPS there | batch / 1T |
|---|---|---|---|---|---|---|
| **text384 100,000** | | | | | | |
| ISONGraph Rust exact | 3 | - | 1.000 | 5,320 (5,206-12,743) | 143 (114-146) | 37x |
| faiss IndexFlatIP | 3 | - | 1.000 | 5,085 (3,070-6,552) | 118 (117-136) | 43x |
| numpy brute force | 3 | - | 1.000 | 673 (543-1,052) | 113 (112-121) | 6x |
| ISONGraph Rust IVF | 3 | 128 | 0.956 | 24,360 (21,141-24,903) | 697 (692-889) | 35x |
| ISONGraph Rust IVF + int8 x4 | 1 | 128 | 0.956 | 11,850 (1 run) | 1,930 (1 run) | 6x |
| faiss IVFFlat | 3 | 128 | 0.955 | 3,014 (2,922-3,561) | 985 (955-1,025) | 3x |
| ISONGraph Rust HNSW | 3 | 128 | 0.963 | 13,989 (13,919-14,608) | 3,526 (2,742-3,644) | 4x |
| faiss HNSWFlat | 3 | 96 | 0.964 | 16,694 (14,638-19,818) | 3,579 (2,966-3,622) | 5x |
| hnswlib | 3 | 128 | 0.963 | 12,743 (12,607-12,757) | 2,175 (2,074-2,226) | 6x |
| usearch | 3 | 128 | 0.963 | 13,656 (13,469-15,408) | 1,780 (1,730-1,816) | 8x |
| **text384 1,000,000** | | | | | | |
| ISONGraph Rust exact | 1 | - | 1.000 | 560 (1 run) | 11 (1 run) | 51x |
| faiss IndexFlatIP | 1 | - | 1.000 | 298 (1 run) | 11 (1 run) | 26x |
| numpy brute force | 1 | - | 1.000 | 61 (1 run) | 11 (1 run) | 5x |
| ISONGraph Rust IVF | 1 | 256 | 0.953 | 10,068 (1 run) | 131 (1 run) | 77x |
| ISONGraph Rust IVF + int8 x4 | 1 | 256 | 0.953 | 6,818 (1 run) | 223 (1 run) | 31x |
| faiss IVFFlat | 1 | 256 | 0.956 | 471 (1 run) | 165 (1 run) | 3x |
| ISONGraph Rust HNSW | 1 | 192 | 0.955 | 5,583 (1 run) | 1,714 (1 run) | 3x |
| faiss HNSWFlat | 1 | 192 | 0.967 | 4,967 (1 run) | 1,357 (1 run) | 4x |
| hnswlib | 1 | 192 | 0.955 | 6,680 (1 run) | 1,461 (1 run) | 5x |
| usearch | 1 | 192 | 0.956 | 3,973 (1 run) | 561 (1 run) | 7x |

Rust concurrent clients (each caller runs single-scan-thread searches against one shared store; 3 s per client count, not pinned), QPS:

| cell | mode | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| text384 100,000 | ISONGraph Rust exact | 57 | 83 | 183 | 341 | 604 |
| text384 100,000 | ISONGraph Rust IVF (knob 128) | 364 | 682 | 1,292 | 2,285 | 4,371 |
| text384 100,000 | ISONGraph Rust HNSW (knob 128) | 1,601 | 2,994 | 5,728 | 10,039 | 14,916 |
| text384 1,000,000 | ISONGraph Rust exact | 5 | 8 | 17 | 37 | 60 |
| text384 1,000,000 | ISONGraph Rust IVF (knob 256) | 129 | 216 | 335 | 419 | 385 |
| text384 1,000,000 | ISONGraph Rust HNSW (knob 192) | 1,336 | 3,037 | 3,765 | 5,535 | 5,343 |

## Q4. int8 scan with exact rerank on real embeddings (exact search, Rust)

`recall vs f32 scan` is `int8_recall_report` (int8 top-10 vs the library's own f32 scan, first 1,000 queries; 300 at 1M); `recall vs truth` is against the float64 ground truth in the timed loop. Speedup = int8 QPS / f32 QPS from the same process (1T = pinned single thread; auto = one query at a time on the automatic thread count; batch = 1,000-query batch, 32 threads). Medians over runs where several exist.

| dataset | factor | recall vs f32 scan (mean / worst query) | recall vs truth | f32 1T QPS | int8 1T QPS | 1T speedup | auto-thread speedup | batch speedup | exact scores |
|---|---|---|---|---|---|---|---|---|---|
| text384 100,000 | x1 | 0.9904 / 0.80 | 0.9905 | 117 | 354 | 3.03x | 4.75x | 0.92x | True |
| text384 100,000 | x2 | 1.0000 / 1.00 | 0.9999 | 117 | 359 | 3.07x | 5.23x | 0.86x | True |
| text384 100,000 | x4 | 1.0000 / 1.00 | 0.9999 | 117 | 364 | 3.12x | 5.27x | 0.78x | True |
| text384 100,000 | x8 | 1.0000 / 1.00 | 0.9999 | 117 | 369 | 3.15x | 4.32x | 0.66x | True |
| text384 100,000 | memory | int8 copy 37 MB; working-set delta 38 MB | | | | | | | |
| text384 1,000,000 | x1 | 0.9917 / 0.90 | 0.9907 | 11 | 33 | 2.97x | 3.13x | 0.98x | True |
| text384 1,000,000 | x2 | 1.0000 / 1.00 | 1.0000 | 11 | 36 | 3.25x | 3.17x | 0.98x | True |
| text384 1,000,000 | x4 | 1.0000 / 1.00 | 1.0000 | 11 | 33 | 2.96x | 3.15x | 0.96x | True |
| text384 1,000,000 | x8 | 1.0000 / 1.00 | 1.0000 | 11 | 33 | 2.95x | 3.20x | 0.93x | True |
| text384 1,000,000 | memory | int8 copy 370 MB; working-set delta 371 MB | | | | | | | |
| glove 100,000 | x1 | 0.9832 / 0.80 | 0.9832 | 546 | 1,971 | 3.61x | 3.65x | 0.61x | True |
| glove 100,000 | x2 | 1.0000 / 1.00 | 1.0000 | 546 | 1,947 | 3.57x | 3.33x | 0.51x | True |
| glove 100,000 | x4 | 1.0000 / 1.00 | 1.0000 | 546 | 1,922 | 3.52x | 3.58x | 0.41x | True |
| glove 100,000 | x8 | 1.0000 / 1.00 | 1.0000 | 546 | 1,851 | 3.39x | 2.97x | 0.39x | True |
| glove 100,000 | memory | int8 copy 10 MB; working-set delta 8 MB | | | | | | | |
| sift 100,000 | x1 | 0.9763 / 0.80 | 0.9763 | 389 | 1,206 | 3.10x | 4.07x | 0.82x | True |
| sift 100,000 | x2 | 1.0000 / 1.00 | 1.0000 | 389 | 1,191 | 3.06x | 4.50x | 0.76x | True |
| sift 100,000 | x4 | 1.0000 / 1.00 | 1.0000 | 389 | 1,170 | 3.01x | 4.80x | 0.64x | True |
| sift 100,000 | x8 | 1.0000 / 1.00 | 1.0000 | 389 | 1,177 | 3.02x | 4.31x | 0.52x | True |
| sift 100,000 | memory | int8 copy 13 MB; working-set delta 12 MB | | | | | | | |

IVF and HNSW combined with int8 (setting `ScanPrecision::Int8 { rerank_factor: 4 }` before the build) are in the Q2 table (`+ int8 x4` rows) with the sweeps above.

## Q5. Persistence: ISONGraph JSON sidecar vs binary snapshot vs faiss / hnswlib / usearch index files

ISONGraph = store only (vectors + keys), index not persisted. `ready` for the ANN rows adds the index rebuild (and a first query) that every ISONGraph load needs. Library rows are native `write_index` / `save_index` / `load`; their load time is to a usable index (the graph is inside the file).

| N | system | save s | file MB | load s | first query after load ms | time to ready s | note |
|---|---|---|---|---|---|---|---|
| 100,000 | ISONGraph Rust JSON sidecar | 1.67 | 453 | 1.27 (runs 1.27, 1.27, 1.28) | 3.8 | 1.28 | exact search ready; answers equal original: True |
| 100,000 | ISONGraph Rust binary snapshot | 0.19 (+0.00 sidecar fingerprint) | 149 | 0.19 (runs 0.19, 0.19, 0.20) | 4.0 | 0.20 | exact search ready; answers equal original: True |
| 100,000 | ISONGraph Rust snapshot + HNSW rebuild | - | - | 0.19 | 0.2 | 6.12 | load + build 5.9 s + first query |
| 100,000 | ISONGraph Rust snapshot + IVF rebuild | - | - | 0.20 | 0.6 | 0.71 | load + build 0.5 s + first query |
| 100,000 | faiss IVFFlat | 0.061 (0.046-0.098) | 149 | 0.094 (0.087-0.170) | 1.24 | 0.095 | native index file |
| 100,000 | faiss HNSWFlat | 0.134 (0.128-0.253) | 176 | 0.091 (0.070-0.158) | 0.46 | 0.091 | native index file |
| 100,000 | hnswlib | 0.312 (0.290-0.367) | 177 | 0.595 (0.572-0.605) | 1.02 | 0.596 | native index file |
| 100,000 | usearch | 0.182 (0.160-0.331) | 177 | 0.139 (0.135-0.231) | 0.98 | 0.140 | native index file |
| 1,000,000 | ISONGraph Rust JSON sidecar | 42.86 | 4,534 | 30.12 (runs 30.12, 30.42, 26.38) | 50.3 | 30.17 | exact search ready; answers equal original: True |
| 1,000,000 | ISONGraph Rust binary snapshot | 7.68 (+0.00 sidecar fingerprint) | 1,489 | 4.56 (runs 4.40, 4.69, 4.56) | 43.6 | 4.60 | exact search ready; answers equal original: True |
| 1,000,000 | ISONGraph Rust snapshot + IVF rebuild | - | - | 4.56 | 31.1 | 21.62 | load + build 17.0 s + first query |
| 1,000,000 | faiss IVFFlat | 0.475 (1 run) | 1,478 | 0.848 (1 run) | 5.81 | 0.854 | native index file |
| 1,000,000 | faiss HNSWFlat | 1.154 (1 run) | 1,618 | 0.888 (1 run) | 1.08 | 0.889 | native index file |
| 1,000,000 | hnswlib | 2.529 (1 run) | 1,626 | 6.048 (1 run) | 3.08 | 6.051 | native index file |
| 1,000,000 | usearch | 3.247 (1 run) | 1,623 | 1.283 (1 run) | 1.80 | 1.285 | native index file |

## Q6. Writes and the first query after writes (text384; index built, recall ~0.95 operating point)

10k inserts, then 10k updates (new vector under an existing id), then 10k deletes, one call at a time on the built index (Rust: `add` / `remove`; libraries as in the competitive harness: calls of 1,000). Then the first search and recall@10 against the exact post-mutation float64 truth at the same knob. `p99/max` are per-write latencies (Rust only).

| N | system | knob | insert/s | update/s | delete/s | insert p99 / max ms | update p99 / max ms | first query after ms | recall before | recall after | QPS 1T after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100,000 | ISONGraph Rust IVF (1) | 128 | 156,140 | 128,696 | 911,295 | 0.01 / 2.1 | 0.01 / 2.2 | 3.8 | 0.9559 | 0.9529 | 857 |
| 100,000 | ISONGraph Rust HNSW (1) | 128 | 944 | 936 | 1,069,039 | 1.89 / 2.8 | 1.95 / 3.4 | 0.7 | 0.9626 | 0.9582 | 3,164 |
| 100,000 | faiss IVFFlat (3) | 128 | 235,791 | 263,826 | 2,687,522 | - | - | 1.0 | 0.9553 | 0.9521 | 1,185 |
| 100,000 | hnswlib (3) | 128 | 6,438 | 1,567 | 375,038 | - | - | 1.0 | 0.9631 | 0.9675 | 1,780 |
| 100,000 | usearch (3) | 128 | 7,693 | 7,682 | 3,068,897 | - | - | 3.7 | 0.9634 | 0.9573 | 1,494 |
| 100,000 | faiss HNSWFlat (3) | 96 | 4,352 | - (unsupported) | - (unsupported) | - | - | - | 0.9639 | - | - |
| 1,000,000 | ISONGraph Rust IVF (1) | 256 | 137,936 | 133,013 | 1,097,249 | 0.01 / 2.6 | 0.01 / 3.0 | 8.0 | 0.9526 | 0.9530 | 129 |
| 1,000,000 | ISONGraph Rust HNSW (1) | 192 | 1,393 | 1,440 | 1,382,170 | 1.49 / 4.7 | 1.35 / 6.6 | 0.6 | 0.9549 | 0.9550 | 1,603 |
| 1,000,000 | faiss IVFFlat (1) | 256 | 86,802 | 68,771 | 1,236,033 | - | - | 8.1 | 0.9561 | 0.9571 | 154 |
| 1,000,000 | hnswlib (1) | 192 | 6,005 | 1,696 | 355,494 | - | - | 1.3 | 0.9552 | 0.9557 | 1,377 |
| 1,000,000 | usearch (1) | 192 | 3,789 | 3,414 | 1,913,729 | - | - | 27.8 | 0.9560 | 0.9562 | 555 |
| 1,000,000 | faiss HNSWFlat (1) | 192 | 2,347 | - (unsupported) | - (unsupported) | - | - | - | 0.9668 | - | - |

## Q7. Memory at 1M x 384 (working set, MB; baseline subtracted)

Raw vectors = 1,000,000 x 384 x 4 = 1,465 MB. Each Rust row is its own process (ingest, then that mode only). Python rows: working set after the index is built and again after the sweep, with the harness' preallocation (hnswlib `max_elements` = N + 20k).

| system | after ingest | after build / first query | peak during build | after the sweep | x raw vectors (after build) |
|---|---|---|---|---|---|
| **text384 100,000** (raw 146 MB) | | | | | |
| ISONGraph Rust exact | 161 | 164 | 188 | 166 | 1.12x |
| ISONGraph Rust IVF | 161 | 166 | 246 | 174 | 1.14x |
| ISONGraph Rust HNSW | 161 | 199 | 201 | 204 | 1.36x |
| ISONGraph Rust IVF + int8 x4 | 161 | 205 | 282 | 213 | 1.40x |
| ISONGraph Rust HNSW + int8 x4 (setting on) | 161 | 238 | 239 | 240 | 1.62x |
| faiss IndexFlatIP | - | 174 | 319 | 185 | 1.19x |
| numpy brute force | - | 159 | 191 | 168 | 1.09x |
| faiss IVFFlat | - | 240 | 461 | 232 | 1.64x |
| faiss HNSWFlat | - | 207 | 348 | 209 | 1.41x |
| hnswlib | - | 201 | 231 | 204 | 1.37x |
| usearch | - | 182 | 211 | 184 | 1.24x |
| ISONGraph Rust exact + int8 copy (any factor) | - | 203 | - | - | 1.39x |
| **text384 1,000,000** (raw 1,465 MB) | | | | | |
| ISONGraph Rust exact | 1,621 | 1,637 | 1,649 | 1,642 | 1.12x |
| ISONGraph Rust IVF | 1,621 | 1,643 | 1,886 | 1,653 | 1.12x |
| ISONGraph Rust HNSW | 1,621 | 1,786 | 1,794 | 1,795 | 1.22x |
| ISONGraph Rust IVF + int8 x4 | 1,621 | 2,011 | 2,255 | 2,058 | 1.37x |
| ISONGraph Rust HNSW + int8 x4 (setting on) | 1,622 | 2,156 | 2,164 | 2,156 | 1.47x |
| faiss IndexFlatIP | - | 1,555 | 2,124 | 1,565 | 1.06x |
| numpy brute force | - | 1,576 | 1,605 | 1,584 | 1.08x |
| faiss IVFFlat | - | 1,882 | 3,588 | 1,884 | 1.28x |
| faiss HNSWFlat | - | 1,816 | 2,298 | 1,827 | 1.24x |
| hnswlib | - | 1,876 | 1,906 | 1,878 | 1.28x |
| usearch | - | 1,714 | 1,744 | 1,716 | 1.17x |
| ISONGraph Rust exact + int8 copy (any factor) | - | 1,998 | - | - | 1.36x |

Rust library-reported breakdown (`memory_stats`, 1M):

* ISONGraph Rust IVF: matrix 1,466 MB, metadata 124 MB, index 17 MB, int8 copy 0 MB
* ISONGraph Rust HNSW: matrix 1,466 MB, metadata 124 MB, index 139 MB, int8 copy 0 MB
* ISONGraph Rust IVF + int8 x4: matrix 1,466 MB, metadata 124 MB, index 17 MB, int8 copy 370 MB
* ISONGraph Rust HNSW + int8 x4 (setting on): matrix 1,466 MB, metadata 124 MB, index 139 MB, int8 copy 370 MB

## Appendix: noisy runs and machine load at job start

| job file | total CPU % | pinned-CPU % | flagged noisy |
|---|---|---|---|
| faiss_flat_glove_100000_r1.json | 19 | 77 | True |
| faiss_flat_glove_100000_r3.json | 18 | 95 | True |
| faiss_flat_sift_100000_r2.json | 17 | 96 | True |
| faiss_hnsw_text384_100000_r2.json | 13 | 69 | True |
| faiss_ivfflat_text384_100000_r2.json | 19 | 69 | True |
| faiss_ivfflat_text384_100000_r3.json | 21 | 72 | True |
| numpy_bf_glove_100000_r3.json | 25 | 7 | False |
| rust_exact_text384_100000_r3.json | 18 | 83 | True |
| rust_hnsw_i8x4_text384_100000_r1.json | 12 | 75 | True |
| usearch_text384_100000_r3.json | 11 | 62 | True |


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

## Limitations, caveats, and what was dropped

* **Not a quiet machine.** See "Machine state". Single-thread QPS of the same cell moved by up to 25% between runs (Rust exact text384 100k: 114 / 146 / 143 QPS; batch QPS of the same cell moved up to 2.5x). Treat differences below ~15% as parity. Where the table shows one run, there is one run.
* **Noisy-start jobs were re-run once.** 22 of 48 jobs in the first pass of the 100k cells started while CPU 8 showed > 15% load from desktop processes; all 100k jobs flagged that way were moved to `raw/discarded_noisy/` (kept, not used) and re-run once. Some re-runs were flagged again (9 of 60 jobs overall are still flagged; list in the appendix of the tables). The four flagged 1M jobs (Rust HNSW, Rust IVF, Rust IVF+int8, faiss HNSW) were re-run once as well; the first attempts are in `raw/discarded_noisy/` and differ a lot from the kept ones (Rust HNSW 1M QPS@0.95 1,531 vs 1,813; HNSW insert rate 650 vs 1,393/s; Rust IVF batch 4,524 vs 10,068 QPS; IVF build 18.8 vs 7.6 s), which is itself a measure of the noise on this machine: single-run 1M cells can move by 20-100% for batch, build and write rates.
* **1M cells are single runs**, 100k text384 cells are 3 process-level runs (int8 study, concurrent clients, 10k-write test: 1 run each, their re-runs were not repeated; the text384 100k int8 numbers come from one run). GloVe and SIFT at 1M were not run (not asked, dropped); chromadb, lancedb, sqlite-vec and qdrant-local were not re-run (outside the questions of this stage).
* **Libraries are driven from Python** (per-call overhead in every library one-query number, none in Rust). hnswlib/usearch/faiss writes were done in batches of 1,000 with parallel threads (the competitive harness), Rust writes are one `add` call at a time on one thread: the write table is not like for like and no per-call hnswlib/usearch measurement was made.
* **Recall at equal knobs differs between graph libraries** (faiss reaches 0.963 at ef 96, hnswlib and Rust at ef 128); that is why every comparison is at interpolated recall 0.95, not at equal ef. Interpolation brackets are as wide as the sweep (ef 24-400/640); at 1M the HNSW + int8 row has no bracket (both measured points are above 0.95) and is read from the ef-192 point (identical recall to the f32 run).
* **HNSW time-to-ready at 1M** after a snapshot load was not run as one measurement (it would repeat the 151 s build); the figure in the verdict is load + build measured separately. IVF 1M time-to-ready (21.6 s) was measured; its 17 s build there was slower than the 7-8 s IVF build in Stage A, and the 18.8 s build in `core` (one run each): IVF build time at 1M varies by more than 2x on this machine.
* **Build-time comparisons use 32 threads for everyone**; `faiss HNSWFlat` has no update/delete (shown as unsupported), memory of the libraries includes the harness' preallocation (N + 20k).
* The Python package versions differ from `benchmarks/competitive/results/environment.json`, so these numbers are not directly comparable with the earlier README's tables.
* `HNSW + int8` is reported as measured: accepted, no effect on search (identical recall at identical ef), memory of the int8 copy wasted.

## Reproduce

```text
cd isongraph-vector-rs
set CARGO_TARGET_DIR=D:/Goldmine/ISON/bench_data/rs_target_stagec
cargo build --release --example stage_c_bench
cd ../benchmarks/rust-stage-c
python collect_env_c.py                                   # machine state -> environment.json
python run_stage_c.py --phase 1 --phase 2 --phase 4       # 100k cells (3 interleaved rounds), glove/sift exact, 100k persistence
python run_stage_c.py --phase 3                           # text384 1M, one run per cell (about 70 minutes)
python rerun_noisy.py && python run_stage_c.py --phase 1 --phase 2 --phase 4   # repeat jobs that started under load
python summarize.py && python make_readme.py              # raw/*.json -> tables.md, summary.json, README.md
```
Files: `isongraph-vector-rs/examples/stage_c_bench.rs` (Rust harness), `bench_c.py` (library harness wrapper), `run_stage_c.py` (driver, load gating),
`rerun_noisy.py`, `summarize.py`, `make_readme.py`, `collect_env_c.py`, `environment.json`, `raw/` (one JSON + stdout + load record per run;
`raw/discarded_noisy/` holds the first attempts that were replaced), `tables.md`, `summary.json`.
