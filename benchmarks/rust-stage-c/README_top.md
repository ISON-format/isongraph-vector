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

