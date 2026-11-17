
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
