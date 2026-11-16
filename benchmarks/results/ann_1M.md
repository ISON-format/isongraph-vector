# Approximate index (IVF-flat) benchmark

Machine: 32 logical CPUs, 47.7 GB RAM, Windows-11-10.0.26300-SP0, Python 3.12.7, numpy 2.4.2. 384-d unit vectors, top_k=10, 200 queries per cell, warm single-thread medians. Synthetic clustered data; see `benchmarks/README.md` for what that does and does not show.

## clustered, N = 1,000,000

* exact scan: first query 4.04 s (builds the cache), warm median 53.73 ms, p95 112.93 ms
* index build: 23.3 s for nlist=4000 (automatic nprobe=200); index itself 14.0 MB on top of 1465 MB of vectors; RSS growth +24 MB (peak during the build +568 MB)
* ingest 90,279 vectors/s (unchanged by `ann`: the index is built lazily)

| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.977 | 1.00 | 0.00 | 0.33 | 0.63 | 162.8x |
| 2 | 0.1 | 0.988 | 1.00 | 0.20 | 0.44 | 0.71 | 122.1x |
| 4 | 0.1 | 0.989 | 1.00 | 0.20 | 0.57 | 0.72 | 94.3x |
| 8 | 0.2 | 0.990 | 1.00 | 0.20 | 0.87 | 1.4 | 61.8x |
| 16 | 0.4 | 0.990 | 1.00 | 0.30 | 1.36 | 2.13 | 39.5x |
| 32 | 0.8 | 0.995 | 1.00 | 0.50 | 2.32 | 2.94 | 23.2x |
| 64 | 1.6 | 0.995 | 1.00 | 0.50 | 4.25 | 6.78 | 12.6x |
| 128 | 3.2 | 0.996 | 1.00 | 0.50 | 7.51 | 11.62 | 7.2x |
| 200 (auto) | 5.0 | 0.997 | 1.00 | 0.80 | 11.36 | 15.15 | 4.7x |
| 256 | 6.4 | 0.997 | 1.00 | 0.80 | 14.35 | 18.04 | 3.7x |

## diffuse, N = 1,000,000

* exact scan: first query 3.58 s (builds the cache), warm median 48.17 ms, p95 55.84 ms
* index build: 23.72 s for nlist=4000 (automatic nprobe=200); index itself 14.0 MB on top of 1465 MB of vectors; RSS growth +14 MB (peak during the build +559 MB)
* ingest 71,175 vectors/s (unchanged by `ann`: the index is built lazily)

| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.030 | 0.00 | 0.00 | 0.64 | 1.16 | 75.3x |
| 2 | 0.1 | 0.055 | 0.00 | 0.00 | 1.32 | 2.03 | 36.5x |
| 4 | 0.1 | 0.095 | 0.00 | 0.00 | 1.91 | 2.91 | 25.2x |
| 8 | 0.2 | 0.168 | 0.00 | 0.00 | 2.9 | 4.1 | 16.6x |
| 16 | 0.4 | 0.284 | 0.10 | 0.00 | 4.78 | 6.68 | 10.1x |
| 32 | 0.8 | 0.468 | 0.20 | 0.10 | 10.96 | 14.11 | 4.4x |
| 64 | 1.6 | 0.704 | 0.50 | 0.30 | 11.83 | 14.38 | 4.1x |
| 128 | 3.2 | 0.930 | 0.80 | 0.60 | 16.6 | 19.57 | 2.9x |
| 200 (auto) | 5.0 | 0.994 | 0.90 | 0.90 | 24.27 | 36.13 | 2.0x |
| 256 | 6.4 | 0.995 | 0.90 | 0.90 | 26.56 | 40.09 | 1.8x |
