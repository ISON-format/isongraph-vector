# Approximate index (IVF-flat) benchmark

Machine: 32 logical CPUs, 47.7 GB RAM, Windows-11-10.0.26300-SP0, Python 3.12.7, numpy 2.4.2. 384-d unit vectors, top_k=10, 200 queries per cell, warm single-thread medians. Synthetic clustered data; see `benchmarks/README.md` for what that does and does not show.

## clustered, N = 20,000

* exact scan: first query 0.07 s (builds the cache), warm median 0.64 ms, p95 1.35 ms
* index build: 0.3 s for nlist=566 (automatic nprobe=28); index itself 1.0 MB on top of 29 MB of vectors; RSS growth +25 MB (peak during the build +128 MB)
* ingest 109,152 vectors/s (unchanged by `ann`: the index is built lazily)

| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|---|
| 1 | 0.2 | 0.648 | 0.20 | 0.10 | 0.07 | 0.09 | 9.1x |
| 2 | 0.4 | 0.897 | 0.60 | 0.40 | 0.07 | 0.1 | 9.1x |
| 4 | 0.7 | 0.993 | 1.00 | 0.60 | 0.09 | 0.12 | 7.1x |
| 8 | 1.4 | 1.000 | 1.00 | 1.00 | 0.13 | 0.16 | 4.9x |
| 16 | 2.8 | 1.000 | 1.00 | 1.00 | 0.17 | 0.23 | 3.8x |
| 28 (auto) | 4.9 | 1.000 | 1.00 | 1.00 | 0.24 | 0.29 | 2.7x |
| 32 | 5.7 | 1.000 | 1.00 | 1.00 | 0.29 | 0.5 | 2.2x |
| 64 | 11.3 | 1.000 | 1.00 | 1.00 | 0.46 | 0.67 | 1.4x |
| 128 | 22.6 | 1.000 | 1.00 | 1.00 | 0.72 | 1.02 | 0.9x |
| 256 | 45.2 | 1.000 | 1.00 | 1.00 | 1.3 | 1.77 | 0.5x |

Under writes (automatic nprobe, recall measured against the exact scan of the current data): +1% new vectors and -1% deletes, new rows still pending (200 scanned exactly): recall 1.0, 0.26 ms. A further +2% crosses the fold threshold and the next search folds them into cells (0 pending): recall 1.0, 0.23 ms. A further +60% pushes churn past 50% of the trained size and retrains (retrained=True): that one query took 0.52 s; afterwards nlist=720, recall 1.0, 0.45 ms.

## diffuse, N = 20,000

* exact scan: first query 0.07 s (builds the cache), warm median 0.62 ms, p95 1.23 ms
* index build: 0.38 s for nlist=566 (automatic nprobe=28); index itself 1.0 MB on top of 29 MB of vectors; RSS growth +2 MB (peak during the build +105 MB)
* ingest 111,637 vectors/s (unchanged by `ann`: the index is built lazily)

| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|---|
| 1 | 0.2 | 0.663 | 0.20 | 0.10 | 0.07 | 0.12 | 8.9x |
| 2 | 0.4 | 0.762 | 0.30 | 0.10 | 0.08 | 0.12 | 7.8x |
| 4 | 0.7 | 0.824 | 0.50 | 0.10 | 0.09 | 0.12 | 6.9x |
| 8 | 1.4 | 0.900 | 0.70 | 0.40 | 0.13 | 0.29 | 4.8x |
| 16 | 2.8 | 0.961 | 0.80 | 0.70 | 0.18 | 0.21 | 3.4x |
| 28 (auto) | 4.9 | 0.991 | 0.90 | 0.80 | 0.24 | 0.27 | 2.6x |
| 32 | 5.7 | 0.994 | 0.90 | 0.90 | 0.26 | 0.33 | 2.4x |
| 64 | 11.3 | 0.997 | 1.00 | 0.90 | 0.42 | 0.57 | 1.5x |
| 128 | 22.6 | 1.000 | 1.00 | 0.90 | 0.75 | 1.47 | 0.8x |
| 256 | 45.2 | 1.000 | 1.00 | 0.90 | 1.34 | 1.79 | 0.5x |

Under writes (automatic nprobe, recall measured against the exact scan of the current data): +1% new vectors and -1% deletes, new rows still pending (200 scanned exactly): recall 0.991, 0.26 ms. A further +2% crosses the fold threshold and the next search folds them into cells (0 pending): recall 0.9905, 0.25 ms. A further +60% pushes churn past 50% of the trained size and retrains (retrained=True): that one query took 0.56 s; afterwards nlist=720, recall 0.9985, 0.4 ms.

## clustered, N = 100,000

* exact scan: first query 0.3 s (builds the cache), warm median 4.63 ms, p95 9.5 ms
* index build: 1.4 s for nlist=1265 (automatic nprobe=63); index itself 2.7 MB on top of 146 MB of vectors; RSS growth +7 MB (peak during the build +221 MB)
* ingest 104,546 vectors/s (unchanged by `ann`: the index is built lazily)

| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.948 | 0.60 | 0.20 | 0.29 | 0.48 | 16.0x |
| 2 | 0.2 | 0.997 | 1.00 | 0.50 | 0.22 | 0.27 | 21.0x |
| 4 | 0.3 | 1.000 | 1.00 | 1.00 | 0.26 | 0.31 | 17.8x |
| 8 | 0.6 | 1.000 | 1.00 | 1.00 | 0.33 | 0.4 | 14.0x |
| 16 | 1.3 | 1.000 | 1.00 | 1.00 | 0.48 | 0.58 | 9.6x |
| 32 | 2.5 | 1.000 | 1.00 | 1.00 | 0.76 | 0.94 | 6.1x |
| 63 (auto) | 5.0 | 1.000 | 1.00 | 1.00 | 1.22 | 1.65 | 3.8x |
| 64 | 5.1 | 1.000 | 1.00 | 1.00 | 1.18 | 1.35 | 3.9x |
| 128 | 10.1 | 1.000 | 1.00 | 1.00 | 2.07 | 2.32 | 2.2x |
| 256 | 20.2 | 1.000 | 1.00 | 1.00 | 3.9 | 5.59 | 1.2x |

Under writes (automatic nprobe, recall measured against the exact scan of the current data): +1% new vectors and -1% deletes, new rows still pending (1000 scanned exactly): recall 1.0, 1.31 ms. A further +2% crosses the fold threshold and the next search folds them into cells (0 pending): recall 1.0, 1.26 ms. A further +60% pushes churn past 50% of the trained size and retrains (retrained=True): that one query took 2.37 s; afterwards nlist=1610, recall 1.0, 1.88 ms.

## diffuse, N = 100,000

* exact scan: first query 0.29 s (builds the cache), warm median 4.99 ms, p95 7.47 ms
* index build: 1.38 s for nlist=1265 (automatic nprobe=63); index itself 2.7 MB on top of 146 MB of vectors; RSS growth +5 MB (peak during the build +219 MB)
* ingest 115,154 vectors/s (unchanged by `ann`: the index is built lazily)

| nprobe | % of cells | recall@10 mean | p5 | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.229 | 0.00 | 0.00 | 0.26 | 0.46 | 19.2x |
| 2 | 0.2 | 0.299 | 0.00 | 0.00 | 0.24 | 0.32 | 20.8x |
| 4 | 0.3 | 0.383 | 0.00 | 0.00 | 0.29 | 0.39 | 17.2x |
| 8 | 0.6 | 0.503 | 0.20 | 0.00 | 0.4 | 0.5 | 12.5x |
| 16 | 1.3 | 0.668 | 0.30 | 0.20 | 0.6 | 1.06 | 8.3x |
| 32 | 2.5 | 0.876 | 0.70 | 0.50 | 0.86 | 1.03 | 5.8x |
| 63 (auto) | 5.0 | 0.987 | 0.90 | 0.80 | 1.23 | 1.44 | 4.1x |
| 64 | 5.1 | 0.988 | 0.90 | 0.80 | 1.21 | 1.42 | 4.1x |
| 128 | 10.1 | 0.996 | 1.00 | 0.80 | 2.2 | 2.56 | 2.3x |
| 256 | 20.2 | 0.998 | 1.00 | 0.90 | 4.13 | 4.7 | 1.2x |

Under writes (automatic nprobe, recall measured against the exact scan of the current data): +1% new vectors and -1% deletes, new rows still pending (1000 scanned exactly): recall 0.986, 1.39 ms. A further +2% crosses the fold threshold and the next search folds them into cells (0 pending): recall 0.987, 1.32 ms. A further +60% pushes churn past 50% of the trained size and retrains (retrained=True): that one query took 2.2 s; afterwards nlist=1610, recall 0.9855, 1.98 ms.
