# bench/ann.mjs output

One run of `npm run bench:ann` (JavaScript build; the TypeScript build is the same code). Synthetic clustered unit vectors, 384-d, k=10, 100 queries, single thread, while other jobs shared the machine, so treat timings as indicative. Real embeddings cluster differently: run `recallCheck()` on your own queries.

## Run: exact vs IVF (js), dim 384, k 10, 100 queries, warm single-thread
node v24.18.0, 32 logical CPUs, 47.7 GB RAM, heap limit 19.7 GB

## N = 10,000
ingest 66276 vectors/s
exact: median 5.10 ms, p95 5.96 ms
index build 7.9 s: nlist 400, automatic nprobe 20, index 0.7 MB + float32 copy 16 MB

| nprobe | % of cells | recall@k mean | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|
| 1 | 0.3 | 0.581 | 0.10 | 0.21 | 0.25 | 24.2x |
| 4 | 1.0 | 0.984 | 0.80 | 0.23 | 0.27 | 22.3x |
| 16 | 4.0 | 1.000 | 1.00 | 0.33 | 0.43 | 15.6x |
| 20 (auto) | 5.0 | 1.000 | 1.00 | 0.32 | 0.39 | 16.0x |
| 64 | 16.0 | 1.000 | 1.00 | 0.72 | 0.89 | 7.0x |

## N = 100,000
ingest 58092 vectors/s
exact: median 54.5 ms, p95 60.8 ms
index build 53.7 s: nlist 1265, automatic nprobe 63, index 2.6 MB + float32 copy 157 MB

| nprobe | % of cells | recall@k mean | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.944 | 0.20 | 0.47 | 0.53 | 116.8x |
| 4 | 0.3 | 1.000 | 1.00 | 0.58 | 0.65 | 94.4x |
| 16 | 1.3 | 1.000 | 1.00 | 1.09 | 1.36 | 49.9x |
| 63 (auto) | 5.0 | 1.000 | 1.00 | 2.97 | 3.31 | 18.3x |
| 64 | 5.1 | 1.000 | 1.00 | 2.68 | 3.28 | 20.3x |

## N = 1,000,000
ingest 57685 vectors/s
exact: median 543 ms, p95 574 ms
index build 502.7 s: nlist 4000, automatic nprobe 200, index 13.7 MB + float32 copy 1570 MB

| nprobe | % of cells | recall@k mean | worst | median ms | p95 ms | speed-up |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 0.753 | 0.00 | 1.43 | 1.57 | 380.2x |
| 4 | 0.1 | 0.803 | 0.00 | 1.81 | 2.01 | 299.7x |
| 16 | 0.4 | 0.861 | 0.10 | 3.41 | 4.10 | 159.3x |
| 64 | 1.6 | 0.904 | 0.20 | 9.49 | 10.5 | 57.2x |
| 200 (auto) | 5.0 | 0.939 | 0.40 | 27.3 | 29.6 | 19.9x |

