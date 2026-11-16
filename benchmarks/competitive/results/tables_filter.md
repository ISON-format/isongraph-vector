#### glove, N = 100,000

| system | build with metadata s | 10% subset: QPS 1T @ recall>=0.90 | max recall | 1% subset: QPS 1T @ recall>=0.90 | max recall |
|---|---|---|---|---|---|
| **ISONGraph C++ (IVF)** | 0.354 | 152 @ 1.000 | 1.000 | 316 @ 1.000 | 1.000 |
| **ISONGraph Python (exact)** | 0.952 | 368 @ 1.000 | 1.000 | 409 @ 1.000 | 1.000 |
| **ISONGraph Python (IVF)** | 1.78 | 366 @ 1.000 | 1.000 | 400 @ 1.000 | 1.000 |
| **ISONGraph Rust (IVF)** | 2.13 | 8377 @ 1.000 | 1.000 | 18699 @ 1.000 | 1.000 |
| chromadb (HNSW) | 11.1 | 4.23 @ 0.912 | 1.000 | 28.0 @ 0.936 | 1.000 |
| faiss IndexFlatIP | 0.036 | 1813 @ 1.000 | 1.000 | 2913 @ 1.000 | 1.000 |
| faiss HNSWFlat | 3.85 | 534 @ 0.957 | 0.957 | - (max recall 0.835 at 541 QPS) | 0.835 |
| faiss IVFFlat | 1.03 | 1083 @ 1.000 | 1.000 | 1910 @ 1.000 | 1.000 |
| hnswlib | 1.31 | 1354 @ 0.935 | 1.000 | 449 @ 0.951 | 1.000 |
| lancedb (no index) | 0.081 | 33.8 @ 1.000 | 1.000 | 37.6 @ 1.000 | 1.000 |
| lancedb IVF_FLAT | 0.740 | 88.3 @ 1.000 | 1.000 | 99.0 @ 1.000 | 1.000 |
| numpy brute force | 0.012 | 552 @ 1.000 | 1.000 | 624 @ 1.000 | 1.000 |
| sqlite-vec (vec0, exact) | 0.579 | 46.1 @ 1.000 | 1.000 | 48.0 @ 1.000 | 1.000 |
| usearch (HNSW) | 1.57 | unsupported | - | unsupported | - |

#### text384, N = 100,000

| system | build with metadata s | 10% subset: QPS 1T @ recall>=0.90 | max recall | 1% subset: QPS 1T @ recall>=0.90 | max recall |
|---|---|---|---|---|---|
| **ISONGraph C++ (IVF)** | 0.986 | 116 @ 1.000 | 1.000 | 345 @ 1.000 | 1.000 |
| **ISONGraph Python (exact)** | 1.97 | 121 @ 1.000 | 1.000 | 131 @ 1.000 | 1.000 |
| **ISONGraph Python (IVF)** | 3.62 | 122 @ 1.000 | 1.000 | 130 @ 1.000 | 1.000 |
| **ISONGraph Rust (IVF)** | 4.77 | 3094 @ 1.000 | 1.000 | 15288 @ 1.000 | 1.000 |
| chromadb (HNSW) | 16.5 | 2.24 @ 0.953 | 1.000 | 20.9 @ 0.971 | 1.000 |
| faiss IndexFlatIP | 0.109 | 464 @ 1.000 | 1.000 | 2491 @ 1.000 | 1.000 |
| faiss HNSWFlat | 13.7 | 1279 @ 0.944 | 0.992 | 377 @ 0.928 | 0.928 |
| faiss IVFFlat | 1.69 | 388 @ 1.000 | 1.000 | 1630 @ 1.000 | 1.000 |
| hnswlib | 6.44 | 1779 @ 0.912 | 1.000 | 259 @ 0.981 | 1.000 |
| lancedb (no index) | 0.217 | 7.09 @ 1.000 | 1.000 | 7.28 @ 1.000 | 1.000 |
| lancedb IVF_FLAT | 1.94 | 73.1 @ 1.000 | 1.000 | 97.8 @ 1.000 | 1.000 |
| numpy brute force | 0.030 | 141 @ 1.000 | 1.000 | 146 @ 1.000 | 1.000 |
| sqlite-vec (vec0, exact) | 1.01 | 13.0 @ 1.000 | 1.000 | 13.3 @ 1.000 | 1.000 |
| usearch (HNSW) | 5.68 | unsupported | - | unsupported | - |

#### text384, N = 1,000,000

| system | build with metadata s | 10% subset: QPS 1T @ recall>=0.90 | max recall | 1% subset: QPS 1T @ recall>=0.90 | max recall |
|---|---|---|---|---|---|
| **ISONGraph C++ (IVF)** | 12.6 | 308 @ 0.943 | 1.000 | 506 @ 0.905 | 1.000 |
| **ISONGraph Python (IVF)** | 117 | 8.64 @ 1.000 | 1.000 | 9.52 @ 1.000 | 1.000 |
| **ISONGraph Rust (IVF)** | 69.5 | 650 @ 0.940 | 1.000 | 1944 @ 0.906 | 1.000 |
| faiss IndexFlatIP | 1.13 | 39.7 @ 1.000 | 1.000 | 172 @ 1.000 | 1.000 |
| faiss HNSWFlat | 268 | 267 @ 0.973 | 0.973 | - (max recall 0.854 at 275 QPS) | 0.854 |
| faiss IVFFlat | 47.0 | 38.1 @ 1.000 | 1.000 | 140 @ 1.000 | 1.000 |
| hnswlib | 151 | 406 @ 0.955 | 0.999 | 135 @ 0.945 | 1.000 |
| lancedb IVF_FLAT | 25.3 | 8.73 @ 1.000 | 1.000 | 13.1 @ 1.000 | 1.000 |
| numpy brute force | 0.300 | 12.9 @ 1.000 | 1.000 | 14.0 @ 1.000 | 1.000 |

