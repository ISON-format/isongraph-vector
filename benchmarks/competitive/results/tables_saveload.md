| dataset | N | system | save s | on-disk MB | load s | first query after load ms | time to ready s |
|---|---|---|---|---|---|---|---|
| glove | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 4.26 | 133 | 2.02 | 236 | 2.25 |
| glove | 100,000 | **ISONGraph C++ (IVF)** | 4.40 | 133 | 2.03 | 302 | 2.33 |
| glove | 100,000 | **ISONGraph C# (IVF)** | 1.36 | 121 | 1.27 | 333 | 1.61 |
| glove | 100,000 | **ISONGraph Python (exact)** | 4.35 | 215 | 3.37 | 173 | 3.54 |
| glove | 100,000 | **ISONGraph Python (IVF)** | 4.42 | 215 | 3.41 | 2017 | 5.43 |
| glove | 100,000 | **ISONGraph Rust (IVF)** | 0.754 | 121 | 0.414 | 237 | 0.651 |
| glove | 100,000 | chromadb (HNSW) | 0.000 | 76.3 | 0.012 | 87.3 | 0.099 |
| glove | 100,000 | faiss IndexFlatIP | 0.045 | 38.9 | 0.025 | 2.13 | 0.027 |
| glove | 100,000 | faiss HNSWFlat | 0.052 | 57.1 | 0.029 | 1.18 | 0.031 |
| glove | 100,000 | faiss IVFFlat | 0.020 | 39.4 | 0.027 | 0.713 | 0.028 |
| glove | 100,000 | hnswlib | 0.036 | 57.5 | 0.142 | 0.708 | 0.142 |
| glove | 100,000 | lancedb (no index) | 0.000 | 46.8 | 0.002 | 16.9 | 0.018 |
| glove | 100,000 | lancedb IVF_FLAT | 0.000 | 86.1 | 0.002 | 26.8 | 0.028 |
| glove | 100,000 | numpy brute force | 0.025 | 42.8 | 0.017 | 2.38 | 0.020 |
| glove | 100,000 | sqlite-vec (vec0, exact) | 0.000 | 48.9 | 0.000 | 30.7 | 0.031 |
| glove | 100,000 | usearch (HNSW) | 0.097 | 57.5 | 0.052 | 1.56 | 0.053 |
| glove | 1,000,000 | **ISONGraph Rust (IVF)** | 23.1 | 1216 | 10.8 | 7592 | 18.3 |
| glove | 1,000,000 | faiss IndexFlatIP | 0.655 | 389 | 0.973 | 66.8 | 1.04 |
| glove | 1,000,000 | faiss HNSWFlat | 0.307 | 524 | 0.228 | 2.15 | 0.230 |
| glove | 1,000,000 | faiss IVFFlat | 0.149 | 391 | 0.246 | 23.8 | 0.270 |
| glove | 1,000,000 | hnswlib | 0.777 | 528 | 3.66 | 3.27 | 3.66 |
| glove | 1,000,000 | usearch (HNSW) | 0.585 | 528 | 0.499 | 3.37 | 0.503 |
| sift | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 12.2 | 134 | 5.14 | 636 | 5.78 |
| sift | 100,000 | **ISONGraph C++ (IVF)** | 12.4 | 134 | 4.97 | 826 | 5.80 |
| sift | 100,000 | **ISONGraph C# (IVF)** | 3.51 | 122 | 3.06 | 1133 | 4.20 |
| sift | 100,000 | **ISONGraph Python (exact)** | 11.0 | 222 | 9.70 | 528 | 10.2 |
| sift | 100,000 | **ISONGraph Python (IVF)** | 4.97 | 222 | 3.66 | 2380 | 6.04 |
| sift | 100,000 | **ISONGraph Rust (IVF)** | 1.67 | 128 | 0.904 | 853 | 1.76 |
| sift | 100,000 | chromadb (HNSW) | 0.000 | 88.5 | 0.015 | 108 | 0.123 |
| sift | 100,000 | faiss IndexFlatIP | 0.060 | 49.6 | 0.087 | 12.9 | 0.100 |
| sift | 100,000 | faiss HNSWFlat | 0.044 | 68.9 | 0.039 | 0.340 | 0.039 |
| sift | 100,000 | faiss IVFFlat | 0.025 | 50.2 | 0.035 | 0.216 | 0.036 |
| sift | 100,000 | hnswlib | 0.067 | 69.3 | 0.158 | 0.214 | 0.158 |
| sift | 100,000 | lancedb (no index) | 0.000 | 59.6 | 0.005 | 74.0 | 0.079 |
| sift | 100,000 | lancedb IVF_FLAT | 0.000 | 110 | 0.008 | 18.6 | 0.027 |
| sift | 100,000 | numpy brute force | 0.031 | 54.6 | 0.026 | 4.71 | 0.030 |
| sift | 100,000 | sqlite-vec (vec0, exact) | 0.001 | 61.8 | 0.001 | 105 | 0.106 |
| sift | 100,000 | usearch (HNSW) | 0.080 | 69.3 | 0.061 | 0.200 | 0.062 |
| sift | 1,000,000 | **ISONGraph Rust (IVF)** | 22.7 | 1285 | 11.2 | 8390 | 19.6 |
| sift | 1,000,000 | faiss HNSWFlat | 1.12 | 626 | 1.10 | 0.567 | 1.10 |
| sift | 1,000,000 | faiss IVFFlat | 0.353 | 498 | 0.723 | 1.63 | 0.725 |
| sift | 1,000,000 | hnswlib | 1.39 | 630 | 4.90 | 0.936 | 4.90 |
| text384 | 100,000 | **ISONGraph C++, /arch:AVX2 (IVF)** | 16.4 | 500 | 6.95 | 531 | 7.48 |
| text384 | 100,000 | **ISONGraph C++ (IVF)** | 16.5 | 500 | 7.34 | 903 | 8.24 |
| text384 | 100,000 | **ISONGraph C# (IVF)** | 4.41 | 453 | 3.95 | 911 | 4.86 |
| text384 | 100,000 | **ISONGraph Python (exact)** | 15.5 | 807 | 10.8 | 291 | 11.1 |
| text384 | 100,000 | **ISONGraph Python (IVF)** | 16.0 | 807 | 10.9 | 5129 | 16.0 |
| text384 | 100,000 | **ISONGraph Rust (IVF)** | 1.94 | 453 | 1.25 | 546 | 1.80 |
| text384 | 100,000 | chromadb (HNSW) | 0.000 | 202 | 0.012 | 175 | 0.187 |
| text384 | 100,000 | faiss IndexFlatIP | 0.128 | 147 | 0.069 | 7.27 | 0.076 |
| text384 | 100,000 | faiss HNSWFlat | 0.134 | 176 | 0.074 | 0.496 | 0.075 |
| text384 | 100,000 | faiss IVFFlat | 0.055 | 149 | 0.091 | 1.29 | 0.092 |
| text384 | 100,000 | hnswlib | 0.147 | 177 | 0.225 | 0.615 | 0.225 |
| text384 | 100,000 | lancedb (no index) | 0.000 | 177 | 0.003 | 110 | 0.113 |
| text384 | 100,000 | lancedb IVF_FLAT | 0.000 | 326 | 0.009 | 23.8 | 0.033 |
| text384 | 100,000 | numpy brute force | 0.110 | 162 | 0.068 | 9.16 | 0.077 |
| text384 | 100,000 | sqlite-vec (vec0, exact) | 0.000 | 180 | 0.000 | 109 | 0.110 |
| text384 | 100,000 | usearch (HNSW) | 0.178 | 177 | 0.136 | 0.949 | 0.137 |
| text384 | 1,000,000 | **ISONGraph Rust (IVF)** | 21.5 | 4534 | 13.0 | 6601 | 19.6 |
| text384 | 1,000,000 | faiss IndexFlatIP | 0.968 | 1472 | 0.736 | 74.1 | 0.811 |
| text384 | 1,000,000 | faiss HNSWFlat | 0.994 | 1618 | 0.805 | 1.21 | 0.806 |
| text384 | 1,000,000 | faiss IVFFlat | 0.417 | 1478 | 0.670 | 6.22 | 0.676 |
| text384 | 1,000,000 | hnswlib | 1.07 | 1623 | 2.12 | 1.58 | 2.12 |
| text384 | 1,000,000 | usearch (HNSW) | 1.54 | 1623 | 1.30 | 1.72 | 1.30 |
