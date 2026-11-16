| Method (mean ms/query; median in brackets) | hotpotqa/minilm | 2wiki/minilm | musique/minilm | hotpotqa/bge |
|---|---|---|---|---|
| vector (numpy brute force top-20) | 0.49 (0.47) | 0.76 (0.73) | 2.06 (1.92) | 1.34 (1.31) |
| ISONGraph similarity_search top-20 (plain vector via the library) | 0.69 (0.68) | 0.94 (0.85) | 2.22 (2.17) | 2.10 (2.01) |
| BM25 (scipy sparse) | 0.15 (0.15) | 0.46 (0.44) | 0.41 (0.38) | 0.53 (0.51) |
| Hybrid RRF (dense + BM25 + fusion) | 5.39 (5.38) | 3.87 (3.79) | 6.22 (6.09) | 6.04 (5.83) |
| PPR re-rank, graph=mention | 11.18 (11.42) | 6.83 (6.73) | 13.70 (13.62) | 13.12 (12.99) |
| ISONGraph semantic_multi_hop (tuned) | 12.34 (12.09) | 3.42 (2.92) | 16.80 (16.36) | 2.15 (2.11) |
| ISONGraph semantic_subgraph(connect=True) | 20.60 (20.79) | 39.91 (35.44) | 31.09 (30.49) | 10.56 (10.06) |
| Vector top-30 + cross-encoder (CPU, 1 thread) | 6843.02 (7238.13) | - | - | - |

Timed multi_hop/subgraph configurations (dev-selected at k=10):

- hotpotqa/minilm: graph `mention`, config `{'hops': 2, 'seeds': 5, 'decay': 0.5, 'blend': 0.3, 'direction': 'both', 'weight_field': None}`
- 2wiki/minilm: graph `mention+entity`, config `{'hops': 2, 'seeds': 5, 'decay': 0.7, 'blend': 0.0, 'direction': 'out', 'weight_field': 'w'}`
- musique/minilm: graph `mention`, config `{'hops': 2, 'seeds': 10, 'decay': 0.9, 'blend': 0.3, 'direction': 'both', 'weight_field': None}`
- hotpotqa/bge: graph `mention`, config `{'hops': 2, 'seeds': 5, 'decay': 0.7, 'blend': 0.0, 'direction': 'both', 'weight_field': None}`
