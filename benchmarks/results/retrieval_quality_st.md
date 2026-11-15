# Retrieval quality benchmark - sample output

Encoder: `st` | seed: 7 | graph: 1710 nodes, 2628 edges | questions: 200 (impact=50, incident_owner=50, incident_rootcause=50, service_deps=50) | mean gold size 5.4 nodes (4.4 supporting)

Seed threshold: 0.10. Budget k = max nodes returned (vector top-k, or top_k_results). Sanity: semantic_subgraph node set == semantic_multi_hop node set in all cases (mismatches: 0); component counts agree with the library's own `connected_components()` (mismatches: 0).


## Budget k = 10

`support|seed-hit` = support recall restricted to the 148 of 200 questions whose seed nodes plain vector top-10 already finds (same subset for every row).

| variant | method | seed recall | support recall | support\|seed-hit | context recall (+-95% CI) | complete | precision | nodes | components | W/T/L vs vector | median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector_only | vector | 0.740 | 0.161 | 0.218 | 0.287 +-0.042 | 0.10 | 0.135 | 10.0 | 8.29 | - | 4.87 |
| hop2_out_decay0.8 | multi_hop | 0.705 | 0.333 | 0.447 | 0.419 +-0.046 | 0.11 | 0.215 | 10.0 | 3.16 | 91/75/34 | 5.06 |
| hop2_out_decay0.8 | subgraph | 0.705 | 0.333 | 0.447 | 0.419 +-0.046 | 0.11 | 0.215 | 10.0 | 3.16 | 91/75/34 | 5.98 |
| hop3_out_decay0.8 | multi_hop | 0.705 | 0.333 | 0.447 | 0.419 +-0.046 | 0.11 | 0.215 | 10.0 | 3.16 | 91/75/34 | 5.21 |
| hop3_out_decay0.8 | subgraph | 0.705 | 0.333 | 0.447 | 0.419 +-0.046 | 0.11 | 0.215 | 10.0 | 3.16 | 91/75/34 | 6.26 |
| hop3_out_decay0.5 | multi_hop | 0.705 | 0.331 | 0.444 | 0.417 +-0.046 | 0.11 | 0.214 | 10.0 | 3.15 | 91/75/34 | 5.50 |
| hop3_out_decay0.5 | subgraph | 0.705 | 0.331 | 0.444 | 0.417 +-0.046 | 0.11 | 0.214 | 10.0 | 3.15 | 91/75/34 | 6.36 |
| hop3_out_decay0.95 | multi_hop | 0.695 | 0.367 | 0.477 | 0.445 +-0.047 | 0.12 | 0.231 | 10.0 | 2.75 | 91/75/34 | 5.15 |
| hop3_out_decay0.95 | subgraph | 0.695 | 0.367 | 0.477 | 0.445 +-0.047 | 0.12 | 0.231 | 10.0 | 2.75 | 91/75/34 | 5.92 |
| hop3_out_blend0.3 | multi_hop | 0.705 | 0.316 | 0.423 | 0.404 +-0.044 | 0.10 | 0.205 | 10.0 | 3.17 | 94/73/33 | 4.98 |
| hop3_out_blend0.3 | subgraph | 0.705 | 0.316 | 0.423 | 0.404 +-0.044 | 0.10 | 0.205 | 10.0 | 3.17 | 94/73/33 | 6.05 |
| hop3_out_seeds1 | multi_hop | 0.590 | 0.344 | 0.434 | 0.407 +-0.048 | 0.14 | 0.234 | 9.0 | 1.00 | 92/61/47 | 4.90 |
| hop3_out_seeds1 | subgraph | 0.590 | 0.344 | 0.434 | 0.407 +-0.048 | 0.14 | 0.234 | 9.0 | 1.00 | 92/61/47 | 5.77 |
| hop3_out_seeds2 | multi_hop | 0.620 | 0.343 | 0.438 | 0.412 +-0.047 | 0.12 | 0.220 | 9.7 | 1.46 | 91/65/44 | 5.04 |
| hop3_out_seeds2 | subgraph | 0.620 | 0.343 | 0.438 | 0.412 +-0.047 | 0.12 | 0.220 | 9.7 | 1.46 | 91/65/44 | 5.99 |
| hop2_in_decay0.8 | multi_hop | 0.705 | 0.253 | 0.342 | 0.340 +-0.054 | 0.23 | 0.166 | 8.1 | 4.32 | 31/149/20 | 6.28 |
| hop2_in_decay0.8 | subgraph | 0.705 | 0.253 | 0.342 | 0.340 +-0.054 | 0.23 | 0.166 | 8.1 | 4.32 | 31/149/20 | 5.66 |
| hop3_both_decay0.8 | multi_hop | 0.705 | 0.483 | 0.650 | 0.523 +-0.056 | 0.30 | 0.260 | 10.0 | 3.13 | 117/74/9 | 5.89 |
| hop3_both_decay0.8 | subgraph | 0.705 | 0.483 | 0.650 | 0.523 +-0.056 | 0.30 | 0.260 | 10.0 | 3.13 | 117/74/9 | 6.62 |
| hop3_both_blend0.3 | multi_hop | 0.720 | 0.387 | 0.520 | 0.449 +-0.052 | 0.23 | 0.216 | 10.0 | 3.94 | 109/85/6 | 6.50 |
| hop3_both_blend0.3 | subgraph | 0.720 | 0.387 | 0.520 | 0.449 +-0.052 | 0.23 | 0.216 | 10.0 | 3.94 | 109/85/6 | 10.04 |

### Context recall by question type (k = 10; multi_hop rows; subgraph returns the same nodes)

| variant | impact | incident_owner | incident_rootcause | service_deps |
|---|---|---|---|---|
| vector_only | 0.713 | 0.130 | 0.075 | 0.229 |
| hop2_out_decay0.8 | 0.528 | 0.235 | 0.140 | 0.772 |
| hop3_out_decay0.8 | 0.528 | 0.235 | 0.140 | 0.772 |
| hop3_out_decay0.5 | 0.528 | 0.235 | 0.140 | 0.765 |
| hop3_out_decay0.95 | 0.491 | 0.315 | 0.198 | 0.776 |
| hop3_out_blend0.3 | 0.537 | 0.230 | 0.138 | 0.710 |
| hop3_out_seeds1 | 0.296 | 0.360 | 0.198 | 0.776 |
| hop3_out_seeds2 | 0.348 | 0.310 | 0.210 | 0.778 |
| hop2_in_decay0.8 | 0.988 | 0.110 | 0.065 | 0.195 |
| hop3_both_decay0.8 | 0.967 | 0.230 | 0.126 | 0.769 |
| hop3_both_blend0.3 | 0.949 | 0.220 | 0.111 | 0.515 |

### Seed recall vs support recall by type (k = 10) - vector_only vs best graph variant

Best graph variant by overall context recall: **hop3_both_decay0.8**.

| type | vector seed | vector support | best seed | best support |
|---|---|---|---|---|
| impact | 1.000 | 0.580 | 1.000 | 0.962 |
| incident_owner | 0.520 | 0.000 | 0.440 | 0.160 |
| incident_rootcause | 0.440 | 0.000 | 0.380 | 0.075 |
| service_deps | 1.000 | 0.065 | 1.000 | 0.736 |

## Budget k = 20

`support|seed-hit` = support recall restricted to the 157 of 200 questions whose seed nodes plain vector top-20 already finds (same subset for every row).

| variant | method | seed recall | support recall | support\|seed-hit | context recall (+-95% CI) | complete | precision | nodes | components | W/T/L vs vector | median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector_only | vector | 0.785 | 0.228 | 0.289 | 0.343 +-0.048 | 0.18 | 0.083 | 20.0 | 15.42 | - | 4.98 |
| hop2_out_decay0.8 | multi_hop | 0.705 | 0.451 | 0.553 | 0.514 +-0.048 | 0.16 | 0.139 | 19.4 | 2.61 | 102/56/42 | 5.00 |
| hop2_out_decay0.8 | subgraph | 0.705 | 0.451 | 0.553 | 0.514 +-0.048 | 0.16 | 0.139 | 19.4 | 2.61 | 102/56/42 | 6.11 |
| hop3_out_decay0.8 | multi_hop | 0.705 | 0.453 | 0.555 | 0.515 +-0.048 | 0.17 | 0.138 | 19.5 | 2.61 | 102/56/42 | 5.13 |
| hop3_out_decay0.8 | subgraph | 0.705 | 0.453 | 0.555 | 0.515 +-0.048 | 0.17 | 0.138 | 19.5 | 2.61 | 102/56/42 | 6.16 |
| hop3_out_decay0.5 | multi_hop | 0.705 | 0.441 | 0.540 | 0.504 +-0.047 | 0.14 | 0.134 | 19.5 | 2.60 | 102/56/42 | 5.34 |
| hop3_out_decay0.5 | subgraph | 0.705 | 0.441 | 0.540 | 0.504 +-0.047 | 0.14 | 0.134 | 19.5 | 2.60 | 102/56/42 | 6.62 |
| hop3_out_decay0.95 | multi_hop | 0.705 | 0.513 | 0.633 | 0.565 +-0.055 | 0.34 | 0.156 | 19.5 | 2.56 | 101/57/42 | 5.25 |
| hop3_out_decay0.95 | subgraph | 0.705 | 0.513 | 0.633 | 0.565 +-0.055 | 0.34 | 0.156 | 19.5 | 2.56 | 101/57/42 | 6.77 |
| hop3_out_blend0.3 | multi_hop | 0.705 | 0.435 | 0.534 | 0.500 +-0.047 | 0.13 | 0.134 | 19.5 | 2.58 | 101/56/43 | 5.41 |
| hop3_out_blend0.3 | subgraph | 0.705 | 0.435 | 0.534 | 0.500 +-0.047 | 0.13 | 0.134 | 19.5 | 2.58 | 101/56/43 | 6.71 |
| hop3_out_seeds1 | multi_hop | 0.590 | 0.431 | 0.526 | 0.482 +-0.057 | 0.34 | 0.191 | 14.2 | 1.00 | 94/51/55 | 5.24 |
| hop3_out_seeds1 | subgraph | 0.590 | 0.431 | 0.526 | 0.482 +-0.057 | 0.34 | 0.191 | 14.2 | 1.00 | 94/51/55 | 6.39 |
| hop3_out_seeds2 | multi_hop | 0.620 | 0.456 | 0.553 | 0.504 +-0.056 | 0.30 | 0.154 | 18.0 | 1.42 | 99/50/51 | 5.80 |
| hop3_out_seeds2 | subgraph | 0.620 | 0.456 | 0.553 | 0.504 +-0.056 | 0.30 | 0.154 | 18.0 | 1.42 | 99/50/51 | 9.55 |
| hop2_in_decay0.8 | multi_hop | 0.705 | 0.258 | 0.328 | 0.344 +-0.055 | 0.25 | 0.101 | 13.2 | 4.34 | 13/147/40 | 4.84 |
| hop2_in_decay0.8 | subgraph | 0.705 | 0.258 | 0.328 | 0.344 +-0.055 | 0.25 | 0.101 | 13.2 | 4.34 | 13/147/40 | 6.22 |
| hop3_both_decay0.8 | multi_hop | 0.710 | 0.589 | 0.735 | 0.608 +-0.054 | 0.35 | 0.155 | 20.0 | 2.73 | 112/80/8 | 5.98 |
| hop3_both_decay0.8 | subgraph | 0.710 | 0.589 | 0.735 | 0.608 +-0.054 | 0.35 | 0.155 | 20.0 | 2.73 | 112/80/8 | 7.07 |
| hop3_both_blend0.3 | multi_hop | 0.720 | 0.487 | 0.615 | 0.530 +-0.055 | 0.33 | 0.133 | 20.0 | 3.42 | 106/78/16 | 7.03 |
| hop3_both_blend0.3 | subgraph | 0.720 | 0.487 | 0.615 | 0.530 +-0.055 | 0.33 | 0.133 | 20.0 | 3.42 | 106/78/16 | 7.74 |

### Context recall by question type (k = 20; multi_hop rows; subgraph returns the same nodes)

| variant | impact | incident_owner | incident_rootcause | service_deps |
|---|---|---|---|---|
| vector_only | 0.834 | 0.145 | 0.094 | 0.298 |
| hop2_out_decay0.8 | 0.528 | 0.385 | 0.307 | 0.834 |
| hop3_out_decay0.8 | 0.528 | 0.390 | 0.307 | 0.834 |
| hop3_out_decay0.5 | 0.528 | 0.390 | 0.307 | 0.791 |
| hop3_out_decay0.95 | 0.537 | 0.440 | 0.305 | 0.978 |
| hop3_out_blend0.3 | 0.545 | 0.350 | 0.277 | 0.829 |
| hop3_out_seeds1 | 0.299 | 0.390 | 0.238 | 1.000 |
| hop3_out_seeds2 | 0.357 | 0.445 | 0.247 | 0.969 |
| hop2_in_decay0.8 | 1.000 | 0.110 | 0.065 | 0.200 |
| hop3_both_decay0.8 | 0.993 | 0.365 | 0.290 | 0.786 |
| hop3_both_blend0.3 | 1.000 | 0.240 | 0.156 | 0.725 |

### Seed recall vs support recall by type (k = 20) - vector_only vs best graph variant

Best graph variant by overall context recall: **hop3_both_decay0.8**.

| type | vector seed | vector support | best seed | best support |
|---|---|---|---|---|
| impact | 1.000 | 0.757 | 1.000 | 0.992 |
| incident_owner | 0.580 | 0.000 | 0.440 | 0.340 |
| incident_rootcause | 0.560 | 0.003 | 0.400 | 0.268 |
| service_deps | 1.000 | 0.151 | 1.000 | 0.755 |
