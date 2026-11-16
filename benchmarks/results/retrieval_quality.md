# Retrieval quality benchmark - sample output

Encoder: `hash` | seed: 7 | graph: 1710 nodes, 2628 edges | questions: 200 (impact=50, incident_owner=50, incident_rootcause=50, service_deps=50) | mean gold size 5.4 nodes (4.4 supporting)

Seed threshold: 0.10. Budget k = max nodes returned (vector top-k, or top_k_results). Sanity: semantic_subgraph node set == semantic_multi_hop node set in all cases (mismatches: 0); component counts agree with the library's own `connected_components()` (mismatches: 0).


## Budget k = 10

`support|seed-hit` = support recall restricted to the 146 of 200 questions whose seed nodes plain vector top-10 already finds (same subset for every row).

| variant | method | seed recall | support recall | support\|seed-hit | context recall (+-95% CI) | complete | precision | nodes | components | W/T/L vs vector | median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector_only | vector | 0.730 | 0.027 | 0.037 | 0.185 +-0.021 | 0.00 | 0.087 | 10.0 | 9.25 | - | 0.53 |
| hop2_out_decay0.8 | multi_hop | 0.675 | 0.254 | 0.342 | 0.356 +-0.044 | 0.09 | 0.188 | 10.0 | 4.03 | 99/90/11 | 0.39 |
| hop2_out_decay0.8 | subgraph | 0.675 | 0.254 | 0.342 | 0.356 +-0.044 | 0.09 | 0.188 | 10.0 | 4.03 | 99/90/11 | 1.05 |
| hop3_out_decay0.8 | multi_hop | 0.675 | 0.254 | 0.342 | 0.356 +-0.044 | 0.09 | 0.188 | 10.0 | 4.03 | 99/90/11 | 0.43 |
| hop3_out_decay0.8 | subgraph | 0.675 | 0.254 | 0.342 | 0.356 +-0.044 | 0.09 | 0.188 | 10.0 | 4.03 | 99/90/11 | 1.15 |
| hop3_out_decay0.5 | multi_hop | 0.675 | 0.254 | 0.341 | 0.355 +-0.044 | 0.09 | 0.187 | 10.0 | 4.18 | 100/89/11 | 0.45 |
| hop3_out_decay0.5 | subgraph | 0.675 | 0.254 | 0.341 | 0.355 +-0.044 | 0.09 | 0.187 | 10.0 | 4.18 | 100/89/11 | 1.10 |
| hop3_out_decay0.95 | multi_hop | 0.675 | 0.268 | 0.356 | 0.367 +-0.045 | 0.09 | 0.194 | 10.0 | 3.65 | 96/93/11 | 0.43 |
| hop3_out_decay0.95 | subgraph | 0.675 | 0.268 | 0.356 | 0.367 +-0.045 | 0.09 | 0.194 | 10.0 | 3.65 | 96/93/11 | 1.11 |
| hop3_out_blend0.3 | multi_hop | 0.685 | 0.266 | 0.358 | 0.367 +-0.044 | 0.08 | 0.196 | 10.0 | 4.25 | 106/83/11 | 1.19 |
| hop3_out_blend0.3 | subgraph | 0.685 | 0.266 | 0.358 | 0.367 +-0.044 | 0.08 | 0.196 | 10.0 | 4.25 | 106/83/11 | 3.03 |
| hop3_out_seeds1 | multi_hop | 0.505 | 0.299 | 0.385 | 0.346 +-0.052 | 0.14 | 0.203 | 9.2 | 1.00 | 83/78/39 | 0.92 |
| hop3_out_seeds1 | subgraph | 0.505 | 0.299 | 0.385 | 0.346 +-0.052 | 0.14 | 0.203 | 9.2 | 1.00 | 83/78/39 | 2.85 |
| hop3_out_seeds2 | multi_hop | 0.565 | 0.300 | 0.389 | 0.366 +-0.048 | 0.10 | 0.199 | 10.0 | 1.75 | 91/83/26 | 1.09 |
| hop3_out_seeds2 | subgraph | 0.565 | 0.300 | 0.389 | 0.366 +-0.048 | 0.10 | 0.199 | 10.0 | 1.75 | 91/83/26 | 2.86 |
| hop2_in_decay0.8 | multi_hop | 0.675 | 0.223 | 0.306 | 0.309 +-0.052 | 0.20 | 0.150 | 8.0 | 4.36 | 46/122/32 | 1.01 |
| hop2_in_decay0.8 | subgraph | 0.675 | 0.223 | 0.306 | 0.309 +-0.052 | 0.20 | 0.150 | 8.0 | 4.36 | 46/122/32 | 2.88 |
| hop3_both_decay0.8 | multi_hop | 0.675 | 0.428 | 0.581 | 0.474 +-0.054 | 0.24 | 0.236 | 10.0 | 3.87 | 131/58/11 | 1.87 |
| hop3_both_decay0.8 | subgraph | 0.675 | 0.428 | 0.581 | 0.474 +-0.054 | 0.24 | 0.236 | 10.0 | 3.87 | 131/58/11 | 4.01 |
| hop3_both_blend0.3 | multi_hop | 0.700 | 0.283 | 0.385 | 0.371 +-0.048 | 0.14 | 0.186 | 10.0 | 6.65 | 91/98/11 | 2.18 |
| hop3_both_blend0.3 | subgraph | 0.700 | 0.283 | 0.385 | 0.371 +-0.048 | 0.14 | 0.186 | 10.0 | 6.65 | 91/98/11 | 4.14 |

### Context recall by question type (k = 10; multi_hop rows; subgraph returns the same nodes)

| variant | impact | incident_owner | incident_rootcause | service_deps |
|---|---|---|---|---|
| vector_only | 0.275 | 0.120 | 0.084 | 0.263 |
| hop2_out_decay0.8 | 0.326 | 0.200 | 0.121 | 0.776 |
| hop3_out_decay0.8 | 0.326 | 0.200 | 0.121 | 0.776 |
| hop3_out_decay0.5 | 0.331 | 0.200 | 0.121 | 0.769 |
| hop3_out_decay0.95 | 0.298 | 0.235 | 0.161 | 0.776 |
| hop3_out_blend0.3 | 0.345 | 0.205 | 0.144 | 0.774 |
| hop3_out_seeds1 | 0.201 | 0.255 | 0.153 | 0.776 |
| hop3_out_seeds2 | 0.258 | 0.250 | 0.182 | 0.776 |
| hop2_in_decay0.8 | 0.875 | 0.105 | 0.062 | 0.193 |
| hop3_both_decay0.8 | 0.802 | 0.200 | 0.119 | 0.776 |
| hop3_both_blend0.3 | 0.595 | 0.120 | 0.080 | 0.691 |

### Seed recall vs support recall by type (k = 10) - vector_only vs best graph variant

Best graph variant by overall context recall: **hop3_both_decay0.8**.

| type | vector seed | vector support | best seed | best support |
|---|---|---|---|---|
| impact | 0.920 | 0.000 | 0.900 | 0.773 |
| incident_owner | 0.480 | 0.000 | 0.420 | 0.127 |
| incident_rootcause | 0.520 | 0.000 | 0.380 | 0.070 |
| service_deps | 1.000 | 0.107 | 1.000 | 0.743 |

## Budget k = 20

`support|seed-hit` = support recall restricted to the 159 of 200 questions whose seed nodes plain vector top-20 already finds (same subset for every row).

| variant | method | seed recall | support recall | support\|seed-hit | context recall (+-95% CI) | complete | precision | nodes | components | W/T/L vs vector | median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector_only | vector | 0.795 | 0.041 | 0.051 | 0.209 +-0.021 | 0.00 | 0.051 | 20.0 | 18.29 | - | 0.55 |
| hop2_out_decay0.8 | multi_hop | 0.680 | 0.412 | 0.499 | 0.486 +-0.052 | 0.21 | 0.136 | 20.0 | 3.70 | 117/70/13 | 0.38 |
| hop2_out_decay0.8 | subgraph | 0.680 | 0.412 | 0.499 | 0.486 +-0.052 | 0.21 | 0.136 | 20.0 | 3.70 | 117/70/13 | 1.28 |
| hop3_out_decay0.8 | multi_hop | 0.680 | 0.412 | 0.499 | 0.486 +-0.052 | 0.21 | 0.136 | 20.0 | 3.69 | 117/70/13 | 0.46 |
| hop3_out_decay0.8 | subgraph | 0.680 | 0.412 | 0.499 | 0.486 +-0.052 | 0.21 | 0.136 | 20.0 | 3.69 | 117/70/13 | 1.35 |
| hop3_out_decay0.5 | multi_hop | 0.680 | 0.367 | 0.442 | 0.446 +-0.047 | 0.12 | 0.119 | 20.0 | 3.67 | 117/70/13 | 0.44 |
| hop3_out_decay0.5 | subgraph | 0.680 | 0.367 | 0.442 | 0.446 +-0.047 | 0.12 | 0.119 | 20.0 | 3.67 | 117/70/13 | 1.41 |
| hop3_out_decay0.95 | multi_hop | 0.680 | 0.424 | 0.515 | 0.497 +-0.054 | 0.28 | 0.140 | 20.0 | 3.52 | 116/70/14 | 0.44 |
| hop3_out_decay0.95 | subgraph | 0.680 | 0.424 | 0.515 | 0.497 +-0.054 | 0.28 | 0.140 | 20.0 | 3.52 | 116/70/14 | 3.64 |
| hop3_out_blend0.3 | multi_hop | 0.685 | 0.379 | 0.458 | 0.460 +-0.049 | 0.16 | 0.128 | 20.0 | 4.09 | 118/69/13 | 1.48 |
| hop3_out_blend0.3 | subgraph | 0.685 | 0.379 | 0.458 | 0.460 +-0.049 | 0.16 | 0.128 | 20.0 | 4.09 | 118/69/13 | 4.51 |
| hop3_out_seeds1 | multi_hop | 0.505 | 0.376 | 0.459 | 0.413 +-0.061 | 0.33 | 0.164 | 13.9 | 1.00 | 84/68/48 | 0.97 |
| hop3_out_seeds1 | subgraph | 0.505 | 0.376 | 0.459 | 0.413 +-0.061 | 0.33 | 0.164 | 13.9 | 1.00 | 84/68/48 | 3.45 |
| hop3_out_seeds2 | multi_hop | 0.565 | 0.416 | 0.505 | 0.463 +-0.059 | 0.33 | 0.142 | 19.1 | 1.64 | 97/74/29 | 0.98 |
| hop3_out_seeds2 | subgraph | 0.565 | 0.416 | 0.505 | 0.463 +-0.059 | 0.33 | 0.142 | 19.1 | 1.64 | 97/74/29 | 3.95 |
| hop2_in_decay0.8 | multi_hop | 0.675 | 0.232 | 0.291 | 0.316 +-0.053 | 0.23 | 0.101 | 12.3 | 4.57 | 46/102/52 | 0.99 |
| hop2_in_decay0.8 | subgraph | 0.675 | 0.232 | 0.291 | 0.316 +-0.053 | 0.23 | 0.101 | 12.3 | 4.57 | 46/102/52 | 2.81 |
| hop3_both_decay0.8 | multi_hop | 0.680 | 0.534 | 0.664 | 0.559 +-0.056 | 0.33 | 0.142 | 20.0 | 3.71 | 143/39/18 | 1.89 |
| hop3_both_decay0.8 | subgraph | 0.680 | 0.534 | 0.664 | 0.559 +-0.056 | 0.33 | 0.142 | 20.0 | 3.71 | 143/39/18 | 4.57 |
| hop3_both_blend0.3 | multi_hop | 0.725 | 0.426 | 0.533 | 0.485 +-0.054 | 0.26 | 0.123 | 20.0 | 8.96 | 135/43/22 | 1.99 |
| hop3_both_blend0.3 | subgraph | 0.725 | 0.426 | 0.533 | 0.485 +-0.054 | 0.26 | 0.123 | 20.0 | 8.96 | 135/43/22 | 4.99 |

### Context recall by question type (k = 20; multi_hop rows; subgraph returns the same nodes)

| variant | impact | incident_owner | incident_rootcause | service_deps |
|---|---|---|---|---|
| vector_only | 0.280 | 0.145 | 0.104 | 0.306 |
| hop2_out_decay0.8 | 0.343 | 0.355 | 0.284 | 0.963 |
| hop3_out_decay0.8 | 0.343 | 0.355 | 0.284 | 0.963 |
| hop3_out_decay0.5 | 0.343 | 0.355 | 0.284 | 0.804 |
| hop3_out_decay0.95 | 0.338 | 0.360 | 0.289 | 1.000 |
| hop3_out_blend0.3 | 0.352 | 0.300 | 0.273 | 0.914 |
| hop3_out_seeds1 | 0.201 | 0.270 | 0.179 | 1.000 |
| hop3_out_seeds2 | 0.258 | 0.365 | 0.228 | 1.000 |
| hop2_in_decay0.8 | 0.900 | 0.105 | 0.062 | 0.197 |
| hop3_both_decay0.8 | 0.910 | 0.305 | 0.240 | 0.781 |
| hop3_both_blend0.3 | 0.823 | 0.210 | 0.132 | 0.774 |

### Seed recall vs support recall by type (k = 20) - vector_only vs best graph variant

Best graph variant by overall context recall: **hop3_both_decay0.8**.

| type | vector seed | vector support | best seed | best support |
|---|---|---|---|---|
| impact | 0.940 | 0.003 | 0.920 | 0.909 |
| incident_owner | 0.580 | 0.000 | 0.420 | 0.267 |
| incident_rootcause | 0.660 | 0.000 | 0.380 | 0.213 |
| service_deps | 1.000 | 0.159 | 1.000 | 0.749 |
