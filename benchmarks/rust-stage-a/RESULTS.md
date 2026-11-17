# Rust stage A: concurrency, one copy, no write stall, deletes

Same-session A/B on the i9-14900HX laptop (8 P + 16 E cores, 32 threads), `text384` / `glove` / `sift`
data from `D:\Goldmine\ISON\bench_data`, method as in `benchmarks/competitive/README_method.md`
(single-thread loops pinned to one P-core, three timed runs per point, ground truth float64 exact).
"old" = the commit before the change (`git archive HEAD`, built into its own target dir); "new" = the working tree.
Old and new ran back to back, one timed job at a time (`run_ab.sh`; raw JSON in `raw/`, tables by `summarize.py`).
The machine drifted by up to ~25% between hours and by ~10% between repeats, so read ratios, not absolute QPS,
and treat differences under ~10% as noise. text384 100k ran 3x per side, 1M 1x (competitive_bench) and 2x
(writes); glove/sift 100k 1x.

Not in the tables: sidecar and graph file bytes are identical old vs new (SemanticGraph built from a shuffled
history with updates and deletes, saved, compared with `cmp`); the `get()` API change is described in the report.

Reading notes:
* "IVF ... after the 30k writes" is `competitive_bench`'s mutation test (10k inserts + 10k updates + 10k deletes
  on the built index at the nprobe that first reached recall 0.95). sift 100k picked nprobe 64 for old
  (0.9860 pre-mutation) but 32 for new (0.9513), so its `recall after` rows compare different operating
  points, not a recall loss; at equal nprobe the sweep tables are within 0.003.
* `exact batch` / `IVF batch` are new (old has no batch call); the old one-query loop with automatic threads is the comparison.
* `update` throughput is lower than before by design: the fold work that used to be deferred to the first search
  after the writes is now paid by the writes (every 256th write folds 256 rows).

### Rust stage A: before / after (same session)

#### text384 n=100,000 (median over 3 old / 3 new run(s))

| metric | old | new | new/old |
|---|---|---|---|
| exact, 1 thread (pinned), QPS | 137.5 | 130.9 | 0.95x |
| exact, one-query loop with auto threads, QPS | 243.4 | 248.6 | 1.02x |
| 8 concurrent clients (1 thread each), exact QPS | 137.8 | 756.0 | 5.49x |
| exact batch, all threads, QPS | - | 10,376 | - |
| IVF build, s | 0.49 | 0.60 | 1.22x |
| IVF 1-thread QPS at recall 0.95 (interpolated) | 749 | 843 | 1.12x |
| IVF one-query loop, auto threads, QPS at ref nprobe | 1,396 | 1,598 | 1.15x |
| IVF batch, all threads, QPS at ref nprobe | - | 41,704 | - |
| ingest, s | 0.06 | 0.12 | 2.08x |
| working set after ingest, MB | 181 | 162 | 0.89x |
| working set after IVF build, MB | 342 | 168 | 0.49x |
| peak working set after IVF build, MB | 420 | 247 | 0.59x |
| insert 10k, per s | 224,393 | 242,247 | 1.08x |
| update 10k, per s | 469,598 | 265,214 | 0.56x |
| delete 10k, per s | 840,845 | 1,629,009 | 1.94x |
| first query after the 30k writes, ms | 224.92 | 1.48 | 0.01x |
| recall@10 after the writes | 0.9490 | 0.9518 | - |

recall@10 / 1-thread QPS by nprobe (median of runs):

| nprobe | old recall | new recall | old QPS | new QPS |
|---|---|---|---|---|
| 1 | 0.2730 | 0.2879 | 21,715 | 26,723 |
| 2 | 0.3918 | 0.4120 | 16,409 | 18,338 |
| 4 | 0.5168 | 0.5379 | 12,143 | 12,750 |
| 8 | 0.6402 | 0.6493 | 7,178 | 8,305 |
| 16 | 0.7450 | 0.7529 | 4,716 | 5,203 |
| 32 | 0.8346 | 0.8434 | 2,697 | 2,450 |
| 64 | 0.9071 | 0.9075 | 1,459 | 1,484 |
| 128 | 0.9529 | 0.9548 | 716 | 790 |
| 256 | 0.9807 | 0.9834 | 385 | 446 |

#### text384 n=1,000,000 (median over 1 old / 1 new run(s))

| metric | old | new | new/old |
|---|---|---|---|
| exact, 1 thread (pinned), QPS | 15.2 | 14.5 | 0.95x |
| exact, one-query loop with auto threads, QPS | 31.8 | 29.3 | 0.92x |
| 8 concurrent clients (1 thread each), exact QPS | 14.8 | 60.6 | 4.08x |
| exact batch, all threads, QPS | - | 1,178 | - |
| IVF build, s | 6.16 | 7.49 | 1.22x |
| IVF 1-thread QPS at recall 0.95 (interpolated) | 103 | 133 | 1.28x |
| IVF one-query loop, auto threads, QPS at ref nprobe | 250 | 316 | 1.27x |
| IVF batch, all threads, QPS at ref nprobe | - | 11,911 | - |
| ingest, s | 0.82 | 1.15 | 1.40x |
| working set after ingest, MB | 1,932 | 1,622 | 0.84x |
| working set after IVF build, MB | 3,532 | 1,643 | 0.47x |
| peak working set after IVF build, MB | 3,779 | 1,888 | 0.50x |
| insert 10k, per s | 26,383 | 125,089 | 4.74x |
| update 10k, per s | 365,429 | 143,493 | 0.39x |
| delete 10k, per s | 564,927 | 969,039 | 1.72x |
| first query after the 30k writes, ms | 616.13 | 7.63 | 0.01x |
| recall@10 after the writes | 0.9521 | 0.9564 | - |

recall@10 / 1-thread QPS by nprobe (median of runs):

| nprobe | old recall | new recall | old QPS | new QPS |
|---|---|---|---|---|
| 1 | 0.2757 | 0.2709 | 5,244 | 4,942 |
| 2 | 0.3806 | 0.3805 | 4,302 | 4,742 |
| 4 | 0.4961 | 0.4949 | 2,797 | 3,312 |
| 8 | 0.6046 | 0.6065 | 1,827 | 2,320 |
| 16 | 0.7022 | 0.7056 | 1,058 | 1,420 |
| 32 | 0.7905 | 0.7865 | 612 | 809 |
| 64 | 0.8600 | 0.8582 | 352 | 436 |
| 128 | 0.9150 | 0.9158 | 187 | 240 |
| 256 | 0.9521 | 0.9526 | 100 | 127 |

#### glove n=100,000 (median over 1 old / 1 new run(s))

| metric | old | new | new/old |
|---|---|---|---|
| exact, 1 thread (pinned), QPS | 664.1 | 579.7 | 0.87x |
| exact, one-query loop with auto threads, QPS | 1,612.0 | 1,913.9 | 1.19x |
| 8 concurrent clients (1 thread each), exact QPS | 671.3 | 3,051.9 | 4.55x |
| exact batch, all threads, QPS | - | 25,961 | - |
| IVF build, s | 0.22 | 0.18 | 0.84x |
| IVF 1-thread QPS at recall 0.95 (interpolated) | 1,857 | 2,251 | 1.21x |
| IVF one-query loop, auto threads, QPS at ref nprobe | 1,853 | 2,634 | 1.42x |
| IVF batch, all threads, QPS at ref nprobe | - | 52,421 | - |
| ingest, s | 0.06 | 0.05 | 0.91x |
| working set after ingest, MB | 72 | 53 | 0.74x |
| working set after IVF build, MB | 125 | 61 | 0.49x |
| peak working set after IVF build, MB | 144 | 80 | 0.56x |
| insert 10k, per s | 675,680 | 649,836 | 0.96x |
| update 10k, per s | 705,696 | 531,547 | 0.75x |
| delete 10k, per s | 1,030,121 | 2,199,107 | 2.13x |
| first query after the 30k writes, ms | 82.47 | 1.02 | 0.01x |
| recall@10 after the writes | 0.9743 | 0.9732 | - |

recall@10 / 1-thread QPS by nprobe (median of runs):

| nprobe | old recall | new recall | old QPS | new QPS |
|---|---|---|---|---|
| 1 | 0.2851 | 0.2832 | 52,731 | 65,848 |
| 2 | 0.4016 | 0.4071 | 40,514 | 51,856 |
| 4 | 0.5209 | 0.5229 | 30,934 | 39,881 |
| 8 | 0.6351 | 0.6361 | 20,568 | 27,274 |
| 16 | 0.7308 | 0.7323 | 13,275 | 16,357 |
| 32 | 0.8134 | 0.8163 | 7,466 | 9,201 |
| 64 | 0.8817 | 0.8825 | 4,323 | 5,297 |
| 128 | 0.9355 | 0.9361 | 2,277 | 2,851 |
| 256 | 0.9740 | 0.9743 | 1,325 | 1,490 |
| 1265 | 1.0000 | 1.0000 | 638 | 578 |

#### sift n=100,000 (median over 1 old / 1 new run(s))

| metric | old | new | new/old |
|---|---|---|---|
| exact, 1 thread (pinned), QPS | 437.7 | 463.8 | 1.06x |
| exact, one-query loop with auto threads, QPS | 1,030.1 | 1,724.3 | 1.67x |
| 8 concurrent clients (1 thread each), exact QPS | 487.6 | 2,214.6 | 4.54x |
| exact batch, all threads, QPS | - | 29,394 | - |
| IVF build, s | 0.20 | 0.18 | 0.90x |
| IVF 1-thread QPS at recall 0.95 (interpolated) | 7,552 | 9,047 | 1.20x |
| IVF one-query loop, auto threads, QPS at ref nprobe | 4,604 | 9,019 | 1.96x |
| IVF batch, all threads, QPS at ref nprobe | - | 117,271 | - |
| ingest, s | 0.06 | 0.05 | 0.88x |
| working set after ingest, MB | 83 | 64 | 0.78x |
| working set after IVF build, MB | 146 | 71 | 0.49x |
| peak working set after IVF build, MB | 172 | 97 | 0.56x |
| insert 10k, per s | 558,403 | 623,756 | 1.12x |
| update 10k, per s | 647,040 | 532,984 | 0.82x |
| delete 10k, per s | 1,041,743 | 2,025,316 | 1.94x |
| first query after the 30k writes, ms | 75.33 | 0.19 | 0.00x |
| recall@10 after the writes | 0.9855 | 0.9480 | - |

recall@10 / 1-thread QPS by nprobe (median of runs):

| nprobe | old recall | new recall | old QPS | new QPS |
|---|---|---|---|---|
| 1 | 0.2912 | 0.2806 | 48,238 | 58,763 |
| 2 | 0.4356 | 0.4286 | 37,905 | 45,946 |
| 4 | 0.5955 | 0.5891 | 28,205 | 35,948 |
| 8 | 0.7474 | 0.7420 | 20,786 | 25,516 |
| 16 | 0.8710 | 0.8711 | 13,859 | 16,166 |
| 32 | 0.9486 | 0.9513 | 7,691 | 8,964 |
| 64 | 0.9860 | 0.9876 | 4,639 | 4,759 |
| 128 | 0.9984 | 0.9984 | 2,558 | 2,712 |
| 256 | 0.9998 | 0.9999 | 1,360 | 1,473 |
| 1265 | 1.0000 | 1.0000 | 544 | 471 |

#### Concurrent single-query callers, 100k x 384 (QPS; efficiency = QPS / (clients x 1-client QPS))

exact:

| clients | old QPS | old eff. | new QPS | new eff. |
|---|---|---|---|---|
| 1 | 145 | 1.00 | 156 | 1.00 |
| 2 | 152 | 0.52 | 238 | 0.76 |
| 4 | 148 | 0.26 | 450 | 0.72 |
| 8 | 149 | 0.13 | 942 | 0.75 |
| 16 | 151 | 0.07 | 1,089 | 0.44 |
| 24 | 149 | 0.04 | 1,410 | 0.38 |
| 32 | 152 | 0.03 | 1,179 | 0.24 |

ivf:

| clients | old QPS | old eff. | new QPS | new eff. |
|---|---|---|---|---|
| 1 | 798 | 1.00 | 1,017 | 1.00 |
| 2 | 773 | 0.48 | 1,678 | 0.82 |
| 4 | 764 | 0.24 | 2,482 | 0.61 |
| 8 | 720 | 0.11 | 3,170 | 0.39 |
| 16 | 798 | 0.06 | 3,333 | 0.20 |
| 24 | 798 | 0.04 | 3,734 | 0.15 |
| 32 | 780 | 0.03 | 3,930 | 0.12 |

#### Batch search, 100k x 384, 2000 queries (QPS by thread cap)

| threads | exact batch | IVF batch (nprobe 128, recall 0.955) |
|---|---|---|
| 1 | 1,217 | 8,006 |
| 2 | 2,432 | 15,497 |
| 4 | 4,648 | 28,464 |
| 8 | 8,269 | 45,112 |
| 16 | 11,514 | 56,782 |
| 24 | 13,136 | 59,332 |
| 32 | 13,553 | 57,944 |

#### One-query exact loop by thread cap, 100k x 384 (QPS, unpinned, median of 5)

| threads | old | new |
|---|---|---|
| 1 | 151.9 | 144.0 |
| 2 | 218.3 | 234.1 |
| 4 | 273.0 | 271.7 |
| 8 | 297.3 | 243.8 |
| auto | 296.8 | 282.3 |

#### Memory, 1M x 384 (working set minus baseline, MB)

| stage | old | new |
|---|---|---|
| raw vectors | 1465 | 1465 |
| after ingest | 1932 (1.32x) | 1621 (1.11x) |
| peak during ingest | 1973 (1.35x) | 1649 (1.13x) |
| after first exact query | 3516 (2.40x) | 1622 (1.11x) |
| after IVF build | 3531 (2.41x) | 1641 (1.12x) |
| peak during IVF build | 3778 (2.58x) | 1884 (1.29x) |

#### Writes on a built IVF index, 1M x 384 (median of 2 / 2 run(s))

| measure | old | new |
|---|---|---|
| insert 30k, per s | 75,848 | 126,149 |
| slowest single insert, ms | 361.44 | 3.28 |
| insert p99, ms | 0.0041 | 0.0070 |
| update 10k, per s | 339,493 | 125,666 |
| slowest single update, ms | 0.14 | 1.87 |
| delete 10k, per s | 447,213 | 1,038,675 |
| first query after 30k inserts (1 thread), ms | 1,015.85 | 5.06 |
| steady query median after them (1 thread), ms | 5.53 | 4.46 |
| first query after all 50k writes (1 thread), ms | 8.38 | 4.87 |
| query median before any write (1 thread), ms | 5.59 | 4.23 |

#### Deleting 10,000 random ids from 1M x 384

| store state | old per s (ns each) | new per s (ns each) |
|---|---|---|
| after_exact_search | 535,240 (1868) | 1,770,883 (565) |
| ivf_built | 567,585 (1762) | 281,683 (3550) |
| never_searched | 978,119 (1022) | 2,062,408 (485) |

