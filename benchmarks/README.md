# isongraph-vector benchmarks: retrieval quality and scale

Two reproducible scripts that fill the repo's evaluation gap. Everything here
only *imports* the Python package (`isongraph_vector`, bound the same way
`conftest.py` does when it is not installed); nothing in the library is
modified. The older `LLMContext_Benchmark/` measures ISON serialization
size; it does not exercise embeddings, so it says nothing about either
question below.

| script | question |
|---|---|
| `retrieval_quality.py` | Does graph-aware retrieval (`semantic_multi_hop`, `semantic_subgraph`) recover the *supporting* context a question needs, compared with plain vector top-k, and at what cost in nodes? |
| `scale.py` | How do latency, ingestion, write-after-read cost, memory and concurrent use behave from 1k to 1M vectors, and where does the exact linear scan stop being acceptable? |
| `ann.py` | What does the opt-in approximate (IVF) index buy at 100k and 1M vectors, in recall@10, latency, build time and memory, and does it survive writes and deletes? |

Sample outputs (all committed under `results/`): `retrieval_quality.{md,json}`,
`retrieval_quality_st.{md,json}` (sentence-transformers), `scale.{md,json}`
(1k/10k/100k plus concurrency) and `scale_1M.{md,json}` (10k/100k/1M, the
pre-optimisation baseline), `scale_1M_after.{md,json}` (the same 1M run after the
exact-path optimisation) and `ann.{md,json}` / `ann_1M.{md,json}` (the approximate index).

## Running

```bash
pip install numpy psutil sqlite-vec         # all optional; numpy is required by scale.py
python benchmarks/retrieval_quality.py             # ~10 s, deterministic (seed 7)
python benchmarks/retrieval_quality.py --quick     # a few seconds
python benchmarks/retrieval_quality.py --encoder st   # real embeddings (all-MiniLM-L6-v2, ~1 min, needs the model)
python benchmarks/scale.py                         # 1k/10k/100k, both backends, concurrency: ~1-2 min
python benchmarks/scale.py --quick                 # 1k and 10k
python benchmarks/scale.py --max-n 1000000         # adds 1M: ~10 minutes, ~9 GB RAM peak
python benchmarks/scale.py --sizes 10000,100000,1000000 --max-n 1000000 --skip-concurrency --out-prefix benchmarks/results/scale_1M
python benchmarks/ann.py                           # approximate index, 20k and 100k, two data regimes: ~6 min
python benchmarks/ann.py --include-1m --sizes 1000000 --no-drift --out-prefix benchmarks/results/ann_1M   # ~10 GB RAM, ~15 min
```

Both write `<prefix>.md` and `<prefix>.json` (default under `benchmarks/results/`;
`--out-prefix ""` prints only). `scale.py` skips the sqlite-vec path if the
extension is missing and skips any size that would not fit in free RAM.

## 1. Retrieval quality

### Setup

* **Graph**: a seeded synthetic operations knowledge graph, 1,710 nodes and
  ~2,600 edges by default: incidents, services, datastores (domain-local and
  shared), runbooks, teams, people. Edges: `AFFECTS`, `RELATED_TO`,
  `DEPENDS_ON`, `OWNED_BY`, `HAS_RUNBOOK`, `ONCALL`, `HAS_MEMBER`. Distractors
  are deliberate: hundreds of incidents share the same symptom vocabulary,
  every team has three non-on-call members, incidents link to other incidents.
* **Questions** (200, 50 per type), each with ground truth computed from the
  graph structure, never from a retrieval method:

  | type | seed (named in the text) | supporting (reachable only via edges) | hops |
  |---|---|---|---|
  | `incident_owner` | the incident | affected service, its team, the team's on-call person | 3 out |
  | `incident_rootcause` | the incident | affected service, its dependencies, the dependencies' teams | 3 out |
  | `service_deps` | the service | its team, direct and indirect dependencies | 2 out |
  | `impact` | a datastore | the services that depend on it | 1 **in** |

  Gold sets are capped at the smallest budget (10 nodes) so a perfect
  retriever can fit them.
* **Encoder**: `TopicHashEncoder`, deterministic and offline: hashed words and
  bigrams into 256 buckets, plus a small synonym lexicon (symptom paraphrases
  share a concept token) and area tokens (payments/billing are both
  "commerce"), L2-normalised. It gives controlled lexical and topical overlap
  with no model download. `--encoder st` uses `SentenceTransformerEncoder`.
* **Methods** (same node budget k for all, k = 10 and 20): `vector_only`
  (`similarity_search(top_k=k)`); `semantic_multi_hop` and `semantic_subgraph`
  across `max_hops` 2/3, decay 0.5/0.8/0.95, `blend` 0.3, `top_k_seeds` 1/2/5,
  direction OUT/IN/BOTH. `semantic_subgraph` is run for every configuration;
  the script asserts its node set equals `semantic_multi_hop`'s (it does in
  every case) and cross-checks the component count against the library's own
  `connected_components()`.

### Metrics

| metric | meaning |
|---|---|
| seed recall | fraction of the question's seed nodes in the returned set |
| support recall | fraction of the supporting (graph-connected) nodes in the returned set |
| `support\|seed-hit` | support recall on the fixed subset of questions where plain vector top-k already finds the seed. Separates "the graph step cannot reach the support" from "the question was never matched" (incident questions are hard to match: many near-duplicate incidents) |
| context recall | recall over seed plus supporting nodes: the headline "did we retrieve what is needed" |
| complete | fraction of questions whose *entire* gold set is returned |
| precision | fraction of returned nodes that are gold |
| nodes | returned set size (budget k, or fewer) |
| components | connected components of the returned nodes under the graph's edges (undirected). 1 means a connected context; vector top-k is typically 9 at k=10 (isolated hits) |
| W/T/L | per-question wins/ties/losses of context recall versus `vector_only` |
| median ms | per-query latency in this 1.7k-node graph (with `st` this includes encoding the query) |

### Findings (hash encoder, seed 7, 200 questions; the 3 other seeds I tried agree)

At k = 10:

| method | seed recall | support recall | context recall | complete | precision | components |
|---|---|---|---|---|---|---|
| vector_only | 0.730 | 0.027 | 0.185 | 0.00 | 0.087 | 9.3 |
| multi_hop / subgraph, OUT, 3 hops, decay 0.8 | 0.675 | 0.254 | 0.356 | 0.09 | 0.188 | 4.0 |
| same, `top_k_seeds=2` | 0.565 | 0.300 | 0.366 | 0.10 | 0.199 | 1.8 |
| same, BOTH directions | 0.675 | 0.428 | 0.474 | 0.24 | 0.236 | 3.9 |

At k = 20, vector-only reaches context recall 0.209 against 0.486 (OUT) and 0.559 (BOTH).

1. **Yes, graph-aware retrieval beats vector-only on supporting-context recall
   in this benchmark.** Context recall roughly doubles (0.19 to 0.36-0.47 at
   k=10; 0.21 to 0.49-0.56 at k=20), support recall goes from about 3% to 25-43%,
   the 95% CI on context recall is about +-0.04, and the W/T/L is 99/90/11 or better.
   Seeds 1, 2 and 3: vector 0.196/0.218/0.199 vs graph (OUT, hop 3) 0.353/0.391/0.368.
   Vector top-k finds the node the question names but almost none of the nodes
   reachable only through relationships.
2. **It is not free.** Seed recall *drops* (0.73 to 0.68, and to 0.57 with
   fewer seeds) because expanded neighbours of the best seeds outrank weaker
   seeds inside a fixed budget. Absolute completeness is low: only 9% of
   questions get their full gold set at k=10 (24% with BOTH). Precision stays
   under 0.25; most returned nodes are still not needed.
3. **The result is structured.** Returned context is 2-4 connected pieces
   rather than 9-18 isolated hits (fewer with fewer seeds). That is the real
   product value of `semantic_subgraph` as LLM context, and it is a property
   `vector_only` cannot have.
4. **Direction and fan-out dominate; the decay/blend knobs barely matter.**
   `service_deps` (out edges, tidy fan-out) goes from 0.26 to 0.78 (k=10) and
   0.31 to 0.96 (k=20). `impact` needs *incoming* edges: the default OUT
   direction gives only 0.33, `Direction.IN` gives 0.88 and BOTH 0.80. Choosing
   decay 0.5/0.8/0.95 or blend 0.3 moves context recall by about +-0.01-0.02
   at k=10. Two or three hop settings give identical results at k=10 because
   hop-3 nodes score below the cut-off; more hops only help with a larger
   budget.
5. **Incident questions stay hard.** `incident_owner`/`incident_rootcause`
   reach 0.12-0.24 context recall at k=10 because (a) the incident itself is
   found only ~50% of the time among near-duplicates and (b) a team's three
   non-on-call members tie with the on-call person under pure hop decay.
   Fewer seeds (`top_k_seeds=1-2`) helps these (0.08 to 0.15-0.18 on root cause)
   and hurts `impact`.
6. **With real embeddings the gain shrinks but holds** (`results/retrieval_quality_st.md`,
   all-MiniLM-L6-v2): context recall at k=10 is 0.287 for vector-only vs 0.419
   (OUT) and 0.523 (BOTH); at k=20, 0.343 vs 0.515 and 0.608. One caveat from
   that run: for `impact`, vector-only already scores 0.71 (the model links a
   datastore to services that mention it) and the default OUT traversal
   *loses* to it (0.53); only IN/BOTH (0.99/0.97) beat it. Graph
   expansion does not help where the text already carries the relation, and
   a wrong direction actively hurts.
7. **Latency of the graph step is small here** (about 0.4 ms for multi_hop and
   about 1.1 ms for subgraph vs 0.6 ms for vector search at 1.7k nodes), but see
   the scale section: `semantic_subgraph` also iterates the whole edge list.

Practical guidance this supports: use the graph-aware calls when answers depend
on relationships, pick `direction` from the question (or use BOTH and accept
more noise), keep `top_k_seeds` small when budget is tight, and do not expect
`decay`/`blend` tuning to matter much.

## 2. Scale

### What it measures

* `similarity_search(top_k=10)`, `semantic_multi_hop` (2 hops, 5 seeds, random
  graph of average out-degree 3; `threshold=-1` so random vectors actually
  produce seeds), `multi_hop` with `blend=0.3` (needs a score for every node),
  and `semantic_subgraph`, each the median of N warm runs (graph queries run
  one third as many).
* Ingestion: `add_batch` throughput in 5,000-vector calls (list conversion
  excluded from the timing).
* Snapshot cost: the first numpy query builds a decoded in-memory matrix of every
  vector; writes patch it in place (before the exact-path optimisation, any write dropped it). Reported: first query, one write, the search
  right after a write, the penalty versus warm, and 100-vector batch writes.
* Memory: RSS growth over a pre-build baseline (psutil sampler, paused during
  timed sections because polling inflated latencies; tracemalloc when psutil
  is missing). Reported after graph build, after ingest, after the first
  query, and peak.
* A GC-off column: the same query with the cyclic garbage collector disabled,
  because it turns out to matter (below).
* A "raw numpy floor": bare `matrix @ q` plus `argpartition` over identical
  vectors, a reference for how fast an exact scan *can* be here (not library code).
* Concurrency (20k vectors, 4 s per scenario): shared `EmbeddingStore` with
  4 readers; 4R+1W; 2R+4W; and a `graph_ops` scenario where readers run
  `semantic_multi_hop` while a writer adds nodes and edges. Reports reads/s,
  writes/s, read p50/p95, exceptions by type, and lost writes (row count versus
  successful writes).
* A least-squares line of latency against N to find where each series crosses
  50 ms and 100 ms (interpolated when bracketed by measurements, otherwise
  labelled extrapolated).

### Results (this machine: 32 logical CPUs, 47.7 GB RAM, Windows, Python 3.12, numpy 2.4, sqlite-vec 0.1.9)

Warm `similarity_search` median, ms (top_k=10, 384-d):

| N | numpy scan | sqlite-vec | raw numpy floor |
|---|---|---|---|
| 1,000 | 0.27 | 0.63 | 0.03 |
| 10,000 | 3.4 | 3.8 | 0.2 |
| 100,000 | 56 | 34 | 4 |
| 1,000,000 | 664 | 893 | 49 |

Other headline numbers:

| | numpy | sqlite-vec |
|---|---|---|
| ingest (vectors/s) at 100k / 1M | 102k / 86k | 40k / 23k |
| first query / snapshot build at 100k / 1M | 0.53 s / 5.8 s | none (0.03 s / 0.84 s is just a search) |
| search right after one write at 100k / 1M | 0.50 s / 5.3 s | 0.034 s / 0.78 s (same as warm) |
| peak RSS growth at 100k / 1M | 0.9 GB / 9.1 GB | 0.7 GB / 7.5 GB |
| raw vectors at 100k / 1M | 146 MB / 1.5 GB | same |
| `semantic_multi_hop` at 100k (GC on) | ~190 ms | 33 ms |
| `semantic_subgraph` at 100k | ~210 ms | 58 ms |

Concurrency at 20k vectors, 0 exceptions and 0 lost writes in every scenario:
numpy 4 readers about 85 reads/s (p50 50 ms); adding one writer drops that to
about 35 reads/s (p50 112 ms) because each write forces a snapshot rebuild; sqlite-vec holds about
135-143 reads/s with or without 1-4 writers (p50 about 7 ms, p95 up to 127 ms). Mixed `semantic_multi_hop` readers plus
a node/edge-adding writer also produced no errors (the graph is not documented
as thread-safe; absence of errors in 4 s is not a guarantee).

### After the exact-path optimisation

The tables above and `results/scale*.md` are the **baseline** measured before
the numpy exact path was vectorised, and are kept as the reference point.
Findings 2 and 4 below describe that baseline. Since then `EmbeddingStore`
scores with one matrix-vector product, selects the top k with
`np.partition`, builds Python objects only for those k, and patches its cached
matrix on writes (append into spare capacity, overwrite in place, tombstone
deletes) instead of discarding it. Re-run at the default sizes (numpy backend,
384-d, a clean run on the same machine):

| metric | baseline | now |
|---|---|---|
| `similarity_search` 10k, ms | 3.24 | 0.35 |
| `similarity_search` 100k, ms | 73 (56 in the committed run) | 5.6 |
| bare matmul + argpartition 100k, ms | 4.4 | 4.4 |
| `semantic_multi_hop` 100k, ms | 210 | 4.7 |
| `semantic_subgraph` 100k, ms | 249 | 30 |
| search right after one write, 100k, ms | 555 | 5.7 |
| write-after-read penalty, 100k, ms | 482 | 0.2 |
| peak memory 100k, MB | 906 | 626 |
| 4 readers, reads/s (20k) | 86 | 1,345 |
| 2 readers + 4 writers, reads/s (20k) | 15 | 1,202 |

Search is now about 1.25x the bare numpy floor instead of about 17x, so (extrapolating) the
exact scan's 50 ms crossover moves from roughly 90,000 vectors to roughly
1,000,000 on this machine. **Update: that has since been measured** (see
"1M re-run after the optimisation" below): the crossover is confirmed at about
960,000. The ANN conclusion in finding 3 has been re-measured too (section 3).
Scores can differ from the old path by one float32 ulp (about 1e-7), which can
swap the order of two results whose scores were already that close.

#### 1M re-run after the optimisation

`results/scale_1M_after.{md,json}` repeats the 1M run (10k/100k/1M, concurrency
skipped) once, with the same script and machine as `scale_1M.{md,json}` (the
baseline, kept). Numpy backend at 1M:

| metric | baseline | now |
|---|---|---|
| `similarity_search`, ms (p95) | 664 (2,143) | **52.0** (55.3) |
| bare matmul + argpartition floor, ms | 49 | 38.6 |
| `semantic_multi_hop`, ms | 2,128 | 44.3 |
| `multi_hop` with `blend=0.3`, ms | 994 | 707 |
| `semantic_subgraph`, ms | 2,320 | 346 |
| first query (cache build), s | 5.8 | 3.8 |
| search right after one write, ms | 5,251 | 49.9 |
| ingest, vectors/s | 85,764 | 84,062 |
| peak RSS growth, MB | 9,075 | 5,996 |

The sqlite-vec series did not change (348 ms at 1M against 893 ms before; run
to run noise on that backend is large). The fitted 50 ms crossover for the numpy
scan is now **about 960,000 vectors** (interpolated between the 100k and 1M
measurements; the 1M median itself is 52 ms), against 90,000 before. So the
extrapolated claim "roughly 1,000,000" holds: it was off by about 4%, and the
floor (the bare matmul) reaches 50 ms at about 1.3M. Two things remain slow at
1M: `blend > 0` (707 ms: `score_map` builds a Python dict over every node, and
it is not touched by the index) and `semantic_subgraph` (346 ms, it iterates
every edge).

### Findings

1. **Where exact search stops being comfortable.** The numpy path crosses
   **50 ms at roughly 90,000 vectors** (fit across 10k-1M; run-to-run range
   about 40k-110k, see caveats) and 100 ms at about 165,000. sqlite-vec crosses 50 ms
   at roughly 120,000-150,000. Up to about 20,000-30,000 vectors (under about 10 ms) none of this
   matters. At 1M, one query is 0.7-0.9 s: unusable interactively.
2. **The cost is mostly not the arithmetic.** A bare matrix product plus
   argpartition over the same data takes 4 ms at 100k and 49 ms at 1M, about 10-14x
   faster than `similarity_search`. The library's `_score_all` builds a Python
   tuple per stored vector on every query, then filters and heaps them; that
   per-row Python work is the bottleneck. A leaner exact path (score,
   `argpartition`, build only k results) would push the 50 ms crossover out
   by roughly 10x before any ANN index is needed. This is the cheapest win
   available.
3. **ANN index recommendation.** Not warranted below roughly 50k vectors, and
   at most a "nice to have" up to 100-200k *if* the exact path is first made
   leaner. Above about 300k vectors an ANN index (or a different vector
   store) is warranted: even a perfectly optimised exact scan costs about 50 ms per
   query at 1M, and memory is already 7-9 GB at that size. Both backends are
   exact linear scans; `sqlite_vec=True` shifts the curve but does not
   change its shape. At 1M the sqlite-vec scan (0.9 s) is slower than the numpy
   snapshot (0.66 s) despite being faster at 100k (34 ms vs 56 ms), so
   its advantage is range-limited, and it is also what makes writes cheap.
4. **Write-after-read is the sharpest edge of the numpy path.** One write
   invalidates the snapshot; the next query costs 35 ms at 10k, 0.5 s at 100k,
   5.3 s at 1M, about 8x the warm query. Interleaved read/write workloads
   on the default path lose most of the benefit of caching (see the concurrency
   numbers). Batch writes, or use `sqlite_vec=True`, where a write after a
   read costs nothing extra.
5. **Memory is about 5-6x the raw vectors.** 1M x 384 floats is 1.5 GB raw, but
   peak RSS growth is 9.1 GB (numpy: the rows are fetched, joined, then
   decoded while the database copy is also resident) and 7.5 GB (sqlite-vec,
   which keeps the vec0 table next to the `embeddings` table). Budget around 6x raw
   vector size for in-memory databases.
6. **Garbage collection is a real tail-latency source in a large process.** With
   100k nodes and 300k edges resident, p95 of `similarity_search` is about 175-200
   ms against a 56 ms median, and `semantic_multi_hop` is about 190 ms at the
   median, because the per-query allocations trigger full collections that walk
   millions of objects. With the GC disabled during the call: 45 ms and 62 ms.
   Per-query allocations (the tuple per vector) are therefore doubly costly,
   and `gc.freeze()` after loading a large graph is worth trying (not tested here).
7. **`semantic_subgraph` cost scales with edges.** It iterates every edge in
   the graph to find those between kept nodes (`self.edges(rel_type)`), so
   it is the slowest call: about 2.3 s at 1M with numpy.
8. **Ingestion**: ~100k vectors/s (numpy path), ~40k/s with the vec0 index,
   falling to 86k and 23k per second at 1M. A 1M-vector load takes 12 s and
   44 s respectively, plus about 6 s for the first query to build the snapshot.

## 3. Approximate index (`ann.py`)

`EmbeddingStore(ann="ivf")` (numpy only, off by default) is an IVF-flat index:
spherical k-means into `nlist` cells (about `4*sqrt(N)` when automatic), then a
query scores only the `nprobe` cells nearest to it (about 5% of the cells when
automatic) with the same cosine formula as the exact scan, and the best k of
those rows are returned. Vectors are not copied. `ann.py` measures what that
buys and costs. Unlike `scale.py` it uses **clustered** synthetic vectors,
because an IVF index's recall depends entirely on cluster structure (on uniform
random 384-d vectors it has none to exploit and would say nothing):

| regime | construction | what it stands for |
|---|---|---|
| `clustered` | N/100 Gaussian clusters; same-cluster cosine about 0.5, different-cluster about 0 | distinct topics: the easy case |
| `diffuse` | N/40 clusters grouped into 20 overlapping super-topics; same-cluster cosine about 0.2, same super-topic about 0.1 | a hard, overlapping case where the 10 nearest neighbours straddle cells |

Queries are fresh draws from the same distribution (not stored vectors), 200 per
cell, warm single-thread medians, top_k = 10, 384-d. Recall@10 is the overlap of
the approximate and exact top-10 sets. Raw tables: `results/ann.md` (20k and
100k, with the index under writes) and `results/ann_1M.md` (1M, run once).

### Results (same machine as above, a shared machine, one run each)

| N | regime | exact scan | build | nprobe=1 | automatic nprobe (5% of cells) | smallest nprobe in the sweep reaching recall >= 0.99 |
|---|---|---|---|---|---|---|
| 100k | clustered | 4.6 ms | 1.4 s | recall 0.948, 0.29 ms (16x) | recall 1.000, 1.22 ms (3.8x) | 2 cells (0.2%): 0.997, 0.22 ms (21x) |
| 100k | diffuse | 5.0 ms | 1.4 s | recall 0.229, 0.26 ms (19x) | recall 0.987, 1.23 ms (4.1x) | 128 cells (10%): 0.996, 2.2 ms (2.3x) |
| 1M | clustered | 53.7 ms | 23.3 s | recall 0.977, 0.33 ms (163x) | recall 0.997, 11.4 ms (4.7x) | 8 cells (0.2%): 0.990, 0.87 ms (62x) |
| 1M | diffuse | 48.2 ms | 23.7 s | recall 0.030, 0.64 ms (75x) | recall 0.994, 24.3 ms (2.0x) | 200 cells (5%): 0.994, 24 ms (2.0x) |

(Intermediate points from the sweep tables: 1M diffuse nprobe 64 = 1.6% of cells
gives recall 0.704 at 11.8 ms and 128 = 3.2% gives 0.930 at 16.6 ms; 100k diffuse
nprobe 32 gives 0.876 at 0.86 ms.)

### Findings

1. **Recall is a property of your data, not of the index.** At identical settings
   recall@10 ranged from 0.03 to 0.98 at nprobe=1 depending only on the regime,
   and the cost of reaching 0.99 ranged from 0.2% of the cells to 5-10%.
   Clustered, topic-like data is where this works well; data whose neighbours
   straddle cell boundaries needs a large nprobe, and then the saving is small.
   There is no setting that is good in general. `recall_check(queries, k)` runs
   your queries both ways and reports recall; use it on your own data and raise
   `ann_nprobe` until the number you need holds.
2. **The saving is smaller than the fraction of rows scanned.** Scanning 5% of
   a 1M-vector store costs about 20-50% of the exact scan, not 5%: the exact
   scan is one multithreaded sequential matrix product (about 0.05 us per row),
   while the index gathers its candidate rows from scattered places in the
   1.5 GB matrix on one thread (about 0.22 us per row at both 100k and 1M).
   Layout work (a cell-contiguous copy of the vectors, which doubles their
   memory, and/or a threaded scan) would recover some of it; neither is done.
   One real trap found along the way: gathering the candidates in a single
   fancy-index copy was 5-7x slower than doing it in blocks of about 384 KB
   (page-faulting the fresh copy dominated); the implementation blocks.
3. **Where it starts to pay off.** Up to about 100k vectors the exact scan is
   4-5 ms and the index saves at most a few ms while risking recall; it is not
   worth enabling. The exact scan reaches 50 ms at about 1M vectors (section
   2), and that is where the index becomes worth it: with clear cluster
   structure it is 20-160x faster at recall 0.98-0.99 (1M clustered, nprobe
   1-8); with hard data it is only about 2-3x faster at recall 0.93-0.99. `ann_min_size` defaults to 50,000 so
   a small store is never approximated, but that is a floor, not a
   recommendation.
4. **Build time and memory.** Built lazily by the first search (or
   `build_ann_index()`): 1.4 s at 100k and about 23 s at 1M (nlist 4000), during
   which the first query blocks. The index itself is small: 2.7 MB at 100k and
   14 MB at 1M (about 1% of the 1.5 GB of vectors, since vectors are not
   copied), and building needs a transient 0.22 GB (100k) to 0.56 GB (1M) on top.
5. **Writes and deletes.** The index follows the incremental cache instead of
   being rebuilt. At 100k with the automatic nprobe, +1% new vectors and -1%
   deletes (new rows scored exactly while "pending") left recall at 1.000
   (clustered) and 0.986 (diffuse) against 1.000 and 0.987 before, at +0.1 ms (the
   1,000 pending rows are scanned exactly on every query); a
   further +2% was folded into cells with the same recall. Pushing churn past
   50% of the trained size (+60% more vectors) retrains on the next search: that
   query took 2.2-2.4 s at 100k (0.5 s at 20k); the retrain at 1M would be
   longer than the 23 s build and was not measured. Recall after the retrain
   matched (1.000 and 0.986). Retraining blocks the query that triggers it;
   call `build_ann_index(force=True)` at a quiet moment to choose when.
6. **Filters.** `node_type`, `threshold` and `node_filter` are applied to the
   probed rows. If fewer than `top_k` survive, the probe widens 4x per step and
   past half the cells the exact scan answers. A rare type or a selective filter
   therefore costs time instead of returning a short list; the tests cover it.
   Not benchmarked here.

Not evaluated: real embeddings and real queries, concurrent read load
(everything is single-threaded and serialised on the store's lock), the
retrain at 1M, and hnswlib. An HNSW graph typically beats IVF-flat on
recall at a given latency; it was not added because it needs a dependency and
the pure-numpy index stays the default.

## Caveats (please read before quoting numbers)

* **Everything is synthetic.** Retrieval quality uses a generated graph and a
  hand-built hashing encoder with a hand-written synonym lexicon. Absolute
  recall values mean nothing for any real corpus; only the *comparisons*
  between methods on the same data are informative, and the
  sentence-transformers run shows the size of the gap depends on the
  encoder (and that graph expansion can lose when the text already encodes the
  relation, e.g. `impact` with the OUT direction).
* Ground truth encodes my view of "what is needed to answer"; with other
  definitions of supporting context the numbers shift. The `incident_*`
  questions are partly limited by seed identification, which is why
  `support|seed-hit` exists. Gold sets are capped at 10 nodes to fit the
  smallest budget; larger real answers would score lower for every method.
* Scale uses random vectors (no cluster structure, so no recall measurement).
  Exact-search latency does not depend on the data distribution, so those
  numbers transfer; an ANN index's recall/latency tradeoff would not, which is
  why `ann.py` (section 3) uses clustered vectors and says so. The ANN numbers
  are synthetic too.
* **Timings vary run to run on a shared machine.** Other processes were
  using the machine while these were collected; numpy `similarity_search` at
  100k measured 30 ms in an ad hoc check, 54-70 ms in repeated runs, and 142 ms
  once while it was busy. The sample files are representative runs, not best
  runs. Re-run on your hardware; the *shape* (linear in N, write-after-read
  penalty about 8x, per-row Python overhead about 10x) is the stable part.
  The 1M figures are a single run each.
* The linear fit for the crossover includes 1M points that behave differently
  from 100k (sqlite-vec jumps 27x for 10x the data, probably memory-bound), so
  crossings that bracket 1M are interpolations over a curved series. Use the
  measured table, not the fitted N, for planning.
* In-memory SQLite only. A file-backed database would add I/O to ingestion and
  to the cold snapshot build.
* Single process, one run per configuration, Windows. Concurrency scenarios
  are 4 s each and show absence of errors in that window, not proof of thread
  safety.
