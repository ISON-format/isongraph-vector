# Multi-hop retrieval benchmark: does graph-aware retrieval beat plain vector search on real data?

Real public multi-hop QA data (HotpotQA, 2WikiMultihopQA, MuSiQue), real embeddings (all-MiniLM-L6-v2 and bge-small-en-v1.5), a graph that is built **without** the question, the answer or any supporting-fact annotation, and the library under test (`isongraph_vector-py`, imported unmodified from this checkout). Plain vector search over the same vectors is what every vector store returns, so it is the baseline that matters. The earlier `benchmarks/retrieval_quality.py` used synthetic data; **nothing from it is reused here**.

Run date 2026-11-15/16, one Windows 11 machine (32 logical cores, RTX 4060 used only for embedding/re-ranking), Python 3.12, numpy 2.4, scipy 1.16, torch 2.5.1+cu121, sentence-transformers 5.1.2, `ison-graph` 1.4.0.

## 1. Result in one page

**Headline.** With a graph built from nothing but passage titles and text (title mentions, a Wikipedia-hyperlink proxy), `semantic_multi_hop` (parameters tuned on a separate dev split) retrieves **all** supporting passages significantly more often than plain vector top-k, at every k (5/10/20), on all three datasets, with both encoders (paired p < 0.001 in all 18 cells; all-supporting@10 vs plain vector shown below, test split, 1,000 questions each). With the *untuned library defaults* the gain at k = 10 is still significant everywhere, but at k = 20 it is not significant on MuSiQue and HotpotQA/bge:

| Dataset / encoder | plain vector | ISONGraph multi_hop (library defaults) | ISONGraph multi_hop (tuned on dev) | cross-encoder rerank | hybrid RRF | independent PPR re-rank (tuned) |
|---|---|---|---|---|---|---|
| HotpotQA / MiniLM | 58.2 | 77.3 (+19.1) | **81.5 (+23.3)** | 68.4 | 72.5 | 81.4 |
| HotpotQA / bge-small | 78.5 | 87.5 (+9.0) | **91.5 (+13.0)** | 84.1 | 83.8 | 91.9 |
| 2Wiki / MiniLM | 32.5 | 79.5 (+47.0) | **81.0 (+48.5)** | 37.5 | 41.1 | 78.7 |
| 2Wiki / bge-small | 43.9 | 94.6 (+50.7) | **94.6 (+50.7)** | 46.0 | 45.6 | 92.8 |
| MuSiQue / MiniLM | 18.2 | 24.5 (+6.3) | **27.4 (+9.2)** | 23.7 | 20.4 | 28.9 |
| MuSiQue / bge-small | 20.9 | 26.3 (+5.4) | **29.6 (+8.7)** | 24.6 | 23.3 | 29.1 |

(All-supporting@10, %. 95% CIs and paired p-values are in the tables below and in `results/`.) Negative controls (same graphs with node labels randomly permuted) give **exactly the plain-vector numbers** after tuning (|delta| <= 0.2 pp), and the library defaults on a permuted graph are *worse* than plain vector (-5 to -11 pp): the gain comes from the graph's content, not from the expansion machinery.

**What this does NOT show - read before quoting the table:**

1. **The library does not beat a simple independent baseline.** A HippoRAG-style personalised-PageRank re-ranker that we implemented separately (no ISONGraph code) on the *same* graph is statistically indistinguishable from `semantic_multi_hop` at k = 10 and 20 on HotpotQA and MuSiQue (differences within about 1.5 pp, CIs include 0). `semantic_multi_hop` is better at small budgets (k = 5: +7.5 pp HotpotQA/MiniLM, +12.0 / +18.9 pp on 2Wiki MiniLM / bge, paired p < 0.001; not significant on HotpotQA/bge and MuSiQue) and slightly better at k = 10 on 2Wiki (+0.9 to +2.3 pp), but that is a statement about *our* PPR re-ranker's score fusion at small k, not a structural advantage. **The defensible differentiator is "graph-aware retrieval", not "the ISONGraph implementation of it".**
2. **The gain depends on graph quality.** Title-mention edges (+9 to +51 pp) >> shared-entity edges (+14 to +30 pp, only run on HotpotQA/MiniLM and 2Wiki) >> kNN edges (+4 to +20 pp). The kNN graph - the only one that needs no text structure at all - gives +4.0 / +4.5 pp on HotpotQA (MiniLM / bge) and +7.3 / +20.0 pp on 2Wiki.
3. **The mention graph is aligned with how these datasets were built** (hyperlink-based question construction; see Threats 1 and 2), so the large numbers are an upper-ish bound. MuSiQue, where only 15% of gold pairs are directly linked by title mentions, shows the realistic scale of the gain when the graph is a weak proxy: **+9 pp** (18 -> 27%).
4. **Graph expansion hurts one question type.** On HotpotQA *comparison* questions (both entities named in the question) tuned `multi_hop` is -5.6 pp (MiniLM) / -0.9 pp (bge, n.s.) vs plain vector, and on 2Wiki comparison -8.0 pp (MiniLM) / -0.8 pp (bge, n.s.). The gain is entirely in *bridge / compositional / inference / bridge-comparison* questions.
5. **Deep chains stay hard.** MuSiQue 3-hop questions: all methods are at 7-17% all-supporting@10 and `multi_hop` adds only +3.4 / +3.8 pp (CIs barely exclude 0); 4-hop questions: 0-3% for every method, no gain.
6. **Not every ISONGraph feature was run to completion.** The union graph `all` (mention+entity+kNN) never finished within the time budget and has **no results**; `mention+entity` exists only for 2Wiki/MiniLM; `entity` is missing for HotpotQA/bge and all MuSiQue runs; `knn` is missing for MuSiQue. Details in Section 4.

## 1b. Claims the evidence supports, and claims it does not

Supported (test split, CIs in `results/tables.md`):

* "Graph-aware retrieval over a title-mention graph retrieves all supporting passages more often than plain vector top-k on HotpotQA, 2WikiMultihopQA and MuSiQue, with two different encoders, at budgets of 5, 10 and 20 passages." (+8.7 to +50.7 pp at k = 10, every paired p < 0.001; every CI excludes 0.)
* "The gain exceeds what a hybrid (vector+BM25) or a cross-encoder reranker over the same candidates gives": `multi_hop` minus RRF = +6.3 to +49.0 pp, minus cross-encoder = +3.7 to +48.6 pp at k = 10 (all p <= 0.005; `results/pairs_vs_nongraph.md`). Exception: MuSiQue/MiniLM at k = 5, where it ties the cross-encoder (+0.6, p = 0.67).
* "The gain is a property of the graph, not of the traversal mechanics": label-permuted graphs give no gain (Section 5).
* "Library defaults already capture most of the gain" (`max_hops=2, top_k_seeds=5, decay=0.8, blend=0`, mention graph): +5.4 to +50.7 pp over vector at k = 10 (all p < 0.001); tuning on dev adds 0 to +4.2 pp on top (at k = 20 the defaults are not significantly better than vector on MuSiQue and HotpotQA/bge).
* "semantic_multi_hop is faster than the PPR re-ranker at equal quality in some configurations and about the same in others" (latency table; single-thread, ms-scale).

Not supported:

* "ISONGraph's traversal is better than other graph-aware re-ranking." It ties an independent PPR re-ranker at k >= 10 (see 1).
* "Graph-aware retrieval helps when the graph is built from embedding similarity alone." Gains are small on HotpotQA (+4 pp), larger on 2Wiki; kNN edges on MuSiQue were not run.
* "Graph-aware retrieval helps on every question type" (comparison questions get worse), or on deep chains (3-4 hops on MuSiQue).
* "These gains will hold in a 5M-passage index" (pooled corpora of 9k-15k passages; Threat 2) or with stronger encoders (only two small ones).
* "`semantic_subgraph(connect=True)` adds value over `semantic_multi_hop`": in these runs the connector step added almost no nodes (mean set size 8-12 vs k = 10), so the set quality equals `multi_hop`'s; the size-matched comparison against vector top-|S| is in Section 3.
* Anything about answer quality / generation: only retrieval of supporting paragraphs is measured.
* Anything about the `all`, and most `mention+entity` / `entity` / `knn` variants outside the cells listed in Section 4.


## 2. What was run

### 2.1 Data (dev sets only, pinned revisions)

| Dataset | Source (HuggingFace dataset repo @ revision, file) | Questions available | Sampled | dev / test | Pooled corpus (unique passages) |
|---|---|---|---|---|---|
| HotpotQA (distractor) | `hotpotqa/hotpot_qa` @ `1908d6afbbea...`, `distractor/validation-00000-of-00001.parquet` | 7,405 | 1,500 | 500 / 1,000 | 14,570 |
| 2WikiMultihopQA | `framolfese/2WikiMultihopQA` @ `fe713bfbd1af...`, `data/validation-00000-of-00001.parquet` (mirror of the official dev set, 12,576 q) | 12,576 | 1,500 | 500 / 1,000 | 9,239 |
| MuSiQue (answerable) | `dgslibisey/MuSiQue` @ `c8f4f8c9465f...`, `musique_ans_v1.0_dev.jsonl` (official v1.0 answerable dev) | 2,417 | 1,500 | 500 / 1,000 | 15,405 |

* Sampling: questions sorted by id, `random.Random(13).sample(..., 1500)`, then a seeded shuffle assigns the first 500 to **dev** (used only to choose hyper-parameters and the graph variant) and the other 1,000 to **test** (all reported numbers). The exact ids and split are in `manifests/<dataset>_sample.json`.
* Corpus: **one corpus per dataset** = every context paragraph (gold + the dataset's own distractors) of all 1,500 sampled questions, de-duplicated by (title, text). A question is searched against the whole pool, so its competitors are its own distractors *and* every other question's gold and distractor paragraphs. Passage text embedded: `"<title>. <text>"`.
* Gold = the dataset's supporting paragraphs (HotpotQA/2Wiki: paragraphs whose title is in `supporting_facts`; MuSiQue: `is_supporting`). 2 gold passages for every HotpotQA question; 2 or 4 for 2Wiki (2: 1,183 q, 4: 317 q); 2/3/4 for MuSiQue (784/459/257 q).
* Hop count = number of gold passages (HotpotQA: always 2).

### 2.2 Embeddings

`sentence-transformers/all-MiniLM-L6-v2` (384-d) and `BAAI/bge-small-en-v1.5` (384-d; documented query instruction prefix on questions only). `max_seq_length=256`, L2-normalised, fp32, computed once on GPU and cached as `.npy` in `D:\Goldmine\ISON\bench_data\multihop\emb_<dataset>__<model>__{passages,queries}.npy`. Identical vectors feed every method.

### 2.3 Graph construction (no gold annotation, no question, no answer)

Built once per dataset from passage text/titles only (kNN: from the passage embeddings). Constants were fixed a priori and not tuned.

| Variant | Edges |
|---|---|
| `mention` | directed A -> B if B's title (parenthetical stripped, >= 4 chars, exact case-sensitive token match) occurs in A's text; titles mentioned by > 30 passages dropped as hubs. (A proxy for Wikipedia hyperlinks: the HF copies carry no link markup.) |
| `entity` | undirected, A -- B if they share an "entity": maximal run of capitalised tokens (with of/the/de connectors; sentence-initial single tokens and a stop list removed) or A/B's own title; entities with document frequency outside [2, 15] dropped. Edge weight `w = n/(n+1)` for `n` shared entities. |
| `knn` | undirected, mutual-or-one-way 5-nearest-neighbours under the *same* encoder; `w` = cosine. |
| `mention+entity`, `all` | unions (separate relationship types). |
| `ctrl_mention`, `ctrl_entity` | **negative controls**: the `mention` / `entity` edge lists with node labels randomly permuted (an isomorphic graph whose edges no longer correspond to the text). |

`results/tables.md` ("Graph diagnostics") reports edge counts, degrees and - for *description only* - how often two gold passages of a question are directly linked.

### 2.4 Methods (all return passage ids; identical budget k, k in {5, 10, 20})

1. **Vector top-k**: exact cosine brute force (numpy). Checked equal to `SemanticGraph.similarity_search` (see Section 6).
2. **BM25**: own Okapi implementation (k1=1.2, b=0.75), lower-cased `[a-z0-9]+` tokens, small stop list, no stemming.
3. **Hybrid**: reciprocal rank fusion (k0=60) of vector and BM25 full rankings.
4. **ISONGraph Vector `semantic_multi_hop`** from this checkout, unmodified, queried with the cached query vector (`threshold=-1.0` so seeds are never dropped). Searched on **dev**: random sample of 100 configurations (seeded; **40 in the runs marked "lean" in Section 4**, which were re-launched with a smaller budget when the first pass ran out of time) from `hops` {1,2,(3 for mention graphs)} x `top_k_seeds` {5,10,15,20} x `decay` {.3,.5,.7,.9} x `blend` {0,.3,.6} x `direction` {OUT, BOTH} (graphs with directed mention edges) x `weight_field` {none, `w`} (weighted graphs), plus the library defaults (`max_hops=2, top_k_seeds=5, decay=0.8, blend=0`) as a separate untuned row. One configuration is selected **per (dataset, encoder, graph, k)** by dev all-supporting@k (ties: dev recall@k) and then evaluated once on test. `top_k_results=20`, truncated to k (identical to asking for k, the list is sorted).
5. **`semantic_subgraph(connect=True)`** with the dev-selected parameters, `connect_max_hops` in {2, 3}. The returned set can be larger than k (connectors are extra), so it is compared with plain vector search at *the same per-query set size* (vector top-|S|), not only at k.
6. **Personalised PageRank re-ranking** (HippoRAG-style, implemented independently in `methods.py`, scipy sparse, 25 power iterations): restart distribution = top-`seeds` cosine hits weighted by cosine, `p <- alpha P^T p + (1-alpha) s` (dangling mass restarts), final score `(1-beta) p/max(p) + beta cos/max(cos)`. **Full** grid on dev (72 configurations; 144 for union graphs): `alpha` {.5,.7,.85} x `seeds` {5,10,20} x `beta` {0,.3,.6,.85} x directed/symmetrised mention edges or unweighted/weighted edges. (PPR gets a comparable or larger search than the library, which is conservative for the library.)
7. **Cross-encoder rerank**: vector top-30 re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-k returned.

"Graph chosen on dev" rows: among the five real graph variants (controls excluded), the one with the best dev all-supporting@k for that method.

### 2.5 Metrics and statistics

Per question: recall@k (|gold in top-k| / |gold|), **all-supporting@k** (every gold passage in top-k; the multi-hop success metric), MRR@k of the first gold passage, number of passages actually returned. Means over the 1,000 test questions; 95% CIs by question-level percentile bootstrap (2,000 resamples). Paired comparison with plain vector search: difference of per-question values, bootstrap CI of the mean difference, and a paired sign-flip permutation test (10,000 permutations), Holm-corrected within each table.

## 3. Results

All numbers: test split (1,000 questions per dataset), percent unless stated; "(+x, p)" is the paired difference to plain vector search in percentage points with the paired sign-flip p-value; 95% CIs (bootstrap) are in the full tables `results/tables.md` and `results/summary.json`. "[graph]" = the graph variant chosen on dev for that method and k. BM25 does not depend on the encoder (the two columns of the same dataset show the same system).


### Headline, k = 10: all-supporting@10 % [95% CI] (Delta vs plain vector in pp, paired p)

| Method | hotpotqa/minilm | hotpotqa/bge | 2wiki/minilm | 2wiki/bge | musique/minilm | musique/bge |
|---|---|---|---|---|---|---|
| Vector top-k (plain) | 58.2 [55.2, 61.3] | 78.5 [76.0, 80.9] | 32.5 [29.6, 35.2] | 43.9 [40.9, 46.9] | 18.2 [15.8, 20.6] | 20.9 [18.4, 23.4] |
| BM25 | 69.1 [66.3, 71.9] (+10.9, p<0.001) | 69.1 [66.3, 71.9] (-9.4, p<0.001) | 41.5 [38.5, 44.5] (+9.0, p<0.001) | 41.5 [38.5, 44.5] (-2.4, p=0.019) | 15.2 [13.0, 17.3] (-3.0, p=0.013) | 15.2 [13.0, 17.3] (-5.7, p<0.001) |
| Hybrid RRF (vector+BM25) | 72.5 [69.7, 75.1] (+14.3, p<0.001) | 83.8 [81.4, 86.0] (+5.3, p<0.001) | 41.1 [38.1, 44.1] (+8.6, p<0.001) | 45.6 [42.4, 48.5] (+1.7, p=0.046) | 20.4 [17.8, 22.9] (+2.2, p=0.033) | 23.3 [20.8, 26.0] (+2.4, p=0.040) |
| Vector top-30 + cross-encoder | 68.4 [65.7, 71.1] (+10.2, p<0.001) | 84.1 [81.9, 86.3] (+5.6, p<0.001) | 37.5 [34.5, 40.2] (+5.0, p<0.001) | 46.0 [43.0, 49.0] (+2.1, p=0.011) | 23.7 [21.2, 26.3] (+5.5, p<0.001) | 24.6 [22.1, 27.1] (+3.7, p<0.001) |
| ISONGraph multi_hop, library defaults | 77.3 [74.7, 79.8] (+19.1, p<0.001) [mention] | 87.5 [85.4, 89.5] (+9.0, p<0.001) [mention] | 79.5 [77.0, 81.9] (+47.0, p<0.001) [mention] | 94.6 [93.2, 96.0] (+50.7, p<0.001) [mention] | 24.5 [21.8, 27.3] (+6.3, p<0.001) [mention] | 26.3 [23.6, 29.0] (+5.4, p<0.001) [mention] |
| ISONGraph multi_hop, tuned | 81.5 [79.1, 83.8] (+23.3, p<0.001) [mention] | 91.5 [89.7, 93.2] (+13.0, p<0.001) [mention] | 81.0 [78.6, 83.4] (+48.5, p<0.001) [mention+entity] | 94.6 [93.2, 96.0] (+50.7, p<0.001) [mention] | 27.4 [24.7, 30.2] (+9.2, p<0.001) [mention] | 29.6 [26.8, 32.4] (+8.7, p<0.001) [mention] |
| PPR re-rank, tuned (independent impl.) | 81.4 [79.0, 83.7] (+23.2, p<0.001) [mention] | 91.9 [90.1, 93.5] (+13.4, p<0.001) [mention] | 78.7 [76.2, 81.1] (+46.2, p<0.001) [mention] | 92.8 [91.2, 94.4] (+48.9, p<0.001) [mention] | 28.9 [26.1, 31.6] (+10.7, p<0.001) [mention] | 29.1 [26.4, 31.9] (+8.2, p<0.001) [mention] |
| CONTROL: multi_hop on label-shuffled mention graph | 58.2 [55.2, 61.3] (+0.0, p=1.000) | 78.4 [75.9, 80.8] (-0.1, p=1.000) | 32.5 [29.6, 35.2] (+0.0, p=1.000) | 43.7 [40.7, 46.7] (-0.2, p=0.497) | 18.2 [15.8, 20.6] (+0.0, p=1.000) | 20.9 [18.4, 23.4] (+0.0, p=1.000) |
| CONTROL: multi_hop on label-shuffled entity graph | n/a | n/a | 32.5 [29.6, 35.2] (+0.0, p=1.000) | 43.8 [40.8, 46.8] (-0.1, p=1.000) | n/a | n/a |

### Headline, k = 5: all-supporting@5 % [95% CI] (Delta vs plain vector in pp, paired p)

| Method | hotpotqa/minilm | hotpotqa/bge | 2wiki/minilm | 2wiki/bge | musique/minilm | musique/bge |
|---|---|---|---|---|---|---|
| Vector top-k (plain) | 47.2 [44.1, 50.3] | 68.4 [65.5, 71.2] | 26.2 [23.5, 28.7] | 38.3 [35.3, 41.2] | 11.9 [10.0, 13.9] | 14.3 [12.2, 16.4] |
| BM25 | 48.7 [45.7, 51.7] (+1.5, p=0.437) | 48.7 [45.7, 51.7] (-19.7, p<0.001) | 33.8 [31.1, 36.7] (+7.6, p<0.001) | 33.8 [31.1, 36.7] (-4.5, p<0.001) | 11.1 [9.3, 12.9] (-0.8, p=0.503) | 11.1 [9.3, 12.9] (-3.2, p=0.003) |
| Hybrid RRF (vector+BM25) | 54.3 [51.1, 57.4] (+7.1, p<0.001) | 69.1 [66.3, 71.9] (+0.7, p=0.621) | 33.2 [30.3, 36.0] (+7.0, p<0.001) | 38.7 [35.8, 41.8] (+0.4, p=0.695) | 12.6 [10.6, 14.6] (+0.7, p=0.451) | 14.9 [12.8, 17.2] (+0.6, p=0.585) |
| Vector top-30 + cross-encoder | 59.7 [56.6, 62.6] (+12.5, p<0.001) | 71.6 [68.8, 74.4] (+3.2, p=0.028) | 33.8 [30.9, 36.5] (+7.6, p<0.001) | 39.4 [36.4, 42.4] (+1.1, p=0.289) | 17.8 [15.5, 20.2] (+5.9, p<0.001) | 18.9 [16.6, 21.2] (+4.6, p<0.001) |
| ISONGraph multi_hop, library defaults | 67.6 [64.8, 70.4] (+20.4, p<0.001) [mention] | 73.4 [70.8, 76.1] (+5.0, p<0.001) [mention] | 53.5 [50.5, 56.6] (+27.3, p<0.001) [mention] | 47.7 [44.7, 50.8] (+9.4, p<0.001) [mention] | 18.2 [15.9, 20.6] (+6.3, p<0.001) [mention] | 17.3 [15.1, 19.7] (+3.0, p<0.001) [mention] |
| ISONGraph multi_hop, tuned | 71.1 [68.3, 73.8] (+23.9, p<0.001) [mention] | 81.3 [78.8, 83.6] (+12.9, p<0.001) [mention] | 63.8 [60.8, 66.8] (+37.6, p<0.001) [mention+entity] | 77.8 [75.3, 80.4] (+39.5, p<0.001) [mention] | 18.4 [16.1, 20.8] (+6.5, p<0.001) [mention] | 21.6 [19.1, 24.2] (+7.3, p<0.001) [mention] |
| PPR re-rank, tuned (independent impl.) | 63.6 [60.8, 66.4] (+16.4, p<0.001) [mention] | 79.6 [77.1, 82.0] (+11.2, p<0.001) [mention] | 51.8 [48.7, 54.9] (+25.6, p<0.001) [mention] | 58.9 [55.9, 61.9] (+20.6, p<0.001) [mention] | 18.3 [16.0, 20.6] (+6.4, p<0.001) [mention] | 20.2 [17.7, 22.6] (+5.9, p<0.001) [mention] |
| CONTROL: multi_hop on label-shuffled mention graph | 47.2 [44.1, 50.3] (+0.0, p=1.000) | 67.5 [64.5, 70.4] (-0.9, p=0.003) | 26.2 [23.5, 28.8] (+0.0, p=1.000) | 38.4 [35.4, 41.3] (+0.1, p=1.000) | 12.0 [10.1, 14.0] (+0.1, p=1.000) | 14.3 [12.2, 16.4] (+0.0, p=1.000) |
| CONTROL: multi_hop on label-shuffled entity graph | n/a | n/a | 26.2 [23.5, 28.8] (+0.0, p=1.000) | 38.4 [35.4, 41.3] (+0.1, p=1.000) | n/a | n/a |

### Headline, k = 20: all-supporting@20 % [95% CI] (Delta vs plain vector in pp, paired p)

| Method | hotpotqa/minilm | hotpotqa/bge | 2wiki/minilm | 2wiki/bge | musique/minilm | musique/bge |
|---|---|---|---|---|---|---|
| Vector top-k (plain) | 66.2 [63.4, 69.1] | 85.4 [83.2, 87.5] | 38.4 [35.3, 41.3] | 48.2 [45.2, 51.2] | 26.3 [23.6, 29.0] | 28.6 [25.9, 31.4] |
| BM25 | 82.2 [79.8, 84.4] (+16.0, p<0.001) | 82.2 [79.8, 84.4] (-3.2, p=0.015) | 46.7 [43.8, 49.8] (+8.3, p<0.001) | 46.7 [43.8, 49.8] (-1.5, p=0.114) | 23.5 [21.0, 26.1] (-2.8, p=0.047) | 23.5 [21.0, 26.1] (-5.1, p<0.001) |
| Hybrid RRF (vector+BM25) | 83.8 [81.6, 86.0] (+17.6, p<0.001) | 89.6 [87.8, 91.5] (+4.2, p<0.001) | 46.7 [43.7, 49.7] (+8.3, p<0.001) | 49.4 [46.4, 52.4] (+1.2, p=0.173) | 31.9 [29.0, 34.8] (+5.6, p<0.001) | 32.9 [30.1, 35.8] (+4.3, p=0.001) |
| Vector top-30 + cross-encoder | 70.3 [67.7, 73.0] (+4.1, p<0.001) | 87.5 [85.4, 89.5] (+2.1, p<0.001) | 39.0 [35.9, 41.9] (+0.6, p=0.216) | 50.1 [47.0, 53.1] (+1.9, p=0.004) | 29.2 [26.4, 32.0] (+2.9, p<0.001) | 31.6 [28.8, 34.4] (+3.0, p<0.001) |
| ISONGraph multi_hop, library defaults | 77.5 [74.9, 80.0] (+11.3, p<0.001) [mention] | 86.5 [84.4, 88.5] (+1.1, p=0.374) [knn] | 79.9 [77.4, 82.3] (+41.5, p<0.001) [mention] | 94.8 [93.4, 96.1] (+46.6, p<0.001) [mention] | 26.4 [23.8, 29.1] (+0.1, p=1.000) [mention] | 28.8 [26.0, 31.6] (+0.2, p=0.942) [mention] |
| ISONGraph multi_hop, tuned | 87.3 [85.2, 89.3] (+21.1, p<0.001) [mention] | 95.2 [93.9, 96.5] (+9.8, p<0.001) [mention] | 87.3 [85.1, 89.3] (+48.9, p<0.001) [mention+entity] | 96.4 [95.2, 97.5] (+48.2, p<0.001) [mention] | 37.7 [34.7, 40.7] (+11.4, p<0.001) [mention] | 38.7 [35.7, 41.5] (+10.1, p<0.001) [mention] |
| PPR re-rank, tuned (independent impl.) | 87.3 [85.3, 89.3] (+21.1, p<0.001) [mention] | 95.2 [93.8, 96.5] (+9.8, p<0.001) [mention] | 85.7 [83.5, 87.9] (+47.3, p<0.001) [mention] | 95.5 [94.2, 96.7] (+47.3, p<0.001) [mention] | 38.1 [35.1, 41.2] (+11.8, p<0.001) [mention] | 39.2 [36.2, 42.2] (+10.6, p<0.001) [mention] |
| CONTROL: multi_hop on label-shuffled mention graph | 66.2 [63.4, 69.1] (+0.0, p=1.000) | 85.4 [83.2, 87.5] (+0.0, p=1.000) | 38.4 [35.3, 41.3] (+0.0, p=1.000) | 48.2 [45.2, 51.2] (+0.0, p=1.000) | 26.3 [23.6, 29.0] (+0.0, p=1.000) | 28.6 [25.9, 31.4] (+0.0, p=1.000) |
| CONTROL: multi_hop on label-shuffled entity graph | n/a | n/a | 38.4 [35.3, 41.3] (+0.0, p=1.000) | 48.2 [45.2, 51.2] (+0.0, p=1.000) | n/a | n/a |

### Every graph variant, k = 10: Delta all-supporting@10 vs plain vector, pp [95% CI] (paired p)

| Method: graph | hotpotqa/minilm | hotpotqa/bge | 2wiki/minilm | 2wiki/bge | musique/minilm | musique/bge |
|---|---|---|---|---|---|---|
| ISONGraph multi_hop (tuned): mention | +23.3 [+20.4, +26.4] (p<0.001) | +13.0 [+10.6, +15.4] (p<0.001) | +47.1 [+43.7, +50.5] (p<0.001) | +50.7 [+47.5, +53.9] (p<0.001) | +9.2 [+7.2, +11.3] (p<0.001) | +8.7 [+6.6, +10.8] (p<0.001) |
| ISONGraph multi_hop (tuned): entity | +13.6 [+11.2, +16.0] (p<0.001) | n/a | +28.1 [+24.9, +31.1] (p<0.001) | +30.3 [+27.3, +33.0] (p<0.001) | n/a | n/a |
| ISONGraph multi_hop (tuned): knn | +4.0 [+1.9, +6.2] (p<0.001) | +4.5 [+2.5, +6.4] (p<0.001) | +7.3 [+4.8, +9.8] (p<0.001) | +20.0 [+17.5, +22.5] (p<0.001) | n/a | n/a |
| ISONGraph multi_hop (tuned): mention+entity | n/a | n/a | +48.5 [+45.2, +51.8] (p<0.001) | n/a | n/a | n/a |
| ISONGraph multi_hop (tuned): ctrl_mention | +0.0 [+0.0, +0.0] (p=1.000) | -0.1 [-0.3, +0.0] (p=1.000) | +0.0 [+0.0, +0.0] (p=1.000) | -0.2 [-0.5, +0.0] (p=0.497) | +0.0 [+0.0, +0.0] (p=1.000) | +0.0 [+0.0, +0.0] (p=1.000) |
| ISONGraph multi_hop (tuned): ctrl_entity | n/a | n/a | +0.0 [+0.0, +0.0] (p=1.000) | -0.1 [-0.3, +0.0] (p=1.000) | n/a | n/a |
| PPR re-rank (tuned): mention | +23.2 [+20.5, +26.1] (p<0.001) | +13.4 [+11.0, +15.8] (p<0.001) | +46.2 [+42.8, +49.7] (p<0.001) | +48.9 [+45.8, +52.1] (p<0.001) | +10.7 [+8.4, +13.3] (p<0.001) | +8.2 [+5.9, +10.5] (p<0.001) |
| PPR re-rank (tuned): entity | +9.2 [+7.1, +11.5] (p<0.001) | n/a | +22.4 [+19.7, +25.1] (p<0.001) | +25.8 [+23.0, +28.6] (p<0.001) | n/a | n/a |
| PPR re-rank (tuned): knn | +4.0 [+2.1, +5.9] (p<0.001) | +3.2 [+1.2, +5.2] (p=0.002) | +7.4 [+5.3, +9.6] (p<0.001) | +14.6 [+12.3, +16.8] (p<0.001) | n/a | n/a |
| PPR re-rank (tuned): mention+entity | n/a | n/a | +39.5 [+36.3, +42.7] (p<0.001) | n/a | n/a | n/a |
| PPR re-rank (tuned): ctrl_mention | +0.1 [-0.5, +0.7] (p=1.000) | -0.3 [-0.9, +0.3] (p=0.544) | +0.2 [-0.3, +0.7] (p=0.688) | +0.0 [+0.0, +0.0] (p=1.000) | +0.0 [-0.3, +0.3] (p=1.000) | -0.1 [-0.3, +0.0] (p=1.000) |
| PPR re-rank (tuned): ctrl_entity | n/a | n/a | -0.9 [-1.7, -0.1] (p=0.052) | +0.0 [-0.3, +0.3] (p=1.000) | n/a | n/a |
| ISONGraph multi_hop (defaults): mention | +19.1 [+16.2, +22.3] (p<0.001) | +9.0 [+6.6, +11.5] (p<0.001) | +47.0 [+43.5, +50.4] (p<0.001) | +50.7 [+47.5, +53.9] (p<0.001) | +6.3 [+3.8, +9.1] (p<0.001) | +5.4 [+3.0, +8.0] (p<0.001) |
| ISONGraph multi_hop (defaults): entity | +4.6 [+1.6, +7.7] (p=0.004) | n/a | +24.0 [+20.6, +27.2] (p<0.001) | +25.4 [+22.4, +28.3] (p<0.001) | n/a | n/a |
| ISONGraph multi_hop (defaults): knn | -0.5 [-3.2, +2.3] (p=0.772) | +2.4 [+0.3, +4.4] (p=0.040) | +7.3 [+4.8, +9.8] (p<0.001) | +17.5 [+15.0, +19.9] (p<0.001) | n/a | n/a |
| ISONGraph multi_hop (defaults): mention+entity | n/a | n/a | +36.9 [+33.5, +40.3] (p<0.001) | n/a | n/a | n/a |
| ISONGraph multi_hop (defaults): ctrl_mention | -11.1 [-13.1, -9.2] (p<0.001) | -10.2 [-12.2, -8.4] (p<0.001) | -6.3 [-7.9, -4.9] (p<0.001) | -5.5 [-6.9, -4.1] (p<0.001) | -6.6 [-8.2, -5.0] (p<0.001) | -6.7 [-8.3, -5.1] (p<0.001) |
| ISONGraph multi_hop (defaults): ctrl_entity | n/a | n/a | -9.6 [-11.5, -7.8] (p<0.001) | -5.8 [-7.3, -4.3] (p<0.001) | n/a | n/a |

### 3.1 ISONGraph multi_hop vs the non-graph alternatives and vs PPR, dev-selected graph (paired, all-supporting@k, pp)

| Dataset/encoder | k | graph (dev-selected) | multi_hop - RRF (pp) [95% CI] (p) | multi_hop - cross-encoder rerank (pp) [95% CI] (p) | multi_hop - PPR (pp) [95% CI] (p) |
|---|---|---|---|---|---|
| hotpotqa/minilm | 5 | mention | +16.8 [+13.6, +20.2] (p=<0.001) | +11.4 [+8.1, +14.9] (p=<0.001) | +7.5 [+4.9, +10.0] (p=<0.001; PPR graph=mention) |
| hotpotqa/minilm | 10 | mention | +9.0 [+6.0, +12.1] (p=<0.001) | +13.1 [+10.1, +16.2] (p=<0.001) | +0.1 [-1.4, +1.5] (p=1.000; PPR graph=mention) |
| hotpotqa/minilm | 20 | mention | +3.5 [+0.9, +6.0] (p=0.013) | +17.0 [+14.2, +19.7] (p=<0.001) | +0.0 [-1.1, +1.1] (p=1.000; PPR graph=mention) |
| hotpotqa/bge | 5 | mention | +12.2 [+9.4, +15.3] (p=<0.001) | +9.7 [+6.8, +12.8] (p=<0.001) | +1.7 [-0.6, +3.9] (p=0.167; PPR graph=mention) |
| hotpotqa/bge | 10 | mention | +7.7 [+5.3, +10.2] (p=<0.001) | +7.4 [+5.1, +9.8] (p=<0.001) | -0.4 [-1.4, +0.6] (p=0.549; PPR graph=mention) |
| hotpotqa/bge | 20 | mention | +5.6 [+3.8, +7.4] (p=<0.001) | +7.7 [+6.1, +9.5] (p=<0.001) | +0.0 [-0.9, +0.9] (p=1.000; PPR graph=mention) |
| 2wiki/minilm | 5 | mention+entity | +30.6 [+26.9, +34.2] (p=<0.001) | +30.0 [+26.3, +33.6] (p=<0.001) | +12.0 [+9.6, +14.4] (p=<0.001; PPR graph=mention) |
| 2wiki/minilm | 10 | mention+entity | +39.9 [+36.0, +43.6] (p=<0.001) | +43.5 [+40.0, +47.2] (p=<0.001) | +2.3 [+1.3, +3.4] (p=<0.001; PPR graph=mention) |
| 2wiki/minilm | 20 | mention+entity | +40.6 [+36.9, +44.2] (p=<0.001) | +48.3 [+45.2, +51.6] (p=<0.001) | +1.6 [+0.3, +2.9] (p=0.019; PPR graph=mention) |
| 2wiki/bge | 5 | mention | +39.1 [+35.7, +42.2] (p=<0.001) | +38.4 [+35.1, +41.7] (p=<0.001) | +18.9 [+15.9, +21.9] (p=<0.001; PPR graph=mention) |
| 2wiki/bge | 10 | mention | +49.0 [+45.7, +52.1] (p=<0.001) | +48.6 [+45.4, +51.8] (p=<0.001) | +1.8 [+0.7, +2.9] (p=0.002; PPR graph=mention) |
| 2wiki/bge | 20 | mention | +47.0 [+43.8, +50.1] (p=<0.001) | +46.3 [+43.3, +49.5] (p=<0.001) | +0.9 [+0.4, +1.5] (p=0.004; PPR graph=mention) |
| musique/minilm | 5 | mention | +5.8 [+3.7, +8.0] (p=<0.001) | +0.6 [-1.6, +2.9] (p=0.672) | +0.1 [-1.6, +1.7] (p=1.000; PPR graph=mention) |
| musique/minilm | 10 | mention | +7.0 [+4.4, +9.8] (p=<0.001) | +3.7 [+1.2, +6.4] (p=0.005) | -1.5 [-3.3, +0.2] (p=0.120; PPR graph=mention) |
| musique/minilm | 20 | mention | +5.8 [+2.6, +9.0] (p=<0.001) | +8.5 [+5.6, +11.5] (p=<0.001) | -0.4 [-1.9, +1.1] (p=0.679; PPR graph=mention) |
| musique/bge | 5 | mention | +6.7 [+4.5, +9.2] (p=<0.001) | +2.7 [+0.2, +5.5] (p=0.046) | +1.4 [-0.1, +2.9] (p=0.091; PPR graph=mention) |
| musique/bge | 10 | mention | +6.3 [+3.5, +9.2] (p=<0.001) | +5.0 [+2.4, +7.7] (p=<0.001) | +0.5 [-1.0, +2.0] (p=0.624; PPR graph=mention) |
| musique/bge | 20 | mention | +5.8 [+2.7, +8.9] (p=<0.001) | +7.1 [+4.2, +9.7] (p=<0.001) | -0.5 [-2.0, +1.0] (p=0.611; PPR graph=mention) |

### 3.2 Library minus independent PPR on the same graph (all-supporting, pp [95% CI], paired p)

#### hotpotqa/minilm: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)

| Graph | k=5 | k=10 | k=20 |
|---|---|---|---|
| mention | +7.5 [+4.9, +10.0] (p=<0.001) | +0.1 [-1.4, +1.5] (p=1.000) | +0.0 [-1.1, +1.1] (p=1.000) |
| entity | +8.9 [+6.5, +11.3] (p=<0.001) | +4.4 [+2.5, +6.4] (p=<0.001) | +1.2 [-0.3, +2.7] (p=0.145) |
| knn | +0.6 [-0.6, +1.8] (p=0.393) | +0.0 [-1.5, +1.6] (p=1.000) | +1.3 [+0.0, +2.7] (p=0.069) |

#### hotpotqa/bge: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)

| Graph | k=5 | k=10 | k=20 |
|---|---|---|---|
| mention | +1.7 [-0.6, +3.9] (p=0.167) | -0.4 [-1.4, +0.6] (p=0.549) | +0.0 [-0.9, +0.9] (p=1.000) |
| knn | +1.4 [-0.7, +3.4] (p=0.200) | +1.3 [+0.1, +2.5] (p=0.052) | +1.9 [+0.6, +3.3] (p=0.011) |

#### 2wiki/minilm: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)

| Graph | k=5 | k=10 | k=20 |
|---|---|---|---|
| mention | +12.0 [+9.6, +14.4] (p=<0.001) | +0.9 [+0.2, +1.7] (p=0.036) | +0.9 [-0.3, +2.0] (p=0.169) |
| entity | +11.7 [+9.2, +14.1] (p=<0.001) | +5.7 [+3.3, +8.1] (p=<0.001) | +6.0 [+3.7, +8.4] (p=<0.001) |
| knn | +0.7 [-0.8, +2.2] (p=0.409) | -0.1 [-1.9, +1.8] (p=1.000) | +1.9 [+0.3, +3.5] (p=0.026) |
| mention+entity | +25.9 [+22.9, +28.9] (p=<0.001) | +9.0 [+7.0, +10.9] (p=<0.001) | +5.8 [+4.0, +7.6] (p=<0.001) |

#### 2wiki/bge: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)

| Graph | k=5 | k=10 | k=20 |
|---|---|---|---|
| mention | +18.9 [+15.9, +21.9] (p=<0.001) | +1.8 [+0.7, +2.9] (p=0.002) | +0.9 [+0.4, +1.5] (p=0.004) |
| entity | +10.2 [+7.6, +12.7] (p=<0.001) | +4.5 [+2.2, +6.7] (p=<0.001) | +1.3 [-0.6, +3.3] (p=0.235) |
| knn | +6.8 [+4.7, +8.8] (p=<0.001) | +5.4 [+3.6, +7.2] (p=<0.001) | +1.6 [+0.3, +2.9] (p=0.034) |

#### musique/minilm: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)

| Graph | k=5 | k=10 | k=20 |
|---|---|---|---|
| mention | +0.1 [-1.6, +1.7] (p=1.000) | -1.5 [-3.3, +0.2] (p=0.120) | -0.4 [-1.9, +1.1] (p=0.679) |

#### musique/bge: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)

| Graph | k=5 | k=10 | k=20 |
|---|---|---|---|
| mention | +1.4 [-0.1, +2.9] (p=0.091) | +0.5 [-1.0, +2.0] (p=0.624) | -0.5 [-2.0, +1.0] (p=0.611) |

### 3.3 By question type and hop count (k = 10, all-supporting %, dev-selected graph, delta vs plain vector in pp [95% CI])

#### hotpotqa/minilm: breakdown by question type, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| bridge | 784 | 51.3 | 66.1 (+14.8 [+12.0, +17.7]) | 62.9 (+11.6 [+9.4, +14.0]) | 82.5 (+31.2 [+27.7, +34.7]) [mention] | 81.6 (+30.4 [+27.0, +33.5]) [mention] |
| comparison | 216 | 83.3 | 95.8 (+12.5 [+8.3, +17.1]) | 88.4 (+5.1 [+2.3, +8.3]) | 77.8 (-5.6 [-9.3, -2.3]) [mention] | 80.6 (-2.8 [-6.0, +0.0]) [mention] |

#### hotpotqa/bge: breakdown by question type, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| bridge | 784 | 73.1 | 79.8 (+6.8 [+4.3, +9.2]) | 80.0 (+6.9 [+4.5, +9.3]) | 89.9 (+16.8 [+13.8, +19.8]) [mention] | 90.4 (+17.3 [+14.4, +20.2]) [mention] |
| comparison | 216 | 98.1 | 98.1 (+0.0 [-1.4, +1.4]) | 99.1 (+0.9 [-0.9, +2.8]) | 97.2 (-0.9 [-2.8, +0.9]) [mention] | 97.2 (-0.9 [-2.8, +0.9]) [mention] |

#### 2wiki/minilm: breakdown by question type, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| bridge_comparison | 221 | 0.0 | 1.4 (+1.4 [+0.0, +3.2]) | 0.0 (+0.0 [+0.0, +0.0]) | 66.5 (+66.5 [+60.2, +72.4]) [mention+entity] | 64.7 (+64.7 [+57.9, +71.0]) [mention] |
| comparison | 238 | 71.0 | 92.0 (+21.0 [+16.0, +26.5]) | 81.5 (+10.5 [+6.7, +14.7]) | 63.0 (-8.0 [-12.6, -3.8]) [mention+entity] | 62.6 (-8.4 [-12.6, -4.6]) [mention] |
| compositional | 422 | 25.1 | 30.1 (+5.0 [+2.1, +7.8]) | 28.2 (+3.1 [+0.9, +5.2]) | 95.0 (+69.9 [+65.2, +74.4]) [mention+entity] | 92.2 (+67.1 [+61.8, +71.8]) [mention] |
| inference | 119 | 42.0 | 52.1 (+10.1 [+4.2, +16.0]) | 52.1 (+10.1 [+4.2, +16.8]) | 94.1 (+52.1 [+42.9, +61.3]) [mention+entity] | 89.1 (+47.1 [+37.8, +56.3]) [mention] |

#### 2wiki/minilm: breakdown by hop count (number of supporting passages), k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| 2 | 779 | 41.7 | 52.4 (+10.7 [+8.3, +13.1]) | 48.1 (+6.4 [+4.6, +8.5]) | 85.1 (+43.4 [+39.5, +47.2]) [mention+entity] | 82.7 (+40.9 [+37.0, +44.7]) [mention] |
| 4 | 221 | 0.0 | 1.4 (+1.4 [+0.0, +3.2]) | 0.0 (+0.0 [+0.0, +0.0]) | 66.5 (+66.5 [+60.2, +72.4]) [mention+entity] | 64.7 (+64.7 [+57.9, +71.0]) [mention] |

#### 2wiki/bge: breakdown by question type, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| bridge_comparison | 221 | 0.9 | 1.8 (+0.9 [+0.0, +2.3]) | 1.4 (+0.5 [+0.0, +1.4]) | 90.5 (+89.6 [+85.5, +93.7]) [mention] | 81.9 (+81.0 [+75.6, +86.0]) [mention] |
| comparison | 238 | 97.9 | 99.2 (+1.3 [+0.0, +2.9]) | 97.9 (+0.0 [-1.3, +1.3]) | 97.1 (-0.8 [-2.5, +0.8]) [mention] | 97.9 (+0.0 [-1.3, +1.3]) [mention] |
| compositional | 422 | 32.9 | 34.4 (+1.4 [-1.7, +4.5]) | 36.7 (+3.8 [+0.7, +6.9]) | 95.3 (+62.3 [+57.6, +67.3]) [mention] | 95.0 (+62.1 [+57.6, +66.8]) [mention] |
| inference | 119 | 54.6 | 59.7 (+5.0 [-0.8, +11.8]) | 58.0 (+3.4 [-3.4, +10.1]) | 95.0 (+40.3 [+31.1, +49.6]) [mention] | 95.0 (+40.3 [+31.9, +48.7]) [mention] |

#### 2wiki/bge: breakdown by hop count (number of supporting passages), k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| 2 | 779 | 56.1 | 58.0 (+1.9 [-0.0, +3.9]) | 58.7 (+2.6 [+0.5, +4.7]) | 95.8 (+39.7 [+36.2, +43.1]) [mention] | 95.9 (+39.8 [+36.3, +43.3]) [mention] |
| 4 | 221 | 0.9 | 1.8 (+0.9 [+0.0, +2.3]) | 1.4 (+0.5 [+0.0, +1.4]) | 90.5 (+89.6 [+85.5, +93.7]) [mention] | 81.9 (+81.0 [+75.6, +86.0]) [mention] |

#### musique/minilm: breakdown by question type, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| 2hop | 519 | 30.8 | 34.1 (+3.3 [+0.2, +6.4]) | 37.8 (+6.9 [+4.4, +9.6]) | 46.2 (+15.4 [+11.9, +19.1]) [mention] | 48.0 (+17.1 [+12.9, +21.2]) [mention] |
| 3hop1 | 213 | 4.7 | 5.6 (+0.9 [-2.8, +4.7]) | 11.7 (+7.0 [+3.3, +10.8]) | 9.4 (+4.7 [+1.4, +8.0]) [mention] | 11.3 (+6.6 [+2.3, +10.8]) [mention] |
| 3hop2 | 78 | 12.8 | 16.7 (+3.8 [-5.1, +11.5]) | 16.7 (+3.8 [-3.8, +11.5]) | 14.1 (+1.3 [-2.6, +6.4]) [mention] | 14.1 (+1.3 [-3.8, +6.4]) [mention] |
| 4hop1 | 113 | 0.9 | 0.0 (-0.9 [-2.7, +0.0]) | 1.8 (+0.9 [-1.8, +3.5]) | 1.8 (+0.9 [+0.0, +2.7]) [mention] | 3.5 (+2.7 [+0.0, +6.2]) [mention] |
| 4hop2 | 30 | 0.0 | 0.0 (+0.0 [+0.0, +0.0]) | 0.0 (+0.0 [+0.0, +0.0]) | 0.0 (+0.0 [+0.0, +0.0]) [mention] | 0.0 (+0.0 [+0.0, +0.0]) [mention] |
| 4hop3 | 47 | 2.1 | 4.3 (+2.1 [-4.3, +10.6]) | 2.1 (+0.0 [+0.0, +0.0]) | 2.1 (+0.0 [+0.0, +0.0]) [mention] | 2.1 (+0.0 [+0.0, +0.0]) [mention] |

#### musique/minilm: breakdown by hop count (number of supporting passages), k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| 2 | 519 | 30.8 | 34.1 (+3.3 [+0.2, +6.4]) | 37.8 (+6.9 [+4.4, +9.6]) | 46.2 (+15.4 [+11.9, +19.1]) [mention] | 48.0 (+17.1 [+12.9, +21.2]) [mention] |
| 3 | 291 | 6.9 | 8.6 (+1.7 [-1.7, +5.2]) | 13.1 (+6.2 [+2.7, +9.6]) | 10.7 (+3.8 [+1.4, +6.5]) [mention] | 12.0 (+5.2 [+1.7, +8.6]) [mention] |
| 4 | 190 | 1.1 | 1.1 (+0.0 [-2.1, +2.1]) | 1.6 (+0.5 [-1.1, +2.1]) | 1.6 (+0.5 [+0.0, +1.6]) [mention] | 2.6 (+1.6 [+0.0, +3.7]) [mention] |

#### musique/bge: breakdown by question type, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| 2hop | 519 | 33.7 | 37.2 (+3.5 [+0.0, +6.9]) | 39.7 (+6.0 [+2.7, +9.1]) | 48.6 (+14.8 [+11.4, +18.5]) [mention] | 49.1 (+15.4 [+11.6, +19.3]) [mention] |
| 3hop1 | 213 | 10.8 | 13.6 (+2.8 [-2.3, +8.0]) | 12.2 (+1.4 [-2.3, +5.2]) | 15.0 (+4.2 [+0.0, +8.5]) [mention] | 13.6 (+2.8 [-0.9, +6.6]) [mention] |
| 3hop2 | 78 | 14.1 | 12.8 (-1.3 [-6.4, +2.6]) | 16.7 (+2.6 [-2.6, +7.7]) | 15.4 (+1.3 [-2.6, +6.4]) [mention] | 9.0 (-5.1 [-11.5, +0.0]) [mention] |
| 4hop1 | 113 | 0.0 | 0.9 (+0.9 [+0.0, +2.7]) | 0.9 (+0.9 [+0.0, +2.7]) | 0.0 (+0.0 [+0.0, +0.0]) [mention] | 0.0 (+0.0 [+0.0, +0.0]) [mention] |
| 4hop2 | 30 | 0.0 | 0.0 (+0.0 [+0.0, +0.0]) | 0.0 (+0.0 [+0.0, +0.0]) | 0.0 (+0.0 [+0.0, +0.0]) [mention] | 0.0 (+0.0 [+0.0, +0.0]) [mention] |
| 4hop3 | 47 | 0.0 | 0.0 (+0.0 [+0.0, +0.0]) | 0.0 (+0.0 [+0.0, +0.0]) | 0.0 (+0.0 [+0.0, +0.0]) [mention] | 0.0 (+0.0 [+0.0, +0.0]) [mention] |

#### musique/bge: breakdown by hop count (number of supporting passages), k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])

| Group | n | Vector top-k (plain) | Hybrid vector+BM25 (RRF) | Vector top-30 + cross-encoder | ISONGraph multi_hop (tuned) | PPR re-rank (tuned, independent impl.) |
|---|---|---|---|---|---|---|
| 2 | 519 | 33.7 | 37.2 (+3.5 [+0.0, +6.9]) | 39.7 (+6.0 [+2.7, +9.1]) | 48.6 (+14.8 [+11.4, +18.5]) [mention] | 49.1 (+15.4 [+11.6, +19.3]) [mention] |
| 3 | 291 | 11.7 | 13.4 (+1.7 [-2.4, +5.9]) | 13.4 (+1.7 [-1.0, +4.8]) | 15.1 (+3.4 [+0.3, +6.5]) [mention] | 12.4 (+0.7 [-2.7, +4.1]) [mention] |
| 4 | 190 | 0.0 | 0.5 (+0.5 [+0.0, +1.6]) | 0.5 (+0.5 [+0.0, +1.6]) | 0.0 (+0.0 [+0.0, +0.0]) [mention] | 0.0 (+0.0 [+0.0, +0.0]) [mention] |

### 3.4 `semantic_subgraph(connect=True)`, set quality vs plain vector of the same size (per query)

`semantic_subgraph` returns an unordered set (no MRR). Parameters = the dev-selected `multi_hop` parameters for that k; `connect_max_hops` 2 and 3. Recall and all-supporting of the returned set are compared with plain vector top-|S_q| (same number of passages for that question). Full table for every k/graph in `results/tables.md` (`semantic_subgraph(connect=True)` sections). Example, k = 10:

**hotpotqa / minilm**

| Graph | connect_max_hops | mean |S| | Recall % | All-supp. % | Vector@|S| recall % | Vector@|S| all % | Δ all (pp) [CI] | p | Δ recall (pp) [CI] | p |
|---|---|---|---|---|---|---|---|---|---|---|
| mention | 2 | 9.5 | 88.8 [87.2, 90.3] | 81.7 [79.3, 84.0] | 76.5 [74.8, 78.3] | 57.1 [54.1, 60.1] | +24.6 [+21.8, +27.7] | <0.001 | +12.3 [+10.8, +13.9] | <0.001 |
| mention | 3 | 9.5 | 88.8 [87.2, 90.3] | 81.8 [79.4, 84.1] | 76.6 [74.8, 78.3] | 57.1 [54.1, 60.1] | +24.7 [+21.8, +27.8] | <0.001 | +12.3 [+10.8, +13.9] | <0.001 |
| entity | 2 | 10.3 | 84.7 [83.0, 86.3] | 72.1 [69.4, 74.8] | 77.5 [75.8, 79.2] | 58.6 [55.6, 61.6] | +13.5 [+11.1, +16.0] | <0.001 | +7.2 [+6.0, +8.4] | <0.001 |
| entity | 3 | 11.5 | 84.9 [83.2, 86.4] | 72.3 [69.6, 75.0] | 78.5 [76.7, 80.1] | 60.0 [57.0, 63.0] | +12.3 [+9.7, +14.9] | <0.001 | +6.4 [+5.1, +7.8] | <0.001 |
| knn | 2 | 10.4 | 78.9 [77.1, 80.5] | 62.2 [59.2, 65.3] | 77.5 [75.8, 79.2] | 58.7 [55.7, 61.8] | +3.5 [+1.4, +5.7] | 0.002 | +1.3 [+0.2, +2.4] | 0.034 |
| knn | 3 | 11.2 | 78.9 [77.1, 80.6] | 62.3 [59.3, 65.3] | 78.2 [76.5, 80.0] | 59.7 [56.7, 62.8] | +2.6 [+0.5, +4.9] | 0.028 | +0.7 [-0.4, +1.9] | 0.285 |
| ctrl_mention | 2 | 10.0 | 77.2 [75.5, 79.0] | 58.2 [55.2, 61.3] | 77.2 [75.5, 79.0] | 58.2 [55.2, 61.3] | +0.0 [+0.0, +0.0] | 1.000 | +0.0 [+0.0, +0.0] | 1.000 |
| ctrl_mention | 3 | 10.0 | 77.2 [75.5, 79.0] | 58.2 [55.2, 61.3] | 77.2 [75.5, 79.0] | 58.2 [55.2, 61.3] | +0.0 [+0.0, +0.0] | 1.000 | +0.0 [+0.0, +0.0] | 1.000 |

**2wiki / bge**

| Graph | connect_max_hops | mean |S| | Recall % | All-supp. % | Vector@|S| recall % | Vector@|S| all % | Δ all (pp) [CI] | p | Δ recall (pp) [CI] | p |
|---|---|---|---|---|---|---|---|---|---|---|
| mention | 2 | 8.2 | 97.4 [96.7, 98.1] | 94.6 [93.2, 96.0] | 71.5 [69.9, 73.0] | 42.4 [39.4, 45.4] | +52.2 [+49.0, +55.3] | <0.001 | +25.9 [+24.3, +27.5] | <0.001 |
| mention | 3 | 8.2 | 97.4 [96.7, 98.1] | 94.6 [93.2, 96.0] | 71.5 [69.9, 73.0] | 42.4 [39.4, 45.4] | +52.2 [+49.0, +55.3] | <0.001 | +25.9 [+24.3, +27.5] | <0.001 |
| entity | 2 | 10.5 | 89.7 [88.6, 90.9] | 74.4 [71.7, 77.1] | 72.6 [71.0, 74.1] | 44.5 [41.4, 47.6] | +29.9 [+26.9, +32.7] | <0.001 | +17.1 [+15.6, +18.5] | <0.001 |
| entity | 3 | 12.3 | 90.0 [88.8, 91.1] | 75.1 [72.4, 77.7] | 73.0 [71.5, 74.6] | 45.0 [42.0, 48.0] | +30.1 [+27.1, +32.9] | <0.001 | +17.0 [+15.4, +18.4] | <0.001 |
| knn | 2 | 10.7 | 84.1 [82.7, 85.5] | 64.3 [61.3, 67.1] | 72.5 [71.0, 74.1] | 44.3 [41.3, 47.3] | +20.0 [+17.4, +22.4] | <0.001 | +11.6 [+10.3, +12.9] | <0.001 |
| knn | 3 | 12.1 | 84.5 [83.2, 85.9] | 65.0 [62.1, 67.8] | 72.8 [71.3, 74.4] | 44.7 [41.7, 47.7] | +20.3 [+17.7, +22.7] | <0.001 | +11.7 [+10.4, +13.0] | <0.001 |
| ctrl_mention | 2 | 10.0 | 72.2 [70.7, 73.7] | 43.7 [40.7, 46.7] | 72.3 [70.8, 73.8] | 43.9 [40.9, 46.9] | -0.2 [-0.5, +0.0] | 0.497 | -0.1 [-0.2, +0.0] | 0.497 |
| ctrl_mention | 3 | 10.0 | 72.2 [70.7, 73.7] | 43.7 [40.7, 46.7] | 72.3 [70.8, 73.8] | 43.9 [40.9, 46.9] | -0.2 [-0.5, +0.0] | 0.497 | -0.1 [-0.2, +0.0] | 0.497 |
| ctrl_entity | 2 | 10.3 | 72.3 [70.7, 73.8] | 43.8 [40.8, 46.8] | 72.4 [70.8, 73.9] | 44.0 [40.9, 47.0] | -0.2 [-0.5, +0.0] | 0.497 | -0.1 [-0.2, +0.0] | 0.497 |
| ctrl_entity | 3 | 15.9 | 72.3 [70.7, 73.8] | 43.8 [40.8, 46.8] | 74.0 [72.5, 75.6] | 46.7 [43.6, 49.9] | -2.9 [-4.0, -1.9] | <0.001 | -1.8 [-2.3, -1.2] | <0.001 |

**musique / minilm**

| Graph | connect_max_hops | mean |S| | Recall % | All-supp. % | Vector@|S| recall % | Vector@|S| all % | Δ all (pp) [CI] | p | Δ recall (pp) [CI] | p |
|---|---|---|---|---|---|---|---|---|---|---|
| mention | 2 | 10.3 | 57.7 [55.7, 59.6] | 27.4 [24.7, 30.2] | 53.6 [51.8, 55.3] | 18.5 [16.1, 20.9] | +8.9 [+6.9, +11.0] | <0.001 | +4.1 [+3.0, +5.2] | <0.001 |
| mention | 3 | 10.8 | 57.9 [56.0, 59.8] | 27.7 [24.9, 30.4] | 54.1 [52.4, 55.9] | 19.0 [16.6, 21.5] | +8.7 [+6.6, +10.9] | <0.001 | +3.8 [+2.6, +4.9] | <0.001 |
| ctrl_mention | 2 | 10.0 | 53.2 [51.4, 54.9] | 18.2 [15.8, 20.6] | 53.2 [51.5, 55.0] | 18.2 [15.8, 20.6] | +0.0 [+0.0, +0.0] | 1.000 | -0.0 [-0.1, +0.0] | 1.000 |
| ctrl_mention | 3 | 10.0 | 53.2 [51.4, 54.9] | 18.2 [15.8, 20.6] | 53.2 [51.5, 55.0] | 18.2 [15.8, 20.6] | +0.0 [+0.0, +0.0] | 1.000 | -0.0 [-0.1, +0.0] | 1.000 |

In every run the connector step added at most about one node on average (mean |S| 8 to 12), so `connect=True` is equivalent to `multi_hop` here; whether connectors matter for LLM context assembly is not something this benchmark tests.

### 3.5 Metrics not in the headline

Recall@k, MRR@k and the mean number of returned passages for every method, graph and k, with Holm-adjusted p-values, are in `results/tables.md`. Observations: (i) MRR of the first supporting passage is *not* improved by the graph methods (tuned `multi_hop` MRR@10 equals plain vector's within 0.002 in all six dataset/encoder cells, because the vector hits stay on top; PPR is lower by 0.015 to 0.09) - graph expansion adds the missing hop *after* the first hit; the cross-encoder is the only method that raises MRR (0.94 vs 0.86); (ii) with library defaults `semantic_multi_hop` returns fewer than k passages (7.8-9.5 of 10 on average), because `top_k_seeds=5` seeds only expand to a few neighbours; this is *not* padded and costs it a little recall.

## 4. What ran and what did not

Graph jobs = (dataset, encoder, graph variant): tune on dev, evaluate on test. "100" = 100 random library configurations + library defaults; "lean" = 40 + defaults, subgraph only for k = 10 (re-launched when the first pass was too slow). PPR always evaluates its whole grid (72 configurations; 144 for union graphs).

| Graph variant | hotpotqa/minilm | hotpotqa/bge | 2wiki/minilm | 2wiki/bge | musique/minilm | musique/bge |
|---|---|---|---|---|---|---|
| mention | 100 | 100 | 100 | 100 | lean | lean |
| entity | 100 | not run | 100 | lean | not run | not run |
| knn | 100 | 100 | 100 | lean | not run | not run |
| mention+entity | not run | not run | 100 | not run | not run | not run |
| all | not run | not run | not run | not run | not run | not run |
| ctrl_mention | 100 | 100 | 100 | 100 | lean | lean |
| ctrl_entity | not run | not run | 100 | 100 | not run | not run |

* Baselines (vector, BM25, RRF, cross-encoder) ran on all 6 dataset/encoder combinations.
* The first pass (22 parallel jobs on 32 logical cores) was far slower than estimated; at the 110-minute mark the unfinished jobs were killed. **Not run / killed unfinished:** `all` (mention+entity+kNN) everywhere; `mention+entity` except 2Wiki/MiniLM; `entity` for HotpotQA/bge and MuSiQue; `knn` for MuSiQue; the entity negative control for HotpotQA and MuSiQue. `ctrl_mention` ran for every dataset/encoder.
* Latency was measured last on a (mostly) idle machine: single thread, warm, 200 queries (100 for subgraph, 8 for the CPU cross-encoder). Other users' background processes (editor / unrelated services) were present, so treat the absolute numbers as +/- 30%: the same cell measured twice (an aborted first attempt) gave 1.8 ms vs 0.7 ms for the library's plain `similarity_search`.

## 5. Negative controls

`ctrl_mention` / `ctrl_entity` use the same edge lists with node labels permuted. After dev tuning the library picks a configuration that is effectively "plain vector" (8 cells = 6 ctrl_mention + 2 ctrl_entity: delta 0.0 to -0.2 pp, every p >= 0.5); the PPR re-ranker likewise (|delta| <= 0.9). With the **library defaults** on a permuted graph the result is *worse* than plain vector by 5.5 to 11.1 pp (all p < 0.001): a random graph injects wrong neighbours into the top-k. In other words: the library defaults are only safe when the graph is informative, and a bad graph costs up to 11 pp of all-supporting@10.

## 6. Latency (ms per query, single thread, warm; query embedding excluded for every method)

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


* Plain vector search over 9k-15k vectors is sub-millisecond to 2 ms; everything graph-aware costs 2-17 ms per query in Python (`semantic_multi_hop` with `blend=0`: 2-3 ms; with `blend>0` it scores every node and builds a dict of all N scores per call: 11-17 ms here, growing linearly with corpus size).
* The cross-encoder is by far the most expensive (about 7 s per query for 30 candidates on one CPU thread; on the GPU it is a few tens of ms, not measured).
* The independent PPR re-ranker costs 7-14 ms per query (25 sparse power iterations over the whole corpus) - the same order as `multi_hop` with blend.

## 7. Library notes (no bug found; behaviours worth knowing)

The library was used unmodified. No incorrect result was found: `SemanticGraph.similarity_search` returns the same top-20 *set* as an independent numpy brute force for 200/200 queries on 4 of 6 setups and 198/200 on both 2Wiki setups (boundary ties). Observations:

1. `semantic_multi_hop` does not guarantee `top_k_results` results: with `top_k_seeds=5` and `top_k_results=10` it returned 7.8-9.5 passages on average. Callers who need exactly k must raise `top_k_seeds` or pad themselves. A documented "fill from remaining seeds / vector ranks" option would remove a recall handicap.
2. With `blend > 0`, `EmbeddingStore.score_map` materialises `dict(zip(cache.refs, scores.tolist()))` for the whole corpus on every query (see `isongraph_vector-py/__init__.py`, `score_map`, `semantic_multi_hop`): measured 11-17 ms per query at 15k nodes vs 2-3 ms with `blend=0`. Linear in N, in Python objects, so it will dominate at 1M nodes; a numpy-array path restricted to the visited nodes would avoid it.
3. `semantic_subgraph` builds a new `SemanticGraph` (with its own in-memory SQLite) per call: 10-40 ms per query vs 2-17 ms for `multi_hop`. For per-query use this dominates.
4. The default `threshold=0.3` of `semantic_multi_hop` was irrelevant here (top-1 cosine >= 0.3 for all 6,000 test queries with both encoders; `results/threshold_check.json`), but it silently depends on the encoder's cosine scale.
5. Library defaults (`decay=0.8`, `top_k_seeds=5`, `max_hops=2`) were close to the tuned optimum on mention graphs; the dev-selected settings at k = 10 on the mention graph varied between cells (hops 1-3, seeds 5-20, decay 0.5-0.9, blend 0-0.3; listed at the end of `results/tables.md`), which suggests the defaults are a reasonable ship-with choice.

## 8. Reproduction

Everything is driven by `run.py` (documented in its docstring). Seeds: sampling `13`, library random search `13`, bootstrap/permutation `20240607`, label permutation for controls `7`. Embeddings and corpora are cached outside the repo in `D:\Goldmine\ISON\bench_data\multihop\` (`corpus_<ds>.json`, `questions_<ds>.json`, `emb_<ds>__<model>__{passages,queries}.npy`, edge pickles); raw per-question results are in `results/raw/*.npz|json`.

```powershell
# one-time: a venv that re-uses the already-installed torch / sentence-transformers / datasets
# (nothing is installed into the global Python or the repo; no extra packages were needed)
python -m venv --system-site-packages D:\Goldmine\ISON\bench_env_mh
$py = "D:\Goldmine\ISON\bench_env_mh\Scripts\python.exe"
cd isongraph-vector\benchmarks\multihop

& $py run.py prepare            # download pinned dev sets, sample 1,500 q per dataset, build pooled corpora
& $py run.py embed              # MiniLM + bge-small embeddings (GPU if available)  ~ 5 min
& $py run.py edges              # mention / entity / kNN edge lists                  ~ 20 s
& $py run.py baselines          # vector, BM25, RRF, cross-encoder                   ~ 8 min
& $py run.py graphs --workers 12   # tune on dev, evaluate on test, all variants + controls
                                   # (the full 100-config budget takes hours of CPU; --variants / --jobs ds:model:variant
                                   #  select subsets; env MH_N_CONFIGS=40 MH_SUB_KS=10 gives the "lean" budget)
$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
& $py run.py latency            # on an idle machine
& $py run.py report             # tables.md, headline.md, summary.json
& $py extra_pairs.py            # paired tests vs RRF / cross-encoder / PPR
& $py threshold_check.py        # default seed-threshold check
```

Files: `common.py` (constants), `prepare_data.py`, `embed.py`, `graphs.py` (edge construction, negative controls, diagnostics), `methods.py` (metrics, BM25, RRF, cross-encoder, PPR, library wrappers), `jobs.py`, `report.py` + `headline.py`, `latency.py`, `extra_pairs.py`, `threshold_check.py`; `manifests/` (sampled ids and split per dataset, corpus hash); `results/`.

## 9. Suggested text for the root README / CHANGELOG (not applied)

README, "Retrieval quality" section - replace any synthetic-benchmark claim with:

> **Does graph-aware retrieval beat plain vector search on real data?** Yes, when the graph carries relevant structure. On HotpotQA, 2WikiMultihopQA and MuSiQue (1,000 held-out dev questions each, MiniLM and bge-small embeddings, a graph built only from title mentions in the passage text, no gold annotations), `semantic_multi_hop` retrieves *all* supporting passages for +9 to +51 percentage points more questions at k=10 than plain vector top-k (paired p < 0.001), and also beats hybrid BM25 fusion and a cross-encoder re-ranker over the same candidates. The gain disappears on a randomly re-labelled graph, is smaller for similarity-only (kNN) graphs (+4 to +20 pp), absent for questions that name both entities (comparison questions, where it costs up to 8 pp), and an independently implemented personalised-PageRank re-ranker over the same graph performs the same at k >= 10. The advantage is therefore *graph-aware retrieval*, not a unique property of this implementation. Numbers, method and threats to validity: `benchmarks/multihop/README.md`.

CHANGELOG:

> - Added `benchmarks/multihop/`: multi-hop retrieval benchmark on real data (HotpotQA, 2WikiMultihopQA, MuSiQue), two encoders, graph variants, negative controls, PPR / BM25 / hybrid / cross-encoder baselines, bootstrap CIs and paired tests. Supersedes the synthetic-data claim in `benchmarks/retrieval_quality`.

Also worth considering for the docs (from Section 7): state that `semantic_multi_hop` may return fewer than `top_k_results`, and that `blend > 0` scores all nodes per query.

## 10. Threats to validity (everything we could identify)

1. **The graph is aligned with how the datasets were made.** HotpotQA and 2WikiMultihopQA questions were *constructed* from pairs of Wikipedia paragraphs linked by a hyperlink / shared entity, and MuSiQue was composed from single-hop questions about chained entities. A title-mention graph recovers exactly that relation, which is why it works so well there (see "gold pairs directly linked" in the diagnostics). That is *legitimate* (the graph never sees the question or the annotation, and the shuffled-label controls collapse to or below plain vector search), but it means the size of the gain is an upper-ish estimate for corpora that have comparably informative links, and says nothing about corpora whose graph is noisier (the `knn`/`entity` rows are the closer analogues).
2. **Corpus pooling effect.** Corpora are 9k-15k passages pooled from 1,500 questions, built from each question's own distractors. In this pool a gold passage's hyperlink neighbours are overwhelmingly *other passages of the same question's context* (the "neighbours of gold inside the same context" column), which makes graph expansion more precise than it would be in a full-Wikipedia (5M+) index, where hub pages and many more same-named entities dilute edges. Absolute numbers are inflated for every method relative to open-domain retrieval; the gaps between methods are the only transferable quantity, and even those may shrink or grow at scale. Not tested.
3. **Encoder strength.** Two small encoders (MiniLM-L6, bge-small). The graph gain is larger for the weaker encoder (it leaves more headroom). Larger / multi-vector encoders and late-interaction models were not tested. `bge-small-en-v1.5` and the MS-MARCO cross-encoder may have seen HotpotQA or MuSiQue *training* data during (fine-)tuning; dev questions were used here, but exposure of the encoder to the training distribution cannot be ruled out and would favour the vector/rerank baselines, not the graph.
4. **Hyper-parameter selection.** The library got a random search of 100 configurations (40 in the lean runs, see Section 4) per graph and k, PPR a full 72/144-point grid, both selected on a 500-question dev split and evaluated on a disjoint 1,000-question test split. Selection optimism (dev vs test) is tabulated in `results/tables.md`. The vector / BM25 / RRF / rerank baselines have no tuned parameters (BM25 k1/b and RRF k0 are the textbook defaults; rerank depth 30 was fixed a priori) - they could be tuned too, and the reranker depth in particular is a lever we did not pull. Random search over 100 points can miss the library's optimum; the *library-defaults* row shows the gain does not depend on tuning.
5. **Edge-construction constants** (hub cap 30 for mentions, entity df window [2,15], kNN k=5, min title length 4) were fixed a priori and not swept; different values would change the `mention`/`entity` results. Entity extraction is a crude capitalisation heuristic.
6. **Multiple comparisons.** Many rows per table; Holm correction is applied within each table but the choice of which graph/k/model to look at is the reader's. We report all variants including those that did not help.
7. **Small effects.** Where differences are within a couple of points (e.g. library vs PPR on the same graph, `knn` graphs, MuSiQue's 3-4 hop strata with n in the tens) the CIs overlap zero and no ranking should be inferred. Strata with few questions are reported with their n.
8. **Metric definition.** "All-supporting@k" counts gold *paragraphs*, not supporting sentences, and treats a paragraph as retrieved when its id is in the top-k. A returned list shorter than k (the library returns fewer than k when seeds have too few reachable neighbours) is *not* padded - this penalises the library slightly. MRR is MRR@k.
9. **Single-machine, single-run latency**, brute-force vector search at 9k-15k vectors (no ANN). Latency ranks of the methods at larger scale could differ.
10. **Dataset copies.** 2Wiki and MuSiQue were read from HuggingFace mirrors rather than the original hosting sites. 2Wiki's mirror carries the official dev split; MuSiQue's file is the official `musique_ans_v1.0_dev.jsonl`. Questions whose supporting title is missing from the context were dropped (none for the samples drawn).
11. **The libraries' two similarity paths differ on ties**: in 2 of 200 checked queries on one setup `SemanticGraph.similarity_search` and our numpy scan return different top-20 *sets* (boundary ties); negligible, but "plain vector" here is the numpy scan.
