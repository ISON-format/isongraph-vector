"""Experiment jobs.  One job = (dataset, model[, graph variant]).  Each job writes
results/raw/<name>.npz (per-question metric arrays, TEST and DEV questions both)
and results/raw/<name>.json (chosen hyper-parameters, full dev grid log)."""
from __future__ import annotations

import itertools
import pickle
import random
import time

import numpy as np

import graphs
import methods
from common import (DATA_DIR, KS, N_DEV, RERANKER, RESULTS_DIR, SEED, load_json)
from embed import load_embeddings

RAW = RESULTS_DIR / "raw"
RAW.mkdir(exist_ok=True)
import os
N_LIB_CONFIGS = int(os.environ.get('MH_N_CONFIGS', '100'))        # random-search budget for the library per graph variant
RERANK_DEPTH = 30


def load_ds(ds, model):
    corpus = load_json(DATA_DIR / f"corpus_{ds}.json")
    qs = load_json(DATA_DIR / f"questions_{ds}.json")
    P, Q = load_embeddings(ds, model)
    dev = np.array([i for i, q in enumerate(qs) if q["split"] == "dev"])
    test = np.array([i for i, q in enumerate(qs) if q["split"] == "test"])
    return corpus, qs, P, Q, dev, test


def get_edges(ds, model, corpus, P, need_knn=True):
    f = DATA_DIR / f"edges_{ds}.pkl"
    if f.exists():
        edges = pickle.load(open(f, "rb"))
    else:
        edges = {"mention": graphs.mention_edges(corpus), "entity": graphs.entity_edges(corpus)}
        pickle.dump(edges, open(f, "wb"))
    edges.update(graphs.permute_edges(edges, len(corpus)))
    if need_knn:
        fk = DATA_DIR / f"edges_knn_{ds}_{model}.pkl"
        if fk.exists():
            edges["knn"] = pickle.load(open(fk, "rb"))
        else:
            edges["knn"] = graphs.knn_edges(P)
            pickle.dump(edges["knn"], open(fk, "wb"))
    return edges


def _save(name, arrays, meta):
    np.savez_compressed(RAW / f"{name}.npz", **arrays)
    import json
    with open(RAW / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)


def _store(arrays, key, rankings, qs, ks=KS):
    for k in ks:
        m = methods.metrics_at(rankings, qs, k)
        for mk, v in m.items():
            arrays[f"{key}|{k}|{mk}"] = v.astype(np.float32)


# ---------------------------------------------------------------------------
# baselines: vector, BM25, RRF, cross-encoder rerank
# ---------------------------------------------------------------------------
def baselines_job(ds, model):
    t0 = time.time()
    corpus, qs, P, Q, dev, test = load_ds(ds, model)
    arrays, meta = {}, {"dataset": ds, "model": model, "corpus_size": len(corpus), "n_questions": len(qs)}
    vr, S = methods.vector_rankings(Q, P, n=50)
    np.save(RAW / f"{ds}__{model}__vector_top50.npy", np.array(vr, dtype=np.int32))
    _store(arrays, "vector", vr, qs)

    docs = [f"{c['title']}. {c['text']}" for c in corpus]
    bm = methods.BM25(docs)
    B = bm.scores([q["question"] for q in qs])
    br = list(methods.topn_from_scores(B, 50))
    np.save(RAW / f"{ds}__{model}__bm25_top50.npy", np.array(br, dtype=np.int32))
    _store(arrays, "bm25", br, qs)
    rr = methods.rrf([S, B], n=50)
    _store(arrays, "rrf", rr, qs)

    cand = np.array(vr)[:, :RERANK_DEPTH]
    rs = methods.rerank_scores([q["question"] for q in qs], cand, corpus, RERANKER)
    np.save(RAW / f"{ds}__{model}__rerank_scores.npy", rs)
    rer = methods.rerank_rankings(cand, rs, n=20)
    _store(arrays, "rerank", rer, qs)

    # sanity: the library's own similarity_search returns the same top-20 as the numpy scan
    lib = methods.import_lib()
    g = lib.SemanticGraph(name="chk", embedding_db=":memory:", auto_embed=False)
    for c in corpus[:0]:
        pass
    for c in corpus:
        g.add_node("passage", c["id"], title=c["title"])
    g.embedding_store.add_batch([(("passage", c["id"]), c["title"], P[c["id"]].tolist()) for c in corpus])
    same = 0
    n_chk = 200
    for i in test[:n_chk]:
        res = g.similarity_search(Q[i].tolist(), top_k=20, threshold=-1.0)
        same += int(set(r.node_ref[1] for r in res) == set(int(x) for x in vr[i][:20]))
    meta["lib_similarity_search_same_top20_set"] = f"{same}/{n_chk}"
    meta["seconds"] = time.time() - t0
    _save(f"{ds}__{model}__baselines", arrays, meta)
    return f"baselines {ds}/{model} done in {time.time()-t0:.0f}s"


# ---------------------------------------------------------------------------
# graph jobs: ISONGraph Vector (multi_hop / subgraph) and PPR on one graph variant
# ---------------------------------------------------------------------------
def lib_grid(variant):
    fams = graphs.ALL_VARIANTS[variant]
    hops = [1, 2, 3] if variant in ("mention", "ctrl_mention") else [1, 2]
    dirs = ["out", "both"] if any(f.startswith("mention") for f in fams) else ["out"]
    wfs = [None] if fams[0].startswith("mention") and len(fams) == 1 else [None, "w"]
    grid = [dict(hops=h, seeds=s, decay=d, blend=b, direction=di, weight_field=w)
            for h, s, d, b, di, w in itertools.product(hops, [5, 10, 15, 20], [0.3, 0.5, 0.7, 0.9],
                                                       [0.0, 0.3, 0.6], dirs, wfs)]
    rng = random.Random(SEED)
    grid = rng.sample(grid, min(N_LIB_CONFIGS, len(grid)))
    default = dict(hops=2, seeds=5, decay=0.8, blend=0.0, direction="out", weight_field=None)  # library defaults
    return [default] + grid


def ppr_grid(variant):
    fams = graphs.ALL_VARIANTS[variant]
    dirs = [True, False] if any(f.startswith("mention") for f in fams) else [True]
    wfs = [False] if fams[0].startswith("mention") and len(fams) == 1 else [False, True]
    g = [dict(alpha=a, seeds=s, beta=b, directed_mention=d, weighted=w)
         for a, s, b, d, w in itertools.product([0.5, 0.7, 0.85], [5, 10, 20], [0.0, 0.3, 0.6, 0.85], dirs, wfs)]
    return g[:int(os.environ.get('MH_N_PPR', '100000'))]


def _score(m):
    return (float(m["all"].mean()), float(m["recall"].mean()))


def graph_job(ds, model, variant, do_subgraph=True):
    t0 = time.time()
    out_name = f"{os.environ.get('MH_PREFIX', '')}{ds}__{model}__{variant.replace('+', '_')}"
    if (RAW / f"{out_name}.npz").exists() and not os.environ.get("MH_FORCE"):
        return f"graph {ds}/{model}/{variant} cached"
    corpus, qs, P, Q, dev, test = load_ds(ds, model)
    edges = get_edges(ds, model, corpus, P, need_knn=("knn" in graphs.ALL_VARIANTS[variant]))
    fams = graphs.ALL_VARIANTS[variant]
    n = len(corpus)
    arrays = {}
    meta = {"dataset": ds, "model": model, "variant": variant, "families": list(fams),
            "diagnostics": graphs.graph_diagnostics(fams, edges, qs, n)}
    dev_qs = [qs[i] for i in dev]
    test_qs = [qs[i] for i in test]

    # -------------------- PPR (independent baseline) --------------------
    S = methods.dense_scores(Q, P)
    log, best = [], {}
    adj_cache = {}
    for cfg in ppr_grid(variant):
        akey = (cfg["directed_mention"], cfg["weighted"])
        if akey not in adj_cache:
            adj_cache[akey] = methods.build_adjacency(n, edges, fams, cfg["directed_mention"], cfg["weighted"])
        r = methods.ppr_rankings(S[dev], adj_cache[akey], cfg["seeds"], cfg["alpha"], cfg["beta"])
        row = {"cfg": cfg}
        for k in KS:
            m = methods.metrics_at(r, dev_qs, k)
            row[f"all@{k}"], row[f"recall@{k}"] = _score(m)
            if k not in best or (row[f"all@{k}"], row[f"recall@{k}"]) > best[k][0]:
                best[k] = ((row[f"all@{k}"], row[f"recall@{k}"]), cfg)
        log.append(row)
    meta["ppr_dev_grid"] = log
    log_ppr = log
    meta["ppr_selected"] = {k: best[k][1] for k in KS}
    for k in KS:
        cfg = best[k][1]
        adj = adj_cache[(cfg["directed_mention"], cfg["weighted"])]
        r_dev = methods.ppr_rankings(S[dev], adj, cfg["seeds"], cfg["alpha"], cfg["beta"])
        r_test = methods.ppr_rankings(S[test], adj, cfg["seeds"], cfg["alpha"], cfg["beta"])
        for split, r, sq in (("dev", r_dev, dev_qs), ("test", r_test, test_qs)):
            m = methods.metrics_at(r, sq, k)
            for mk, v in m.items():
                arrays[f"ppr|{split}|{k}|{mk}"] = v.astype(np.float32)

    # -------------------- ISONGraph Vector --------------------
    g = methods.build_lib_graph(corpus, P, edges, fams)
    grid = lib_grid(variant)
    log, best, times = [], {}, []
    for ci, cfg in enumerate(grid):
        tm = []
        r = methods.lib_multihop_rankings(g, Q, cfg, idx=dev, timing=tm)
        row = {"cfg": cfg, "ms_per_query": 1000 * float(np.mean(tm)), "default": ci == 0}
        for k in KS:
            m = methods.metrics_at(r, dev_qs, k)
            row[f"all@{k}"], row[f"recall@{k}"] = _score(m)
            row[f"n_ret@{k}"] = float(m["n_ret"].mean())
            if k not in best or (row[f"all@{k}"], row[f"recall@{k}"]) > best[k][0]:
                best[k] = ((row[f"all@{k}"], row[f"recall@{k}"]), cfg)
        log.append(row)
    meta["lib_dev_grid"] = log
    meta["lib_selected"] = {k: best[k][1] for k in KS}
    uniq = {}
    for k in KS:
        uniq.setdefault(tuple(sorted(best[k][1].items(), key=lambda x: x[0])), best[k][1])
    cache_test = {}
    for key, cfg in list(uniq.items()) + [("default", grid[0])]:
        ck = tuple(sorted(cfg.items(), key=lambda x: x[0]))
        if ck not in cache_test:
            cache_test[ck] = (methods.lib_multihop_rankings(g, Q, cfg, idx=dev),
                              methods.lib_multihop_rankings(g, Q, cfg, idx=test))
    for k in KS:
        ck = tuple(sorted(best[k][1].items(), key=lambda x: x[0]))
        r_dev, r_test = cache_test[ck]
        for split, r, sq in (("dev", r_dev, dev_qs), ("test", r_test, test_qs)):
            m = methods.metrics_at(r, sq, k)
            for mk, v in m.items():
                arrays[f"lib_mh|{split}|{k}|{mk}"] = v.astype(np.float32)
    ck = tuple(sorted(grid[0].items(), key=lambda x: x[0]))
    r_dev, r_test = cache_test[ck]
    for k in KS:
        for split, r, sq in (("dev", r_dev, dev_qs), ("test", r_test, test_qs)):
            m = methods.metrics_at(r, sq, k)
            for mk, v in m.items():
                arrays[f"lib_default|{split}|{k}|{mk}"] = v.astype(np.float32)

    # -------------------- semantic_subgraph(connect=True) with the tuned params --------------------
    if do_subgraph:
        for k in [int(x) for x in os.environ.get('MH_SUB_KS', '5,10,20').split(',')]:
            cfg = best[k][1]
            for ch in (2, 3):
                sets = methods.lib_subgraph_sets(g, Q, cfg, k, True, ch, idx=test)
                m = methods.set_metrics(sets, test_qs)
                for mk, v in m.items():
                    arrays[f"lib_sub_c{ch}|test|{k}|{mk}"] = v.astype(np.float32)
                # per-query sets sizes also needed for size-matched vector baseline: stored as 'size'
    meta["n_lib_configs"] = len(grid)
    meta["n_ppr_configs"] = len(log_ppr)
    meta["sub_ks"] = os.environ.get("MH_SUB_KS", "5,10,20")
    meta["seconds"] = time.time() - t0
    _save(f"{os.environ.get('MH_PREFIX', '')}{ds}__{model}__{variant.replace('+', '_')}", arrays, meta)
    return f"graph {ds}/{model}/{variant} done in {time.time()-t0:.0f}s"
