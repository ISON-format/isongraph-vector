"""Per-query latency, single thread, warm, query embeddings precomputed (encoding
cost is identical for every method and excluded).  Run on an otherwise idle
machine with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1."""
from __future__ import annotations

import json
import platform
import time

import numpy as np

import graphs
import methods
import report
from common import DATA_DIR, RERANKER, RESULTS_DIR, load_json
from jobs import get_edges, load_ds

N_Q = 200
N_Q_RERANK = 8


def _stat(ts):
    a = np.array(ts) * 1000.0
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "n": len(a)}


def timeit(fn, idx, warm=10):
    for i in idx[:warm]:
        fn(i)
    ts = []
    for i in idx:
        t = time.perf_counter()
        fn(i)
        ts.append(time.perf_counter() - t)
    return _stat(ts)


def run(combos=(("hotpotqa", "minilm"), ("2wiki", "minilm"), ("musique", "minilm"), ("hotpotqa", "bge"))):
    import torch
    torch.set_num_threads(1)
    data = report.load_all()
    out = {"machine": {"platform": platform.platform(), "processor": platform.processor(), "threads": 1,
                       "note": "single-threaded, warm, query vectors precomputed; timed on a machine that may have "
                               "had other light background processes (see README)"}, "results": {}}
    k = 10
    for ds, model in combos:
        corpus, qs, P, Q, dev, test = load_ds(ds, model)
        edges = get_edges(ds, model, corpus, P)
        e = data[(ds, model)]
        idx = list(test[:N_Q])
        res = {}
        # vector (numpy brute force)
        def vec(i):
            s = P @ Q[i]
            t = np.argpartition(-s, 20)[:20]
            return t[np.argsort(-s[t])]
        res["vector (numpy brute force top-20)"] = timeit(vec, idx)
        # library similarity_search
        lib = methods.import_lib()
        g0 = lib.SemanticGraph(name="x", embedding_db=":memory:", auto_embed=False)
        for c in corpus:
            g0.add_node("passage", c["id"], title=c["title"])
        g0.embedding_store.add_batch([(("passage", c["id"]), c["title"], P[c["id"]].tolist()) for c in corpus])
        res["ISONGraph similarity_search top-20 (plain vector via the library)"] = timeit(
            lambda i: g0.similarity_search(Q[i].tolist(), top_k=20, threshold=-1.0), idx)
        # BM25
        docs = [f"{c['title']}. {c['text']}" for c in corpus]
        bm = methods.BM25(docs)
        qtxt = [q["question"] for q in qs]
        def bm25(i):
            s = bm.scores([qtxt[i]])
            return methods.topn_from_scores(s, 20)
        res["BM25 (scipy sparse)"] = timeit(bm25, idx)
        def hyb(i):
            s1 = P @ Q[i]
            s2 = bm.scores([qtxt[i]])
            return methods.rrf([s1[None, :], s2], n=20)
        res["Hybrid RRF (dense + BM25 + fusion)"] = timeit(hyb, idx)
        # PPR + ISONGraph with the dev-selected graph at k=10
        for meth in ("ppr", "lib_mh"):
            v = report.best_variant(e, meth, k)
            fams = graphs.VARIANTS[v]
            sel = e["meta"][v]["ppr_selected" if meth == "ppr" else "lib_selected"][str(k)]
            if meth == "ppr":
                A = methods.build_adjacency(len(corpus), edges, fams, sel["directed_mention"], sel["weighted"])
                def ppr(i):
                    s = (P @ Q[i])[None, :]
                    return methods.ppr_rankings(s, A, sel["seeds"], sel["alpha"], sel["beta"], n=20)
                res[f"PPR re-rank, graph={v}"] = timeit(ppr, idx)
            else:
                t0 = time.perf_counter()
                g = methods.build_lib_graph(corpus, P, edges, fams)
                res["_graph_build_seconds"] = {v: time.perf_counter() - t0}
                tm = []
                methods.lib_multihop_rankings(g, Q, sel, idx=idx[:10])
                methods.lib_multihop_rankings(g, Q, sel, idx=idx, timing=tm)
                res[f"ISONGraph semantic_multi_hop (tuned), graph={v}"] = _stat(tm)
                tm = []
                methods.lib_subgraph_sets(g, Q, sel, k, True, 2, idx=idx[:5])
                methods.lib_subgraph_sets(g, Q, sel, k, True, 2, idx=idx[:100], timing=tm)
                res[f"ISONGraph semantic_subgraph(connect=True), graph={v}"] = _stat(tm)
                res["_selected_cfg"] = {"variant": v, "cfg": sel}
        # cross-encoder rerank on CPU (1 thread), 30 candidates (cost depends only on the model and
        # passage length, so measured once, on the first combination)
        if (ds, model) != combos[0]:
            out["results"][f"{ds}/{model}"] = res
            print(ds, model, json.dumps({k_: (v_["mean_ms"] if isinstance(v_, dict) and "mean_ms" in v_ else None)
                                        for k_, v_ in res.items()}, indent=1), flush=True)
            continue
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(RERANKER, device="cpu", max_length=256)
        vr = np.load(RESULTS_DIR / "raw" / f"{ds}__{model}__vector_top50.npy")
        def rr(i):
            cand = vr[i][:30]
            pairs = [(qtxt[i], f"{corpus[int(p)]['title']}. {corpus[int(p)]['text']}") for p in cand]
            return ce.predict(pairs, batch_size=32, show_progress_bar=False)
        res["Vector top-30 + cross-encoder (CPU, 1 thread)"] = timeit(rr, idx[:N_Q_RERANK], warm=2)
        out["results"][f"{ds}/{model}"] = res
        print(ds, model, json.dumps({k_: (v_["mean_ms"] if isinstance(v_, dict) and "mean_ms" in v_ else None)
                                    for k_, v_ in res.items()}, indent=1), flush=True)
    with open(RESULTS_DIR / "latency.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
