"""How much does the library's default seed threshold (0.3) matter on real embeddings?
semantic_multi_hop(threshold=0.3) drops any seed whose cosine to the query is < 0.3.
Runs the library-default configuration (max_hops=2, top_k_seeds=5, decay=0.8, top_k_results=10)
on the `mention` graph with threshold=0.3 (the true default) vs -1.0 (used everywhere else)."""
import json
import numpy as np
import graphs, methods
from common import RESULTS_DIR
from jobs import get_edges, load_ds
from ison_graph import Direction

out = {}
for ds in ("hotpotqa", "2wiki", "musique"):
    for model in ("minilm", "bge"):
        corpus, qs, P, Q, dev, test = load_ds(ds, model)
        edges = get_edges(ds, model, corpus, P, need_knn=False)
        g = methods.build_lib_graph(corpus, P, edges, ("mention",))
        S = Q @ P.T
        top1 = S.max(1)[test]
        res = {"frac_test_queries_top1_cos_below_0.3": float((top1 < 0.3).mean()),
               "mean_top1_cos": float(top1.mean())}
        for thr in (0.3, -1.0):
            rk = []
            for i in test:
                r = g.semantic_multi_hop(Q[i].tolist(), max_hops=2, top_k_seeds=5, top_k_results=10,
                                         direction=Direction.OUT, decay=0.8, threshold=thr)
                rk.append(np.array([x.node_ref[1] for x in r]))
            m = methods.metrics_at(rk, [qs[i] for i in test], 10)
            res[f"threshold={thr}"] = {"all@10": float(m["all"].mean()), "recall@10": float(m["recall"].mean()),
                                       "mean_returned": float(m["n_ret"].mean()),
                                       "frac_empty": float((m["n_ret"] == 0).mean())}
        out[f"{ds}/{model}"] = res
        print(ds, model, json.dumps(res), flush=True)
json.dump(out, open(RESULTS_DIR / "threshold_check.json", "w"), indent=1)
