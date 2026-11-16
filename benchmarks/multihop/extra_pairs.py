"""Paired comparisons of the dev-selected ISONGraph multi_hop against the non-graph baselines
(hybrid RRF, cross-encoder rerank) and against plain vector, per k.  -> results/pairs_vs_nongraph.md/json"""
import json
import numpy as np
import report
from common import KS, RESULTS_DIR

data = report.load_all()
rows, js = [], {}
rows.append("| Dataset/encoder | k | graph (dev-selected) | multi_hop - RRF (pp) [95% CI] (p) | multi_hop - cross-encoder rerank (pp) [95% CI] (p) | multi_hop - PPR (pp) [95% CI] (p) |")
rows.append("|---|---|---|---|---|---|")
for (ds, m), e in data.items():
    st = report.Stats(len(e["test"]))
    for k in KS:
        v = report.best_variant(e, "lib_mh", k)
        a = e["arr"][(f"lib_mh:{v}", k, "all")]
        cells = []
        for other in ("rrf", "rerank"):
            d = st.diff(a, e["arr"][(other, k, "all")])
            cells.append(f"{report.fmt_d(d)} (p={report.fp(d[3])})")
            js[f"{ds}/{m}/k{k}/lib_mh-{other}"] = d
        vp = report.best_variant(e, "ppr", k)
        d = st.diff(a, e["arr"][(f"ppr:{vp}", k, "all")])
        cells.append(f"{report.fmt_d(d)} (p={report.fp(d[3])}; PPR graph={vp})")
        js[f"{ds}/{m}/k{k}/lib_mh-ppr_best"] = d
        rows.append(f"| {ds}/{m} | {k} | {v} | " + " | ".join(cells) + " |")
(RESULTS_DIR / "pairs_vs_nongraph.md").write_text("\n".join(rows), encoding="utf-8")
json.dump(js, open(RESULTS_DIR / "pairs_vs_nongraph.json", "w"), indent=1, default=float)
print("\n".join(rows))
