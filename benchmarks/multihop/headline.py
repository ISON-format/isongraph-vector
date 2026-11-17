"""Compact README tables built from the computed summary (called by report.py)."""


def _p(p):
    return "p<0.001" if p < 0.001 else "p=" + format(p, ".3f")


def headline_md(out, DATASETS, MODELS, ALLV, fmt_ci, fmt_d):
    L = []
    order = [("vector", "Vector top-k (plain)"), ("bm25", "BM25"), ("rrf", "Hybrid RRF (vector+BM25)"),
             ("rerank", "Vector top-30 + cross-encoder"),
             ("lib_default@best", "ISONGraph multi_hop, library defaults"), ("lib_mh@best", "ISONGraph multi_hop, tuned"),
             ("ppr@best", "PPR re-rank, tuned (independent impl.)"),
             ("lib_mh:ctrl_mention", "CONTROL: multi_hop on label-shuffled mention graph"),
             ("lib_mh:ctrl_entity", "CONTROL: multi_hop on label-shuffled entity graph")]
    cols = [(ds, m) for ds in DATASETS for m in MODELS if m in out["results"].get(ds, {})]
    for k in (10, 5, 20):
        L.append("")
        L.append(f"### Headline, k = {k}: all-supporting@{k} % [95% CI] (Delta vs plain vector in pp, paired p)")
        L.append("")
        L.append("| Method | " + " | ".join(f"{ds}/{m}" for ds, m in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for key, label in order:
            cells = []
            for ds, m in cols:
                rows = {r["key"]: r for r in out["results"][ds][m].get("tables", {}).get(str(k), [])}
                r = rows.get(key)
                if r is None:
                    cells.append("n/a")
                    continue
                c = fmt_ci(r["all"])
                if "d_all" in r:
                    d = r["d_all"]
                    c += f" ({100 * d[0]:+.1f}, {_p(d[3])})"
                if "@best" in key:
                    c += f" [{r['variant']}]"
                cells.append(c)
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Every graph variant, k = 10: Delta all-supporting@10 vs plain vector, pp [95% CI] (paired p)")
    L.append("")
    L.append("| Method: graph | " + " | ".join(f"{ds}/{m}" for ds, m in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for meth, mlabel in (("lib_mh", "ISONGraph multi_hop (tuned)"), ("ppr", "PPR re-rank (tuned)"),
                         ("lib_default", "ISONGraph multi_hop (defaults)")):
        for v in ALLV:
            cells, seen = [], False
            for ds, m in cols:
                rows = {r["key"]: r for r in out["results"][ds][m].get("tables", {}).get("10", [])}
                r = rows.get(f"{meth}:{v}")
                if r is None:
                    cells.append("n/a")
                else:
                    seen = True
                    cells.append(f"{fmt_d(r['d_all'])} ({_p(r['d_all'][3])})")
            if seen:
                L.append(f"| {mlabel}: {v} | " + " | ".join(cells) + " |")
    return "\n".join(L)
