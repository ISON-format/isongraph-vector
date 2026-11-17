"""Aggregate raw per-question results: bootstrap CIs, paired sign-flip tests,
Holm correction, breakdowns by question type / hop count.  Writes
results/summary.json and results/tables.md."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import graphs
from headline import headline_md
from common import DATASETS, KS, MODELS, DATA_DIR, RESULTS_DIR, load_json
from jobs import RAW

B_BOOT = 2000
B_PERM = 10000
RNG_SEED = 20240607
BASELINES = ["vector", "bm25", "rrf", "rerank"]
VARIANTS = list(graphs.VARIANTS)          # real graphs (eligible for dev-based selection)
ALLV = list(graphs.ALL_VARIANTS)           # + negative controls
NICE = {"vector": "Vector top-k (plain)", "bm25": "BM25", "rrf": "Hybrid vector+BM25 (RRF)",
        "rerank": "Vector top-30 + cross-encoder", "lib_mh": "ISONGraph multi_hop (tuned)",
        "lib_default": "ISONGraph multi_hop (library defaults)", "ppr": "PPR re-rank (tuned, independent impl.)"}


def vname(v):
    return v


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


class Stats:
    def __init__(self, n, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.n = n
        self.boot = rng.integers(0, n, size=(B_BOOT, n))
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(B_PERM, n))

    def ci(self, x):
        means = x[self.boot].mean(1)
        return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def diff(self, x, y):
        d = (x - y).astype(np.float32)
        obs = float(d.mean())
        means = d[self.boot].mean(1)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        perm = np.abs((self.signs @ d) / self.n)
        p = (1 + int((perm >= abs(obs) - 1e-12).sum())) / (1 + B_PERM)
        return obs, lo, hi, p


def fmt_ci(t, pct=True, nd=1):
    m, lo, hi = t
    if pct:
        return f"{100*m:.{nd}f} [{100*lo:.{nd}f}, {100*hi:.{nd}f}]"
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_d(t, nd=1):
    d, lo, hi, p = t
    return f"{100*d:+.{nd}f} [{100*lo:+.{nd}f}, {100*hi:+.{nd}f}]"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def load_all():
    data = {}
    for ds in DATASETS:
        qs = load_json(DATA_DIR / f"questions_{ds}.json")
        test = np.array([i for i, q in enumerate(qs) if q["split"] == "test"])
        dev = np.array([i for i, q in enumerate(qs) if q["split"] == "dev"])
        for m in MODELS:
            e = {"qs": qs, "test": test, "dev": dev, "meta": {}, "arr": {}, "devarr": {}, "sets": {}}
            bf = RAW / f"{ds}__{m}__baselines.npz"
            if not bf.exists():
                continue
            z = np.load(bf)
            for key in z.files:
                meth, k, mk = key.split("|")
                e["arr"][(meth, int(k), mk)] = z[key][test]
                e["devarr"][(meth, int(k), mk)] = z[key][dev]
            e["meta"]["baselines"] = json.load(open(RAW / f"{ds}__{m}__baselines.json", encoding="utf-8"))
            for v in ALLV:
                f = RAW / f"{ds}__{m}__{v.replace('+', '_')}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                for key in z.files:
                    meth, split, k, mk = key.split("|")
                    if meth.startswith("lib_sub"):
                        e["sets"][(meth, v, int(k), mk)] = z[key]
                    else:
                        (e["arr"] if split == "test" else e["devarr"])[(f"{meth}:{v}", int(k), mk)] = z[key]
                e["meta"][v] = json.load(open(f.with_suffix(".json"), encoding="utf-8"))
            data[(ds, m)] = e
    return data


def best_variant(e, meth, k):
    """Graph variant chosen on DEV (all@k, then recall@k) for a method."""
    best, bv = None, None
    for v in VARIANTS:
        if (f"{meth}:{v}", k, "all") not in e["devarr"]:
            continue
        s = (e["devarr"][(f"{meth}:{v}", k, "all")].mean(), e["devarr"][(f"{meth}:{v}", k, "recall")].mean())
        if best is None or s > best:
            best, bv = s, v
    return bv


def get(e, name, k, mk):
    """name: 'vector' | 'lib_mh@dev-best' | 'lib_mh:variant' ..."""
    if name.endswith("@best"):
        meth = name[:-5]
        v = best_variant(e, meth, k)
        return e["arr"].get((f"{meth}:{v}", k, mk)), v
    return e["arr"].get((name, k, mk)), None


def run():
    data = load_all()
    out = {"B_boot": B_BOOT, "B_perm": B_PERM, "results": {}}
    md = []
    for (ds, model), e in data.items():
        n_test = len(e["test"])
        st = Stats(n_test)
        res = out["results"].setdefault(ds, {}).setdefault(model, {})
        res["n_test"] = n_test
        res["corpus_size"] = e["meta"]["baselines"]["corpus_size"]
        res["lib_similarity_search_same_top20_set"] = e["meta"]["baselines"]["lib_similarity_search_same_top20_set"]
        res["graphs"] = {v: e["meta"][v]["diagnostics"] for v in ALLV if v in e["meta"]}
        res["selected"] = {v: {"lib": e["meta"][v]["lib_selected"], "ppr": e["meta"][v]["ppr_selected"]}
                           for v in ALLV if v in e["meta"]}
        md.append(f"\n## {ds} / {model}  (corpus {res['corpus_size']} passages, {n_test} test questions)\n")
        for k in KS:
            rows = []   # (label, variant, per-question arrays dict)
            names = BASELINES + [f"lib_mh@best", f"ppr@best", f"lib_default@best"] + \
                    [f"lib_mh:{v}" for v in ALLV] + [f"ppr:{v}" for v in ALLV] + [f"lib_default:{v}" for v in ALLV]
            vec = {mk: e["arr"][("vector", k, mk)] for mk in ("recall", "all", "mrr", "n_ret")}
            table = []
            for nm in names:
                arrs = {}
                var = None
                for mk in ("recall", "all", "mrr", "n_ret"):
                    a, var = get(e, nm, k, mk)
                    if a is None:
                        break
                    arrs[mk] = a
                if len(arrs) < 4:
                    continue
                label = nm
                base = nm.split(":")[0].split("@")[0]
                if "@best" in nm:
                    label = f"{NICE[base]} [graph chosen on dev: {var}]"
                elif ":" in nm:
                    label = f"{NICE[base]} [graph: {nm.split(':')[1]}]"
                else:
                    label = NICE[nm]
                rec = {"label": label, "key": nm, "variant": var or (nm.split(":")[1] if ":" in nm else None)}
                for mk in ("recall", "all", "mrr"):
                    rec[mk] = st.ci(arrs[mk])
                rec["n_ret"] = float(arrs["n_ret"].mean())
                if nm != "vector":
                    rec["d_all"] = st.diff(arrs["all"], vec["all"])
                    rec["d_recall"] = st.diff(arrs["recall"], vec["recall"])
                    rec["d_mrr"] = st.diff(arrs["mrr"], vec["mrr"])
                table.append(rec)
            # Holm over the main comparisons (non-@best rows are exploratory; Holm family = all rows)
            for dk in ("d_all", "d_recall"):
                idx = [i for i, r in enumerate(table) if dk in r]
                adj = holm(np.array([table[i][dk][3] for i in idx]))
                for i, a in zip(idx, adj):
                    table[i][dk + "_holm"] = float(a)
            res.setdefault("tables", {})[str(k)] = table
            md.append(f"\n### k = {k}  (returned-passage budget = {k})\n")
            md.append("| Method | Recall@k % [95% CI] | All-supporting@k % [95% CI] | MRR@k [95% CI] | mean returned | "
                      "Δ all vs vector (pp) [95% CI] | p (Holm) | Δ recall vs vector (pp) [95% CI] | p (Holm) |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for r in table:
                dall = fmt_d(r["d_all"]) if "d_all" in r else "-"
                pa = f"{fp(r['d_all'][3])} ({fp(r['d_all_holm'])})" if "d_all" in r else "-"
                drec = fmt_d(r["d_recall"]) if "d_recall" in r else "-"
                pr = f"{fp(r['d_recall'][3])} ({fp(r['d_recall_holm'])})" if "d_recall" in r else "-"
                md.append(f"| {r['label']} | {fmt_ci(r['recall'])} | {fmt_ci(r['all'])} | {fmt_ci(r['mrr'], pct=False)} | "
                          f"{r['n_ret']:.1f} | {dall} | {pa} | {drec} | {pr} |")

            # ---- semantic_subgraph(connect=True): size-matched comparison ----
            vtop = np.load(RAW / f"{ds}__{model}__vector_top50.npy")[e["test"]]
            sub_rows = []
            for v in ALLV:
                for ch in (2, 3):
                    key = (f"lib_sub_c{ch}", v, k, "recall")
                    if key not in e["sets"]:
                        continue
                    srec, sall, ssize = (e["sets"][(f"lib_sub_c{ch}", v, k, x)] for x in ("recall", "all", "size"))
                    # vector top-|S_q| per query
                    vrec, vall = np.zeros(n_test), np.zeros(n_test)
                    for qi, q in enumerate(e["test"]):
                        gold = set(e["qs"][q]["gold"])
                        s = int(min(ssize[qi], 50))
                        hit = len(gold.intersection(int(x) for x in vtop[qi][:s]))
                        vrec[qi] = hit / len(gold)
                        vall[qi] = float(hit == len(gold))
                    sub_rows.append({
                        "variant": v, "connect_max_hops": ch, "size": float(ssize.mean()),
                        "recall": st.ci(srec), "all": st.ci(sall),
                        "vec_matched_recall": st.ci(vrec), "vec_matched_all": st.ci(vall),
                        "d_all": st.diff(sall, vall), "d_recall": st.diff(srec, vrec)})
            if sub_rows:
                res.setdefault("subgraph", {})[str(k)] = sub_rows
                md.append(f"\n#### semantic_subgraph(connect=True), top_k_results={k}: set quality vs plain vector top-|S| (size-matched per query)\n")
                md.append("| Graph | connect_max_hops | mean |S| | Recall % | All-supp. % | Vector@|S| recall % | Vector@|S| all % | Δ all (pp) [CI] | p | Δ recall (pp) [CI] | p |")
                md.append("|---|---|---|---|---|---|---|---|---|---|---|")
                for r in sub_rows:
                    md.append(f"| {r['variant']} | {r['connect_max_hops']} | {r['size']:.1f} | {fmt_ci(r['recall'])} | {fmt_ci(r['all'])} | "
                              f"{fmt_ci(r['vec_matched_recall'])} | {fmt_ci(r['vec_matched_all'])} | {fmt_d(r['d_all'])} | {fp(r['d_all'][3])} | "
                              f"{fmt_d(r['d_recall'])} | {fp(r['d_recall'][3])} |")

        # ---- library vs independent PPR on the same graph (paired) ----
        md.append(f"\n#### {ds}/{model}: ISONGraph multi_hop (tuned) minus PPR re-rank (tuned), same graph; all-supporting (pp [95% CI], paired p)\n")
        md.append("| Graph | k=5 | k=10 | k=20 |\n|---|---|---|---|")
        lp = res.setdefault("lib_minus_ppr", {})
        for v in VARIANTS:
            cells = []
            for k in KS:
                a = e["arr"].get((f"lib_mh:{v}", k, "all"))
                b = e["arr"].get((f"ppr:{v}", k, "all"))
                if a is None or b is None:
                    cells.append("n/a")
                    continue
                d = st.diff(a, b)
                lp.setdefault(v, {})[str(k)] = d
                cells.append(f"{fmt_d(d)} (p={fp(d[3])})")
            if any(c != "n/a" for c in cells):
                md.append(f"| {v} | " + " | ".join(cells) + " |")

        # ---- breakdowns at k=10 ----
        k = 10
        qs_test = [e["qs"][i] for i in e["test"]]
        for attr, title in (("type", "question type"), ("hops", "hop count (number of supporting passages)")):
            groups = defaultdict(list)
            for i, q in enumerate(qs_test):
                groups[q[attr]].append(i)
            md.append(f"\n#### {ds}/{model}: breakdown by {title}, k=10 -- All-supporting@10 % (Δ vs vector, pp [95% CI])\n")
            cols = ["vector", "rrf", "rerank", "lib_mh@best", "ppr@best"]
            md.append("| Group | n | " + " | ".join(NICE[c.split('@')[0]] if c != "vector" else NICE[c] for c in cols) + " |")
            md.append("|---|---|" + "---|" * len(cols))
            bd = res.setdefault("breakdown", {}).setdefault(attr, {})
            for gname in sorted(groups, key=str):
                ix = np.array(groups[gname])
                sts = Stats(len(ix), seed=RNG_SEED + 1)
                vecall = e["arr"][("vector", k, "all")][ix]
                cells = []
                bd[str(gname)] = {"n": int(len(ix))}
                for c in cols:
                    a, var = get(e, c, k, "all")
                    if a is None:
                        cells.append("n/a")
                        continue
                    a = a[ix]
                    cell = f"{100*a.mean():.1f}"
                    if c != "vector":
                        d = sts.diff(a, vecall)
                        cell += f" ({fmt_d(d)})"
                    if var:
                        cell += f" [{var}]"
                    cells.append(cell)
                    bd[str(gname)][c] = float(a.mean())
                md.append(f"| {gname} | {len(ix)} | " + " | ".join(cells) + " |")

        # ---- hyper-parameters / dev-vs-test optimism ----
        md.append(f"\n#### {ds}/{model}: dev vs test All-supporting@10 for tuned configs (selection optimism)\n")
        md.append("| Method | Graph | dev % | test % |\n|---|---|---|---|")
        for meth in ("lib_mh", "ppr"):
            for v in ALLV:
                kd = (f"{meth}:{v}", 10, "all")
                if kd in e["devarr"]:
                    md.append(f"| {meth} | {v} | {100*e['devarr'][kd].mean():.1f} | {100*e['arr'][kd].mean():.1f} |")
        md.append(f"| vector | - | {100*e['devarr'][('vector',10,'all')].mean():.1f} | {100*e['arr'][('vector',10,'all')].mean():.1f} |")

    md.append("\n## Graph diagnostics (gold used for REPORTING only)\n")
    md.append("| Dataset | model | Graph | undirected edges | mean degree | isolated % | gold pairs directly linked % | gold pairs linked within 2 hops % | nbrs of gold inside same question context % |\n|---|---|---|---|---|---|---|---|---|")
    for (ds, model), e in data.items():
        for v in ALLV:
            if v in e["meta"]:
                d = e["meta"][v]["diagnostics"]
                md.append(f"| {ds} | {model} | {v} | {d['n_edges_undirected_union']} | {d['mean_degree']:.1f} | {100*d['isolated_frac']:.0f} | "
                          f"{100*d['gold_pairs_directly_linked']:.0f} | {100*d['gold_pairs_linked_within_2']:.0f} | "
                          f"{100*(d['neighbours_of_gold_that_are_in_same_question_context'] or 0):.0f} |")
    md.append("\n## Selected hyper-parameters (dev), ISONGraph multi_hop, k=10\n")
    for (ds, model), e in data.items():
        for v in ALLV:
            if v in e["meta"]:
                md.append(f"- {ds}/{model}/{v}: lib `{e['meta'][v]['lib_selected']['10']}`; ppr `{e['meta'][v]['ppr_selected']['10']}`")
    (RESULTS_DIR / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS_DIR / "headline.md").write_text(headline_md(out, DATASETS, MODELS, ALLV, fmt_ci, fmt_d), encoding="utf-8")
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    print("wrote tables.md, summary.json")
