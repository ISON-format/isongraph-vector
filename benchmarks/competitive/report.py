"""Turn results/raw/*.json into results/*.md tables, results/summary.json and results/plots/*.png.

    python report.py [--raw results/raw] [--out results]
Nothing here measures anything; every number in the tables is read from the raw JSON files.
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

DISPLAY = {
    "isong_py": "ISONGraph Python (exact)", "isong_py_ivf": "ISONGraph Python (IVF)",
    "rust_default_exact": "ISONGraph Rust (exact)", "rust_default_ivf": "ISONGraph Rust (IVF)",
    "rust_native_exact": "ISONGraph Rust, target-cpu=native (exact)", "rust_native_ivf": "ISONGraph Rust, target-cpu=native (IVF)",
    "cpp_default_exact": "ISONGraph C++ (exact)", "cpp_default_ivf": "ISONGraph C++ (IVF)",
    "cpp_avx2_exact": "ISONGraph C++, /arch:AVX2 (exact)", "cpp_avx2_ivf": "ISONGraph C++, /arch:AVX2 (IVF)",
    "csharp_exact": "ISONGraph C# (exact)", "csharp_ivf": "ISONGraph C# (IVF)",
    "js_exact": "ISONGraph JS (exact)", "js_ivf": "ISONGraph JS (IVF)",
    "numpy_bf": "numpy brute force", "faiss_flat": "faiss IndexFlatIP", "faiss_ivfflat": "faiss IVFFlat",
    "faiss_hnsw": "faiss HNSWFlat", "hnswlib": "hnswlib", "usearch": "usearch (HNSW)", "chromadb": "chromadb (HNSW)",
    "qdrant_local": "qdrant-client local mode (exact numpy)", "lancedb_ivfflat": "lancedb IVF_FLAT", "lancedb_flat": "lancedb (no index)",
    "sqlite_vec": "sqlite-vec (vec0, exact)",
}
OURS = lambda s: s.startswith(("isong", "rust", "cpp", "csharp", "js_"))  # noqa: E731
EXACT_SYSTEMS = {"numpy_bf", "isong_py", "faiss_flat", "lancedb_flat", "sqlite_vec", "qdrant_local"}


def load(raw):
    recs = defaultdict(dict)  # (system, ds, n) -> merged dict
    status = {}
    for f in sorted(glob.glob(os.path.join(raw, "*.json"))):
        if f.endswith(".meta.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        runs = d["runs"] if "runs" in d else [d]
        for r in runs:
            sysname = r.get("system") or d.get("system")
            if not sysname:
                continue
            key = (sysname, r.get("ds", d.get("ds")), r.get("n", d.get("n")))
            st = d.get("status", "ok")
            if st != "ok":
                status[key] = st
                continue
            m = recs[key]
            for k, v in r.items():
                if k in ("filter", "mutation", "saveload", "sweep", "build") and k in m and k != "filter":
                    continue
                m[k] = v
            m.setdefault("_files", []).append(os.path.basename(f))
            for kk in ("load_before_pct", "load_contaminated", "wall_s"):
                if kk in d:
                    m.setdefault(kk, d[kk])
    return recs, status


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if x >= 100000:
        return f"{x:,.0f}"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def multi_qps(p):
    if "batch" in p and p["batch"]:
        return p["batch"]["qps_batch"], "batch"
    if "allthreads" in p:
        return p["allthreads"]["qps"], "auto-threads loop"
    return None, None


def best_at(rec, thr):
    """highest-QPS sweep point with recall >= thr; returns the point or None."""
    ok = [p for p in rec.get("sweep", []) if p.get("recall") is not None and p["recall"] >= thr]
    return max(ok, key=lambda p: p["qps_1t"]) if ok else None


def interp_at(rec, thr, getter):
    """QPS at recall == thr by log-linear interpolation between the two measured points that bracket it.
    No extrapolation: None when the sweep never reaches thr. Exact systems (one point at recall ~1.0) return that point."""
    import math
    pts = [(p["recall"], getter(p)) for p in rec.get("sweep", []) if p.get("recall") is not None and getter(p)]
    if not pts:
        return None
    pts.sort()
    if len(pts) == 1:
        return pts[0][1] if pts[0][0] >= thr else None
    best = None
    for (r1, q1), (r2, q2) in zip(pts, pts[1:]):
        if r1 <= thr <= r2:
            if r2 == r1:
                v = max(q1, q2)
            elif r2 - r1 > 0.05:
                v = q2          # bracket too wide to interpolate honestly: report the measured point above (conservative)
                v = (v, "*")
            else:
                f = (thr - r1) / (r2 - r1)
                v = math.exp(math.log(q1) * (1 - f) + math.log(q2) * f)
            if isinstance(v, tuple):
                best = v if best is None else best
            else:
                best = v if best is None or isinstance(best, tuple) else max(best, v)
    if best is None:
        above = [q for r, q in pts if r >= thr]
        if above and pts[0][0] >= thr:      # lowest measured point already above thr: report it (a lower bound)
            return max(above)
    return best


def name(s):
    return DISPLAY.get(s, s)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_median(rec):
    b = rec.get("build", {})
    runs = b.get("total_s_runs")
    return statistics.median(runs) if runs else b.get("total_s")


def section_cell(recs, ds, n):
    keys = sorted([k for k in recs if k[1] == ds and k[2] == n and "sweep" in recs[k]], key=lambda k: (not OURS(k[0]), k[0]))
    if not keys:
        return ""
    md = [f"### {ds}, N = {n:,}\n"]
    # headline: qps at recall thresholds
    rows = []
    for k in keys:
        rec = recs[k]
        row = [("**" + name(k[0]) + "**") if OURS(k[0]) else name(k[0])]
        def _f(v):
            if not v:
                return "-"
            return fmt(v[0]) + "*" if isinstance(v, tuple) else fmt(v)
        for thr in (0.90, 0.95, 0.99):
            v = interp_at(rec, thr, lambda p: p["qps_1t"])
            row.append(_f(v))
        for thr in (0.90, 0.95):
            v = interp_at(rec, thr, lambda p: multi_qps(p)[0])
            mode = next((multi_qps(p)[1] for p in rec.get("sweep", []) if multi_qps(p)[1]), None)
            row.append((_f(v) + (" †" if mode and mode != "batch" else "")) if v else "-")
        p95 = best_at(rec, 0.95)
        row.append(f"{fmt(p95['median_ms'])} / {fmt(p95['p95_ms'])} / {fmt(p95['p99_ms'])} (recall {p95['recall']:.3f})" if p95 else "-")
        rows.append(row)
    md.append("Queries/second at recall@10 = 0.90 / 0.95 / 0.99, log-linearly interpolated between the two measured sweep points that bracket the target (no extrapolation; `*` = the two bracketing points are more than 0.05 recall apart, so the measured point above the target is shown instead, a conservative value; `-` = never reached; exact scans appear at their single recall-1.0 point). 1T = one pinned P-core, one query at a time. All-core = batched call on 32 logical CPUs (`†` = the ISONGraph ports have no batch API, so this is the same one-query loop with automatic scan threads). Latency columns are for the first measured sweep point with recall >= 0.95.\n")
    md.append(table(["system", "QPS 1T @0.90", "@0.95", "@0.99", "all-core QPS @0.90", "@0.95", "1T latency ms median / p95 / p99"], rows))
    md.append("")
    # build + memory
    rows = []
    for k in keys:
        rec = recs[k]
        b = rec.get("build", {})
        mb = rec.get("mem_baseline", {})
        ma = rec.get("mem_after_build", {})
        mq = rec.get("mem_after_first_query", ma)
        base = mb.get("rss_mb", 0)
        srv = ""
        if mq.get("server_rss_mb"):
            srv = f" (+{mq['server_rss_mb']:.0f} server)"
        reps = b.get("total_s_runs")
        rows.append([("**" + name(k[0]) + "**") if OURS(k[0]) else name(k[0]), fmt(b.get("ingest_s")), fmt(b.get("index_s")), fmt(b.get("total_s")),
                     (fmt(statistics.median(reps)) + f" ({fmt(min(reps))}-{fmt(max(reps))})") if reps else "-", fmt(b.get("ready_s")),
                     f"{mq.get('rss_mb', 0) - base:,.0f}{srv}", f"{ma.get('peak_mb', 0) - base:,.0f}"])
    md.append("Build and memory (RSS growth over the process baseline after imports; the raw vectors are streamed in from disk and are not held by the harness).\n")
    md.append(table(["system", "ingest s", "index s", "total build s", "3 builds: median (min-max) s", "ready s (incl. first query)", "steady RSS +MB", "peak RSS +MB (build)"], rows))
    md.append("")
    # full sweeps
    md.append("<details><summary>Full accuracy sweeps (recall@10, single-thread QPS, latency, all-core QPS)</summary>\n")
    for k in keys:
        rec = recs[k]
        rows = []
        for p in rec["sweep"]:
            q, mode = multi_qps(p)
            rows.append([p.get("knob") if p.get("knob") is not None else "-", f"{p['recall']:.4f}", f"{fmt(p['qps_1t'])} ({fmt(p.get('qps_1t_min'))}-{fmt(p.get('qps_1t_max'))})",
                         fmt(p["median_ms"]), fmt(p["p95_ms"]), fmt(p["p99_ms"]), (fmt(q) + ("" if mode == "batch" else " †")) if q else "-", p.get("nq")])
        md.append(f"**{name(k[0])}** ({rec.get('knob_name') or 'exact'})\n")
        md.append(table(["knob", "recall@10", "QPS 1T (min-max)", "median ms", "p95 ms", "p99 ms", "all-core QPS", "queries/run"], rows))
        md.append("")
    md.append("</details>\n")
    return "\n".join(md)


def section_exact(recs):
    md = []
    for ds in ("text384", "glove", "sift"):
        rows = []
        ns = sorted({k[2] for k in recs if k[1] == ds})
        for n in ns:
            for k in sorted([k for k in recs if k[1] == ds and k[2] == n], key=lambda k: (not OURS(k[0]), k[0])):
                rec = recs[k]
                sw = rec.get("sweep", [])
                ex = None
                if k[0].endswith("_exact") or k[0] in EXACT_SYSTEMS:
                    ex = sw[0] if sw else None
                else:
                    continue
                if not ex:
                    continue
                la = rec.get("loop_allthreads")
                b = ex.get("batch")
                rows.append([f"{n:,}", ("**" + name(k[0]) + "**") if OURS(k[0]) else name(k[0]), fmt(ex["median_ms"]), fmt(ex["p95_ms"]), fmt(ex["p99_ms"]),
                             f"{fmt(ex['qps_1t'])} ({fmt(ex.get('qps_1t_min'))}-{fmt(ex.get('qps_1t_max'))})",
                             (fmt(la["median_ms"]) + f" ms ({fmt(la['qps'])} QPS)") if la else "-", fmt(b["qps_batch"]) if b else "-", f"{ex['recall']:.3f}", fmt(rec.get("build", {}).get("first_query_ms"))])
        if rows:
            md.append(f"#### {ds}\n")
            md.append(table(["N", "system", "median ms (1T)", "p95", "p99", "QPS 1T (min-max)", "all-threads single query", "batch QPS", "recall", "first query ms"], rows))
            md.append("")
    return "\n".join(md)


def section_mutation(recs):
    rows = []
    for k in sorted(recs, key=lambda k: (k[1], k[2], not OURS(k[0]), k[0])):
        m = recs[k].get("mutation")
        if not m:
            continue
        if "skipped" in m:
            continue

        def cell(key):
            v = m.get(key)
            if v is None:
                return "-"
            if v.get("unsupported"):
                return "unsupported"
            if v.get("error"):
                return "error"
            return f"{fmt(v['seconds'])} s ({fmt(v['per_s'])}/s)"
        rows.append([k[1], f"{k[2]:,}", ("**" + name(k[0]) + "**") if OURS(k[0]) else name(k[0]), cell("insert_10k"), cell("update_10k"), cell("delete_10k"),
                     fmt(m.get("recall_before")) if m.get("recall_before") is not None else "-", f"{m['recall_after']:.3f}" if m.get("recall_after") is not None else "n/a",
                     fmt(m.get("first_query_after_ms"))])
    return table(["dataset", "N", "system", "insert 10k", "update 10k", "delete 10k", "recall@10 before", "after", "first query after ms"], rows)


def section_filter(recs):
    md = []
    for ds, n in sorted({(k[1], k[2]) for k in recs if recs[k].get("filter")}):
        rows = []
        for k in sorted([k for k in recs if k[1] == ds and k[2] == n and recs[k].get("filter")], key=lambda k: (not OURS(k[0]), k[0])):
            f = recs[k]["filter"]
            row = [("**" + name(k[0]) + "**") if OURS(k[0]) else name(k[0]), fmt(f.get("build_with_metadata_s"))]
            for lab in ("f10", "f01"):
                pts = f.get("subsets", {}).get(lab)
                if pts is None:
                    row += ["-", "-"]
                    continue
                if isinstance(pts, dict) and pts.get("unsupported"):
                    row += ["unsupported", "-"]
                    continue
                ok = [p for p in pts if p["recall"] >= 0.9]
                best = max(ok, key=lambda p: p["qps_1t"]) if ok else None
                top = max(pts, key=lambda p: p["recall"])
                row.append(f"{fmt(best['qps_1t'])} @ {best['recall']:.3f}" if best else f"- (max recall {top['recall']:.3f} at {fmt(top['qps_1t'])} QPS)")
                row.append(f"{top['recall']:.3f}")
            rows.append(row)
        md.append(f"#### {ds}, N = {n:,}\n")
        md.append(table(["system", "build with metadata s", "10% subset: QPS 1T @ recall>=0.90", "max recall", "1% subset: QPS 1T @ recall>=0.90", "max recall"], rows))
        md.append("")
    return "\n".join(md)


def section_saveload(recs):
    rows = []
    for k in sorted(recs, key=lambda k: (k[1], k[2], not OURS(k[0]), k[0])):
        s = recs[k].get("saveload")
        if not s:
            continue
        if s.get("unsupported") or s.get("error"):
            rows.append([k[1], f"{k[2]:,}", name(k[0]), s.get("error", "unsupported")[:60], "-", "-", "-", "-"])
            continue
        rows.append([k[1], f"{k[2]:,}", ("**" + name(k[0]) + "**") if OURS(k[0]) else name(k[0]), fmt(s.get("save_s")), fmt(s.get("disk_mb")), fmt(s.get("load_s")),
                     fmt(s.get("first_query_after_load_ms")), fmt(s.get("time_to_ready_s"))])
    return table(["dataset", "N", "system", "save s", "on-disk MB", "load s", "first query after load ms", "time to ready s"], rows)


def plots(recs, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    os.makedirs(out, exist_ok=True)
    made = []
    cells = sorted({(k[1], k[2]) for k in recs if "sweep" in recs[k] and len(recs[k]["sweep"]) > 2})
    for ds, n in cells:
        for mode in ("1t", "batch"):
            fig, ax = plt.subplots(figsize=(8, 5.2))
            drew = False
            for k in sorted([k for k in recs if k[1] == ds and k[2] == n and "sweep" in recs[k]], key=lambda k: k[0]):
                pts = recs[k]["sweep"]
                if len(pts) < 3 and not OURS(k[0]):
                    continue
                xs, ys = [], []
                for p in pts:
                    q = p["qps_1t"] if mode == "1t" else (multi_qps(p)[0] if multi_qps(p)[1] == "batch" else None)
                    if q:
                        xs.append(p["recall"])
                        ys.append(q)
                if len(xs) < 2:
                    continue
                ax.plot(xs, ys, marker="o" if OURS(k[0]) else "s", ms=3, lw=2.4 if OURS(k[0]) else 1.2, ls="-" if OURS(k[0]) else "--", label=name(k[0]))
                drew = True
            if not drew:
                plt.close(fig)
                continue
            ax.set_yscale("log")
            ax.set_xlim(0.5, 1.005)
            ax.set_xlabel("recall@10 (vs exact float64 cosine)")
            ax.set_ylabel("queries / second (" + ("1 thread" if mode == "1t" else "batched, all cores") + ")")
            ax.set_title(f"{ds}, N={n:,}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc="lower left")
            f = os.path.join(out, f"{ds}_{n}_{mode}.png")
            fig.savefig(f, dpi=120, bbox_inches="tight")
            plt.close(fig)
            made.append(os.path.relpath(f, os.path.dirname(out)))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    recs, status = load(a.raw)
    os.makedirs(a.out, exist_ok=True)
    parts = {}
    parts["cells"] = "\n".join(section_cell(recs, ds, n) for ds, n in [("text384", 100000), ("glove", 100000), ("sift", 100000), ("text384", 1000000), ("glove", 1000000), ("sift", 1000000)])
    parts["exact"] = section_exact(recs)
    parts["mutation"] = section_mutation(recs)
    parts["filter"] = section_filter(recs)
    parts["saveload"] = section_saveload(recs)
    parts["failures"] = table(["system", "dataset", "N", "status"], [[name(k[0]), k[1], f"{k[2]:,}", v] for k, v in sorted(status.items())])
    for k, v in parts.items():
        open(os.path.join(a.out, f"tables_{k}.md"), "w", encoding="utf-8").write(v + "\n")
    made = plots(recs, os.path.join(a.out, "plots"))
    summary = {f"{k[0]}|{k[1]}|{k[2]}": {kk: vv for kk, vv in v.items() if kk in ("build", "mem_baseline", "mem_after_build", "mem_after_first_query", "sweep", "mutation", "saveload", "filter", "loop_allthreads", "version", "load_before_pct", "load_contaminated")}
               for k, v in recs.items()}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("systems/cells:", len(recs), "failures:", len(status), "plots:", len(made))


if __name__ == "__main__":
    main()
