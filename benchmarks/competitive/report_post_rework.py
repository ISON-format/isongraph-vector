"""Tables for the post-rework competitive run: results/post_rework/raw/*.json -> results/post_rework/tables_*.md

    python report_post_rework.py [--raw results/post_rework/raw] [--old results/raw] [--out results/post_rework] [--n 100000]

Nothing here measures anything; every number is read from the raw JSON files (the old, pre-rework run is read from
results/raw, read-only, only for the before/after table). Repeat runs of the same job (file names ending `__r2`, `__r3`)
are kept apart: the tables use the first run of each job, tables_repeat.md compares the repeats.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

import report as R
from report import fmt, interp_at, table

HERE = os.path.dirname(os.path.abspath(__file__))

DISPLAY = dict(R.DISPLAY)
DISPLAY.update({
    "rust_default_hnsw": "ISONGraph Rust (HNSW)", "rust_native_hnsw": "ISONGraph Rust, target-cpu=native (HNSW)",
    "rust_default_exact_int8": "ISONGraph Rust (exact, int8+rerank)", "rust_native_exact_int8": "ISONGraph Rust, target-cpu=native (exact, int8+rerank)",
    "cpp_default_hnsw": "ISONGraph C++ (HNSW)", "cpp_avx2_hnsw": "ISONGraph C++, /arch:AVX2 (HNSW)",
    "csharp_hnsw": "ISONGraph C# (HNSW)", "js_hnsw": "ISONGraph JS (HNSW)",
    "ts_exact": "ISONGraph TS (exact)", "ts_ivf": "ISONGraph TS (IVF)", "ts_hnsw": "ISONGraph TS (HNSW)",
})
OURS = lambda s: s.startswith(("isong", "rust", "cpp", "csharp", "js_", "ts_"))  # noqa: E731
EXACT = {"numpy_bf", "isong_py", "faiss_flat"}


def name(s):
    return ("**" + DISPLAY.get(s, s) + "**") if OURS(s) else DISPLAY.get(s, s)


def is_exact(s):
    return s in EXACT or s.endswith("_exact") or s.endswith("_exact_int8")


def load(raw):
    """system -> list of records (first = primary). Also failures."""
    recs, fails, loads = {}, [], []
    for f in sorted(glob.glob(os.path.join(raw, "*.json"))):
        base = os.path.basename(f)
        if base.endswith(".meta.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            fails.append((base, f"unreadable: {e}"))
            continue
        if d.get("status") != "ok":
            fails.append((base, d.get("status", "?") + (": " + d["error"] if d.get("error") else "") + (f" (limit {d.get('limit_s')} s)" if d.get("limit_s") else "")))
            continue
        m = re.search(r"__(r\d+)\.json$", base)
        rep = m.group(1) if m else "r1"
        loads.append((base, d.get("load_before_pct"), d.get("load_contaminated"), d.get("wall_s")))
        for r in (d["runs"] if "runs" in d else [d]):
            s = r.get("system") or d.get("system")
            if not s:
                continue
            r = dict(r, _rep=rep, _file=base, _job_exp=d.get("job_exp"))
            recs.setdefault((s, r.get("n", d.get("n"))), []).append(r)
    for k in recs:
        recs[k].sort(key=lambda r: r["_rep"])
    return recs, fails, loads


def merged(rlist, rep="r1"):
    """merge the records of one repetition (one system can come from several jobs: core job + saveload job)."""
    out = {}
    for r in rlist:
        if r["_rep"] != rep:
            continue
        for k, v in r.items():
            if k in ("sweep", "build", "mem_baseline", "mem_after_build", "mem_after_ingest", "mem_after_sweep") and k in out and r.get("sweep") is None:
                continue   # saveload-only record: do not overwrite the measured build with the ingest-only one
            out[k] = v
        out.setdefault("_files", []).append(r["_file"])
    return out


def first_at(rec, thr=0.95):
    for p in rec.get("sweep", []):
        if p.get("recall") is not None and p["recall"] >= thr:
            return p
    return None


def batch_of(rec, p):
    if p is None:
        return None
    if p.get("batch"):
        return p["batch"]["qps_batch"]
    if p.get("batch_allthreads"):
        return p["batch_allthreads"]["qps"]
    return None


def exact_batch(rec):
    if rec.get("batch_allthreads"):
        return rec["batch_allthreads"]["qps"]
    sw = rec.get("sweep") or []
    return batch_of(rec, sw[0]) if sw else None


def mem_plus(rec, key):
    base = rec.get("mem_baseline", {}).get("rss_mb")
    v = rec.get(key, {}).get("rss_mb")
    if base is None or v is None:
        return None
    return v - base


def peak_plus(rec):
    base = rec.get("mem_baseline", {}).get("rss_mb")
    v = rec.get("mem_after_build", {}).get("peak_mb")
    return None if base is None or v is None else v - base


def star(v):
    if not v:
        return "-"
    return fmt(v[0]) + "*" if isinstance(v, tuple) else fmt(v)


def val(v):
    return v[0] if isinstance(v, tuple) else v


def build_s(rec):
    b = rec.get("build", {})
    reps = b.get("total_s_runs")
    return (statistics.median(reps) if reps else b.get("total_s")), reps


ANN_ORDER = ["faiss_hnsw", "hnswlib", "faiss_ivfflat", "rust_default_hnsw", "rust_native_hnsw", "cpp_default_hnsw", "cpp_avx2_hnsw", "csharp_hnsw", "js_hnsw", "ts_hnsw",
             "rust_default_ivf", "rust_native_ivf", "cpp_default_ivf", "cpp_avx2_ivf", "csharp_ivf", "isong_py_ivf", "js_ivf", "ts_ivf"]
EXACT_ORDER = ["numpy_bf", "faiss_flat", "isong_py", "rust_default_exact", "rust_default_exact_int8", "rust_native_exact", "rust_native_exact_int8",
               "cpp_default_exact", "cpp_avx2_exact", "csharp_exact", "js_exact", "ts_exact"]


def t_ann(recs, n):
    rows = []
    for s in ANN_ORDER:
        if (s, n) not in recs:
            continue
        rec = merged(recs[(s, n)])
        if not rec.get("sweep"):
            continue
        p = first_at(rec)
        qi = interp_at(rec, 0.95, lambda p_: p_["qps_1t"])
        b, reps = build_s(rec)
        bq = batch_of(rec, p)
        thr = None
        if p and p.get("batch"):
            thr = p["batch"].get("threads")
        elif p and p.get("batch_allthreads"):
            thr = p["batch_allthreads"].get("threads")
        rows.append([name(s), (f"{p['knob']} ({p['recall']:.3f})" if p else "not reached"), star(qi),
                     (f"{fmt(p['qps_1t'])} ({fmt(p.get('qps_1t_min'))}-{fmt(p.get('qps_1t_max'))})" if p else "-"),
                     (f"{fmt(p['median_ms'])} / {fmt(p['p99_ms'])}" if p else "-"),
                     (fmt(bq) + (f" ({thr} thr)" if bq and thr else "") if bq else "-") + " / " + star(interp_at(rec, 0.95, lambda p_: batch_of(rec, p_))),
                     fmt(rec["build"].get("total_s")) + (f" ({fmt(min(reps))}-{fmt(max(reps))} over {len(reps)} builds)" if reps else ""),
                     f"{mem_plus(rec, 'mem_after_build'):,.0f}" if mem_plus(rec, "mem_after_build") is not None else "-",
                     f"{mem_plus(rec, 'mem_after_sweep'):,.0f}" if mem_plus(rec, "mem_after_sweep") is not None else "-"])
    hdr = ["system", "knob at first recall>=0.95 (recall)", "QPS 1T @0.95 (interpolated)", "QPS 1T at that point (3-run min-max)", "latency ms median / p99",
           "batch QPS (all CPUs): at that knob / interpolated @0.95", "build s (ingest + index)", "RSS +MB after build", "RSS +MB after sweep"]
    return table(hdr, rows)


def t_exact(recs, n):
    rows = []
    for s in EXACT_ORDER:
        if (s, n) not in recs:
            continue
        rec = merged(recs[(s, n)])
        sw = rec.get("sweep") or []
        if not sw:
            continue
        e = sw[0]
        la = rec.get("loop_allthreads")
        b, reps = build_s(rec)
        rows.append([name(s), f"{fmt(e['qps_1t'])} ({fmt(e.get('qps_1t_min'))}-{fmt(e.get('qps_1t_max'))})", f"{fmt(e['median_ms'])} / {fmt(e['p99_ms'])}",
                     f"{e['recall']:.3f}", (f"{fmt(la['qps'])}" if la else "-"), fmt(exact_batch(rec)), fmt(rec["build"].get("total_s")),
                     fmt(rec["build"].get("first_query_ms")),
                     f"{mem_plus(rec, 'mem_after_first_query') if mem_plus(rec, 'mem_after_first_query') is not None else float('nan'):,.0f}",
                     (rec["concurrent_clients"] and f"{fmt(rec['concurrent_clients']['qps'])} ({rec['concurrent_clients']['clients']} clients)") if isinstance(rec.get("concurrent_clients"), dict) else "-"])
    hdr = ["system", "QPS 1T (3-run min-max)", "latency ms median / p99", "recall", "QPS, one query at a time, library may use all threads", "batch QPS (all CPUs)",
           "build s", "first query ms", "RSS +MB (steady)", "N concurrent single-thread clients"]
    return table(hdr, rows)


def t_saveload(recs, n):
    rows = []
    for s in ["faiss_flat", "faiss_ivfflat", "faiss_hnsw", "hnswlib", "numpy_bf", "isong_py", "isong_py_ivf", "rust_default_ivf", "cpp_default_ivf", "csharp_ivf", "js_exact", "ts_exact"]:
        if (s, n) not in recs:
            continue
        rec = merged(recs[(s, n)])
        sl = rec.get("saveload")
        if not sl:
            continue
        if sl.get("error"):
            rows.append([name(s), "error: " + sl["error"][:80]] + ["-"] * 8)
            continue
        rows.append([name(s), fmt(sl.get("save_s")) if not sl.get("json_error") else "FAILED: " + sl["json_error"], fmt(sl.get("disk_mb")), fmt(sl.get("load_s")), fmt(sl.get("time_to_ready_s")),
                     fmt(sl.get("snap_save_s")), fmt(sl.get("snap_mb")), fmt(sl.get("snap_load_s")),
                     (fmt(sl["load_s"] / sl["snap_load_s"]) + "x") if sl.get("snap_load_s") and sl.get("load_s") else "-",
                     (fmt(sl["save_s"] / sl["snap_save_s"]) + "x") if sl.get("snap_save_s") and sl.get("save_s") else "-"])
    hdr = ["system", "JSON/native save s", "file MB", "load s", "time to ready s (load + first query; IVF/index rebuilt, never persisted)",
           "snapshot save s", "snapshot MB", "snapshot load s", "snapshot load speed-up vs JSON", "snapshot save speed-up vs JSON"]
    return table(hdr, rows)


def t_concurrent(recs, n):
    rows = []
    for s in ["rust_default_hnsw", "rust_native_hnsw", "cpp_default_hnsw", "cpp_avx2_hnsw", "csharp_hnsw"]:
        if (s, n) not in recs:
            continue
        rec = merged(recs[(s, n)])
        c = rec.get("concurrent_clients")
        p = first_at(rec)
        if not c or not p:
            continue
        rows.append([name(s), p["knob"], fmt(p["qps_1t"]), fmt(batch_of(rec, p))] + [f"{fmt(x['qps'])} ({x['clients']} clients)" for x in c])
    return table(["system", "ef (first >=0.95)", "1T QPS", "batch QPS (all CPUs)", "concurrent clients A", "concurrent clients B"], rows)


def t_ratios(recs, n):
    def g(s, f):
        if (s, n) not in recs:
            return None
        rec = merged(recs[(s, n)])
        return f(rec)

    def q95(rec):
        return val(interp_at(rec, 0.95, lambda p: p["qps_1t"]))

    def q95b(rec):
        return val(interp_at(rec, 0.95, lambda p: batch_of(rec, p)))

    def exq(rec):
        return rec["sweep"][0]["qps_1t"] if rec.get("sweep") else None

    refs = {"faiss HNSW": "faiss_hnsw", "hnswlib": "hnswlib", "faiss IVFFlat": "faiss_ivfflat"}
    rows = []
    for s in ["rust_default_hnsw", "rust_native_hnsw", "cpp_default_hnsw", "cpp_avx2_hnsw", "csharp_hnsw", "js_hnsw", "ts_hnsw",
              "rust_default_ivf", "cpp_default_ivf", "csharp_ivf", "isong_py_ivf", "js_ivf", "ts_ivf"]:
        v = g(s, q95)
        if not v:
            continue
        row = [name(s), fmt(v)]
        for r in refs.values():
            rv = g(r, q95)
            row.append(f"{v / rv:.2f}x" if rv else "-")
        vb = g(s, q95b)
        fb = g("faiss_hnsw", q95b)
        row.append(f"{vb / fb:.2f}x" if vb and fb else "-")
        rb = g(s, lambda rec: rec["build"]["total_s"])
        fbld = g("faiss_hnsw", lambda rec: rec["build"]["total_s"])
        hbld = g("hnswlib", lambda rec: rec["build"]["total_s"])
        row.append(f"{rb / fbld:.2f}x / {rb / hbld:.2f}x" if rb and fbld and hbld else "-")
        m = g(s, lambda rec: mem_plus(rec, "mem_after_build"))
        mf = g("faiss_hnsw", lambda rec: mem_plus(rec, "mem_after_build"))
        mh = g("hnswlib", lambda rec: mem_plus(rec, "mem_after_build"))
        row.append(f"{m / mf:.2f}x / {m / mh:.2f}x" if m and mf and mh else "-")
        rows.append(row)
    ann = table(["ISONGraph system", "QPS 1T @0.95", "vs faiss HNSW", "vs hnswlib", "vs faiss IVFFlat", "batch QPS @0.95 vs faiss HNSW batch", "build time vs faiss HNSW / hnswlib (first build in a fresh process, as for ISONGraph)", "RSS after build vs faiss HNSW / hnswlib"], rows)
    rows = []
    for s in ["isong_py", "rust_default_exact", "rust_default_exact_int8", "rust_native_exact", "rust_native_exact_int8", "cpp_default_exact", "cpp_avx2_exact", "csharp_exact", "js_exact", "ts_exact"]:
        v = g(s, exq)
        if not v:
            continue
        row = [name(s), fmt(v)]
        for r in ("numpy_bf", "faiss_flat"):
            rv = g(r, exq)
            row.append(f"{v / rv:.2f}x" if rv else "-")
        vb = g(s, exact_batch)
        for r in ("numpy_bf", "faiss_flat"):
            rv = g(r, exact_batch)
            row.append(f"{vb / rv:.3f}x" if vb and rv else "-")
        rows.append(row)
    ex = table(["ISONGraph exact", "QPS 1T", "vs numpy (1T)", "vs faiss Flat (1T)", "batch QPS vs numpy batch", "batch QPS vs faiss Flat batch"], rows)
    return ann + "\n\n" + ex


def old_vs_new(recs, n, old_raw):
    """before/after for the ISONGraph systems (old numbers are read from the pre-rework raw files, same cell)."""
    if not old_raw or not os.path.isdir(old_raw):
        return "(pre-rework raw directory not found)"
    orecs, _, _ = load(old_raw)
    rows = []
    drift_vals = []
    pairs = [("isong_py", "isong_py"), ("isong_py_ivf", "isong_py_ivf"), ("rust_default_exact", "rust_default_exact"), ("rust_default_ivf", "rust_default_ivf"),
             ("cpp_default_exact", "cpp_default_exact"), ("cpp_default_ivf", "cpp_default_ivf"), ("csharp_exact", "csharp_exact"), ("csharp_ivf", "csharp_ivf"),
             ("js_exact", "js_exact"), ("js_ivf", "js_ivf"), ("faiss_hnsw", "faiss_hnsw"), ("hnswlib", "hnswlib"), ("faiss_ivfflat", "faiss_ivfflat"), ("faiss_flat", "faiss_flat"), ("numpy_bf", "numpy_bf")]
    for new_s, old_s in pairs:
        if (new_s, n) not in recs or (old_s, n) not in orecs:
            continue
        a = merged(orecs[(old_s, n)])
        b = merged(recs[(new_s, n)])
        if not a.get("sweep") or not b.get("sweep"):
            continue
        if is_exact(new_s):
            qa, qb = a["sweep"][0]["qps_1t"], b["sweep"][0]["qps_1t"]
        else:
            qa, qb = val(interp_at(a, 0.95, lambda p: p["qps_1t"])), val(interp_at(b, 0.95, lambda p: p["qps_1t"]))
        if new_s in ("faiss_hnsw", "hnswlib", "faiss_ivfflat", "faiss_flat", "numpy_bf") and qa and qb:
            drift_vals.append(qb / qa)
        ba, bb = a["build"].get("total_s"), b["build"].get("total_s")
        ma, mb = mem_plus(a, "mem_after_first_query") if a.get("mem_after_first_query") else mem_plus(a, "mem_after_build"), mem_plus(b, "mem_after_build")
        rows.append([name(new_s), fmt(qa), fmt(qb), f"{qb / qa:.2f}x" if qa and qb else "-", "@@DRIFT@@" + (f"{qb / qa:.4f}" if qa and qb else "-"), fmt(ba), fmt(bb), (f"{ma:,.0f}" if ma is not None else "-"), (f"{mb:,.0f}" if mb is not None else "-")])
    drift = math.exp(sum(math.log(v) for v in drift_vals) / len(drift_vals)) if drift_vals else None
    for r in rows:
        raw = r[4].replace("@@DRIFT@@", "")
        r[4] = f"{float(raw) / drift:.2f}x" if drift and raw != "-" else "-"
    head = (f"Machine drift between the two runs, taken as the geometric mean of the QPS change of the five non-ISONGraph reference systems (faiss Flat/IVF/HNSW, hnswlib, numpy; code unchanged): **{drift:.2f}x**. "
            "`change / drift` divides that out; read it, not the raw `change`, as the effect of the rework. Build times and RSS are not drift-corrected.\n\n") if drift else ""
    return head + table(["system", "QPS 1T before (exact, or @0.95)", "after", "change", "change / drift", "build s before", "after", "RSS +MB before", "after"], rows)


def t_repeat(recs, n):
    rows = []
    for (s, nn), lst in sorted(recs.items()):
        if nn != n:
            continue
        reps = sorted({r["_rep"] for r in lst})
        if len(reps) < 2:
            continue
        vals = []
        for rp in reps:
            rec = merged(lst, rp)
            if not rec.get("sweep"):
                continue
            vals.append(rec["sweep"][0]["qps_1t"] if is_exact(s) else val(interp_at(rec, 0.95, lambda p: p["qps_1t"])))
        b = [merged(lst, rp).get("build", {}).get("total_s") for rp in reps]
        vals = [v for v in vals if v]
        if len(vals) >= 2:
            rows.append([name(s), ", ".join(fmt(v) for v in vals), f"{(max(vals) / min(vals) - 1) * 100:.1f}%", ", ".join(fmt(x) for x in b if x)])
    return table(["system", "QPS 1T (exact) or @0.95 per whole-process repeat", "spread max/min", "build s per repeat"], rows) if rows else "(no repeats were run)"


def full_sweeps(recs, n):
    md = []
    for s in ANN_ORDER:
        if (s, n) not in recs:
            continue
        rec = merged(recs[(s, n)])
        rows = []
        for p in rec.get("sweep", []):
            rows.append([p.get("knob"), f"{p['recall']:.4f}", f"{fmt(p['qps_1t'])} ({fmt(p.get('qps_1t_min'))}-{fmt(p.get('qps_1t_max'))})", fmt(p["median_ms"]), fmt(p["p99_ms"]),
                         fmt(batch_of(rec, p)), p.get("nq")])
        md.append(f"**{DISPLAY.get(s, s)}** ({rec.get('knob_name')})\n")
        md.append(table(["knob", "recall@10", "QPS 1T (min-max)", "median ms", "p99 ms", "batch QPS", "queries/run"], rows))
        md.append("")
        dp = rec.get("default_ef_point")
        if dp:
            md.append(f"Library default ef_search ({dp.get('knob')}): recall {dp['recall']:.4f}, 1T QPS {fmt(dp['qps_1t'])}, batch QPS {fmt(batch_of(rec, dp))}\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "results", "post_rework", "raw"))
    ap.add_argument("--old", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "post_rework"))
    ap.add_argument("--n", type=int, nargs="*", default=[100000, 1000000])
    a = ap.parse_args()
    recs, fails, loads = load(a.raw)
    ns = [n for n in a.n if any(k[1] == n for k in recs)]
    for n in ns:
        parts = [f"### text384, N = {n:,}  (post-rework run)\n"]
        parts.append("#### Approximate systems: queries/second at recall@10 >= 0.95, build, memory\n")
        parts.append("`QPS 1T @0.95` is log-linearly interpolated between the two measured sweep points that bracket 0.95 (as in the original tables; `*` = bracket wider than 0.05 recall, the measured point above 0.95 is shown). The next column is the raw first measured point with recall >= 0.95, with the min-max of its three timed runs. Single-thread = one pinned P-core, one query at a time. Batch = one batched call over all 32 logical CPUs (ISONGraph: the port's batch search; faiss/hnswlib: their multi-threaded batch query). RSS is the process working set above the baseline taken before any data is touched: after the build (before any query) and after the whole sweep.\n")
        parts.append(t_ann(recs, n))
        parts.append("\n#### Exact systems\n")
        parts.append(t_exact(recs, n))
        parts.append("\n#### Save / load\n")
        parts.append(t_saveload(recs, n))
        parts.append("\n#### Concurrent clients (HNSW stores)\n")
        parts.append(t_concurrent(recs, n))
        parts.append("\n#### Ratios (ISONGraph / competitor; > 1 means ISONGraph is faster, or for build/RSS larger)\n")
        parts.append(t_ratios(recs, n))
        parts.append("\n#### Before / after the rework (same cell, same machine; before = results/raw, pre-rework run)\n")
        parts.append(old_vs_new(recs, n, a.old))
        parts.append("\n#### Whole-process repeats\n")
        parts.append(t_repeat(recs, n))
        parts.append("\n<details><summary>Full accuracy sweeps</summary>\n")
        parts.append(full_sweeps(recs, n))
        parts.append("</details>\n")
        open(os.path.join(a.out, f"tables_{n}.md"), "w", encoding="utf-8").write("\n".join(parts))
    lines = ["### Jobs: machine load before each job (all other processes), wall time, failures\n", table(["file", "CPU load before %", "contaminated", "wall s"], [[f, fmt(l), c, fmt(w)] for f, l, c, w in loads])]
    if fails:
        lines += ["\n#### Failed jobs\n", table(["file", "reason"], [[f, r] for f, r in fails])]
    open(os.path.join(a.out, "tables_jobs.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote tables for", ns, "failures:", len(fails))


if __name__ == "__main__":
    main()
