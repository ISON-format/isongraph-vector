"""Read raw/*.json (Rust stage_c_bench and Python bench_c.py results) and write tables.md + summary.json.

    python summarize.py            # from benchmarks/rust-stage-c/

Rules: medians over the repeated runs of a cell (min-max shown); QPS at recall 0.95 is log-linearly interpolated
between the two sweep points that bracket it (no extrapolation; '-' = never reached); memory = working set after
the build minus the baseline taken after imports (Python) / before any data (Rust).
"""
import glob
import json
import math
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

PRETTY = {
    "rust_exact": "ISONGraph Rust exact", "rust_ivf": "ISONGraph Rust IVF", "rust_hnsw": "ISONGraph Rust HNSW",
    "rust_ivf_i8x4": "ISONGraph Rust IVF + int8 x4", "rust_hnsw_i8x4": "ISONGraph Rust HNSW + int8 x4 (setting on)",
    "numpy_bf": "numpy brute force", "faiss_flat": "faiss IndexFlatIP", "faiss_ivfflat": "faiss IVFFlat",
    "faiss_hnsw": "faiss HNSWFlat", "hnswlib": "hnswlib", "usearch": "usearch",
}


def load_all():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        if p.endswith(".load.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if d.get("status") != "ok":
            continue
        sysname = d.get("system", "")
        if d.get("exps") == ["persist"]:
            sysname = "rust_persist"
        ld = {}
        lp = p[:-5] + ".load.json"
        if os.path.exists(lp):
            ld = json.load(open(lp))
        d["_file"] = os.path.basename(p)
        d["_load"] = ld
        runs.setdefault((sysname, d["ds"], d["n"]), []).append(d)
    return runs


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def rng(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def f(x, n=1, comma=True):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    return f"{x:,.{n}f}" if comma else f"{x:.{n}f}"


def cell(xs, n=1):
    """median (min-max) across runs"""
    m = med(xs)
    lo, hi = rng(xs)
    if m is None:
        return "-"
    if len(xs) < 2:
        return f"{f(m, n)} (1 run)"
    return f"{f(m, n)} ({f(lo, n)}-{f(hi, n)})"


def qps_of(pt):
    return pt.get("qps_1t")


def batch_of(pt):
    b = pt.get("batch")
    return None if not b else b.get("qps_batch")


def interp(sweep, key, target=0.95):
    """log-linear interpolation of key(pt) at recall=target; None if not bracketed."""
    pts = [p for p in sweep if p.get("knob") is not None and key(p) is not None]
    pts.sort(key=lambda p: p["recall"])
    for a, b in zip(pts, pts[1:]):
        if a["recall"] <= target <= b["recall"] and b["recall"] > a["recall"]:
            t = (target - a["recall"]) / (b["recall"] - a["recall"])
            return math.exp(math.log(key(a)) * (1 - t) + math.log(key(b)) * t)
    hit = [p for p in pts if p["recall"] >= target]
    if len(pts) and pts[0]["recall"] >= target:
        return None   # target below the first measured point: not bracketed
    return None


def ref_point(sweep, target=0.95):
    for p in sorted([p for p in sweep if p.get("knob") is not None], key=lambda p: p["knob"]):
        if p["recall"] >= target:
            return p
    return None


def mem_delta(d, which="mem_after_build"):
    b = d.get("mem_baseline", {}).get("rss_mb")
    a = d.get(which, {}).get("rss_mb")
    return None if a is None or b is None else a - b


def peak_delta(d, which="mem_after_build"):
    b = d.get("mem_baseline", {}).get("rss_mb")
    a = d.get(which, {}).get("peak_mb")
    return None if a is None or b is None else a - b


def build_s(d):
    b = d.get("build", {})
    if "total_s" in b:
        return b["total_s"]
    return b.get("ingest_s")


def index_s(d):
    return d.get("build", {}).get("index_s")


def exact_pt(d):
    sw = d.get("sweep", [])
    return sw[0] if sw else None


def auto_qps(d):
    if "loop_allthreads" in d:
        return d["loop_allthreads"].get("qps")
    pt = exact_pt(d)
    if pt and "auto_threads" in pt:
        return pt["auto_threads"].get("qps")
    return None


def exact_batch(d):
    pt = exact_pt(d)
    if pt and pt.get("batch"):
        return pt["batch"].get("qps_batch")
    return None


def main():
    R = load_all()
    md = []
    S = {}   # summary.json
    out = md.append

    def runs(sysn, ds, n):
        return R.get((sysn, ds, n), [])

    # ---------------------------------------------------------------- Q1 exact
    out("## Q1. Exact scan: ISONGraph Rust vs numpy vs faiss IndexFlatIP (single thread, one pinned P-core)\n")
    out("QPS, median over runs (min-max). `ratio` = Rust / other, > 1 means Rust is faster. Rounds were interleaved "
        "(Rust, numpy, faiss, Rust, ...) in one session.\n")
    out("| cell | runs | Rust exact | numpy | faiss Flat | Rust/numpy | Rust/faiss | Rust auto-thread loop | faiss Flat batch (32 thr) | numpy batch | Rust batch |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    S["q1"] = {}
    for ds, n in (("text384", 100_000), ("text384", 1_000_000), ("glove", 100_000), ("sift", 100_000)):
        rr, nn, ff = runs("rust_exact", ds, n), runs("numpy_bf", ds, n), runs("faiss_flat", ds, n)
        if not rr:
            continue
        q = lambda L: [qps_of(exact_pt(d)) for d in L if exact_pt(d)]  # noqa: E731
        mr, mn, mf = med(q(rr)), med(q(nn)), med(q(ff))
        S["q1"][f"{ds}_{n}"] = {"rust": mr, "numpy": mn, "faiss_flat": mf, "rust_runs": q(rr), "numpy_runs": q(nn), "faiss_runs": q(ff),
                                "ratio_numpy": (mr / mn if mr and mn else None), "ratio_faiss": (mr / mf if mr and mf else None)}
        out(f"| {ds} {n:,} | {len(rr)}/{len(nn)}/{len(ff)} | {cell(q(rr))} | {cell(q(nn))} | {cell(q(ff))} | "
            f"{f(mr / mn, 2) if mr and mn else '-'}x | {f(mr / mf, 2) if mr and mf else '-'}x | "
            f"{cell([auto_qps(d) for d in rr], 0)} | {cell([exact_batch(d) for d in ff], 0)} | {cell([exact_batch(d) for d in nn], 0)} | {cell([exact_batch(d) for d in rr], 0)} |")
    out("")

    # ---------------------------------------------------------------- Q2 ANN at recall 0.95
    ann_sys = ["rust_hnsw", "rust_ivf", "rust_ivf_i8x4", "rust_hnsw_i8x4", "hnswlib", "faiss_hnsw", "usearch", "faiss_ivfflat"]
    out("## Q2. Approximate search at recall@10 = 0.95, single thread (pinned), text384\n")
    out("QPS interpolated (log-linear) between the two sweep points that bracket recall 0.95; `-` = not bracketed by the "
        "sweep (recall 0.95 not reached, or the lowest point already above it: then the lowest measured point is shown in the "
        "`first point` column). Build = ingest + index build on the library's own path with 32 threads. Memory = working set "
        "after the build minus baseline.\n")
    for ds, n in (("text384", 100_000), ("text384", 1_000_000)):
        out(f"### {ds} {n:,}\n")
        out("| system | runs | QPS @0.95 (1T) | knob at first recall>=0.95 | recall at that knob | p99 ms there | build s | of which index s | RSS after build MB | peak MB | first point (knob: recall, QPS) | best recall reached |")
        out("|---|---|---|---|---|---|---|---|---|---|---|---|")
        S.setdefault("q2", {})[f"{ds}_{n}"] = {}
        for sn in ann_sys + ["rust_exact", "faiss_flat"]:
            rs = runs(sn, ds, n)
            if not rs:
                continue
            if sn in ("rust_exact", "faiss_flat"):
                q1 = [qps_of(exact_pt(d)) for d in rs if exact_pt(d)]
                out(f"| {PRETTY[sn]} (exact, recall 1.0) | {len(rs)} | {cell(q1)} | - | 1.000 | {f(med([exact_pt(d)['p99_ms'] for d in rs if exact_pt(d)]), 2)} | {cell([build_s(d) for d in rs], 2)} | 0 | {cell([mem_delta(d) for d in rs], 0)} | - | - | 1.000 |")
                continue
            qs = [interp(d["sweep"], qps_of) for d in rs]
            refs = [ref_point(d["sweep"]) for d in rs]
            kn = [r["knob"] for r in refs if r]
            rc = [r["recall"] for r in refs if r]
            p99 = [r["p99_ms"] for r in refs if r]
            fp = rs[0]["sweep"][0] if rs[0].get("sweep") else None
            fps = f"{fp['knob']}: {fp['recall']:.3f}, {fp['qps_1t']:,.0f}" if fp else "-"
            best = max((p["recall"] for d in rs for p in d["sweep"]), default=None)
            out(f"| {PRETTY[sn]} | {len(rs)} | {cell([x for x in qs if x], 0)} | {f(med(kn), 0) if kn else '-'} | {f(med(rc), 4) if rc else '-'} | {f(med(p99), 2)} | "
                f"{cell([build_s(d) for d in rs], 1)} | {cell([index_s(d) for d in rs], 1)} | {cell([mem_delta(d) for d in rs], 0)} | {cell([peak_delta(d) for d in rs], 0)} | {fps} | {f(best, 4)} |")
            S["q2"][f"{ds}_{n}"][sn] = {"qps_at_095": med([x for x in qs if x]), "runs": [x for x in qs if x], "build_s": med([build_s(d) for d in rs]),
                                        "rss_after_build_mb": med([mem_delta(d) for d in rs]), "best_recall": best}
        out("")
    # sweeps
    out("### Sweeps (median over runs; recall / QPS 1T)\n")
    for ds, n in (("text384", 100_000), ("text384", 1_000_000)):
        out(f"**{ds} {n:,}**\n")
        for sn in ann_sys:
            rs = runs(sn, ds, n)
            if not rs:
                continue
            knobs = sorted({p["knob"] for d in rs for p in d["sweep"]})
            cells = []
            for kn in knobs:
                pts = [p for d in rs for p in d["sweep"] if p["knob"] == kn]
                cells.append(f"{kn}: {med([p['recall'] for p in pts]):.3f} / {med([qps_of(p) for p in pts]):,.0f}")
            out(f"* {PRETTY[sn]} ({rs[0].get('knob_name', '')}): " + "; ".join(cells))
        out("")

    # ---------------------------------------------------------------- Q3 batch / all-core
    out("## Q3. Batch / all-core throughput, text384 100k (32 logical CPUs, 1,000 queries per batch, k=10)\n")
    out("Batch QPS is the median of 3 timed batches after a warm-up batch, at the sweep point nearest recall 0.95 "
        "(first knob with recall >= 0.95; its recall is shown). Exact rows are recall 1.0.\n")
    out("| system | runs | knob | recall there | batch QPS (32 thr) | 1T QPS there | batch / 1T |")
    out("|---|---|---|---|---|---|---|")
    S["q3"] = {}
    for ds, n in (("text384", 100_000), ("text384", 1_000_000)):
        out(f"| **{ds} {n:,}** | | | | | | |")
        for sn in ["rust_exact", "faiss_flat", "numpy_bf", "rust_ivf", "rust_ivf_i8x4", "faiss_ivfflat", "rust_hnsw", "faiss_hnsw", "hnswlib", "usearch"]:
            rs = runs(sn, ds, n)
            if not rs:
                continue
            if sn in ("rust_exact", "faiss_flat", "numpy_bf"):
                b = [exact_batch(d) for d in rs]
                o = [qps_of(exact_pt(d)) for d in rs if exact_pt(d)]
                out(f"| {PRETTY[sn]} | {len(rs)} | - | 1.000 | {cell(b, 0)} | {cell(o, 0)} | {f(med(b) / med(o), 0) + 'x' if med(b) and med(o) else '-'} |")
                S["q3"][f"{sn}_{ds}_{n}"] = {"batch_qps": med(b)}
                continue
            pts = [ref_point(d["sweep"]) for d in rs]
            pts = [p for p in pts if p]
            if not pts:
                continue
            b = [batch_of(p) for p in pts]
            o = [qps_of(p) for p in pts]
            out(f"| {PRETTY[sn]} | {len(rs)} | {pts[0]['knob']} | {f(med([p['recall'] for p in pts]), 3)} | {cell(b, 0)} | {cell(o, 0)} | {f(med(b) / med(o), 0) + 'x' if med(b) and med(o) else '-'} |")
            S["q3"][f"{sn}_{ds}_{n}"] = {"batch_qps": med(b), "knob": pts[0]["knob"]}
    out("")
    out("Rust concurrent clients (each caller runs single-scan-thread searches against one shared store; 3 s per client count, not pinned), QPS:\n")
    out("| cell | mode | 1 | 2 | 4 | 8 | 16 |")
    out("|---|---|---|---|---|---|---|")
    for ds, n in (("text384", 100_000), ("text384", 1_000_000)):
        for sn in ("rust_exact", "rust_ivf", "rust_hnsw"):
            rs = [d for d in runs(sn, ds, n) if "concurrent_clients" in d]
            if not rs:
                continue
            row = []
            for i in range(5):
                row.append(f(med([d["concurrent_clients"][i]["qps"] for d in rs]), 0))
            kn = rs[0].get("ref_knob")
            out(f"| {ds} {n:,} | {PRETTY[sn]}{' (knob ' + str(kn) + ')' if kn else ''} | " + " | ".join(row) + " |")
    out("")

    # ---------------------------------------------------------------- Q4 int8
    out("## Q4. int8 scan with exact rerank on real embeddings (exact search, Rust)\n")
    out("`recall vs f32 scan` is `int8_recall_report` (int8 top-10 vs the library's own f32 scan, first 1,000 queries; 300 at 1M); "
        "`recall vs truth` is against the float64 ground truth in the timed loop. Speedup = int8 QPS / f32 QPS from the same process "
        "(1T = pinned single thread; auto = one query at a time on the automatic thread count; batch = 1,000-query batch, 32 threads). "
        "Medians over runs where several exist.\n")
    out("| dataset | factor | recall vs f32 scan (mean / worst query) | recall vs truth | f32 1T QPS | int8 1T QPS | 1T speedup | auto-thread speedup | batch speedup | exact scores |")
    out("|---|---|---|---|---|---|---|---|---|---|")
    S["q4"] = {}
    for ds, n in (("text384", 100_000), ("text384", 1_000_000), ("glove", 100_000), ("sift", 100_000)):
        rs = [d for d in runs("rust_exact", ds, n) if "int8_study" in d]
        if not rs:
            continue
        S["q4"][f"{ds}_{n}"] = {}
        for fac in (1, 2, 4, 8):
            by = [next(b for b in d["int8_study"]["by_factor"] if b["rerank_factor"] == fac) for d in rs]
            f1 = [d["int8_study"]["f32"]["one_thread"]["qps_1t"] for d in rs]
            fa = [d["int8_study"]["f32"]["auto_threads"]["qps_1t"] for d in rs]
            fb = [d["int8_study"]["f32"]["batch"]["qps_batch"] for d in rs]
            i1 = [b["one_thread"]["qps_1t"] for b in by]
            ia = [b["auto_threads"]["qps_1t"] for b in by]
            ib = [b["batch"]["qps_batch"] for b in by]
            rec = med([b["recall_vs_f32_scan"] for b in by])
            mn = min(b["min_recall_vs_f32_scan"] for b in by)
            rt = med([b["one_thread"]["recall"] for b in by])
            sp1 = med([a / b for a, b in zip(i1, f1)])
            spa = med([a / b for a, b in zip(ia, fa)])
            spb = med([a / b for a, b in zip(ib, fb)])
            ex = all(b["scores_exact"] for b in by)
            out(f"| {ds} {n:,} | x{fac} | {f(rec, 4)} / {f(mn, 2)} | {f(rt, 4)} | {f(med(f1), 0)} | {f(med(i1), 0)} | {f(sp1, 2)}x | {f(spa, 2)}x | {f(spb, 2)}x | {ex} |")
            S["q4"][f"{ds}_{n}"][f"x{fac}"] = {"recall_vs_f32": rec, "worst": mn, "recall_vs_truth": rt, "speedup_1t": sp1, "speedup_auto": spa, "speedup_batch": spb}
        mm = med([d["int8_study"]["rss_delta_mb"] for d in rs])
        out(f"| {ds} {n:,} | memory | int8 copy {f(med([d['int8_study']['quant_bytes_mb'] for d in rs]), 0)} MB; working-set delta {f(mm, 0)} MB | | | | | | | |")
    out("")
    out("IVF and HNSW combined with int8 (setting `ScanPrecision::Int8 { rerank_factor: 4 }` before the build) are in the Q2 table (`+ int8 x4` rows) with the sweeps above.\n")

    # ---------------------------------------------------------------- Q5 persistence
    out("## Q5. Persistence: ISONGraph JSON sidecar vs binary snapshot vs faiss / hnswlib / usearch index files\n")
    out("ISONGraph = store only (vectors + keys), index not persisted. `ready` for the ANN rows adds the index rebuild "
        "(and a first query) that every ISONGraph load needs. Library rows are native `write_index` / `save_index` / `load`; "
        "their load time is to a usable index (the graph is inside the file).\n")
    out("| N | system | save s | file MB | load s | first query after load ms | time to ready s | note |")
    out("|---|---|---|---|---|---|---|---|")
    S["q5"] = {}
    for n in (100_000, 1_000_000):
        rs = [d for d in runs("rust_persist", "text384", n)]
        if rs:
            p = rs[0]["persist"]
            rd = p.get("time_to_ready_after_snapshot_load", {})
            S["q5"][f"rust_{n}"] = p
            out(f"| {n:,} | ISONGraph Rust JSON sidecar | {f(p['json_save_s'], 2)} | {f(p['json_mb'], 0)} | {f(p['json_load_s'], 2)} (runs {', '.join(f(x, 2) for x in p['json_load_s_runs'])}) | {f(p['json_first_query_ms'], 1)} | {f(p['json_load_s'] + p['json_first_query_ms'] / 1000, 2)} | exact search ready; answers equal original: {p['json_answers_equal_original']} |")
            out(f"| {n:,} | ISONGraph Rust binary snapshot | {f(p['snapshot_save_s'], 2)} (+{f(p['sidecar_fingerprint_s'], 2)} sidecar fingerprint) | {f(p['snapshot_mb'], 0)} | {f(p['snapshot_load_s'], 2)} (runs {', '.join(f(x, 2) for x in p['snapshot_load_s_runs'])}) | {f(p['snapshot_first_query_ms'], 1)} | {f(p['snapshot_load_s'] + p['snapshot_first_query_ms'] / 1000, 2)} | exact search ready; answers equal original: {p['snapshot_answers_equal_original']} |")
            for md_, v in rd.items():
                out(f"| {n:,} | ISONGraph Rust snapshot + {md_.upper()} rebuild | - | - | {f(v['load_s'], 2)} | {f(v['first_query_ms'], 1)} | {f(v['ready_s'], 2)} | load + build {f(v['build_s'], 1)} s + first query |")
        for sn in ("faiss_flat", "faiss_ivfflat", "faiss_hnsw", "hnswlib", "usearch"):
            rr = [d for d in runs(sn, "text384", n) if "saveload" in d and "save_s" in d["saveload"]]
            if not rr:
                continue
            sl = [d["saveload"] for d in rr]
            out(f"| {n:,} | {PRETTY[sn]} | {cell([s['save_s'] for s in sl], 3)} | {f(med([s['disk_mb'] for s in sl]), 0)} | {cell([s['load_s'] for s in sl], 3)} | {f(med([s['first_query_after_load_ms'] for s in sl]), 2)} | {f(med([s['time_to_ready_s'] for s in sl]), 3)} | native index file |")
    out("")

    # ---------------------------------------------------------------- Q6 writes
    out("## Q6. Writes and the first query after writes (text384; index built, recall ~0.95 operating point)\n")
    out("10k inserts, then 10k updates (new vector under an existing id), then 10k deletes, one call at a time on the built index "
        "(Rust: `add` / `remove`; libraries as in the competitive harness: calls of 1,000). Then the first search and recall@10 against "
        "the exact post-mutation float64 truth at the same knob. `p99/max` are per-write latencies (Rust only).\n")
    out("| N | system | knob | insert/s | update/s | delete/s | insert p99 / max ms | update p99 / max ms | first query after ms | recall before | recall after | QPS 1T after |")
    out("|---|---|---|---|---|---|---|---|---|---|---|---|")
    S["q6"] = {}
    for n in (100_000, 1_000_000):
        for sn in ("rust_ivf", "rust_hnsw", "faiss_ivfflat", "hnswlib", "usearch", "faiss_hnsw"):
            rs = [d for d in runs(sn, "text384", n) if isinstance(d.get("mutation"), dict) and "insert_10k" in d["mutation"]]
            if not rs:
                continue
            ms = [d["mutation"] for d in rs]
            def g(key, sub="per_s"):
                return med([m_[key].get(sub) for m_ in ms if isinstance(m_.get(key), dict) and sub in m_[key]])
            def lat(key):
                a = med([m_[key]["lat"]["p99_ms"] for m_ in ms if "lat" in m_.get(key, {})])
                b = med([m_[key]["lat"]["max_ms"] for m_ in ms if "lat" in m_.get(key, {})])
                return "-" if a is None else f"{f(a, 2)} / {f(b, 1)}"
            def un(key):
                return any(isinstance(m_.get(key), dict) and ("unsupported" in m_[key] or "error" in m_[key]) for m_ in ms)
            fqa = med([m_.get("first_query_after_ms") for m_ in ms])
            ra = med([m_.get("recall_after") for m_ in ms])
            rb = med([m_.get("recall_before") for m_ in ms])
            qa = med([(m_.get("after") or {}).get("qps_1t") for m_ in ms])
            out(f"| {n:,} | {PRETTY[sn]} ({len(rs)}) | {ms[0].get('ref_knob')} | {f(g('insert_10k'), 0)} | {f(g('update_10k'), 0)}{' (unsupported)' if un('update_10k') else ''} | {f(g('delete_10k'), 0)}{' (unsupported)' if un('delete_10k') else ''} | {lat('insert_10k')} | {lat('update_10k')} | {f(fqa, 1)} | {f(rb, 4)} | {f(ra, 4)} | {f(qa, 0)} |")
            S["q6"][f"{sn}_{n}"] = {"insert_s": g("insert_10k"), "update_s": g("update_10k"), "delete_s": g("delete_10k"), "first_query_after_ms": fqa, "recall_after": ra}
    out("")

    # ---------------------------------------------------------------- Q7 memory
    out("## Q7. Memory at 1M x 384 (working set, MB; baseline subtracted)\n")
    out("Raw vectors = 1,000,000 x 384 x 4 = 1,465 MB. Each Rust row is its own process (ingest, then that mode only). "
        "Python rows: working set after the index is built and again after the sweep, with the harness' preallocation (hnswlib `max_elements` = N + 20k).\n")
    out("| system | after ingest | after build / first query | peak during build | after the sweep | x raw vectors (after build) |")
    out("|---|---|---|---|---|---|")
    S["q7"] = {}
    raw_mb = 1_000_000 * 384 * 4 / 2**20
    for n in (100_000, 1_000_000):
        raw_mb = n * 384 * 4 / 2**20
        out(f"| **text384 {n:,}** (raw {f(raw_mb, 0)} MB) | | | | | |")
        for sn in ["rust_exact", "rust_ivf", "rust_hnsw", "rust_ivf_i8x4", "rust_hnsw_i8x4", "faiss_flat", "numpy_bf", "faiss_ivfflat", "faiss_hnsw", "hnswlib", "usearch"]:
            rs = runs(sn, "text384", n)
            if not rs:
                continue
            ingest = med([mem_delta(d, "mem_after_ingest") for d in rs]) if sn.startswith("rust") else None
            if sn.startswith("rust"):
                aft = [mem_delta(d, "mem_after_build") if "mem_after_build" in d else mem_delta(d, "mem_after_first_exact_query") for d in rs]
                # the exact process runs the int8 study before it measures, so its own peak would include the int8 copy: use the peak after ingest
                pk = [peak_delta(d, "mem_after_build") if "mem_after_build" in d else peak_delta(d, "mem_after_ingest") for d in rs]
            else:
                aft = [mem_delta(d, "mem_after_first_query") for d in rs]
                pk = [peak_delta(d, "mem_after_first_query") for d in rs]
            sweep = [mem_delta(d, "mem_after_sweep") for d in rs]
            a = med(aft)
            out(f"| {PRETTY[sn]} | {f(ingest, 0)} | {f(a, 0)} | {f(med(pk), 0)} | {f(med(sweep), 0)} | {f(a / raw_mb, 2) + 'x' if a else '-'} |")
            S["q7"][f"{sn}_{n}"] = {"after_build_mb": a, "peak_mb": med(pk), "x_raw": (a / raw_mb if a else None)}
        ex = [d for d in runs("rust_exact", "text384", n) if "int8_study" in d]
        if ex:
            a8 = med([d["int8_study"]["mem_after_int8"]["rss_mb"] - d["mem_baseline"]["rss_mb"] for d in ex])
            out(f"| ISONGraph Rust exact + int8 copy (any factor) | - | {f(a8, 0)} | - | - | {f(a8 / raw_mb, 2)}x |")
            S["q7"][f"rust_exact_int8_{n}"] = {"after_build_mb": a8, "x_raw": a8 / raw_mb}
    out("")
    # Rust lib-reported breakdown
    out("Rust library-reported breakdown (`memory_stats`, 1M):\n")
    for sn in ("rust_ivf", "rust_hnsw", "rust_ivf_i8x4", "rust_hnsw_i8x4"):
        rs = [d for d in runs(sn, "text384", 1_000_000) if "memory_stats" in d]
        if rs:
            m_ = rs[0]["memory_stats"]
            out(f"* {PRETTY[sn]}: matrix {m_['matrix_bytes'] / 2**20:,.0f} MB, metadata {m_['metadata_bytes'] / 2**20:,.0f} MB, index {m_['index_bytes'] / 2**20:,.0f} MB, int8 copy {m_['quant_bytes'] / 2**20:,.0f} MB")
    out("")

    # ---------------------------------------------------------------- build time / noise
    out("## Appendix: noisy runs and machine load at job start\n")
    out("| job file | total CPU % | pinned-CPU % | flagged noisy |")
    out("|---|---|---|---|")
    for (sn, ds, n), L in sorted(R.items()):
        for d in L:
            ld = d.get("_load", {})
            if ld.get("noisy") or ld.get("total_pct", 0) > 25:
                out(f"| {d['_file']} | {f(ld.get('total_pct'), 0)} | {f(ld.get('pin_cpu_pct'), 0)} | {ld.get('noisy')} |")
    out("")
    open(os.path.join(HERE, "tables.md"), "w").write("\n".join(md))
    json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
