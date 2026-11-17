"""Turn raw/*.json from run_ab.sh into the markdown tables in RESULTS.md.

    python summarize.py > RESULTS_tables.md
"""
import glob
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, pattern))):
        try:
            out.append(json.load(open(f, encoding="utf-8")))
        except Exception:
            pass
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def fmt(x, digits=0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.{digits}f}"


def cb_runs(label, ds, n):
    """competitive_bench result files (one per repeat) -> list of {exact, ivf} dicts."""
    runs = []
    for d in load(f"cb_{label}_{ds}_{n}*.json"):
        by = {r["system"].split("_")[-1]: r for r in d["runs"]}
        runs.append(by)
    return runs


def at_recall(sweep, target):
    """QPS interpolated (log-linear in qps) at the recall target; None if never reached."""
    pts = sorted(((p["recall"], p["qps_1t"], p["knob"]) for p in sweep), key=lambda t: t[0])
    for (r0, q0, k0), (r1, q1, k1) in zip(pts, pts[1:]):
        if r0 <= target <= r1 and r1 > r0:
            w = (target - r0) / (r1 - r0)
            return math.exp(math.log(q0) * (1 - w) + math.log(q1) * w)
    return None


def table_cb(ds, n):
    old, new = cb_runs("old", ds, n), cb_runs("new", ds, n)
    if not old or not new:
        return ""
    lines = [f"#### {ds} n={n:,} (median over {len(old)} old / {len(new)} new run(s))", ""]
    lines += ["| metric | old | new | new/old |", "|---|---|---|---|"]

    def row(name, f, digits=0, ratio=True):
        a = med([f(r) for r in old])
        b = med([f(r) for r in new])
        lines.append(f"| {name} | {fmt(a, digits)} | {fmt(b, digits)} | {fmt(b / a, 2) + 'x' if ratio and a else '-'} |")

    row("exact, 1 thread (pinned), QPS", lambda r: r["exact"]["sweep"][0]["qps_1t"], 1)
    row("exact, one-query loop with auto threads, QPS", lambda r: r["exact"]["loop_allthreads"]["qps"], 1)
    row("8 concurrent clients (1 thread each), exact QPS", lambda r: r["exact"]["concurrent_clients"]["qps"], 1)
    if "batch_allthreads" in new[0]["exact"]:
        lines.append(f"| exact batch, all threads, QPS | - | {fmt(med([r['exact']['batch_allthreads']['qps'] for r in new]), 0)} | - |")
    row("IVF build, s", lambda r: r["ivf"]["build"]["index_s"], 2)
    row("IVF 1-thread QPS at recall 0.95 (interpolated)", lambda r: at_recall(r["ivf"]["sweep"], 0.95), 0)
    row("IVF one-query loop, auto threads, QPS at ref nprobe", lambda r: next((p["allthreads"]["qps"] for p in r["ivf"]["sweep"] if p["knob"] == r["ivf"]["ref_knob"]), None), 0)
    if "batch_allthreads" in new[0]["ivf"]["sweep"][0]:
        lines.append("| IVF batch, all threads, QPS at ref nprobe | - | " + fmt(med([next(p["batch_allthreads"]["qps"] for p in r["ivf"]["sweep"] if p["knob"] == r["ivf"]["ref_knob"]) for r in new]), 0) + " | - |")
    row("ingest, s", lambda r: r["ivf"]["build"]["ingest_s"], 2)
    row("working set after ingest, MB", lambda r: r["ivf"]["mem_after_ingest"]["rss_mb"] - r["ivf"]["mem_baseline"]["rss_mb"], 0)
    row("working set after IVF build, MB", lambda r: r["ivf"]["mem_after_build"]["rss_mb"] - r["ivf"]["mem_baseline"]["rss_mb"], 0)
    row("peak working set after IVF build, MB", lambda r: r["ivf"]["mem_after_build"]["peak_mb"] - r["ivf"]["mem_baseline"]["rss_mb"], 0)
    if all("mutation" in r["ivf"] and "insert_10k" in r["ivf"]["mutation"] for r in old + new):
        row("insert 10k, per s", lambda r: r["ivf"]["mutation"]["insert_10k"]["per_s"], 0)
        row("update 10k, per s", lambda r: r["ivf"]["mutation"]["update_10k"]["per_s"], 0)
        row("delete 10k, per s", lambda r: r["ivf"]["mutation"]["delete_10k"]["per_s"], 0)
        row("first query after the 30k writes, ms", lambda r: r["ivf"]["mutation"]["first_query_after_ms"], 2)
        row("recall@10 after the writes", lambda r: r["ivf"]["mutation"]["recall_after"], 4, ratio=False)
    lines += ["", "recall@10 / 1-thread QPS by nprobe (median of runs):", "", "| nprobe | old recall | new recall | old QPS | new QPS |", "|---|---|---|---|---|"]
    knobs = [p["knob"] for p in old[0]["ivf"]["sweep"]]
    for k in knobs:
        def pick(runs, key):
            return med([next((p[key] for p in r["ivf"]["sweep"] if p["knob"] == k), None) for r in runs])
        lines.append(f"| {k} | {pick(old, 'recall'):.4f} | {pick(new, 'recall'):.4f} | {fmt(pick(old, 'qps_1t'))} | {fmt(pick(new, 'qps_1t'))} |")
    lines.append("")
    return "\n".join(lines)


def table_eb():
    out = []
    for kind in ["scaling"]:
        old = load(f"eb_old_100000_scaling*.json")
        new = load(f"eb_new_100000_scaling*.json")
        if old and new and "scaling" in old[0] and "scaling" in new[0]:
            out += ["#### Concurrent single-query callers, 100k x 384 (QPS; efficiency = QPS / (clients x 1-client QPS))", ""]
            for mode in ["exact", "ivf"]:
                out += [f"{mode}:", "", "| clients | old QPS | old eff. | new QPS | new eff. |", "|---|---|---|---|---|"]
                for a, b in zip(old[0]["scaling"][mode], new[0]["scaling"][mode]):
                    out.append(f"| {a['clients']} | {fmt(a['qps'])} | {a['efficiency']:.2f} | {fmt(b['qps'])} | {b['efficiency']:.2f} |")
                out.append("")
    new = load("eb_new_100000_*batch*.json")
    if new and "batch" in new[0]:
        out += ["#### Batch search, 100k x 384, 2000 queries (QPS by thread cap)", "", "| threads | exact batch | IVF batch (nprobe 128, recall 0.955) |", "|---|---|---|"]
        for a, b in zip(new[0]["batch"]["exact"], new[0]["batch"]["ivf"]):
            out.append(f"| {a['threads']} | {fmt(a['qps'])} | {fmt(b['qps'])} |")
        out.append("")
    old, new = load("eb_old_100000_*loop*.json"), load("eb_new_100000_*loop*.json")
    if old and new and "loop" in old[0]:
        out += ["#### One-query exact loop by thread cap, 100k x 384 (QPS, unpinned, median of 5)", "", "| threads | old | new |", "|---|---|---|"]
        for a, b in zip(old[0]["loop"], new[0]["loop"]):
            out.append(f"| {a['threads'] or 'auto'} | {fmt(a['qps'], 1)} | {fmt(b['qps'], 1)} |")
        out.append("")
    old, new = load("eb_old_1000000_mem*.json"), load("eb_new_1000000_mem*.json")
    if old and new:
        o, n = old[0]["mem"], new[0]["mem"]
        out += ["#### Memory, 1M x 384 (working set minus baseline, MB)", "", "| stage | old | new |", "|---|---|---|"]
        out.append(f"| raw vectors | {o['raw_mb']:.0f} | {n['raw_mb']:.0f} |")
        for key, name in [("after_ingest_mb", "after ingest"), ("peak_after_ingest_mb", "peak during ingest"), ("after_first_query_mb", "after first exact query"), ("after_ivf_build_mb", "after IVF build"), ("peak_after_build_mb", "peak during IVF build")]:
            out.append(f"| {name} | {o[key]:.0f} ({o[key] / o['raw_mb']:.2f}x) | {n[key]:.0f} ({n[key] / n['raw_mb']:.2f}x) |")
        out.append("")
    old, new = load("eb_old_1000000_writes*.json"), load("eb_new_1000000_writes*.json")
    if old and new:
        out += [f"#### Writes on a built IVF index, 1M x 384 (median of {len(old)} / {len(new)} run(s))", "", "| measure | old | new |", "|---|---|---|"]

        def w(runs, path):
            vals = []
            for r in runs:
                x = r["writes"]
                for p in path:
                    x = x[p]
                vals.append(x)
            return med(vals)

        for name, path, d in [
            ("insert 30k, per s", ["insert_30k", "per_s"], 0),
            ("slowest single insert, ms", ["insert_30k", "max_ms"], 2),
            ("insert p99, ms", ["insert_30k", "p99_ms"], 4),
            ("update 10k, per s", ["update_10k", "per_s"], 0),
            ("slowest single update, ms", ["update_10k", "max_ms"], 2),
            ("delete 10k, per s", ["delete_10k", "per_s"], 0),
            ("first query after 30k inserts (1 thread), ms", ["after_30k_inserts", "first_ms"], 2),
            ("steady query median after them (1 thread), ms", ["after_30k_inserts", "median_ms"], 2),
            ("first query after all 50k writes (1 thread), ms", ["after_all", "first_ms"], 2),
            ("query median before any write (1 thread), ms", ["before_1t", "median_ms"], 2),
        ]:
            out.append(f"| {name} | {fmt(w(old, path), d)} | {fmt(w(new, path), d)} |")
        out.append("")
    old, new = load("eb_old_1000000_deletes*.json"), load("eb_new_1000000_deletes*.json")
    if old and new:
        out += ["#### Deleting 10,000 random ids from 1M x 384", "", "| store state | old per s (ns each) | new per s (ns each) |", "|---|---|---|"]
        for k in old[0]["deletes"]:
            a, b = old[0]["deletes"][k], new[0]["deletes"][k]
            out.append(f"| {k} | {fmt(a['per_s'])} ({a['ns_each']:.0f}) | {fmt(b['per_s'])} ({b['ns_each']:.0f}) |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print("### Rust stage A: before / after (same session)\n")
    for ds, n in [("text384", 100000), ("text384", 1000000), ("glove", 100000), ("sift", 100000)]:
        print(table_cb(ds, n))
    print(table_eb())
