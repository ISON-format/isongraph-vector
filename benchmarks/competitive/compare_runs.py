"""Cross-process repeatability: compare two independent runs (results/raw vs results/raw_rerun) of the same cells.

    python compare_runs.py > results/tables_repeat.md
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report as R

HERE = os.path.dirname(os.path.abspath(__file__))
a, _ = R.load(os.path.join(HERE, "results", "raw"))
b, _ = R.load(os.path.join(HERE, "results", "raw_rerun"))
rows = []
for k in sorted(b, key=lambda k: (not R.OURS(k[0]), k[0])):
    if k not in a or "sweep" not in a[k]:
        continue
    ra, rb = a[k], b[k]
    for thr in (0.95,):
        pa, pb = R.best_at(ra, thr), R.best_at(rb, thr)
        if not pa or not pb:
            continue
        d = (pb["qps_1t"] / pa["qps_1t"] - 1) * 100
        ba, bb = ra["build"]["total_s"], rb["build"]["total_s"]
        rows.append([R.name(k[0]), f"{k[1]} {k[2]:,}", f"{R.fmt(pa['qps_1t'])} @ {pa['recall']:.3f}", f"{R.fmt(pb['qps_1t'])} @ {pb['recall']:.3f}", f"{d:+.0f}%",
                     f"{R.fmt(ba)} / {R.fmt(bb)}"])
print(R.table(["system", "cell", "run 1: QPS 1T @ recall>=0.95", "run 2 (fresh process, later)", "difference", "build s run1 / run2"], rows))
