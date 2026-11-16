#!/usr/bin/env python3
"""
Approximate-index benchmark for isongraph-vector: recall@10 and latency of the
opt-in IVF index (``EmbeddingStore(ann="ivf")``) against the exact numpy scan.

For each size and each data regime it reports

  * exact scan: first-query (cache build) cost and warm latency
  * index build time and memory (index bytes, and RSS growth + peak while it
    builds, via psutil when installed)
  * a sweep over ``nprobe``: recall@10 versus exact, and median / p95 latency
    and the speed-up, plus the automatic default (``nprobe=None``)
  * (unless --no-drift) the index under writes: +1% vectors and -1% deletes
    (new rows still pending), +2% more (folded into cells), and the cost of
    the retrain that a +60% churn triggers, with recall at each stage

Unlike scale.py, the vectors here are CLUSTERED: an IVF index's recall depends
entirely on how the data clusters, so uniform random vectors (where it
degenerates to a coin flip) would say nothing. Two synthetic regimes are
used, both 384-d and unit length:

  * ``clustered``  well-separated topics: N/100 Gaussian clusters; two points
                   of one cluster have cosine about 0.5, of different ones
                   about 0
  * ``diffuse``    hierarchical and overlapping: N/40 clusters grouped into 20
                   super-topics, noisy members (same-cluster cosine about 0.2,
                   same super-topic about 0.1): a harder, more embedding-like case

Both are synthetic. Real embeddings (a sentence-transformer over a real corpus)
will sit somewhere between or outside them; use ``store.recall_check`` on your
own queries before relying on a setting.

Usage:
    python benchmarks/ann.py                       # 20k and 100k, both regimes (a few minutes)
    python benchmarks/ann.py --quick               # 20k only
    python benchmarks/ann.py --include-1m          # adds 1,000,000 vectors (~10 GB RAM, ~15-25 minutes)
    python benchmarks/ann.py --sizes 1000000 --include-1m --no-drift --out-prefix benchmarks/results/ann_1M
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import environment_info, import_package  # noqa: E402

ivec = import_package()

try:
    import numpy as np
except ImportError:
    raise SystemExit("ann.py needs numpy (pip install numpy)")

try:
    import psutil
except ImportError:
    psutil = None

DIM = 384
CHUNK = 5000
SEED = 4242
K = 10
EmbeddingStore = ivec.EmbeddingStore

# regime -> (points per cluster, noise norm, super-topics or None, super spread)
REGIMES = {
    "clustered": dict(per_cluster=100, noise=1.0, supers=None, spread=0.0),
    "diffuse": dict(per_cluster=40, noise=2.0, supers=20, spread=1.0),
}


# =============================================================================
# data
# =============================================================================

def unit(x: np.ndarray) -> np.ndarray:
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


class Regime:
    """Deterministic generator of clustered unit vectors for one regime and size."""

    def __init__(self, name: str, n: int, dim: int = DIM):
        spec = REGIMES[name]
        self.name, self.dim, self.noise = name, dim, spec["noise"]
        rng = np.random.default_rng([SEED, 1, n % 1000003, len(name)])
        clusters = max(8, n // spec["per_cluster"])
        if spec["supers"]:
            supers = unit(rng.standard_normal((spec["supers"], dim), dtype=np.float32))
            parent = rng.integers(0, spec["supers"], clusters)
            centers = unit(supers[parent] + spec["spread"] * unit(
                rng.standard_normal((clusters, dim), dtype=np.float32)))
        else:
            centers = unit(rng.standard_normal((clusters, dim), dtype=np.float32))
        self.centers = centers.astype(np.float32)

    def sample(self, count: int, stream: int) -> np.ndarray:
        rng = np.random.default_rng([SEED, 2, stream])
        which = rng.integers(0, self.centers.shape[0], count)
        noise = rng.standard_normal((count, self.dim), dtype=np.float32)
        noise *= self.noise / np.sqrt(self.dim)
        return unit(self.centers[which] + noise)


def load(store, regime: Regime, n: int, first: int = 0) -> float:
    """add_batch rows [first, first+n) in CHUNK-sized calls; vectors/second."""
    spent = 0.0
    for start in range(0, n, CHUNK):
        count = min(CHUNK, n - start)
        vectors = regime.sample(count, stream=1000 + (first + start) // CHUNK).tolist()
        entries = [(("doc", first + start + i), "", vectors[i]) for i in range(count)]
        t0 = time.perf_counter()
        store.add_batch(entries)
        spent += time.perf_counter() - t0
    return n / spent


# =============================================================================
# measuring
# =============================================================================

class PeakRSS:
    """Poll process RSS on a thread; report growth over the start value."""

    def __init__(self):
        self.proc = psutil.Process() if psutil else None
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        if self.proc:
            self.base = self.proc.memory_info().rss
            self.peak = self.base
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.02)

    def __exit__(self, *exc):
        if self.proc:
            self._stop.set()
            self._thread.join()
            self.end = self.proc.memory_info().rss
            self.peak = max(self.peak, self.end)

    def growth_mb(self) -> Optional[float]:
        return (self.end - self.base) / 2**20 if self.proc else None

    def peak_mb(self) -> Optional[float]:
        return (self.peak - self.base) / 2**20 if self.proc else None


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def timed_searches(store, queries: List[List[float]], k: int, exact: bool):
    """Per-query latencies (ms) and the node-ref sets returned."""
    for q in queries[:3]:                      # warm up
        store.similarity_search(q, k, exact=exact)
    times, found = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = store.similarity_search(q, k, exact=exact)
        times.append((time.perf_counter() - t0) * 1000)
        found.append({r.node_ref for r in res})
    return times, found


def recall(found: List[set], truth: List[set]) -> Dict[str, float]:
    per = [len(f & t) / len(t) for f, t in zip(found, truth) if t]
    return {"mean": sum(per) / len(per), "min": min(per),
            "p5": percentile(per, 0.05)}


def run_one(n: int, regime_name: str, args) -> dict:
    regime = Regime(regime_name, n)
    queries = regime.sample(args.queries, stream=7).tolist()
    out: dict = {"n": n, "regime": regime_name}
    print(f"\n=== {regime_name}: {n:,} vectors ===", flush=True)

    store = EmbeddingStore(db_path=":memory:", ann="ivf", ann_min_size=0, ann_seed=0)
    out["ingest_per_s"] = round(load(store, regime, n))
    print(f"  ingested ({out['ingest_per_s']:,}/s)", flush=True)

    t0 = time.perf_counter()
    store.similarity_search(queries[0], K, exact=True)
    out["exact_first_query_s"] = round(time.perf_counter() - t0, 2)

    gc.collect()
    exact_t, truth = timed_searches(store, queries, K, exact=True)
    out["exact_ms"] = {"median": round(percentile(exact_t, 0.5), 2),
                       "p95": round(percentile(exact_t, 0.95), 2)}
    print(f"  exact: median {out['exact_ms']['median']} ms", flush=True)

    with PeakRSS() as mem:
        stats = store.build_ann_index()
    out["build"] = {"seconds": round(stats["build_seconds"], 2), "nlist": stats["nlist"],
                    "auto_nprobe": stats["nprobe"], "index_mb": round(stats["index_bytes"] / 2**20, 1),
                    "rss_growth_mb": None if mem.growth_mb() is None else round(mem.growth_mb()),
                    "rss_peak_growth_mb": None if mem.peak_mb() is None else round(mem.peak_mb()),
                    "raw_vectors_mb": round(n * DIM * 4 / 2**20)}
    print(f"  build {out['build']['seconds']} s, nlist {stats['nlist']}, "
          f"index {out['build']['index_mb']} MB, peak RSS +{out['build']['rss_peak_growth_mb']} MB", flush=True)

    rows = []
    nlist = stats["nlist"]
    auto = stats["nprobe"]
    probes = sorted({p for p in args.nprobes if p < nlist} | {auto})
    for p in probes:
        store.ann_nprobe = p
        ann_t, found = timed_searches(store, queries, K, exact=False)
        r = recall(found, truth)
        row = {"nprobe": p, "auto": p == auto, "pct_cells": round(100 * p / nlist, 1),
               "recall_mean": round(r["mean"], 4), "recall_min": round(r["min"], 2),
               "recall_p5": round(r["p5"], 2),
               "ms_median": round(percentile(ann_t, 0.5), 2),
               "ms_p95": round(percentile(ann_t, 0.95), 2)}
        row["speedup"] = round(out["exact_ms"]["median"] / row["ms_median"], 1)
        rows.append(row)
        print(f"  nprobe {p:>4} ({row['pct_cells']:>4}% cells): recall@{K} {row['recall_mean']:.3f} "
              f"(p5 {row['recall_p5']:.2f})  {row['ms_median']:.2f} ms  x{row['speedup']}", flush=True)
    out["sweep"] = rows
    store.ann_nprobe = None                    # automatic from here on

    if not args.no_drift:
        out["drift"] = run_drift(store, regime, n, queries)
    store.close()
    return out


def run_drift(store, regime: Regime, n: int, queries) -> dict:
    """Index under writes: pending rows, folding, deletes, retrain.

    Stage 1 stays under the fold threshold (max(512, 2% of the trained size)),
    so the new rows are scanned exactly alongside the probed cells. Stage 2
    crosses it: the next search folds them into cells. Stage 3 pushes the
    churn past 50% of the trained size, which retrains.
    """
    drift: dict = {}

    def measure(label: str) -> dict:
        t, found = timed_searches(store, queries, K, exact=False)
        _, truth = timed_searches(store, queries, K, exact=True)
        s = store.ann_stats()
        row = {"pending": s["pending"], "recall": round(recall(found, truth)["mean"], 4),
               "ms_median": round(percentile(t, 0.5), 2), "nlist": s["nlist"]}
        drift[label] = row
        return row

    one = max(100, n // 100)
    load(store, regime, one, first=n)                                    # +1%
    for i in range(one):                                                 # -1%, spread out
        store.remove(("doc", (i * 97) % n))
    measure("stage1_pending")
    load(store, regime, 2 * one, first=n + one)                          # +2% more: over the threshold
    store.similarity_search(queries[0], K)                               # this search folds
    measure("stage2_folded")
    builds = store.ann_stats()["searches"]["builds"]
    load(store, regime, int(0.6 * n), first=n + 3 * one)                 # churn past 50%
    t0 = time.perf_counter()
    store.similarity_search(queries[0], K)
    drift["retrain_query_s"] = round(time.perf_counter() - t0, 2)
    drift["retrained"] = store.ann_stats()["searches"]["builds"] > builds
    measure("stage3_retrained")
    print(f"  drift: pending {drift['stage1_pending']['pending']} -> recall {drift['stage1_pending']['recall']}; "
          f"folded -> {drift['stage2_folded']['recall']}; retrain query {drift['retrain_query_s']} s "
          f"(retrained={drift['retrained']}) -> recall {drift['stage3_retrained']['recall']}", flush=True)
    return drift


# =============================================================================
# report
# =============================================================================

def render(results: List[dict], env: dict, args) -> str:
    lines = ["# Approximate index (IVF-flat) benchmark", "",
             f"Machine: {env.get('cpu_count')} logical CPUs, {env.get('ram_gb')} GB RAM, "
             f"{env['platform']}, Python {env['python']}, numpy {env['numpy']}. "
             f"{DIM}-d unit vectors, top_k={K}, {args.queries} queries per cell, "
             "warm single-thread medians. Synthetic clustered data; see "
             "`benchmarks/README.md` for what that does and does not show.", ""]
    for r in results:
        b = r["build"]
        lines += [f"## {r['regime']}, N = {r['n']:,}", "",
                  f"* exact scan: first query {r['exact_first_query_s']} s (builds the cache), "
                  f"warm median {r['exact_ms']['median']} ms, p95 {r['exact_ms']['p95']} ms",
                  f"* index build: {b['seconds']} s for nlist={b['nlist']} "
                  f"(automatic nprobe={b['auto_nprobe']}); index itself {b['index_mb']} MB on top of "
                  f"{b['raw_vectors_mb']} MB of vectors; RSS growth +{b['rss_growth_mb']} MB "
                  f"(peak during the build +{b['rss_peak_growth_mb']} MB)",
                  f"* ingest {r['ingest_per_s']:,} vectors/s (unchanged by `ann`: the index is built lazily)", "",
                  f"| nprobe | % of cells | recall@{K} mean | p5 | worst | median ms | p95 ms | speed-up |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in r["sweep"]:
            tag = " (auto)" if s["auto"] else ""
            lines.append(f"| {s['nprobe']}{tag} | {s['pct_cells']} | {s['recall_mean']:.3f} | "
                         f"{s['recall_p5']:.2f} | {s['recall_min']:.2f} | {s['ms_median']} | "
                         f"{s['ms_p95']} | {s['speedup']}x |")
        d = r.get("drift")
        if d:
            a, f, t = d["stage1_pending"], d["stage2_folded"], d["stage3_retrained"]
            lines += ["",
                      f"Under writes (automatic nprobe, recall measured against the exact scan of the "
                      f"current data): +1% new vectors and -1% deletes, new rows still pending "
                      f"({a['pending']} scanned exactly): recall {a['recall']}, {a['ms_median']} ms. "
                      f"A further +2% crosses the fold threshold and the next search folds them into "
                      f"cells ({f['pending']} pending): recall {f['recall']}, {f['ms_median']} ms. "
                      f"A further +60% pushes churn past 50% of the trained size and retrains "
                      f"(retrained={d['retrained']}): that one query took {d['retrain_query_s']} s; "
                      f"afterwards nlist={t['nlist']}, recall {t['recall']}, {t['ms_median']} ms."]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="20000,100000")
    ap.add_argument("--include-1m", action="store_true",
                    help="add (or, with --sizes, allow) 1,000,000 vectors: ~10 GB RAM, many minutes")
    ap.add_argument("--regimes", default="clustered,diffuse")
    ap.add_argument("--nprobes", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="20k only, 60 queries")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--out-prefix", default=str(Path(__file__).resolve().parent / "results" / "ann"))
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    if args.quick:
        sizes, args.queries = [20000], 60
    if args.include_1m and 1_000_000 not in sizes:
        sizes.append(1_000_000)
    if any(s >= 500_000 for s in sizes) and not args.include_1m:
        raise SystemExit("sizes of 500,000 or more need --include-1m (about 10 GB of RAM)")
    args.nprobes = [int(p) for p in args.nprobes.split(",") if p]

    results = []
    for n in sizes:
        for name in args.regimes.split(","):
            results.append(run_one(n, name, args))
    env = environment_info()
    text = render(results, env, args)
    print("\n" + text)
    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".md").write_text(text, encoding="utf-8")
        prefix.with_suffix(".json").write_text(
            json.dumps({"environment": env, "k": K, "dim": DIM, "results": results}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {prefix.with_suffix('.md')} and {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
