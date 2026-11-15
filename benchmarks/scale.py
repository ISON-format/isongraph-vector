#!/usr/bin/env python3
"""
Scale benchmark for isongraph-vector.

Measures, for the in-memory numpy scan path and (when available) the
sqlite-vec path, over random normalised 384-d vectors:

  * ingestion throughput (EmbeddingStore.add_batch, 5,000 per call)
  * warm query latency: similarity_search(top_k=10), semantic_multi_hop
    (pure and blend=0.3) and semantic_subgraph - median of N runs
  * first-query-after-load cost and write-after-read cost (the numpy path
    patches its cached matrix in place on writes)
  * memory: RSS delta (psutil sampler) or tracemalloc peak when psutil is absent
  * a concurrent read/write section (threads): throughput, latency, errors,
    and lost writes

It then fits latency against N and reports where the linear scan crosses a
latency budget (default 50 ms), with a recommendation about an ANN index.

Random vectors have no cluster structure, so this measures cost, not
quality (see retrieval_quality.py). Exact search cost does not depend on the
data distribution, so the latency numbers transfer; recall numbers would not.

Usage:
    python benchmarks/scale.py                       # 1k, 10k, 100k (about a minute or two)
    python benchmarks/scale.py --quick               # 1k and 10k only
    python benchmarks/scale.py --max-n 1000000       # adds 1M (needs ~8 GB free RAM, several minutes)
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import environment_info, import_package  # noqa: E402

ivec = import_package()

try:
    import numpy as np
except ImportError:  # the whole benchmark is about the vectorised path
    raise SystemExit("scale.py needs numpy (pip install numpy)")

try:
    import psutil
except ImportError:
    psutil = None

import tracemalloc  # noqa: E402

DIM = 384
CHUNK = 5000
SEED = 1234
SemanticGraph = ivec.SemanticGraph
MockEncoder = ivec.MockEncoder
Direction = __import__("ison_graph").Direction


# =============================================================================
# helpers
# =============================================================================

def chunk_vectors(start: int, count: int, dim: int = DIM) -> np.ndarray:
    """Deterministic random normalised float32 vectors for rows [start, start+count)."""
    rng = np.random.default_rng([SEED, start])
    x = rng.standard_normal((count, dim), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def query_vectors(n: int, dim: int = DIM) -> List[List[float]]:
    rng = np.random.default_rng([SEED, 999_999_937])
    q = rng.standard_normal((n, dim), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.tolist()


def timeit(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def timeit_nogc(fn: Callable[[], object]) -> float:
    """Like timeit, with the cyclic garbage collector off for the call."""
    gc.disable()
    try:
        return timeit(fn)
    finally:
        gc.enable()


def summarize(ms: List[float]) -> dict:
    s = sorted(ms)
    return {"median_ms": statistics.median(s), "p95_ms": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
            "min_ms": s[0], "max_ms": s[-1], "runs": len(s)}


class PeakMemory:
    """Peak resident-set growth while active (psutil), else tracemalloc peak."""

    def __init__(self):
        self.mode = "rss" if psutil else "tracemalloc"
        self._proc = psutil.Process() if psutil else None
        self._stop = threading.Event()
        # The sampler holds the GIL for a moment on every poll, which roughly
        # doubled measured query latency, so it only runs while "active" and
        # is paused around every timed section.
        self._active = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.base = 0
        self.peak = 0

    def rss_mb(self) -> float:
        if self._proc:
            return self._proc.memory_info().rss / 2**20
        cur, _ = tracemalloc.get_traced_memory()
        return cur / 2**20

    def __enter__(self):
        gc.collect()
        if self.mode == "tracemalloc":
            tracemalloc.start()
            tracemalloc.reset_peak()
        self.base = self.rss_mb()
        self.peak = self.base
        if self._proc:
            def poll():
                while not self._stop.is_set():
                    if self._active.is_set():
                        self.peak = max(self.peak, self._proc.memory_info().rss / 2**20)
                    time.sleep(0.02)
            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        self._active.set()
        return self

    def pause(self):
        self._active.clear()

    def resume(self):
        self._active.set()

    def delta_mb(self) -> float:
        """Current growth over the starting point."""
        cur = self.rss_mb()
        self.peak = max(self.peak, cur)
        return cur - self.base

    def peak_delta_mb(self) -> float:
        if self.mode == "tracemalloc":
            return tracemalloc.get_traced_memory()[1] / 2**20 - self.base
        self.peak = max(self.peak, self.rss_mb())
        return self.peak - self.base

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if self.mode == "tracemalloc":
            tracemalloc.stop()


def build_graph(n: int, avg_degree: float, sqlite_vec: bool) -> SemanticGraph:
    """A SemanticGraph with n nodes and ~avg_degree*n directed edges (no embeddings yet)."""
    g = SemanticGraph(name="scale", embedding_db=":memory:", encoder=MockEncoder(DIM),
                      auto_embed=False, sqlite_vec=sqlite_vec)
    for i in range(n):
        g.add_node("n", i)
    rng = np.random.default_rng([SEED, 31337])
    m = int(n * avg_degree)
    src = rng.integers(0, n, m).tolist()
    dst = rng.integers(0, n, m).tolist()
    seen = set()
    for s, d in zip(src, dst):
        if s == d or (s, d) in seen:
            continue
        seen.add((s, d))
        g.add_edge("LINK", ("n", s), ("n", d))
    return g


def ingest(g: SemanticGraph, n: int) -> float:
    """Embed n nodes with add_batch in CHUNK-sized calls. Returns seconds inside add_batch."""
    spent = 0.0
    for start in range(0, n, CHUNK):
        count = min(CHUNK, n - start)
        vecs = chunk_vectors(start, count).tolist()          # generation + list conversion not timed
        entries = [(("n", start + j), "", v) for j, v in enumerate(vecs)]
        t0 = time.perf_counter()
        g.embedding_store.add_batch(entries)
        spent += time.perf_counter() - t0
    return spent


# =============================================================================
# single-configuration benchmark
# =============================================================================

def bench_config(backend: str, n: int, args) -> dict:
    use_vec = backend == "sqlite_vec"
    out: dict = {"backend": backend, "n": n, "dim": DIM}
    runs = args.runs
    heavy_runs = max(3, runs // 3)
    qs = query_vectors(max(runs, 32))

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        g = build_graph(n, args.avg_degree, use_vec)
        out["graph_build_s"] = time.perf_counter() - t0
        out["rss_after_graph_mb"] = mem.delta_mb()

        secs = ingest(g, n)
        out["ingest_s"] = secs
        out["ingest_vectors_per_s"] = n / secs
        out["rss_after_ingest_mb"] = mem.delta_mb()
        store = g.embedding_store
        assert store.count() == n, "ingestion lost rows"

        # first query pays the snapshot build on the numpy path
        out["first_query_ms"] = timeit(lambda: store.similarity_search(qs[0], top_k=10))
        out["rss_after_first_query_mb"] = mem.delta_mb()
        mem.pause()  # no sampler thread while latencies are being timed

        # warm similarity_search
        for q in qs[:2]:
            store.similarity_search(q, top_k=10)
        ms = [timeit(lambda q=q: store.similarity_search(q, top_k=10)) for q in qs[:runs]]
        out["similarity_search"] = summarize(ms)

        # graph-aware queries (threshold=-1: random vectors never reach the default 0.3)
        kw = dict(max_hops=2, top_k_seeds=5, top_k_results=10, threshold=-1.0, direction=Direction.OUT)
        g.semantic_multi_hop(qs[0], **kw)
        out["semantic_multi_hop"] = summarize(
            [timeit(lambda q=q: g.semantic_multi_hop(q, **kw)) for q in qs[:heavy_runs]])
        out["semantic_multi_hop_blend0.3"] = summarize(
            [timeit(lambda q=q: g.semantic_multi_hop(q, blend=0.3, **kw)) for q in qs[:heavy_runs]])
        sub_ms = []
        for q in qs[:heavy_runs]:
            sub_ms.append(timeit(lambda q=q: g.semantic_subgraph(q, **kw).close()))
        out["semantic_subgraph"] = summarize(sub_ms)

        # The same two queries with the cyclic GC off. A graph with hundreds of
        # thousands of live objects makes every full collection that a query's
        # own allocations trigger expensive; this separates that from the work.
        out["similarity_search_gc_off"] = summarize(
            [timeit_nogc(lambda q=q: store.similarity_search(q, top_k=10)) for q in qs[:runs]])
        out["semantic_multi_hop_gc_off"] = summarize(
            [timeit_nogc(lambda q=q: g.semantic_multi_hop(q, **kw)) for q in qs[:heavy_runs]])

        # write-after-read: one write, then the next search
        warm = out["similarity_search"]["median_ms"]
        add_ms, after_ms = [], []
        wr_runs = max(3, min(5, runs))
        for i in range(wr_runs):
            v = chunk_vectors(n + 7 * i, 1)[0].tolist()
            add_ms.append(timeit(lambda i=i, v=v: store.add(("w", i), "", vector=v)))
            after_ms.append(timeit(lambda: store.similarity_search(qs[i], top_k=10)))
        out["write_one"] = summarize(add_ms)
        out["search_after_write"] = summarize(after_ms)
        out["write_after_read_penalty_ms"] = max(0.0, statistics.median(after_ms) - warm)

        # batch write then search
        batch = [(("b", j), "", v) for j, v in enumerate(chunk_vectors(n + 100_000, 100).tolist())]
        out["add_batch_100_ms"] = timeit(lambda: store.add_batch(batch))
        out["search_after_batch_ms"] = timeit(lambda: store.similarity_search(qs[0], top_k=10))

        mem.delta_mb()
        out["memory_mode"] = mem.mode
        out["peak_memory_mb"] = mem.peak_delta_mb()
        out["raw_vector_bytes_mb"] = n * DIM * 4 / 2**20
    g.close()
    del g, store
    gc.collect()
    return out


def raw_numpy_floor(n: int, runs: int) -> dict:
    """The cost a bare matrix-vector product + argpartition would have: no SQLite, no Python loop."""
    mat = np.concatenate([chunk_vectors(s, min(CHUNK, n - s)) for s in range(0, n, CHUNK)])
    qs = np.asarray(query_vectors(max(runs, 8)), dtype=np.float32)

    def one(q):
        scores = mat @ q
        top = np.argpartition(scores, -10)[-10:]
        return top[np.argsort(-scores[top])]

    one(qs[0])
    ms = [timeit(lambda q=q: one(q)) for q in qs[:runs]]
    del mat
    gc.collect()
    return summarize(ms)


# =============================================================================
# concurrency
# =============================================================================

def run_concurrent(backend: str, n: int, readers: int, writers: int, duration: float,
                   graph_ops: bool = False) -> dict:
    use_vec = backend == "sqlite_vec"
    g = build_graph(n, 3.0, use_vec)
    ingest(g, n)
    store = g.embedding_store
    qs = query_vectors(64)
    store.similarity_search(qs[0], top_k=10)  # warm the snapshot
    stop = threading.Event()
    lock = threading.Lock()
    read_ms: List[float] = []
    writes_ok = [0]
    writes_attempted = [0]
    errors: Counter = Counter()
    first_error: Dict[str, str] = {}

    def record_error(kind: str, exc: BaseException):
        with lock:
            errors[kind] += 1
            first_error.setdefault(kind, f"{type(exc).__name__}: {exc}"[:200])

    kw = dict(max_hops=2, top_k_seeds=5, top_k_results=10, threshold=-1.0)

    def reader(idx: int):
        local = []
        i = idx
        while not stop.is_set():
            q = qs[i % len(qs)]
            i += 1
            t0 = time.perf_counter()
            try:
                if graph_ops:
                    g.semantic_multi_hop(q, **kw)
                else:
                    store.similarity_search(q, top_k=10)
                local.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:  # noqa: BLE001 - the point is to count them
                record_error("read:" + type(exc).__name__, exc)
        with lock:
            read_ms.extend(local)

    def writer(idx: int):
        j = 0
        rng = np.random.default_rng([SEED, idx, 5])
        while not stop.is_set():
            v = rng.standard_normal(DIM, dtype=np.float32)
            v = (v / np.linalg.norm(v)).tolist()
            ref = ("cw", f"{idx}-{j}")
            j += 1
            with lock:
                writes_attempted[0] += 1
            try:
                if graph_ops:
                    g.add_node("cw", ref[1])
                    g.add_edge("LINK", ref, ("n", j % n))
                store.add(ref, "", vector=v)
                with lock:
                    writes_ok[0] += 1
            except Exception as exc:  # noqa: BLE001
                record_error("write:" + type(exc).__name__, exc)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(readers)]
    threads += [threading.Thread(target=writer, args=(i,)) for i in range(writers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    final = store.count()
    res = {
        "backend": backend, "n": n, "readers": readers, "writers": writers,
        "scenario": "graph_ops" if graph_ops else "store_ops",
        "duration_s": round(elapsed, 2),
        "reads": len(read_ms), "reads_per_s": len(read_ms) / elapsed,
        "writes_ok": writes_ok[0], "writes_per_s": writes_ok[0] / elapsed,
        "errors": dict(errors), "first_error": first_error,
        "rows_expected": n + writes_ok[0], "rows_found": final,
        "lost_writes": (n + writes_ok[0]) - final,
    }
    if read_ms:
        res["read_latency"] = summarize(read_ms)
    g.close()
    return res


# =============================================================================
# analysis
# =============================================================================

def fit_linear(points: Sequence[tuple]) -> Optional[dict]:
    """Least-squares ms = a + b*N over (N, ms) points; None if <2 points."""
    big = [p for p in points if p[0] >= 10_000]
    points = big if len(big) >= 2 else list(points)
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * xs
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum()) or 1.0
    return {"intercept_ms": float(a), "ms_per_1k_vectors": float(b * 1000), "r2": 1 - ss_res / ss_tot}


def crossing(points: Sequence[tuple], fit: Optional[dict], budget_ms: float) -> dict:
    pts = sorted(points)
    if pts and pts[0][1] >= budget_ms:
        return {"n": pts[0][0], "kind": "already above budget at smallest size"}
    for (n0, m0), (n1, m1) in zip(pts, pts[1:]):
        if m0 < budget_ms <= m1:
            n = n0 + (budget_ms - m0) * (n1 - n0) / (m1 - m0)
            return {"n": int(n), "kind": "interpolated between measured sizes"}
    if fit and fit["ms_per_1k_vectors"] > 0:
        n = (budget_ms - fit["intercept_ms"]) / (fit["ms_per_1k_vectors"] / 1000)
        return {"n": int(n), "kind": "extrapolated (linear fit)"}
    return {"n": None, "kind": "no crossing found"}


def fmt_n(n) -> str:
    return "n/a" if n is None else f"{n:,}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="1000,10000,100000,1000000",
                    help="comma list of vector counts; anything above --max-n is dropped")
    ap.add_argument("--max-n", type=int, default=100_000,
                    help="largest N to run (default 100000; pass 1000000 for the 1M run)")
    ap.add_argument("--runs", type=int, default=15, help="timed runs per latency figure (median reported)")
    ap.add_argument("--avg-degree", type=float, default=3.0, help="average out-degree of the random graph")
    ap.add_argument("--backends", default="numpy,sqlite_vec")
    ap.add_argument("--threshold-ms", default="50,100", help="latency budgets for the crossover analysis")
    ap.add_argument("--concurrency-n", type=int, default=20_000)
    ap.add_argument("--duration", type=float, default=4.0, help="seconds per concurrency scenario")
    ap.add_argument("--skip-concurrency", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1k and 10k only, short concurrency run")
    ap.add_argument("--out-prefix", default=str(Path(__file__).resolve().parent / "results" / "scale"),
                    help="writes <prefix>.md and <prefix>.json; pass '' to skip")
    args = ap.parse_args(argv)
    if args.quick:
        args.sizes, args.concurrency_n, args.duration, args.runs = "1000,10000", 5000, 2.0, 9
    sizes = [s for s in (int(x) for x in args.sizes.split(",")) if s <= args.max_n]
    backends = [b.strip() for b in args.backends.split(",")]
    budgets = [float(x) for x in args.threshold_ms.split(",")]

    env = environment_info()
    report: dict = {"config": vars(args), "environment": env, "results": [], "skipped": [], "concurrency": [], "floor": {}}
    print(f"sizes={sizes} backends={backends} runs={args.runs} memory={'psutil RSS' if psutil else 'tracemalloc'}",
          file=sys.stderr)

    # sqlite-vec availability
    avail = {"numpy": True, "sqlite_vec": False}
    if "sqlite_vec" in backends:
        try:
            ivec.EmbeddingStore(sqlite_vec=True, dimension=DIM).close()
            avail["sqlite_vec"] = True
        except Exception as exc:  # noqa: BLE001 - not installed / extension cannot load
            report["skipped"].append({"backend": "sqlite_vec", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"sqlite-vec path skipped: {exc}", file=sys.stderr)

    free_gb = (psutil.virtual_memory().available / 2**30) if psutil else None
    for backend in backends:
        if not avail.get(backend):
            continue
        for n in sizes:
            est_gb = n * DIM * 4 * 5 / 2**30
            if free_gb is not None and est_gb > 0.8 * free_gb:
                msg = f"{backend} N={n:,}: needs ~{est_gb:.1f} GB (5x raw vectors), {free_gb:.1f} GB free; skipped"
                report["skipped"].append({"backend": backend, "n": n, "reason": msg})
                print(msg, file=sys.stderr)
                continue
            print(f"[{backend}] N={n:,} ...", file=sys.stderr, flush=True)
            t0 = time.time()
            try:
                r = bench_config(backend, n, args)
            except MemoryError:
                report["skipped"].append({"backend": backend, "n": n, "reason": "MemoryError"})
                print("  MemoryError, skipped", file=sys.stderr)
                gc.collect()
                continue
            r["wall_s"] = time.time() - t0
            report["results"].append(r)
            print(f"  search {r['similarity_search']['median_ms']:.1f} ms, ingest {r['ingest_vectors_per_s']:.0f}/s, "
                  f"peak {r['peak_memory_mb']:.0f} MB ({r['wall_s']:.0f}s)", file=sys.stderr, flush=True)

    for n in sizes:
        if n * DIM * 4 * 2 / 2**30 < (0.4 * free_gb if free_gb else 4):
            report["floor"][str(n)] = raw_numpy_floor(n, args.runs)

    if not args.skip_concurrency:
        cn = args.concurrency_n
        scenarios = [(4, 0, False), (4, 1, False), (2, 4, False), (3, 1, True)]
        for backend in backends:
            if not avail.get(backend):
                continue
            for readers, writers, graph_ops in scenarios:
                print(f"[{backend}] concurrency {readers}R/{writers}W{' graph' if graph_ops else ''} ...",
                      file=sys.stderr, flush=True)
                report["concurrency"].append(
                    run_concurrent(backend, cn, readers, writers, args.duration, graph_ops))

    # ---- analysis ----------------------------------------------------------
    analysis: dict = {}
    for backend in backends:
        for metric in ("similarity_search", "semantic_multi_hop", "semantic_multi_hop_blend0.3", "semantic_subgraph",
                       "similarity_search_gc_off"):
            pts = [(r["n"], r[metric]["median_ms"]) for r in report["results"] if r["backend"] == backend]
            if not pts:
                continue
            fit = fit_linear(pts)
            analysis[f"{backend}/{metric}"] = {
                "points": pts, "fit": fit,
                "crossings": {str(int(b)): crossing(pts, fit, b) for b in budgets},
            }
    pts = [(int(n), v["median_ms"]) for n, v in report["floor"].items()]
    if pts:
        fit = fit_linear(pts)
        analysis["raw_numpy_floor"] = {"points": pts, "fit": fit,
                                        "crossings": {str(int(b)): crossing(pts, fit, b) for b in budgets}}
    report["analysis"] = analysis

    md = render_markdown(report, budgets)
    print(md)
    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".md").write_text(md, encoding="utf-8")
        prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {prefix.with_suffix('.md')} and {prefix.with_suffix('.json')}", file=sys.stderr)
    return 0


def render_markdown(report: dict, budgets: List[float]) -> str:
    env = report["environment"]
    md: List[str] = ["# Scale benchmark - sample output\n"]
    md.append(f"Python {env['python']} | numpy {env['numpy']} | sqlite-vec {env['sqlite_vec']} | "
              f"{env.get('cpu_count', '?')} logical CPUs | {env.get('ram_gb', '?')} GB RAM | {env['platform']}\n")
    md.append(f"{DIM}-d random normalised float32 vectors, in-memory SQLite, medians of {report['config']['runs']} runs "
              f"(graph queries: {max(3, report['config']['runs'] // 3)} runs), top_k=10. Memory = "
              f"{'process RSS growth (psutil)' if psutil else 'tracemalloc peak'}.\n")
    for s in report["skipped"]:
        md.append(f"> Skipped: {s.get('backend')} {s.get('n', '')} - {s['reason']}\n")

    res = report["results"]
    md.append("\n## Latency (ms, warm)\n")
    md.append("| backend | N | similarity_search | p95 | semantic_multi_hop | multi_hop blend=0.3 | semantic_subgraph | search, GC off | multi_hop, GC off |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in res:
        md.append(f"| {r['backend']} | {r['n']:,} | {r['similarity_search']['median_ms']:.2f} | "
                  f"{r['similarity_search']['p95_ms']:.2f} | {r['semantic_multi_hop']['median_ms']:.2f} | "
                  f"{r['semantic_multi_hop_blend0.3']['median_ms']:.2f} | {r['semantic_subgraph']['median_ms']:.2f} | "
                  f"{r['similarity_search_gc_off']['median_ms']:.2f} | {r['semantic_multi_hop_gc_off']['median_ms']:.2f} |")
    if report["floor"]:
        md.append("\nReference, not part of the library: bare `matrix @ q` + `argpartition` over the same vectors "
                  "(no SQLite, no Python loop): " +
                  ", ".join(f"N={int(n):,}: {v['median_ms']:.2f} ms" for n, v in sorted(report["floor"].items(), key=lambda kv: int(kv[0]))) + "\n")

    md.append("\n## Ingestion, snapshot build and write-after-read\n")
    md.append("| backend | N | add_batch vec/s | first query (snapshot build) ms | write 1 vector ms | search right after write ms | penalty vs warm ms | add_batch(100) ms | search after batch ms |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in res:
        md.append(f"| {r['backend']} | {r['n']:,} | {r['ingest_vectors_per_s']:,.0f} | {r['first_query_ms']:.1f} | "
                  f"{r['write_one']['median_ms']:.2f} | {r['search_after_write']['median_ms']:.1f} | "
                  f"{r['write_after_read_penalty_ms']:.1f} | {r['add_batch_100_ms']:.1f} | {r['search_after_batch_ms']:.1f} |")

    md.append("\n## Memory (MB growth over a pre-build baseline)\n")
    md.append("| backend | N | raw vectors | after graph build | after ingest | after first query | peak |")
    md.append("|---|---|---|---|---|---|---|")
    for r in res:
        md.append(f"| {r['backend']} | {r['n']:,} | {r['raw_vector_bytes_mb']:.0f} | {r['rss_after_graph_mb']:.0f} | "
                  f"{r['rss_after_ingest_mb']:.0f} | {r['rss_after_first_query_mb']:.0f} | {r['peak_memory_mb']:.0f} |")

    if report["concurrency"]:
        md.append(f"\n## Concurrency (N={report['concurrency'][0]['n']:,}, {report['concurrency'][0]['duration_s']} s per scenario)\n")
        md.append("`store_ops`: threads share one EmbeddingStore (documented thread-safe). `graph_ops`: readers run "
                  "semantic_multi_hop while a writer adds nodes and edges to the SemanticGraph (the graph itself is not documented as thread-safe).\n")
        md.append("| backend | scenario | readers/writers | reads/s | writes/s | read p50 ms | read p95 ms | errors | lost writes |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for c in report["concurrency"]:
            lat = c.get("read_latency", {})
            errs = ", ".join(f"{k}={v}" for k, v in c["errors"].items()) or "0"
            md.append(f"| {c['backend']} | {c['scenario']} | {c['readers']}/{c['writers']} | {c['reads_per_s']:.0f} | "
                      f"{c['writes_per_s']:.0f} | {lat.get('median_ms', float('nan')):.1f} | {lat.get('p95_ms', float('nan')):.1f} | "
                      f"{errs} | {c['lost_writes']} |")
        for c in report["concurrency"]:
            for k, v in c["first_error"].items():
                md.append(f"\n- first `{k}` ({c['backend']}, {c['scenario']}, {c['readers']}R/{c['writers']}W): `{v}`")

    md.append("\n## Where does the linear scan cross the latency budget?\n")
    md.append("Least-squares fit of median latency against N over the measured sizes of at least 10,000 vectors (all sizes if fewer than two).\n")
    md.append("| series | ms per 1k vectors | intercept ms | R^2 | " + " | ".join(f"N at {int(b)} ms" for b in budgets) + " |")
    md.append("|---|---|---|---|" + "---|" * len(budgets))
    for name, a in report["analysis"].items():
        f = a["fit"]
        if not f:
            continue
        cells = [f"{fmt_n(a['crossings'][str(int(b))]['n'])} ({a['crossings'][str(int(b))]['kind'].split(' ')[0]})" for b in budgets]
        md.append(f"| {name} | {f['ms_per_1k_vectors']:.3f} | {f['intercept_ms']:.2f} | {f['r2']:.3f} | " + " | ".join(cells) + " |")

    md.append("\n" + recommendation(report, budgets))
    return "\n".join(md) + "\n"


def recommendation(report: dict, budgets: List[float]) -> str:
    a = report["analysis"].get("numpy/similarity_search")
    b50 = str(int(budgets[0]))
    lines = ["## Recommendation (auto-generated from the numbers above)\n"]
    if a and a["crossings"].get(b50, {}).get("n"):
        n50 = a["crossings"][b50]["n"]
        kind = a["crossings"][b50]["kind"]
        lines.append(f"* The numpy exact scan reaches {b50} ms at about **{n50:,} vectors** ({kind}). Below roughly "
                     f"{max(1000, n50 // 5):,} vectors (one fifth of that) an ANN index buys nothing you would notice.")
        floor = report["analysis"].get("raw_numpy_floor")
        if floor and floor["crossings"].get(b50, {}).get("n"):
            lines.append(f"* A bare matrix product over the same vectors reaches {b50} ms at about "
                         f"{floor['crossings'][b50]['n']:,}: that is the ceiling for any exact scan here, so the gap between the two "
                         "lines is the library's own per-query overhead (the vectorised exact path sits close to this floor).")
        lines.append(f"* Past ~{n50:,} vectors, or if you need many concurrent queries per second, an ANN index (or a "
                     "different vector store) becomes worth evaluating; exact search stays correct but scales linearly.")
    else:
        lines.append("* The linear scan did not reach the budget in the measured range.")
    v = report["analysis"].get("sqlite_vec/similarity_search")
    if v and a and v["fit"] and a["fit"]:
        lines.append(f"* sqlite-vec (exact, in C) fits {v['fit']['ms_per_1k_vectors']:.3f} ms per 1k vectors vs "
                     f"{a['fit']['ms_per_1k_vectors']:.3f} for numpy: it is still a linear scan, so it moves the crossover but does not remove it.")
    g = report["analysis"].get("numpy/similarity_search_gc_off")
    if g and g["crossings"].get(b50, {}).get("n"):
        lines.append(f"* With the cyclic GC disabled during the call the same numpy scan reaches {b50} ms at about "
                     f"{g['crossings'][b50]['n']:,}: in a process holding a large graph, part of the tail latency is Python "
                     "garbage collection triggered by the per-query allocations, not search work (see the GC-off columns).")
    pen = [r for r in report["results"] if r["backend"] == "numpy"]
    if pen:
        big = max(pen, key=lambda r: r["n"])
        lines.append(f"* On the numpy path a write patches the cached matrix in place: at N={big['n']:,} the search right after a write costs "
                     f"~{big['search_after_write']['median_ms']:.0f} ms against {big['similarity_search']['median_ms']:.0f} ms warm. "
                     "A large gap here would mean the cache is being rebuilt instead of patched.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
