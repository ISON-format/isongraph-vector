#!/usr/bin/env python3
"""
Stage D benchmark for the Python port: memory, single-query and batch search,
snapshot against JSON.

Each measurement runs in a fresh child process (RSS is only meaningful there)
and is repeated ``--runs`` times; the parent prints medians and ranges.

    python benchmarks/stage-d-python/bench_stage_d.py --impl new --runs 3
    python benchmarks/stage-d-python/bench_stage_d.py --impl path/to/old/__init__.py --runs 3

``--impl new`` (default) is this checkout's package; any other value is the path
of an ``__init__.py`` to load instead (used to compare against the code as it was
before Stage D). Data are random unit vectors (fixed seed), 5,000 per add_batch
call as the competitive benchmark does. Timings are indicative only when the
machine is shared; RSS figures are not affected by that.
"""

import argparse
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_impl(spec: str):
    path = os.path.join(REPO, "isongraph_vector-py", "__init__.py") if spec == "new" else spec
    name = "isongraph_vector_bench_" + str(abs(hash(path)))
    module_spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[os.path.dirname(path)])
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


def rss_mb(proc) -> float:
    return proc.memory_info().rss / 2**20


def child(args) -> dict:
    import numpy as np
    import psutil

    mod = load_impl(args.impl)
    proc = psutil.Process()
    gc.collect()
    base = rss_mb(proc)
    out = {"impl": args.impl, "n": args.n, "dim": args.dim, "baseline_mb": round(base, 1),
           "raw_mb": round(args.n * args.dim * 4 / 2**20, 1)}

    kwargs = {}
    if args.ann:
        kwargs["ann"] = "ivf"
    db_dir = tempfile.TemporaryDirectory() if args.file_db else None
    db_path = os.path.join(db_dir.name, "emb.db") if db_dir else ":memory:"
    store = mod.EmbeddingStore(db_path, dimension=args.dim, **kwargs)
    rng = np.random.default_rng(1234)
    centers = None
    if args.clustered:      # 256 cluster centres + noise: the shape of real embeddings, for IVF recall
        centers = rng.standard_normal((256, args.dim)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def draw(count, gen):
        if centers is None:
            return gen.standard_normal((count, args.dim)).astype(np.float32)
        which = gen.integers(0, 256, count)
        return centers[which] + 0.5 * gen.standard_normal((count, args.dim)).astype(np.float32) / np.sqrt(args.dim)

    started = time.perf_counter()
    for start in range(0, args.n, 5000):
        count = min(5000, args.n - start)
        block = draw(count, rng)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        store.add_batch([(("doc", start + i), "", v) for i, v in enumerate(block.tolist())])
        del block
    out["ingest_s"] = round(time.perf_counter() - started, 2)
    gc.collect()
    out["rss_after_ingest_mb"] = round(rss_mb(proc) - base, 1)

    qrng = np.random.default_rng(99)
    queries = draw(512, qrng)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    qlist = queries.tolist()

    started = time.perf_counter()
    store.similarity_search(qlist[0], 10)
    out["first_query_s"] = round(time.perf_counter() - started, 3)
    gc.collect()
    out["rss_steady_mb"] = round(rss_mb(proc) - base, 1)
    out["rss_ratio_to_raw"] = round(out["rss_steady_mb"] / out["raw_mb"], 2)

    if args.ann:
        started = time.perf_counter()
        store.build_ann_index()
        out["ivf_build_s"] = round(time.perf_counter() - started, 2)
        gc.collect()
        out["rss_with_ivf_mb"] = round(rss_mb(proc) - base, 1)

    for q in qlist[:20]:
        store.similarity_search(q, 10)
    timed = qlist[:args.queries]
    started = time.perf_counter()
    for q in timed:
        store.similarity_search(q, 10)
    elapsed = time.perf_counter() - started
    out["single_qps"] = round(len(timed) / elapsed, 1)
    out["single_ms"] = round(1000 * elapsed / len(timed), 3)
    if args.ann:
        report = store.recall_check(qlist[:50], k=10)
        out["ivf_recall_at_10"] = round(report.recall, 4)

    if hasattr(store, "similarity_search_batch") and not args.ann:
        for batch in (16, 64, 256):
            best = None
            for _ in range(3):
                started = time.perf_counter()
                store.similarity_search_batch(qlist[:batch], 10)
                took = time.perf_counter() - started
                best = took if best is None else min(best, took)
            out[f"batch{batch}_qps"] = round(batch / best, 1)
        # the same queries one at a time, for a like-for-like ratio
        started = time.perf_counter()
        for q in qlist[:256]:
            store.similarity_search(q, 10)
        out["single_256_qps"] = round(256 / (time.perf_counter() - started), 1)

    if args.files and not args.ann:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "e.embeddings.json")
            started = time.perf_counter()
            store.save_embeddings(json_path)
            out["json_save_s"] = round(time.perf_counter() - started, 2)
            out["json_mb"] = round(os.path.getsize(json_path) / 2**20, 1)
            gc.collect()
            other = mod.EmbeddingStore(":memory:", dimension=args.dim)
            started = time.perf_counter()
            other.load_embeddings(json_path)
            out["json_load_s"] = round(time.perf_counter() - started, 2)
            assert other.count() == args.n
            other.close()
            del other
            gc.collect()
            if hasattr(store, "save_snapshot"):
                snap_path = os.path.join(tmp, "e.embeddings.snap")
                started = time.perf_counter()
                store.save_snapshot(snap_path)
                out["snap_save_s"] = round(time.perf_counter() - started, 2)
                out["snap_mb"] = round(os.path.getsize(snap_path) / 2**20, 1)
                other = mod.EmbeddingStore(":memory:")
                started = time.perf_counter()
                other.load_snapshot(snap_path)
                out["snap_load_s"] = round(time.perf_counter() - started, 2)
                assert other.count() == args.n
                other.close()
    store.close()
    if db_dir:
        db_dir.cleanup()
    return out


def parent(args) -> None:
    runs = []
    for i in range(args.runs):
        cmd = [sys.executable, os.path.abspath(__file__), "--child", "--impl", args.impl,
               "--n", str(args.n), "--dim", str(args.dim), "--queries", str(args.queries)]
        if args.ann:
            cmd.append("--ann")
        if args.files:
            cmd.append("--files")
        if args.clustered:
            cmd.append("--clustered")
        if args.file_db:
            cmd.append("--file-db")
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            sys.stderr.write(done.stderr)
            raise SystemExit(done.returncode)
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
        print(f"run {i + 1}/{args.runs}: {runs[-1]}", flush=True)
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and k not in ("n", "dim")]
    summary = {}
    for k in keys:
        values = [r[k] for r in runs if k in r]
        summary[k] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    print(f"== impl={args.impl} ann={args.ann} clustered={args.clustered} file_db={args.file_db} n={args.n} dim={args.dim} runs={args.runs}")
    for k, v in summary.items():
        print(f"{k:24s} median {v['median']:>10}   range [{v['min']} .. {v['max']}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="new")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--ann", action="store_true", help="measure the IVF index instead of the exact scan")
    ap.add_argument("--files", action="store_true", help="also time JSON and snapshot save/load")
    ap.add_argument("--file-db", action="store_true",
                    help="back the store with a SQLite file instead of :memory:")
    ap.add_argument("--clustered", action="store_true",
                    help="clustered data instead of uniform random (uniform is the worst case for IVF recall)")
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args()
    if args.child:
        print(json.dumps(child(args)))
    else:
        parent(args)


if __name__ == "__main__":
    main()
