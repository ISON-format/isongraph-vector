#!/usr/bin/env python3
"""Entry point for the multi-hop retrieval benchmark.

    python run.py prepare            # download pinned dev sets, sample questions, build pooled corpora
    python run.py embed              # embed passages + questions (MiniLM, bge-small), cache .npy
    python run.py edges              # build and cache graph edge lists (mention / entity / kNN)
    python run.py baselines          # vector, BM25, RRF hybrid, cross-encoder rerank   (GPU for rerank)
    python run.py graphs [--workers N] [--datasets ..] [--models ..] [--variants ..]
                                     # tune on dev, evaluate on test: ISONGraph multi_hop/subgraph + PPR
    python run.py latency            # single-thread per-query latency (run on an otherwise idle machine)
    python run.py report             # bootstrap CIs, paired tests, tables -> results/*.json, results/tables.md
    python run.py all                # everything above in order

Run with the benchmark venv:  D:\\Goldmine\\ISON\\bench_env_mh\\Scripts\\python.exe run.py <command>
(a venv created with --system-site-packages: torch, sentence-transformers, datasets, scipy, numpy,
ison-graph; the library under test is imported from this checkout, unmodified).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import DATASETS, MODELS  # noqa: E402


def _graph_task(args):
    import jobs
    return jobs.graph_job(*args)


def _baseline_task(args):
    import jobs
    return jobs.baselines_job(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "embed", "edges", "baselines", "graphs", "latency", "report", "all"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--jobs", nargs="*", default=None, help="explicit ds:model:variant triples")
    a = ap.parse_args()
    cmds = ["prepare", "embed", "edges", "baselines", "graphs", "latency", "report"] if a.cmd == "all" else [a.cmd]
    for cmd in cmds:
        t0 = time.time()
        if cmd == "prepare":
            import prepare_data
            for ds in a.datasets:
                prepare_data.build(ds)
        elif cmd == "embed":
            import embed
            for ds in a.datasets:
                for m in a.models:
                    embed.embed(ds, m)
        elif cmd == "edges":
            import jobs
            from jobs import load_ds, get_edges
            for ds in a.datasets:
                for m in a.models:
                    corpus, qs, P, Q, dev, test = load_ds(ds, m)
                    e = get_edges(ds, m, corpus, P)
                    print(ds, m, {k: len(v[0]) for k, v in e.items()})
        elif cmd == "baselines":
            tasks = [(ds, m) for ds in a.datasets for m in a.models]
            # reranking uses the GPU: run sequentially to avoid GPU contention
            import jobs
            for t in tasks:
                print(jobs.baselines_job(*t), flush=True)
        elif cmd == "graphs":
            import graphs
            variants = a.variants or list(graphs.VARIANTS)
            tasks = [(ds, m, v) for ds in a.datasets for m in a.models for v in variants]
            if a.jobs:
                tasks = [tuple(x.split(":")) for x in a.jobs]
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(_graph_task, t): t for t in tasks}
                for f in as_completed(futs):
                    try:
                        print(f.result(), flush=True)
                    except Exception as e:  # keep going; report at the end
                        print("FAILED", futs[f], repr(e), flush=True)
        elif cmd == "latency":
            import latency
            latency.run()
        elif cmd == "report":
            import report
            report.run()
        print(f"[{cmd}] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
