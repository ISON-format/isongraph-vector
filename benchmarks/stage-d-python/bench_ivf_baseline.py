#!/usr/bin/env python3
"""
Python IVF at 100k x 384 real text vectors: recall@10 and single-query QPS against nprobe.

The baseline the Stage D part 3 HNSW prototype had to beat. Ground truth is the float64
cosine top-100 of benchmarks/competitive (gt_100000.i32), queries are the real HotpotQA
question embeddings.

    python benchmarks/stage-d-python/bench_ivf_baseline.py --runs 3
"""

import argparse
import importlib.util
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PREP = os.environ.get("BENCH_PREP", r"D:\Goldmine\ISON\bench_data\prep\text384")


def load_impl():
    path = os.path.join(REPO, "isongraph_vector-py", "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "isongraph_vector_bench", path, submodule_search_locations=[os.path.dirname(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["isongraph_vector_bench"] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--nprobe", type=int, nargs="*", default=[32, 63, 100, 160, 250])
    args = ap.parse_args()
    d = 384
    base = np.fromfile(os.path.join(PREP, "base.f32"), dtype=np.float32, count=args.n * d).reshape(-1, d)
    q = np.fromfile(os.path.join(PREP, "queries.f32"), dtype=np.float32, count=1000 * d).reshape(-1, d)
    gt = np.fromfile(os.path.join(PREP, f"gt_{args.n}.i32"), dtype=np.int32).reshape(-1, 100)[:1000, :10]
    mod = load_impl()
    store = mod.EmbeddingStore(":memory:", dimension=d, ann="ivf")
    for s in range(0, args.n, 5000):
        store.add_batch([(("doc", s + i), "", v) for i, v in enumerate(base[s:s + 5000].tolist())])
    t = time.perf_counter()
    store.build_ann_index()
    print(f"IVF build {time.perf_counter() - t:.2f} s  stats nlist={store.ann_stats()['nlist']}")
    ql = q.tolist()
    for nprobe in args.nprobe:
        store.ann_nprobe = nprobe
        rec = 0
        for i in range(1000):
            r = store.similarity_search(ql[i], 10)
            rec += len({x.node_ref[1] for x in r} & set(gt[i].tolist()))
        qps = []
        for _ in range(args.runs):
            t = time.perf_counter()
            for w in ql[:args.queries]:
                store.similarity_search(w, 10)
            qps.append(args.queries / (time.perf_counter() - t))
        print(f"nprobe {nprobe:4d}  recall@10 {rec / 10000:.4f}  single QPS median {statistics.median(qps):7.1f} "
              f"({min(qps):.1f}-{max(qps):.1f})", flush=True)


if __name__ == "__main__":
    main()
