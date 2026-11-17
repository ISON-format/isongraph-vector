"""Thin wrapper around benchmarks/competitive/bench_py.py for Stage C.

Same adapters, same pinning (logical CPU 8), same timed loops, same ground truth; the only change is a shorter
accuracy-knob list for the ANN libraries (all the points that matter around recall 0.95), so a cell takes
minutes, not tens of minutes. Nothing under benchmarks/competitive/ is modified.

    python bench_c.py --system hnswlib --ds text384 --n 100000 --exp core,mutation,saveload --out raw/x.json
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(os.path.dirname(HERE), "competitive")
sys.path.insert(0, COMP)
import adapters  # noqa: E402

HN = [24, 32, 48, 64, 96, 128, 192, 256, 400]
IVF = [16, 32, 64, 128, 256, 512]
for name, kn in (("faiss_hnsw", HN), ("hnswlib", HN), ("usearch", HN), ("faiss_ivfflat", IVF)):
    adapters.ADAPTERS[name].knobs = (lambda self, n, kn=kn: list(kn))

import bench_py  # noqa: E402

if __name__ == "__main__":
    bench_py.main()
