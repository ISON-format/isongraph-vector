"""Convert an ann-benchmarks HDF5 file into the benchmark's raw layout (see common.py).

The file's own neighbour lists are NOT used: ground truth is recomputed by prep_gt.py in float64 cosine
on the normalised vectors that every system is fed.

    python prep_ann.py glove-100-angular D:/Goldmine/ISON/bench_data/raw/glove-100-angular.hdf5
"""
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

name, path = sys.argv[1], sys.argv[2]
nq = 2000
short = name.split("-")[0]
out = common.ds_dir(short)
os.makedirs(out, exist_ok=True)
f = h5py.File(path, "r")
train, test = f["train"], f["test"]
d = train.shape[1]


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


with open(os.path.join(out, "base.f32"), "wb") as o:
    for s in range(0, train.shape[0], 100000):
        norm(train[s:s + 100000]).tofile(o)
norm(test[:nq]).tofile(os.path.join(out, "queries.f32"))
json.dump({"name": short, "source": f"ann-benchmarks {name}.hdf5", "n": int(train.shape[0]), "d": int(d), "q": nq,
           "metric": "cosine on L2-normalised vectors (original metric: " + name.split("-")[-1] + ")"},
          open(os.path.join(out, "meta.json"), "w"), indent=1)
print(name, train.shape, "->", out)
