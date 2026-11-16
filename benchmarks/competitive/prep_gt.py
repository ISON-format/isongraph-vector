"""Write ground truth, filter groups and mutation plans for a dataset (see common.py).

    python prep_gt.py glove 10000,100000,1000000
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ds = sys.argv[1]
ns = [int(x) for x in sys.argv[2].split(",")]
m = C.meta(ds)
d = m["d"]
Q = C.queries(ds)
out = C.ds_dir(ds)
base_path = os.path.join(out, "base.f32")


def chunks_of(n, step=50000):
    for s in range(0, n, step):
        yield s, C.read_f32(base_path, s, min(step, n - s), d)


for n in ns:
    t0 = time.time()
    if n > m["n"]:
        print("skip", n)
        continue
    C.exact_topk(chunks_of(n), Q).tofile(os.path.join(out, f"gt_{n}.i32"))
    print(ds, n, "gt", round(time.time() - t0, 1), "s", flush=True)
    if n < 100000:
        continue
    g = C.make_grp(n)
    g.tofile(os.path.join(out, f"grp_{n}.u8"))
    X = C.read_f32(base_path, 0, n, d)
    for kind, code in (("f10_", 0), ("f01_", 1)):
        ids = np.nonzero(g == code)[0]
        C.exact_topk([(ids, X[ids])], Q).tofile(os.path.join(out, f"gt_{kind}{n}.i32"))
    del X
    if m["n"] >= n + 2 * C.MUT:
        upd, dele = C.make_mut(n)
        upd.tofile(os.path.join(out, f"mut_{n}_upd.i32"))
        dele.tofile(os.path.join(out, f"mut_{n}_del.i32"))
        Y = C.read_f32(base_path, 0, n + 2 * C.MUT, d)  # rows n..n+10k-1 inserts, n+10k.. update vectors
        row_of = np.arange(n + C.MUT, dtype=np.int64)
        row_of[upd] = n + C.MUT + np.arange(C.MUT)
        alive = np.ones(n + C.MUT, dtype=bool)
        alive[dele] = False
        ids = np.nonzero(alive)[0]
        C.exact_topk([(ids, Y[row_of[ids]])], Q).tofile(os.path.join(out, f"gt_mut_{n}.i32"))
        del Y
    print(ds, n, "filters+mutation", round(time.time() - t0, 1), "s", flush=True)
