"""Shared data layout, deterministic plans and float64 ground truth for the competitive benchmark.

Layout (all raw little-endian, row-major):
  <prep>/<ds>/base.f32        N_total x d   float32, L2-normalised
  <prep>/<ds>/queries.f32     Q x d         float32, L2-normalised
  <prep>/<ds>/meta.json
  <prep>/<ds>/gt_<n>.i32      Q x 100       exact float64 cosine top-100 ids over base[0:n]
  <prep>/<ds>/grp_<n>.u8      n             0 = 10% subset, 1 = 1% subset (disjoint), 2 = the rest
  <prep>/<ds>/gt_f10_<n>.i32, gt_f01_<n>.i32   Q x 100 top-100 ids inside the 10% / 1% subset
  <prep>/<ds>/mut_<n>_upd.i32, mut_<n>_del.i32   10k ids updated / 10k ids deleted (disjoint)
  <prep>/<ds>/gt_mut_<n>.i32  Q x 100  top-100 ids after: insert ids n..n+10k-1 (rows n..n+10k-1),
                              update <upd> (row n+10k+j supplies the new vector of id upd[j]),
                              delete <del>.
"""
import json
import os

import numpy as np

PREP = os.environ.get("BENCH_PREP", r"D:\Goldmine\ISON\bench_data\prep")
K_GT = 100
MUT = 10_000
SEED = 20261115


def ds_dir(ds):
    return os.path.join(PREP, ds)


def meta(ds):
    return json.load(open(os.path.join(ds_dir(ds), "meta.json")))


def read_f32(path, start, count, d):
    """rows [start, start+count) of a raw float32 matrix with d columns."""
    a = np.fromfile(path, dtype=np.float32, count=count * d, offset=start * d * 4)
    return a.reshape(-1, d)


def base_rows(ds, start, count):
    m = meta(ds)
    return read_f32(os.path.join(ds_dir(ds), "base.f32"), start, count, m["d"])


def queries(ds, q=None):
    m = meta(ds)
    a = read_f32(os.path.join(ds_dir(ds), "queries.f32"), 0, m["q"], m["d"])
    return a if q is None else a[:q]


def gt(ds, n, kind="", q=None):
    """kind in '', 'f10_', 'f01_', 'mut_' -> (Q, 100) int32"""
    a = np.fromfile(os.path.join(ds_dir(ds), f"gt_{kind}{n}.i32"), dtype=np.int32).reshape(-1, K_GT)
    return a if q is None else a[:q]


def grp(ds, n):
    return np.fromfile(os.path.join(ds_dir(ds), f"grp_{n}.u8"), dtype=np.uint8)


def mut_plan(ds, n):
    d = ds_dir(ds)
    return (np.fromfile(os.path.join(d, f"mut_{n}_upd.i32"), dtype=np.int32),
            np.fromfile(os.path.join(d, f"mut_{n}_del.i32"), dtype=np.int32))


def make_grp(n):
    rng = np.random.default_rng([SEED, 11, n])
    perm = rng.permutation(n)
    g = np.full(n, 2, dtype=np.uint8)
    g[perm[: n // 10]] = 0
    g[perm[n // 10: n // 10 + n // 100]] = 1
    return g


def make_mut(n):
    rng = np.random.default_rng([SEED, 12, n])
    ids = rng.choice(n, size=2 * MUT, replace=False).astype(np.int32)
    return ids[:MUT], ids[MUT:]


def exact_topk(chunks, queries32, k=K_GT):
    """Exact float64 cosine top-k. `chunks` yields (first_id_or_id_array, matrix32). Returns ids (Q,k)."""
    q = queries32.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    nq = q.shape[0]
    best_s = np.full((nq, k), -np.inf)
    best_i = np.zeros((nq, k), dtype=np.int64)
    def pieces():
        for ids, mat in chunks:
            if len(mat) <= 50000:
                yield ids, mat
            else:
                for s0 in range(0, len(mat), 50000):
                    sub = ids[s0:s0 + 50000] if not isinstance(ids, (int, np.integer)) else ids + s0
                    yield sub, mat[s0:s0 + 50000]

    for ids, mat in pieces():
        m = mat.astype(np.float64)
        nrm = np.linalg.norm(m, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        m /= nrm
        s = (m @ q.T).T                      # (nq, c)
        c = s.shape[1]
        if isinstance(ids, (int, np.integer)):
            ids_arr = np.arange(ids, ids + c, dtype=np.int64)
        else:
            ids_arr = np.asarray(ids, dtype=np.int64)
        kk = min(k, c)
        part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(s, part, axis=1)
        pi = ids_arr[part]
        cs = np.concatenate([best_s, ps], axis=1)
        ci = np.concatenate([best_i, pi], axis=1)
        sel = np.argpartition(-cs, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cs, sel, axis=1)
        best_i = np.take_along_axis(ci, sel, axis=1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_i, order, axis=1).astype(np.int32)


def recall_at(found, truth, k=10):
    """mean recall@k; found (Q,>=k) ids (-1 padded ok), truth (Q,>=k)."""
    hit = 0
    for f, t in zip(found, truth):
        hit += len(set(int(x) for x in f[:k] if x >= 0) & set(int(x) for x in t[:k]))
    return hit / (k * len(truth))
