#!/usr/bin/env python3
"""
What does ``canonical_scores`` cost, and what does it buy? (Stage D part 3, Python)

Data: the first N rows of the real text384 matrix (MiniLM, L2-normalised) and its
queries, see benchmarks/competitive/common.py. ``--synthetic`` uses random unit
vectors instead.

    python benchmarks/stage-d-python/bench_canonical.py cost --runs 3     # timing (take the bench lock)
    python benchmarks/stage-d-python/bench_canonical.py diff               # no timing: how often results differ

``cost``  exact single-query QPS, batch QPS, IVF single and batch, canonical on vs off,
          plus a reference that no store offers: a raw BLAS matrix-matrix batch with no
          canonical rescoring (what a batch costs when it is allowed to differ from the
          single-query answer).
``diff``  with canonical_scores OFF, how often do the answers of one query differ when asked
          (a) one at a time vs as a matrix-matrix batch, (b) after the rows moved to other slots
          (rotation / permutation of insertion order), and (c) from canonical ON; reported as
          ids, order and score bits over the top-10 of every query.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PREP = os.environ.get("BENCH_PREP", r"D:\Goldmine\ISON\bench_data\prep")


def load_impl():
    path = os.path.join(REPO, "isongraph_vector-py", "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "isongraph_vector_bench", path, submodule_search_locations=[os.path.dirname(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["isongraph_vector_bench"] = module
    spec.loader.exec_module(module)
    return module


def data(n, synthetic, nq):
    d = 384
    if synthetic:
        rng = np.random.default_rng(7)
        base = rng.standard_normal((n, d)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        q = rng.standard_normal((nq, d)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return base, q
    base = np.fromfile(os.path.join(PREP, "text384", "base.f32"), dtype=np.float32,
                       count=n * d).reshape(n, d)
    q = np.fromfile(os.path.join(PREP, "text384", "queries.f32"), dtype=np.float32,
                    count=nq * d).reshape(nq, d)
    return base, q


def fill(mod, base, ids=None, ann=None):
    store = mod.EmbeddingStore(":memory:", dimension=base.shape[1], ann=ann)
    ids = range(base.shape[0]) if ids is None else ids
    for s in range(0, base.shape[0], 5000):
        store.add_batch([(("doc", int(i)), "", v) for i, v in
                         zip(list(ids)[s:s + 5000], base[s:s + 5000].tolist())])
    return store


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def summarize(values):
    return f"{statistics.median(values):9.1f} ({min(values):.1f}-{max(values):.1f})"


def cost(args):
    mod = load_impl()
    base, q = data(args.n, args.synthetic, 600)
    ql = q.tolist()
    rows = {}
    for label, ann in (("exact", None), ("ivf", "ivf")):
        store = fill(mod, base, ann=ann)
        if ann:
            store.build_ann_index()
        res = {True: ([], []), False: ([], [])}
        for canon in (True, False):
            store.canonical_scores = canon
            for w in ql[:20]:
                store.similarity_search(w, 10)
        for _ in range(args.runs):            # on/off alternate inside every run, so drift hits both
            for canon in (True, False):
                store.canonical_scores = canon
                t = time.perf_counter()
                for w in ql[:args.queries]:
                    store.similarity_search(w, 10)
                res[canon][0].append(args.queries / (time.perf_counter() - t))
                nb = args.batch_queries
                t = time.perf_counter()
                store.similarity_search_batch(ql[:nb], 10)
                res[canon][1].append(nb / (time.perf_counter() - t))
        for canon in (True, False):
            rows[(label, canon)] = res[canon]
            print(f"{label:6s} canonical={'on ' if canon else 'off'}  single QPS {summarize(res[canon][0])}"
                  f"   batch({args.batch_queries}) QPS {summarize(res[canon][1])}", flush=True)

    # reference: a raw matrix-matrix batch, no canonical rescoring (not an API of the store)
    store = fill(mod, base)
    cache = store._ensure_cache()
    n = cache.n
    mt = cache.matrix[:n].T
    norms = cache.norms[:n]

    def raw_batch(qs):
        for s in range(0, len(qs), 64):
            block = np.asarray(qs[s:s + 64], dtype=np.float32)
            dots = block @ mt
            for row, vec in zip(dots, block):
                qn = float(np.linalg.norm(vec))
                sc = np.zeros(n, dtype=np.float32)
                np.divide(row, norms * qn, out=sc, where=norms * qn > 0)
                store._rank(sc, 10)
    raw = []
    for _ in range(args.runs):
        t = time.perf_counter()
        raw_batch(ql[:args.batch_queries])
        raw.append(args.batch_queries / (time.perf_counter() - t))
    print(f"exact  raw BLAS gemm batch, no rescoring (reference)        batch QPS {summarize(raw)}")
    print(json.dumps({f"{a}_{'on' if c else 'off'}": v for (a, c), v in rows.items()}))


def top(results):
    return [(r.node_ref[1], r.score) for r in results]


def bits(x):
    return np.float32(x).view(np.uint32)


def compare(a, b):
    """-> (ids as a set, ids in order, scores bit-equal per id, whole list identical)"""
    ia, ib = [i for i, _ in a], [i for i, _ in b]
    same_set = set(ia) == set(ib)
    same_order = ia == ib
    sa = {i: bits(s) for i, s in a}
    sb = {i: bits(s) for i, s in b}
    common = set(ia) & set(ib)
    score_diff = sum(1 for i in common if sa[i] != sb[i])
    return same_set, same_order, score_diff, len(common), same_order and score_diff == 0


def tally(name, pairs):
    n = len(pairs)
    set_ne = sum(1 for p in pairs if not p[0])
    ord_ne = sum(1 for p in pairs if not p[1])
    lists_bits_ne = sum(1 for p in pairs if p[2] > 0)
    bits_ne = sum(p[2] for p in pairs)
    total = sum(p[3] for p in pairs)
    any_ne = sum(1 for p in pairs if not p[4])
    print(f"{name:48s} queries={n}  id-set differs {set_ne}  order differs {ord_ne}  "
          f"queries with a score-bit difference {lists_bits_ne} ({100.0 * lists_bits_ne / n:.1f}%)  "
          f"score bits differ {bits_ne}/{total} ({100.0 * bits_ne / max(1, total):.2f}%)  "
          f"any difference {any_ne} ({100.0 * any_ne / n:.1f}%)")


def diff(args):
    mod = load_impl()
    base, q = data(args.n, args.synthetic, args.queries)
    ql = q.tolist()
    n = base.shape[0]
    store = fill(mod, base)
    cache = store._ensure_cache()

    # (a) single (gemv) vs raw gemm batch, canonical off
    store.canonical_scores = False
    single = [top(store.similarity_search(w, 10)) for w in ql]
    mt = cache.matrix[:n].T
    norms = cache.norms[:n]
    batched = []
    for s in range(0, len(ql), 64):
        block = np.asarray(ql[s:s + 64], dtype=np.float32)
        dots = block @ mt
        for row, vec in zip(dots, block):
            qn = float(np.linalg.norm(vec))
            sc = np.zeros(n, dtype=np.float32)
            np.divide(row, norms * qn, out=sc, where=norms * qn > 0)
            best = store._rank(sc, 10)
            batched.append([(cache.refs[i][1], float(sc[i])) for i in best.tolist()])
    tally("(a) OFF: one at a time vs matrix-matrix batch", [compare(x, y) for x, y in zip(single, batched)])
    # the store's own batch with canonical off answers query by query (== single by construction)
    own = [top(r) for r in store.similarity_search_batch(ql, 10)]
    tally("    OFF: similarity_search_batch vs single (own API)", [compare(x, y) for x, y in zip(single, own)])

    # (b) slot dependence, canonical off: rotate the insertion order by `shift`
    for shift in (1, 3, 5, 7, 13, 1000, 12345):
        order = np.roll(np.arange(n), shift)
        other = fill(mod, base[order], ids=order.tolist())
        other.canonical_scores = False
        moved = [top(other.similarity_search(w, 10)) for w in ql]
        tally(f"(b) OFF: rows rotated by {shift} slots vs original", [compare(x, y) for x, y in zip(single, moved)])
        other.close()
    rng = np.random.default_rng(5)
    order = rng.permutation(n)
    other = fill(mod, base[order], ids=order.tolist())
    other.canonical_scores = False
    moved = [top(other.similarity_search(w, 10)) for w in ql]
    tally("(b) OFF: random permutation of slots vs original", [compare(x, y) for x, y in zip(single, moved)])
    # same permuted store, canonical ON, vs original ON  (the guarantee)
    other.canonical_scores = True
    store.canonical_scores = True
    on_orig = [top(store.similarity_search(w, 10)) for w in ql]
    on_moved = [top(other.similarity_search(w, 10)) for w in ql]
    tally("    ON : random permutation vs original", [compare(x, y) for x, y in zip(on_orig, on_moved)])
    on_batch = [top(r) for r in store.similarity_search_batch(ql, 10)]
    tally("    ON : similarity_search_batch vs single", [compare(x, y) for x, y in zip(on_orig, on_batch)])

    # (d) exact duplicates at different slots: the same vector stored twice should tie, and a
    #     tie is ordered by insertion; with raw BLAS the two copies can score differently
    dup = 20000
    both = np.concatenate([base, base[:dup]])
    ids = list(range(n)) + [n + i for i in range(dup)]
    twin = fill(mod, both, ids=ids)
    for canon in (False, True):
        twin.canonical_scores = canon
        pairs = split = inverted = 0
        for w in ql:
            got = [(r.node_ref[1], r.score) for r in twin.similarity_search(w, 10)]
            where = {i: (k, bits(s)) for k, (i, s) in enumerate(got)}
            for i, (k, b) in where.items():
                if i >= n and (i - n) in where:
                    k0, b0 = where[i - n]
                    pairs += 1
                    split += b != b0
                    inverted += k < k0
        print(f"(d) {'ON ' if canon else 'OFF'}: duplicate pairs both in a top-10: {pairs}; "
              f"copies with different score bits: {split}; copy ranked above original: {inverted}")
    twin.close()

    # (c) canonical ON vs OFF on the same store
    tally("(c) ON vs OFF, same store, one at a time", [compare(x, y) for x, y in zip(on_orig, single)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cost", "diff"])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--batch-queries", type=int, default=256)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.mode == "diff" and args.queries == 200:
        args.queries = 2000
    (cost if args.mode == "cost" else diff)(args)


if __name__ == "__main__":
    main()
