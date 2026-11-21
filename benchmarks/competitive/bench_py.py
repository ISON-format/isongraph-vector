"""Run one (system, dataset, n) cell of the competitive benchmark in a fresh process and write JSON.

    python bench_py.py --system faiss_hnsw --ds text384 --n 100000 --exp core,mutation,saveload --out result.json
    python bench_py.py --system faiss_hnsw --ds text384 --n 100000 --exp filter --out result_filter.json

Experiments (each is its own block in the JSON):
  core      build time, memory, exact/ANN accuracy sweep (1 thread query loop and all-cores batch), x3 runs
  mutation  insert 10k / update 10k / delete 10k into the built index, then recall again
  saveload  native save + load, on-disk size, time to first query after load
  filter    a separate build with a metadata column; search restricted to a random 10% and 1% subset
Memory is the process RSS (working set) and its peak, read with psutil.
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from adapters import ADAPTERS  # noqa: E402

PROC = psutil.Process()
K = 10
ALL_THREADS = os.cpu_count() or 1


PIN_CPU = int(os.environ.get("BENCH_PIN_CPU", "8"))   # a P-core logical CPU on the i9-14900HX (0-15 = P-cores, 16-31 = E-cores)
ALL_CPUS = list(range(os.cpu_count() or 1))


class pinned:
    """Pin the process to one P-core for a strictly single-threaded timed loop (and raise priority); restore after.
    Unpinned single-thread loops on this hybrid CPU were measured to vary up to 3x between processes (E-core placement,
    power state), so every 1-thread number in the report is taken pinned."""

    def __enter__(self):
        try:
            PROC.cpu_affinity([PIN_CPU])
        except Exception:  # noqa: BLE001
            pass

    def __exit__(self, *a):
        try:
            PROC.cpu_affinity(ALL_CPUS)
        except Exception:  # noqa: BLE001
            pass


def mem():
    mi = PROC.memory_info()
    return {"rss_mb": mi.rss / 2**20, "peak_mb": getattr(mi, "peak_wset", mi.rss) / 2**20}


def pct(a, p):
    return float(np.percentile(a, p))


def lat_stats(lat_s):
    a = np.asarray(lat_s) * 1000.0
    return {"median_ms": pct(a, 50), "p95_ms": pct(a, 95), "p99_ms": pct(a, 99), "mean_ms": float(a.mean())}


def loop(fn, Q, k, collect=True):
    lat = np.empty(len(Q))
    res = np.empty((len(Q), k), dtype=np.int64) if collect else None
    for i in range(len(Q)):
        t = time.perf_counter()
        r = fn(Q[i], k)
        lat[i] = time.perf_counter() - t
        if collect:
            m = min(len(r), k)
            res[i, :m] = r[:m]
            res[i, m:] = -1
    return lat, res


def ingest(A, ds, n, with_grp=None, chunk=20000, start=0):
    secs = 0.0
    for s in range(start, start + n, chunk):
        m = min(chunk, start + n - s)
        X = np.ascontiguousarray(C.base_rows(ds, s, m))
        ids = np.arange(s, s + m, dtype=np.int64)
        g = None if with_grp is None else with_grp[s:s + m]
        t = time.perf_counter()
        A.add(ids, X, g)
        secs += time.perf_counter() - t
    return secs


def build_once(name, ds, n, d, workdir, n_cap):
    A = ADAPTERS[name](d, n_cap, workdir)
    A.set_threads(ALL_THREADS)
    t_ing = ingest(A, ds, n)
    t = time.perf_counter()
    A.finalize()
    t_idx = time.perf_counter() - t
    return A, t_ing, t_idx


def warm(fn, Q, k, max_q=50, max_s=3.0):
    """warm-up: up to max_q queries or max_s seconds, whichever first."""
    t0 = time.perf_counter()
    for i in range(min(max_q, len(Q))):
        fn(Q[i], k)
        if time.perf_counter() - t0 > max_s:
            break


def sweep_point(A, Q, truth, k, budget_s, repeats=3):
    with pinned():
        return _sweep_point(A, Q, truth, k, budget_s, repeats)


def _sweep_point(A, Q, truth, k, budget_s, repeats=3):
    """1-thread loop at the current knob. returns dict."""
    A.set_threads(1)
    warm(A.search1, Q, k)
    # probe latency to size the query count
    probe, _ = loop(A.search1, Q[:5], k, collect=False)
    est = float(np.median(probe))
    nq = int(min(len(Q), max(30, budget_s / max(est, 1e-6))))
    q = Q[:nq]
    qps_runs, lat_all = [], []
    rec = None
    for r in range(repeats):
        lat, res = loop(A.search1, q, k, collect=(r == 0))
        if r == 0:
            rec = C.recall_at(res, truth[:nq], k)
        qps_runs.append(nq / lat.sum())
        lat_all.append(lat)
        if lat.sum() > budget_s * 1.5:   # too slow to repeat more
            break
    lat_cat = np.concatenate(lat_all)
    rec_q = nq
    want = min(len(Q), 1000)
    if nq < want and est * (want - nq) < 20:
        # recall over at least 1000 queries even when the timed loop was shortened (untimed extra pass)
        _, res2 = loop(A.search1, Q[nq:want], k, collect=True)
        rec = (rec * nq + C.recall_at(res2, truth[nq:want], k) * (want - nq)) / want
        rec_q = want
    out = {"nq": nq, "recall_queries": rec_q, "recall": rec, "qps_1t_runs": qps_runs, "qps_1t": statistics.median(qps_runs),
           "qps_1t_min": min(qps_runs), "qps_1t_max": max(qps_runs)}
    out.update(lat_stats(lat_cat))
    return out


def batch_point(A, Q, truth, k, repeats=3):
    A.set_threads(ALL_THREADS)
    try:
        r0 = A.searchN(Q[:min(64, len(Q))], k)
    except NotImplementedError:
        return None
    if r0 is None:
        return None
    runs = []
    rec = None
    for r in range(repeats):
        t = time.perf_counter()
        res = A.searchN(Q, k)
        dt = time.perf_counter() - t
        runs.append(len(Q) / dt)
        if r == 0:
            rec = C.recall_at(res, truth[:len(Q)], k)
        if dt > 90:
            break
    return {"recall": rec, "qps_batch_runs": runs, "qps_batch": statistics.median(runs),
            "qps_batch_min": min(runs), "qps_batch_max": max(runs), "threads": ALL_THREADS}


def core(args, name, ds, n, d, workdir, out):
    Qall = C.queries(ds, args.queries)
    truth = C.gt(ds, n, "", args.queries)
    n_cap = n + 2 * C.MUT
    gc.collect()
    out["mem_baseline"] = mem()
    A, t_ing, t_idx = build_once(name, ds, n, d, workdir, n_cap)
    gc.collect()
    out["version"] = A.version
    out["build"] = {"ingest_s": t_ing, "index_s": t_idx, "total_s": t_ing + t_idx}
    out["mem_after_build"] = mem()
    # first query (some systems build lazily here)
    A.set_threads(1)
    knobs = A.knobs(n)
    if knobs[0] is not None:
        A.set_knob(knobs[-1])
    t = time.perf_counter()
    A.search1(Qall[0], K)
    out["build"]["first_query_ms"] = (time.perf_counter() - t) * 1000
    out["build"]["ready_s"] = out["build"]["total_s"] + out["build"]["first_query_ms"] / 1000
    out["mem_after_first_query"] = mem()
    if hasattr(A, "stats"):
        try:
            out["index_stats"] = {k_: v for k_, v in A.stats().items() if isinstance(v, (int, float, str, type(None)))}
        except Exception as e:  # noqa: BLE001
            out["index_stats"] = {"error": str(e)}
    pts = []
    budget = args.budget
    for kn in knobs:
        if kn is not None:
            A.set_knob(kn)
        p = sweep_point(A, Qall, truth, K, budget)
        p["knob"] = kn
        b = batch_point(A, Qall, truth, K)
        if b:
            p["batch"] = b
        pts.append(p)
        print(f"  {name} {ds} {n} knob={kn} recall={p['recall']:.4f} qps1t={p['qps_1t']:.0f} med={p['median_ms']:.3f}ms"
              + (f" batch={b['qps_batch']:.0f}" if b else ""), flush=True)
        if p["recall"] >= 0.9995 and len(pts) > 2 and kn is not None and pts[-2]["recall"] >= 0.9995:
            break
    out["sweep"] = pts
    out["knob_name"] = A.knob_name
    # exact systems: also "one query at a time, library may use all cores" (latency mode)
    if A.exact or name in ("isong_py",):
        A.set_threads(ALL_THREADS)
        warm(A.search1, Qall, K, 20)
        probe, _ = loop(A.search1, Qall[:5], K, collect=False)
        nq = int(min(len(Qall), max(30, budget / max(float(np.median(probe)), 1e-6))))
        runs = []
        for r in range(3):
            lat, _ = loop(A.search1, Qall[:nq], K, collect=False)
            runs.append(lat)
            if lat.sum() > budget * 1.5:
                break
        cat = np.concatenate(runs)
        out["loop_allthreads"] = dict(nq=nq, threads=ALL_THREADS, qps=statistics.median([nq / r_.sum() for r_ in runs]), **lat_stats(cat))
    out["mem_after_sweep"] = mem()
    # reference knob: first with recall >= 0.95 else best
    ref = None
    for p in pts:
        if p["recall"] >= 0.95:
            ref = p["knob"]
            break
    if ref is None and pts and pts[0]["knob"] is not None:
        ref = pts[-1]["knob"]
    out["ref_knob"] = ref
    out["ref_recall_pre"] = next((p["recall"] for p in pts if p["knob"] == ref), None)
    # build repeats (fresh adapters) at small sizes only
    if args.build_repeats > 1:
        rep = [out["build"]["total_s"]]
        try:
            for _ in range(args.build_repeats - 1):
                A2, a, b = build_once(name, ds, n, d, workdir + "_rep", n_cap)
                rep.append(a + b)
                A2.close()
                del A2
                gc.collect()
        except Exception as e:  # noqa: BLE001
            out["build"]["repeat_error"] = f"{type(e).__name__}: {e}"
        out["build"]["total_s_runs"] = rep
    return A


def mutation(args, A, name, ds, n, out):
    if not os.path.exists(os.path.join(C.ds_dir(ds), f"gt_mut_{n}.i32")):
        out["mutation"] = {"skipped": "dataset has no spare rows for this n"}
        return
    Qall = C.queries(ds, args.queries)
    truth = C.gt(ds, n, "mut_", args.queries)
    upd, dele = C.mut_plan(ds, n)
    ref = out.get("ref_knob")
    res = {"ref_knob": ref, "recall_before": out.get("ref_recall_pre")}
    A.set_threads(ALL_THREADS if name not in ("isong_py", "isong_py_ivf") else 1)
    ok = True

    def timed(label, fn, ids_all, X_all=None, batch=1000):
        nonlocal ok
        t0 = time.perf_counter()
        try:
            for s in range(0, len(ids_all), batch):
                if X_all is None:
                    fn(ids_all[s:s + batch])
                else:
                    fn(ids_all[s:s + batch], X_all[s:s + batch])
        except NotImplementedError:
            res[label] = {"unsupported": True}
            ok = False
            return
        except Exception as e:  # noqa: BLE001
            res[label] = {"error": f"{type(e).__name__}: {e}"}
            ok = False
            return
        dt = time.perf_counter() - t0
        res[label] = {"seconds": dt, "per_s": len(ids_all) / dt, "batch": batch}

    Xi = np.ascontiguousarray(C.base_rows(ds, n, C.MUT))
    timed("insert_10k", A_add_wrapper(A), np.arange(n, n + C.MUT, dtype=np.int64), Xi)
    Xu = np.ascontiguousarray(C.base_rows(ds, n + C.MUT, C.MUT))
    timed("update_10k", A.upsert, upd.astype(np.int64), Xu)
    timed("delete_10k", A.delete, dele.astype(np.int64))
    res["mem_after"] = mem()
    if ok:
        A.set_threads(1)
        if ref is not None:
            A.set_knob(ref)
        t = time.perf_counter()
        A.search1(Qall[0], K)
        res["first_query_after_ms"] = (time.perf_counter() - t) * 1000
        p = sweep_point(A, Qall, truth, K, args.budget)
        res["after"] = p
        res["recall_after"] = p["recall"]
        if hasattr(A, "stats"):
            try:
                res["index_stats_after"] = {k_: v for k_, v in A.stats().items() if isinstance(v, (int, float, str, type(None)))}
            except Exception:  # noqa: BLE001
                pass
    out["mutation"] = res


def A_add_wrapper(A):
    def f(ids, X):
        A.add(ids, X, None)
    return f


def saveload(args, A, name, ds, n, out):
    Qall = C.queries(ds, args.queries)
    path = os.path.join(args.workdir, f"{name}_{ds}_{n}.saved")
    res = {}
    ref = out.get("ref_knob")
    try:
        A.set_threads(ALL_THREADS)
        t = time.perf_counter()
        A.save(path)
        res["save_s"] = time.perf_counter() - t
        res["disk_mb"] = A.disk_bytes(path) / 2**20
        if hasattr(A, "save_alt"):   # the binary snapshot of the same store (ISONGraph ports only)
            t = time.perf_counter()
            A.save_alt(path + ".snap")
            res["snap_save_s"] = time.perf_counter() - t
            res["snap_mb"] = os.path.getsize(path + ".snap") / 2**20
        t = time.perf_counter()
        A.load(path)
        res["load_s"] = time.perf_counter() - t
        A.set_threads(1)
        if ref is not None:
            A.set_knob(ref)
        t = time.perf_counter()
        A.finalize() if name in ("isong_py_ivf",) else None
        A.search1(Qall[0], K)
        res["first_query_after_load_ms"] = (time.perf_counter() - t) * 1000
        res["time_to_ready_s"] = res["load_s"] + res["first_query_after_load_ms"] / 1000
        if hasattr(A, "load_alt"):
            t = time.perf_counter()
            A.load_alt(path + ".snap")
            res["snap_load_s"] = time.perf_counter() - t
            try:
                os.remove(path + ".snap")
            except OSError:
                pass
    except NotImplementedError:
        res["unsupported"] = True
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    out["saveload"] = res


def filt(args, name, ds, n, d, workdir, out):
    Qall = C.queries(ds, args.queries)
    g = C.grp(ds, n)
    n_cap = n + 2 * C.MUT
    A = ADAPTERS[name](d, n_cap, workdir)
    A.set_threads(ALL_THREADS)
    t_ing = ingest(A, ds, n, with_grp=g)
    t = time.perf_counter()
    A.finalize()
    t_idx = time.perf_counter() - t
    if hasattr(A, "prepare_filter"):
        A.prepare_filter(g)
    res = {"build_with_metadata_s": t_ing + t_idx, "mem": mem(), "subsets": {}}
    knobs = A.knobs(n)
    if knobs[0] is not None and len(knobs) > 5:
        pick = [knobs[i] for i in sorted(set([1, len(knobs) // 3, len(knobs) // 2, 2 * len(knobs) // 3, len(knobs) - 1]))]
    else:
        pick = knobs
    A.set_threads(1)
    _pin = pinned()
    _pin.__enter__()
    for gi, label in ((0, "f10"), (1, "f01")):
        truth = C.gt(ds, n, label + "_", args.queries)
        pts = []
        for kn in pick:
            if kn is not None:
                A.set_knob(kn)
            fn = (lambda q, k, gi=gi: A.search_filtered1(q, k, gi))
            try:
                warm(fn, Qall, K, 10, 2.0)
                probe, _ = loop(fn, Qall[:5], K, collect=False)
            except NotImplementedError:
                res["subsets"][label] = {"unsupported": True}
                pts = None
                break
            nq = int(min(len(Qall), max(30, args.budget / max(float(np.median(probe)), 1e-6))))
            runs = []
            rec = None
            lats = []
            for r in range(3):
                lat, rs = loop(fn, Qall[:nq], K, collect=(r == 0))
                if r == 0:
                    rec = C.recall_at(rs, truth[:nq], K)
                runs.append(nq / lat.sum())
                lats.append(lat)
                if lat.sum() > args.budget * 1.5:
                    break
            p = {"knob": kn, "nq": nq, "recall": rec, "qps_1t": statistics.median(runs), "qps_1t_runs": runs}
            p.update(lat_stats(np.concatenate(lats)))
            pts.append(p)
            print(f"  filter {label} {name} knob={kn} recall={rec:.4f} qps={p['qps_1t']:.0f}", flush=True)
        if pts is not None:
            res["subsets"][label] = pts
    _pin.__exit__()
    out["filter"] = res
    A.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--exp", default="core,mutation,saveload")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=15.0, help="target seconds per timed 1-thread loop; fewer queries if slower")
    ap.add_argument("--build-repeats", type=int, default=1)
    ap.add_argument("--workdir", default=r"D:\Goldmine\ISON\bench_data\work")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exps = args.exp.split(",")
    m = C.meta(args.ds)
    d = m["d"]
    os.makedirs(args.workdir, exist_ok=True)
    wd = os.path.join(args.workdir, f"{args.system}_{args.ds}_{args.n}")
    os.makedirs(wd, exist_ok=True)
    out = {"system": args.system, "ds": args.ds, "n": args.n, "d": d, "k": K, "queries": args.queries,
           "all_threads": ALL_THREADS, "exps": exps, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    A = None
    try:
        if "core" in exps:
            A = core(args, args.system, args.ds, args.n, d, wd, out)
        if "mutation" in exps and A is not None:
            mutation(args, A, args.system, args.ds, args.n, out)
        if "saveload" in exps and A is not None:
            saveload(args, A, args.system, args.ds, args.n, out)
        if "filter" in exps:
            if A is not None:
                A.close()
                A = None
                gc.collect()
            filt(args, args.system, args.ds, args.n, d, wd, out)
        out["mem_end"] = mem()
        out["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        import traceback
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        print(out["traceback"], flush=True)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=float)
    if A is not None:
        try:
            A.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
