"""Run the whole competitive matrix, one process per (system, dataset, n, experiment-group), sequentially.

    python run_all.py --list                       # show the job list
    python run_all.py --out results/raw            # run everything not yet done (resumable)
    python run_all.py --only text384:100000 --systems faiss_hnsw,hnswlib
    python run_all.py --wait-flag C:/.../B_DONE --wait-minutes 120

Timed runs must not overlap anything else. Before every job the runner measures machine CPU load (all other
processes) and waits until it is low; the load it saw is stored beside the result (`load_before_pct`).
Never run two copies of this script at once.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
DATA = r"D:\Goldmine\ISON\bench_data"
PREP = os.path.join(DATA, "prep")
RS_DEFAULT = os.path.join(DATA, "rs_target_default", "release", "examples", "competitive_bench.exe")
RS_NATIVE = os.path.join(DATA, "rs_target_native", "release", "examples", "competitive_bench.exe")
CPP_DEFAULT = os.path.join(DATA, "cpp_build", "cb_default.exe")
CPP_AVX2 = os.path.join(DATA, "cpp_build", "cb_avx2.exe")
CS_DLL = os.path.join(DATA, "cs_build", "IsonGraph.Vector.CompetitiveBench.dll")
JS_ENTRY = os.path.join(os.path.dirname(HERE), "..", "isongraph-vector-js", "bench", "competitive.mjs")

PY_ANN = ["isong_py_ivf", "faiss_ivfflat", "faiss_hnsw", "hnswlib", "usearch", "chromadb", "lancedb_ivfflat"]
PY_EXACT = ["numpy_bf", "isong_py", "faiss_flat", "lancedb_flat", "sqlite_vec"]
NATIVE = ["rust_default", "rust_native", "cpp_default", "cpp_avx2", "csharp", "js"]

CELLS = [("text384", 100_000), ("glove", 100_000), ("sift", 100_000), ("text384", 1_000_000), ("glove", 1_000_000), ("sift", 1_000_000)]


def jobs():
    out = []

    def add(system, ds, n, exp, timeout_min, prio, extra=None):
        out.append(dict(system=system, ds=ds, n=n, exp=exp, timeout=timeout_min * 60, prio=prio, extra=extra or []))

    for ds, n in CELLS:
        big = n >= 1_000_000
        mut = not (ds == "sift" and big)
        base_prio = (0 if ds == "text384" else 1 if ds == "glove" else 22) + (10 if big else 0)
        for s in PY_ANN + PY_EXACT + NATIVE:
            if s == "js" and big and ds != "text384":
                continue  # JS k-means over 1M plain arrays takes ~30 min per cell; one 1M row (text) is enough
            if s in ("sqlite_vec", "chromadb") and big and ds != "text384":
                continue  # one 1M row (text) is enough: sqlite-vec is exact at ~0.5 s/query; chroma 1M ingest is very slow
            exp = ["core"]
            if mut:
                exp.append("mutation")
            if not (big and s in ("isong_py", "isong_py_ivf", "cpp_default", "cpp_avx2", "csharp", "js", "chromadb", "lancedb_flat", "lancedb_ivfflat", "sqlite_vec", "numpy_bf")):
                exp.append("saveload")
            if s in ("rust_native",):
                exp = [e for e in exp if e != "saveload"]   # save/load is not SIMD-dependent; measured on rust_default
            to = (90 if s == "chromadb" else 150) if big else 60
            extra = ["--build-repeats", "3"] if (not big and s in PY_ANN + PY_EXACT) else []
            add(s, ds, n, ",".join(exp), to, base_prio, extra)
    # filtered search
    for ds, n in [("text384", 100_000), ("glove", 100_000), ("text384", 1_000_000)]:
        big = n >= 1_000_000
        for s in PY_ANN + PY_EXACT + ["rust_default", "cpp_default"]:
            if s in ("usearch",):
                pass  # still run: records "unsupported"
            if big and s not in ("isong_py_ivf", "faiss_ivfflat", "faiss_hnsw", "hnswlib", "lancedb_ivfflat", "numpy_bf", "faiss_flat", "rust_default", "cpp_default"):
                continue
            add(s, ds, n, "filter", 150 if big else 60, (3 if ds == "text384" else 4) + (10 if big else 0))
    # tiny exact-latency cells
    for s in PY_EXACT + ["qdrant_local", "rust_default", "rust_native", "cpp_default", "cpp_avx2", "csharp", "js"]:
        add(s, "text384", 10_000, "core", 30, 5)
    add("qdrant_local", "text384", 100_000, "core,mutation", 90, 6)
    return sorted(out, key=lambda j: j["prio"])


def job_name(j):
    return f"{j['system']}__{j['ds']}__{j['n']}__{j['exp'].replace(',', '+')}"


def command(j, out_path, queries, budget):
    s, ds, n = j["system"], j["ds"], j["n"]
    ddir = os.path.join(PREP, ds)
    common = ["--ds-dir", ddir, "--ds", ds, "--n", str(n), "--exp", j["exp"], "--queries", str(queries), "--budget", str(budget), "--out", out_path]
    if s.startswith("rust"):
        return [RS_NATIVE if s == "rust_native" else RS_DEFAULT] + common + ["--label", s]
    if s.startswith("cpp"):
        return [CPP_AVX2 if s == "cpp_avx2" else CPP_DEFAULT] + common + ["--label", s]
    if s == "csharp":
        return ["dotnet", CS_DLL] + common + ["--label", "csharp"]
    if s == "js":
        return ["node", "--max-old-space-size=24000", JS_ENTRY] + common + ["--label", "js"]
    return [PY, os.path.join(HERE, "bench_py.py"), "--system", s, "--ds", ds, "--n", str(n), "--exp", j["exp"], "--queries", str(queries),
            "--budget", str(budget), "--out", out_path] + j["extra"]


def cpu_load(seconds=3.0):
    me = os.getpid()
    mine = {p.pid for p in psutil.Process(me).children(recursive=True)} | {me}
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except Exception:  # noqa: BLE001
            pass
    time.sleep(seconds)
    tot = 0.0
    top = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            c = p.cpu_percent(None)
            if p.pid in mine or p.pid == 0:
                continue
            tot += c
            if c > 20:
                top.append((round(c), p.info["name"], p.pid))
        except Exception:  # noqa: BLE001
            pass
    return tot / (psutil.cpu_count() or 1), top


def wait_quiet(max_wait_min, thresh=25.0):
    t0 = time.time()
    while True:
        load, top = cpu_load()
        if load <= thresh:
            return load, False
        if time.time() - t0 > max_wait_min * 60:
            return load, True
        time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "raw"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="", help="ds:n[,ds:n]")
    ap.add_argument("--systems", default="")
    ap.add_argument("--exp", default="", help="only jobs whose exp contains this")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--wait-flag", default="")
    ap.add_argument("--wait-minutes", type=float, default=120)
    ap.add_argument("--quiet-wait-min", type=float, default=20, help="per job: max minutes to wait for low CPU load")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--dry", action="store_true", help="tiny queries/budget; output to --out")
    a = ap.parse_args()
    js = jobs()
    if a.only:
        keep = {tuple(x.split(":")) for x in a.only.split(",")}
        js = [j for j in js if (j["ds"], str(j["n"])) in keep]
    if a.systems:
        keep = set(a.systems.split(","))
        js = [j for j in js if j["system"] in keep]
    if a.exp:
        js = [j for j in js if a.exp in j["exp"].split(",")]
    if a.list:
        for j in js:
            print(job_name(j), "prio", j["prio"], "timeout_min", j["timeout"] // 60)
        print(len(js), "jobs")
        return
    os.makedirs(a.out, exist_ok=True)
    flag_waited = None
    if a.wait_flag:
        t0 = time.time()
        while not os.path.exists(a.wait_flag):
            if time.time() - t0 > a.wait_minutes * 60:
                flag_waited = f"flag file did not appear within {a.wait_minutes} min; ran anyway (timings may be contaminated)"
                break
            time.sleep(60)
        if flag_waited is None:
            flag_waited = "flag file present"
        print(flag_waited, flush=True)
    log = open(os.path.join(a.out, "..", "run_log.txt"), "a")
    for j in js:
        name = job_name(j)
        out_path = os.path.join(a.out, name + ".json")
        if os.path.exists(out_path) and not a.redo:
            continue
        if not all(os.path.exists(p) for p in [os.path.join(PREP, j["ds"], "meta.json")]):
            continue
        exe = command(j, out_path, a.queries, a.budget)
        if not a.dry:
            load, contaminated = wait_quiet(a.quiet_wait_min)
        else:
            load, contaminated = cpu_load(1.0)[0], False
        t0 = time.time()
        msg = f"[{time.strftime('%H:%M:%S')}] START {name} load_before={load:.1f}%{' CONTAMINATED' if contaminated else ''}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        env = dict(os.environ, RUST_LOG="error", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        stdout_path = out_path[:-5] + ".stdout.txt"
        status = "ok"
        with open(stdout_path, "w", encoding="utf-8", errors="replace") as so:
            p = subprocess.Popen(exe, stdout=so, stderr=subprocess.STDOUT, env=env, creationflags=0x00000080)  # HIGH_PRIORITY_CLASS
            if j["system"] == "js":
                try:   # node is single-threaded for these loops: keep it on four P-cores (8,10,12,14), one per physical core
                    psutil.Process(p.pid).cpu_affinity([8, 10, 12, 14])
                except Exception:  # noqa: BLE001
                    pass
            try:
                p.wait(timeout=j["timeout"])
            except subprocess.TimeoutExpired:
                status = "timeout"
                try:
                    for ch in psutil.Process(p.pid).children(recursive=True):
                        ch.kill()
                except Exception:  # noqa: BLE001
                    pass
                p.kill()
        dt = time.time() - t0
        if status == "timeout" or not os.path.exists(out_path):
            if not os.path.exists(out_path) or status == "timeout":
                json.dump({"system": j["system"], "ds": j["ds"], "n": j["n"], "exps": j["exp"].split(","),
                           "status": "timeout" if status == "timeout" else "crashed", "limit_s": j["timeout"], "exit": p.returncode}, open(out_path, "w"))
        # annotate with run context
        try:
            d = json.load(open(out_path))
            d["load_before_pct"] = load
            d["load_contaminated"] = contaminated
            d["wall_s"] = dt
            d["flag"] = flag_waited
            json.dump(d, open(out_path, "w"), indent=1)
        except Exception as e:  # noqa: BLE001
            print("annotate failed", e)
        msg = f"[{time.strftime('%H:%M:%S')}] END   {name} {status} {dt:.0f}s"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()


if __name__ == "__main__":
    main()
