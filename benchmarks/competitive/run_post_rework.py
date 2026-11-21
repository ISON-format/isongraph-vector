"""Post-rework competitive run: the reworked ISONGraph ports (HNSW, batch/concurrent search, int8, snapshots)
against faiss / hnswlib / numpy on text384, with freshly built binaries.

This is a small sibling of run_all.py (same machine-load check, same job isolation, same drivers); it exists because
run_all.py points at the stale prebuilt binaries and has no job for the new HNSW configurations. It does not touch
run_all.py or the existing results.

    python run_post_rework.py --list
    python run_post_rework.py --cell 100000 --out results/post_rework/raw             # resumable
    python run_post_rework.py --cell 100000 --systems faiss_hnsw,hnswlib --redo --tag r2
    python run_post_rework.py --smoke                                                  # 10k rows, tiny budgets, to scratch dir

Binaries (build them first, see README_post_rework.md):  BUILD = D:\\Goldmine\\ISON\\bench_data\\post_rework_build
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

import run_all as R

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BUILD = r"D:\Goldmine\ISON\bench_data\post_rework_build"
RS_DEFAULT = os.path.join(BUILD, "rs_default", "release", "examples", "competitive_bench.exe")
RS_NATIVE = os.path.join(BUILD, "rs_native", "release", "examples", "competitive_bench.exe")
CPP_DEFAULT = os.path.join(BUILD, "cpp", "cb_default.exe")
CPP_AVX2 = os.path.join(BUILD, "cpp", "cb_avx2.exe")
CS_DLL = os.path.join(BUILD, "cs", "IsonGraph.Vector.CompetitiveBench.dll")
REPO = os.path.dirname(os.path.dirname(HERE))
JS_ENTRY = os.path.join(REPO, "isongraph-vector-js", "bench", "competitive.mjs")
TS_DIST = os.path.join(REPO, "isongraph-vector-ts", "dist", "index.js")
DS = "text384"

PY_SYSTEMS = ["numpy_bf", "faiss_flat", "faiss_ivfflat", "faiss_hnsw", "hnswlib", "isong_py", "isong_py_ivf"]


def jobs(n):
    big = n >= 1_000_000
    out = []

    def add(system, exp, timeout_min, extra=None):
        out.append(dict(system=system, ds=DS, n=n, exp=exp, timeout=timeout_min * 60, extra=extra or []))

    # competitors and the Python port
    for s in PY_SYSTEMS:
        exp = "core" if big and s in ("numpy_bf", "isong_py", "isong_py_ivf") else "core,saveload"
        add(s, exp, 90 if big else 60, [] if big else ["--build-repeats", "3"])
    # ISONGraph native ports: exact + IVF (+ int8 for Rust) + JSON/snapshot save-load, then HNSW in a separate process (clean memory)
    add("rust_default", "core,int8,saveload" if not big else "core,int8,saveload", 90 if big else 60)
    add("rust_default", "hnsw", 90 if big else 60)
    if not big:
        add("rust_native", "core,int8", 60)
        add("rust_native", "hnsw", 60)
    add("cpp_default", "core" if big else "core,saveload", 90 if big else 60)
    add("cpp_default", "hnsw", 90 if big else 60)
    if not big:
        add("cpp_avx2", "core", 60)
        add("cpp_avx2", "hnsw", 60)
    add("csharp", "core" if big else "core,saveload", 90 if big else 60)
    add("csharp", "hnsw", 45 if big else 60)   # 1M: ~10+ min expected; give it 45, record a timeout rather than wait longer
    add("js", "core", 60 if big else 60)
    # JS/TS HNSW at 1M is estimated at over half an hour (100k took ~5 min): skipped at 1M, see README_post_rework.md
    if not big:
        add("js", "hnsw", 60)
        add("js", "saveload", 60)
        add("ts", "core", 60)
        add("ts", "hnsw", 60)
        add("ts", "saveload", 60)
    return out


def job_name(j, tag=""):
    return f"{j['system']}__{j['ds']}__{j['n']}__{j['exp'].replace(',', '+')}{('__' + tag) if tag else ''}"


def command(j, out_path, queries, budget):
    s, ds, n = j["system"], j["ds"], j["n"]
    ddir = os.path.join(R.PREP, ds)
    common = ["--ds-dir", ddir, "--ds", ds, "--n", str(n), "--exp", j["exp"], "--queries", str(queries), "--budget", str(budget), "--out", out_path]
    if s.startswith("rust"):
        return [RS_NATIVE if s == "rust_native" else RS_DEFAULT] + common + ["--label", s]
    if s.startswith("cpp"):
        return [CPP_AVX2 if s == "cpp_avx2" else CPP_DEFAULT] + common + ["--label", s]
    if s == "csharp":
        return ["dotnet", CS_DLL] + common + ["--label", "csharp"]
    if s == "js":
        return ["node", "--max-old-space-size=24000", JS_ENTRY] + common + ["--label", "js"]
    if s == "ts":
        return ["node", "--max-old-space-size=24000", JS_ENTRY, "--entry", TS_DIST] + common + ["--label", "ts"]
    return [PY, os.path.join(HERE, "bench_py.py"), "--system", s, "--ds", ds, "--n", str(n), "--exp", j["exp"], "--queries", str(queries),
            "--budget", str(budget), "--out", out_path] + j["extra"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "post_rework", "raw"))
    ap.add_argument("--cell", type=int, default=100_000)
    ap.add_argument("--systems", default="")
    ap.add_argument("--exp", default="", help="only jobs whose exp list contains this")
    ap.add_argument("--tag", default="", help="suffix for the result file names (repeat runs)")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--quiet-wait-min", type=float, default=20)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="10k rows, 100 queries, tiny budget; writes under <out>/../smoke")
    a = ap.parse_args()
    n = 10_000 if a.smoke else a.cell
    js = jobs(n)
    if a.systems:
        keep = set(a.systems.split(","))
        js = [j for j in js if j["system"] in keep]
    if a.exp:
        js = [j for j in js if a.exp in j["exp"].split(",")]
    if a.list:
        for j in js:
            print(job_name(j, a.tag), "timeout_min", j["timeout"] // 60)
        print(len(js), "jobs")
        return
    out_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "smoke") if a.smoke else a.out
    os.makedirs(out_dir, exist_ok=True)
    queries, budget = (100, 0.3) if a.smoke else (a.queries, a.budget)
    log = open(os.path.join(out_dir, "smoke_log.txt") if a.smoke else os.path.join(out_dir, "..", "run_log.txt"), "a")
    for j in js:
        name = job_name(j, a.tag)
        out_path = os.path.join(out_dir, name + ".json")
        if os.path.exists(out_path) and not a.redo:
            continue
        if a.smoke:
            j = dict(j, extra=[])
        exe = command(j, out_path, queries, budget)
        if a.smoke:
            load, contaminated = R.cpu_load(1.0)[0], False
        else:
            load, contaminated = R.wait_quiet(a.quiet_wait_min)
        t0 = time.time()
        msg = f"[{time.strftime('%H:%M:%S')}] START {name} load_before={load:.1f}%{' CONTAMINATED' if contaminated else ''}"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()
        env = dict(os.environ, RUST_LOG="error", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        status = "ok"
        with open(out_path[:-5] + ".stdout.txt", "w", encoding="utf-8", errors="replace") as so:
            p = subprocess.Popen(exe, stdout=so, stderr=subprocess.STDOUT, env=env, creationflags=0x00000080)  # HIGH_PRIORITY_CLASS
            if j["system"] in ("js", "ts"):
                try:   # node's timed loops are single-threaded: keep it on four P-cores (8,10,12,14), as run_all.py does
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
            json.dump({"system": j["system"], "ds": j["ds"], "n": j["n"], "exps": j["exp"].split(","),
                       "status": "timeout" if status == "timeout" else "crashed", "limit_s": j["timeout"], "exit": p.returncode}, open(out_path, "w"))
        try:
            d = json.load(open(out_path))
            d["load_before_pct"] = load
            d["load_contaminated"] = contaminated
            d["wall_s"] = dt
            d["job_exp"] = j["exp"]
            json.dump(d, open(out_path, "w"), indent=1)
        except Exception as e:  # noqa: BLE001
            print("annotate failed", e)
        msg = f"[{time.strftime('%H:%M:%S')}] END   {name} {status} {dt:.0f}s"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()


if __name__ == "__main__":
    main()
