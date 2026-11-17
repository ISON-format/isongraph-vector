"""Stage C driver: runs every cell sequentially (never two timed jobs at once), resumable, logs machine load.

    python run_stage_c.py --list
    python run_stage_c.py --phase 1          # text384 100k, 3 interleaved rounds
    python run_stage_c.py --phase 2          # glove / sift 100k exact (Q1, Q4)
    python run_stage_c.py --phase 3          # text384 1M
    python run_stage_c.py --phase 4          # persistence 100k / 1M (Rust)

Before every job it samples total CPU load and the load on the pinned CPU; if the machine is busy it waits up to
WAIT_MAX seconds. The load it saw is stored in raw/<job>.load.json; a job started at high load is flagged noisy
and should be re-run (rm its json and run again: finished jobs are skipped).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
DATA = r"D:\Goldmine\ISON\bench_data"
PREP = DATA + r"\prep"
RS = DATA + r"\rs_target_stagec\release\examples\stage_c_bench.exe"
PY = sys.executable
PIN = int(os.environ.get("BENCH_PIN_CPU", "8"))
LOAD_OK = float(os.environ.get("STAGEC_LOAD_OK", "30"))    # total % CPU considered quiet enough
WAIT_MAX = 45


def measure_load(sec=4.0):
    per = psutil.cpu_percent(interval=sec, percpu=True)
    return {"total_pct": sum(per) / len(per), "pin_cpu_pct": per[PIN], "max_cpu_pct": max(per)}


def wait_quiet():
    t0 = time.time()
    while True:
        ld = measure_load()
        if ld["total_pct"] < LOAD_OK and ld["pin_cpu_pct"] < 15:
            ld["noisy"] = False
            return ld
        if time.time() - t0 > WAIT_MAX:
            ld["noisy"] = True
            return ld
        time.sleep(5)


def rust(mode, ds, n, exp, rep, int8=0, extra=None, tag=None):
    name = f"rust_{mode}{'_i8x%d' % int8 if int8 else ''}_{ds}_{n}{'_' + tag if tag else ''}_r{rep}"
    cmd = [RS, "--ds-dir", f"{PREP}/{ds}", "--ds", ds, "--n", str(n), "--mode", mode, "--exp", exp, "--queries", "1000",
           "--budget", "3", "--label", f"rust_{mode}{'_i8x%d' % int8 if int8 else ''}", "--out", os.path.join(RAW, name + ".json")]
    if int8:
        cmd += ["--int8", str(int8)]
    cmd += extra or []
    return name, cmd


def py(system, ds, n, exp, rep):
    name = f"{system}_{ds}_{n}_r{rep}"
    cmd = [PY, os.path.join(HERE, "bench_c.py"), "--system", system, "--ds", ds, "--n", str(n), "--exp", exp,
           "--queries", "1000", "--budget", "3", "--out", os.path.join(RAW, name + ".json")]
    return name, cmd


def jobs(phase):
    out = []
    if phase == 1:
        ds, n = "text384", 100_000
        for r in (1, 2, 3):
            first = r == 1   # int8 study, concurrent clients and 10k writes once; the timed sweeps three times
            out += [rust("exact", ds, n, "core,int8,conc" if first else "core", r),
                    py("numpy_bf", ds, n, "core", r),
                    py("faiss_flat", ds, n, "core", r),
                    rust("ivf", ds, n, "core,conc,writes" if first else "core", r),
                    py("faiss_ivfflat", ds, n, "core,mutation,saveload", r),
                    rust("hnsw", ds, n, "core,conc,writes" if first else "core", r),
                    py("faiss_hnsw", ds, n, "core,mutation,saveload", r),
                    py("hnswlib", ds, n, "core,mutation,saveload", r),
                    py("usearch", ds, n, "core,mutation,saveload", r)]
            if r == 1:
                out += [rust("ivf", ds, n, "core", r, int8=4),
                        rust("hnsw", ds, n, "core", r, int8=4, extra=["--knobs", "64,96,128,192"])]
    elif phase == 2:
        for ds in ("glove", "sift"):
            for r in (1, 2, 3):
                out += [rust("exact", ds, 100_000, "core,int8" if r == 1 else "core", r),
                        py("numpy_bf", ds, 100_000, "core", r),
                        py("faiss_flat", ds, 100_000, "core", r)]
    elif phase == 3:
        ds, n = "text384", 1_000_000
        out += [rust("exact", ds, n, "core,int8,conc", 1),
                py("numpy_bf", ds, n, "core", 1),
                py("faiss_flat", ds, n, "core", 1),
                rust("hnsw", ds, n, "core,conc,writes", 1, extra=["--knobs", "48,64,96,128,192,256,400,640"]),
                py("hnswlib", ds, n, "core,mutation,saveload", 1),
                rust("exact", ds, n, "persist", 1, extra=["--ready", "ivf"], tag="persist"),
                rust("ivf", ds, n, "core,conc,writes", 1),
                py("faiss_ivfflat", ds, n, "core,mutation,saveload", 1),
                py("faiss_hnsw", ds, n, "core,mutation,saveload", 1),
                py("usearch", ds, n, "core,mutation,saveload", 1),
                rust("ivf", ds, n, "core", 1, int8=4, extra=["--knobs", "64,128,256,512"]),
                rust("hnsw", ds, n, "core", 1, int8=4, extra=["--knobs", "192,256"])]
    elif phase == 4:
        out += [rust("exact", "text384", 100_000, "persist", 1, extra=["--ready", "ivf,hnsw"], tag="persist")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, action="append")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    js = []
    for ph in a.phase or [1, 2, 3, 4]:
        js += jobs(ph)
    if a.only:
        js = [j for j in js if a.only in j[0]]
    if a.list:
        for nm, cmd in js:
            print(nm)
        return
    for nm, cmd in js:
        outp = os.path.join(RAW, nm + ".json")
        if os.path.exists(outp) and json.load(open(outp)).get("status") == "ok":
            print(f"skip {nm} (done)", flush=True)
            continue
        ld = wait_quiet()
        print(f"== {nm}  load total {ld['total_pct']:.0f}% pin-cpu {ld['pin_cpu_pct']:.0f}% noisy={ld['noisy']}", flush=True)
        t0 = time.time()
        with open(os.path.join(RAW, nm + ".txt"), "w") as fh:
            fh.write(" ".join(cmd) + "\n")
            fh.flush()
            try:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
            except subprocess.TimeoutExpired:
                fh.write("TIMEOUT\n")
        ld["seconds"] = time.time() - t0
        ld["after"] = psutil.cpu_percent(interval=None)
        json.dump(ld, open(os.path.join(RAW, nm + ".load.json"), "w"))
        print(f"   done in {ld['seconds']:.0f}s ok={os.path.exists(outp) and json.load(open(outp)).get("status") == "ok"}", flush=True)


if __name__ == "__main__":
    main()
