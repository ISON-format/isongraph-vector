"""Machine state for the Stage C README: CPU, library versions, background load sample, top processes."""
import json
import platform
import subprocess
import sys
import time
from importlib import metadata

import psutil

def ver(p):
    try:
        return metadata.version(p)
    except Exception:  # noqa: BLE001
        return None

per = psutil.cpu_percent(interval=10, percpu=True)
procs = []
for p in psutil.process_iter(["name"]):
    try:
        p.cpu_percent(None)
        procs.append(p)
    except Exception:  # noqa: BLE001
        pass
time.sleep(5)
rows = []
for p in procs:
    try:
        rows.append((p.cpu_percent(None), p.info["name"]))
    except Exception:  # noqa: BLE001
        pass
top = sorted(rows, reverse=True)[:8]
def cmd(c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return str(e)
env = {
    "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version.split()[0], "platform": platform.platform(),
    "cpu": "Intel Core i9-14900HX (8 P-cores with SMT + 16 E-cores = 24 cores / 32 logical)",
    "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    "ram_available_gb": round(psutil.virtual_memory().available / 2**30, 1),
    "packages": {p: ver(p) for p in ("numpy", "faiss-cpu", "hnswlib", "usearch", "chromadb", "lancedb", "sqlite-vec", "psutil", "threadpoolctl")},
    "rustc": cmd(["rustc", "--version"]), "cargo": cmd(["cargo", "--version"]),
    "rust_build": "cargo build --release --example stage_c_bench (default flags, CARGO_TARGET_DIR=bench_data/rs_target_stagec); AVX2/FMA kernels chosen at run time",
    "load_sample_per_cpu_pct_15s": [round(x, 1) for x in per],
    "load_mean_pct": round(sum(per) / len(per), 1), "pin_cpu8_pct": per[8],
    "top_processes_cpu_pct_5s": [[round(c, 1), n] for c, n in top],
    "power": cmd(["powercfg", "/getactivescheme"]),
}
json.dump(env, open("environment.json", "w"), indent=1)
print(json.dumps(env, indent=1))
