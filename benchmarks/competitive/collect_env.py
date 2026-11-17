"""Write results/environment.json: hardware, OS, toolchains and exact library versions used."""
import json
import os
import platform
import subprocess
import sys
from importlib import metadata

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, shell=isinstance(cmd, str)).stdout.strip() or "n/a"
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def main():
    env = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu_model": run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]),
        "cpu_cores": psutil.cpu_count(logical=False), "cpu_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "note": "i9-14900HX is a hybrid CPU (8 P-cores with SMT + 16 E-cores); single-thread loops are not pinned to a core type",
        "packages": {p: metadata.version(p) for p in ("numpy", "faiss-cpu", "hnswlib", "usearch", "chromadb", "qdrant-client", "lancedb", "sqlite-vec", "pyarrow", "threadpoolctl", "psutil")},
        "rustc": run(["rustc", "--version"]), "cargo": run(["cargo", "--version"]),
        "rust_build": "cargo build --release --example competitive_bench (default flags = baseline x86-64) and RUSTFLAGS='-C target-cpu=native'",
        "msvc": run('cmd /c ""C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul && cl 2>&1"').splitlines()[0:1],
        "cpp_build": "cl /std:c++17 /O2 /EHsc /MD /DNDEBUG (default SSE2) and the same with /arch:AVX2",
        "dotnet": run(["dotnet", "--version"]), "node": run(["node", "--version"]),
    }
    try:
        import sqlite3
        env["sqlite3"] = sqlite3.sqlite_version
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy
        import faiss
        env["numpy_blas"] = str(numpy.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", ""))
        env["faiss_avx"] = [m for m in ("avx512", "avx2", "sse4") if hasattr(faiss, "supported_instruction_sets") and m in str(faiss.supported_instruction_sets()).lower()]
    except Exception as e:  # noqa: BLE001
        env["numpy_blas"] = f"n/a ({e})"
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(env, open(os.path.join(HERE, "results", "environment.json"), "w"), indent=1)
    print(json.dumps(env, indent=1))


if __name__ == "__main__":
    main()
