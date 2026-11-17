#!/usr/bin/env python3
"""Throughput / latency of semantic search through the HTTP service, with N tenants.

What it does
  1. creates N tenants (each with its own API key) and one graph per tenant holding M random
     D-dimensional vectors (client-mode graphs: ingestion cost is not what is being measured)
  2. starts the server(s) -- as separate OS processes by default, so the load generator and the
     server do not fight over one GIL
  3. runs C closed-loop client threads for a fixed duration, each over its own keep-alive
     connection, each request a POST .../search for a random query vector against a random tenant
  4. prints a table and (with --out) writes JSON a benchmark wave can ingest

Modes
  --mode spawn      (default) server in a child process
  --mode inproc     server in this process (client and server share the GIL: pessimistic)
  --mode external   --url and --keys-file point at a running deployment (keys file: JSON list of
                    {"key": ..., "graph": ...}); no setup is done
  --shards K        spawn K storage nodes plus a router in front (tenants placed by the router's
                    hash); shows the router's overhead and scale-out of independent tenants

Examples
  python scripts/load_test.py --tenants 8 --nodes 2000 --dim 64 --clients 16 --duration 10
  python scripts/load_test.py --shards 2 --tenants 16 --out results.json
  python scripts/load_test.py --text                # text queries: the server encodes (MockEncoder)

Honest caveats: closed-loop load measures peak sustainable throughput, not latency under a fixed
arrival rate; the Python/stdlib server is GIL-bound, so throughput does not grow with --clients
past a handful, and does grow with --shards (separate processes).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401

from isongraph_vector_server import auth  # noqa: E402
from isongraph_vector_server.router import rendezvous  # noqa: E402

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def request(host, port, method, path, key=None, body=None, conn=None, timeout=60):
    own = conn is None
    conn = conn or http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        payload = resp.read()
        return resp.status, (json.loads(payload) if payload else None)
    finally:
        if own:
            conn.close()


def wait_ready(host, port, proc=None, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            raise SystemExit("server exited: " + (proc.stderr.read().decode()[-1500:] if proc.stderr else ""))
        try:
            if request(host, port, "GET", "/readyz")[0] == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise SystemExit("server did not become ready")


def start_node(data_dir: str, port: int, args) -> subprocess.Popen:
    cfg = {
        "data_dir": data_dir, "host": "127.0.0.1", "port": port, "log_level": "CRITICAL",
        "autosave_interval_seconds": 0, "max_workers": max(32, args.clients + 8),
        "accept_queue": 128, "max_concurrent_requests": max(32, args.clients + 8),
        "failed_auth_per_minute": 10 ** 6,
        "encoder": {"kind": "mock", "dimension": args.dim},
        "quotas": {"rate_per_second": 10 ** 6, "rate_burst": 10 ** 6, "max_inflight": args.clients + 8,
                   "max_nodes": args.nodes * 2, "max_dimension": max(args.dim, 16)},
    }
    path = Path(data_dir).parent / f"cfg-{port}.json"
    path.write_text(json.dumps(cfg))
    boot = ("import sys; sys.path.insert(0, %r); import _bootstrap; "
            "from isongraph_vector_server.cli import main; sys.exit(main(['serve', '--config', %r]))"
            % (str(HERE), str(path)))
    return subprocess.Popen([sys.executable, "-c", boot], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def start_router(shards, port: int, args) -> subprocess.Popen:
    cfg = {"host": "127.0.0.1", "port": port, "log_level": "CRITICAL", "max_workers": max(32, args.clients + 8),
           "accept_queue": 128, "router": {"shards": shards, "upstream_timeout_seconds": 60}}
    path = Path(tempfile.gettempdir()) / f"igv-router-{port}.json"
    path.write_text(json.dumps(cfg))
    boot = ("import sys; sys.path.insert(0, %r); import _bootstrap; "
            "from isongraph_vector_server.cli import main; sys.exit(main(['router', '--config', %r]))"
            % (str(HERE), str(path)))
    return subprocess.Popen([sys.executable, "-c", boot], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def populate(host, port, key, graph, args, rnd: random.Random):
    s, r = request(host, port, "POST", "/v1/graphs", key, {"name": graph, "embedding": {"mode": "client", "dimension": args.dim}})
    assert s == 201, (s, r)
    batch = 400
    for start in range(0, args.nodes, batch):
        items = [{"type": "doc", "id": i, "properties": {"name": f"doc {i}", "bucket": i % 10},
                  "vector": [rnd.gauss(0, 1) for _ in range(args.dim)]}
                 for i in range(start, min(args.nodes, start + batch))]
        s, r = request(host, port, "POST", f"/v1/graphs/{graph}/nodes", key, {"nodes": items})
        assert s == 200, (s, r)


def _gen_proc(host, port, targets, args, begin_epoch):
    return run_load(host, port, targets, args, begin_epoch)


def percentile(sorted_ms, p):
    if not sorted_ms:
        return float("nan")
    k = min(len(sorted_ms) - 1, max(0, int(round(p / 100 * len(sorted_ms) + 0.5)) - 1))
    return sorted_ms[k]


def run_load(host, port, targets, args, begin_epoch=None):
    rnd = random.Random(args.seed)
    queries = [[rnd.gauss(0, 1) for _ in range(args.dim)] for _ in range(512)]
    texts = [f"doc {rnd.randrange(args.nodes)}" for _ in range(512)]
    stop_at = [0.0]
    measure_start = [0.0]
    results = []
    lock = threading.Lock()
    started = threading.Barrier(args.clients + 1)

    def worker(wid):
        r = random.Random(args.seed * 1000 + wid)
        conn = http.client.HTTPConnection(host, port, timeout=60)
        lat, errs, per_tenant = [], {}, {}
        started.wait()
        while time.perf_counter() < stop_at[0]:
            t_idx = r.randrange(len(targets))
            key, graph = targets[t_idx]["key"], targets[t_idx]["graph"]
            q = r.randrange(512)
            body = ({"query": texts[q], "top_k": args.top_k, "threshold": -1.0} if args.text
                    else {"vector": queries[q], "top_k": args.top_k, "threshold": -1.0})
            t0 = time.perf_counter()
            try:
                conn.request("POST", f"/v1/graphs/{graph}/search", body=json.dumps(body).encode(),
                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                resp = conn.getresponse()
                resp.read()
                status = resp.status
            except (OSError, http.client.HTTPException):
                status = -1
                conn.close()
                conn = http.client.HTTPConnection(host, port, timeout=60)
            now = time.perf_counter()
            dt = (now - t0) * 1000
            if t0 < measure_start[0]:
                continue                                  # warm-up sample: not counted
            if status == 200:
                lat.append(dt)
                per_tenant[t_idx] = per_tenant.get(t_idx, 0) + 1
            else:
                errs[status] = errs.get(status, 0) + 1
        conn.close()
        with lock:
            results.append((lat, errs, per_tenant))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(args.clients)]
    [t.start() for t in threads]
    if begin_epoch is None:
        begin_epoch = time.time() + 0.3
    begin = time.perf_counter() + max(0.0, begin_epoch - time.time())
    measure_start[0] = begin + args.warmup
    stop_at[0] = measure_start[0] + args.duration
    started.wait()
    [t.join() for t in threads]
    return results


def summarize(results, args, wall):
    all_lat, errs, per_tenant = [], {}, {}
    for lat, e, pt in results:
        all_lat += lat
        for k, v in e.items():
            errs[k] = errs.get(k, 0) + v
        for k, v in pt.items():
            per_tenant[k] = per_tenant.get(k, 0) + v
    all_lat.sort()
    n = len(all_lat)
    counts = sorted(per_tenant.values()) or [0]
    return {
        "requests_ok": n,
        "errors": {str(k): v for k, v in errs.items()},
        "throughput_rps": round(n / wall, 1),
        "latency_ms": {
            "mean": round(statistics.fmean(all_lat), 3) if n else None,
            "p50": round(percentile(all_lat, 50), 3), "p95": round(percentile(all_lat, 95), 3),
            "p99": round(percentile(all_lat, 99), 3), "max": round(all_lat[-1], 3) if n else None},
        "per_tenant_requests": {"min": counts[0], "max": counts[-1], "tenants_hit": len(per_tenant)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenants", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=2000, help="vectors per tenant graph")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--clients", type=int, default=16)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--mode", choices=["spawn", "inproc", "external"], default="spawn")
    ap.add_argument("--shards", type=int, default=0, help="spawn K nodes + a router (spawn mode)")
    ap.add_argument("--text", action="store_true", help="text queries (server encodes) instead of vectors")
    ap.add_argument("--url", help="external mode: base URL")
    ap.add_argument("--keys-file", help="external mode: JSON list of {key, graph}")
    ap.add_argument("--gen-procs", type=int, default=1,
                    help="load-generator processes (clients are split across them); use >1 to keep the "
                         "generator's own GIL from being the bottleneck")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="igv-load-"))
    procs, targets, server = [], [], None
    host, port = "127.0.0.1", 0
    try:
        if args.mode == "external":
            u = args.url.replace("http://", "").split(":")
            host, port = u[0], int(u[1])
            targets = json.loads(Path(args.keys_file).read_text())
        else:
            rnd = random.Random(args.seed)
            n_nodes = max(1, args.shards)
            data_dirs = [str(tmp / f"node{i}") for i in range(n_nodes)]
            names_ = [f"s{i}" for i in range(n_nodes)]
            quotas = {"rate_per_second": 10 ** 6, "rate_burst": 10 ** 6, "max_inflight": args.clients + 8,
                      "max_nodes": args.nodes * 2, "max_dimension": max(args.dim, 16)}
            tenant_owner = {}
            for t in range(args.tenants):
                tid = f"tenant-{t}"
                owner = rendezvous(tid, names_) if args.shards else "s0"
                idx = names_.index(owner)
                auth.create_tenant(data_dirs[idx], tid, quotas)
                targets.append({"tid": tid, "key": auth.add_key(data_dirs[idx], tid), "graph": "kb", "node": idx})
                tenant_owner[tid] = idx
            if args.mode == "inproc":
                from isongraph_vector_server.config import ServerConfig
                from isongraph_vector_server.server import Server
                assert not args.shards, "inproc supports a single node"
                cfg = ServerConfig(data_dir=data_dirs[0], host="127.0.0.1", port=0, log_level="CRITICAL")
                cfg.autosave_interval_seconds = 0
                cfg.max_workers = cfg.max_concurrent_requests = max(32, args.clients + 8)
                cfg.accept_queue = 128
                cfg.failed_auth_per_minute = 10 ** 6
                cfg.encoder.kind, cfg.encoder.dimension = "mock", args.dim
                cfg.quotas.rate_per_second = cfg.quotas.rate_burst = 10 ** 6
                cfg.quotas.max_inflight = args.clients + 8
                server = Server(cfg, configure_logging=False).start()
                ports = [server.address[1]]
            else:
                ports = [free_port() for _ in range(n_nodes)]
                for dd, p in zip(data_dirs, ports):
                    procs.append(start_node(dd, p, args))
                for p, proc in zip(ports, procs):
                    wait_ready(host, p, proc)
            # populate each tenant directly on its node
            print(f"populating {args.tenants} tenants x {args.nodes} vectors (dim {args.dim}) ...", flush=True)
            for t in targets:
                p = ports[t["node"]]
                if args.text:
                    s, r = request(host, p, "POST", "/v1/graphs", t["key"], {"name": "kb"})
                    assert s == 201, r
                    for start in range(0, args.nodes, 400):
                        items = [{"type": "doc", "id": i, "properties": {"name": f"doc {i}"}}
                                 for i in range(start, min(args.nodes, start + 400))]
                        s, r = request(host, p, "POST", "/v1/graphs/kb/nodes", t["key"], {"nodes": items})
                        assert s == 200, r
                else:
                    populate(host, p, t["key"], "kb", args, rnd)
            if args.shards:
                port = free_port()
                shards = [{"name": names_[i], "primary": f"http://127.0.0.1:{ports[i]}"} for i in range(n_nodes)]
                procs.append(start_router(shards, port, args))
                for _ in range(100):
                    try:
                        if request(host, port, "GET", "/healthz")[0] == 200:
                            break
                    except OSError:
                        time.sleep(0.1)
            else:
                port = ports[0]

        print(f"load: {args.clients} clients, {len(targets)} tenants, {args.warmup}s warm-up + "
              f"{args.duration}s measured, {'text' if args.text else 'vector'} queries, top_k={args.top_k}", flush=True)
        if args.gen_procs > 1:
            import copy
            import multiprocessing as mp
            per = copy.copy(args)
            per.clients = max(1, -(-args.clients // args.gen_procs))
            jobs = []
            begin = time.time() + 3.0
            for i in range(args.gen_procs):
                a = copy.copy(per)
                a.seed = args.seed + i
                jobs.append((host, port, targets, a, begin))
            with mp.get_context("spawn").Pool(args.gen_procs) as pool:
                results = [x for part in pool.starmap(_gen_proc, jobs) for x in part]
        else:
            results = run_load(host, port, targets, args)
        summary = summarize(results, args, args.duration)
        report = {
            "config": {k: v for k, v in vars(args).items() if k not in ("out", "keys_file")},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "cpu_count": os.cpu_count(), "server_processes": max(1, args.shards)},
            "result": summary,
        }
        r = summary
        print()
        print(f"  throughput      {r['throughput_rps']:>10.1f} req/s   (measured window only)")
        lat = r["latency_ms"]
        print(f"  latency ms      mean {lat['mean']}  p50 {lat['p50']}  p95 {lat['p95']}  p99 {lat['p99']}  max {lat['max']}")
        print(f"  ok / errors     {r['requests_ok']} / {r['errors'] or 0}")
        print(f"  per-tenant      min {r['per_tenant_requests']['min']}  max {r['per_tenant_requests']['max']}  "
              f"({r['per_tenant_requests']['tenants_hit']} tenants hit)")
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))
            print(f"  wrote {args.out}")
        return 0
    finally:
        if server is not None:
            server.stop()
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
