"""igv-server command line: run nodes and the router, manage tenants and keys.

Tenant and key management is a *local, offline* operation on the data
directory (it edits ``tenant.json``); there is deliberately no HTTP admin
API, so there is no remotely reachable credential that can mint other
credentials. A running server notices the change within a second.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import auth, names
from .config import ConfigError, load_config
from .router import rendezvous


def _data_dir(args) -> str:
    return args.data_dir or os.environ.get("IGV_DATA_DIR") or "./igv-data"


def _parse_quotas(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"quota must look like name=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            raise SystemExit(f"quota value for {k} is not a number: {v!r}") from None
    return out


def _wait_for_signal(stop: threading.Event) -> None:
    def handler(signum, frame):
        stop.set()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
    while not stop.wait(0.5):
        pass


def cmd_serve(args) -> int:
    from .server import Server
    cfg = load_config(args.config)
    for attr, key in (("host", "host"), ("port", "port"), ("data_dir", "data_dir"), ("role", "role")):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(cfg, key, value)
    cfg.validate()
    server = Server(cfg).start()
    stop = threading.Event()
    _wait_for_signal(stop)
    server.stop()
    return 0


def cmd_router(args) -> int:
    from .server import RouterServer
    cfg = load_config(args.config)
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    server = RouterServer(cfg).start()
    stop = threading.Event()
    _wait_for_signal(stop)
    server.stop()
    return 0


def cmd_tenant(args) -> int:
    d = _data_dir(args)
    if args.action == "create":
        auth.create_tenant(d, args.tenant_id, _parse_quotas(args.quota))
        print(f"created tenant {args.tenant_id}")
    elif args.action == "list":
        base = Path(d) / "tenants"
        for entry in sorted(os.scandir(base), key=lambda e: e.name) if base.is_dir() else []:
            if names.is_tenant_id(entry.name) and (Path(entry.path) / "tenant.json").is_file():
                info = auth.describe_tenant(d, entry.name)
                active = sum(1 for k in info["keys"] if k["revoked_at"] is None)
                print(f"{entry.name}\t{'disabled' if info['disabled'] else 'active'}\t{active} key(s)")
    elif args.action == "show":
        print(json.dumps(auth.describe_tenant(d, args.tenant_id), indent=2))
    elif args.action in ("disable", "enable"):
        auth.set_disabled(d, args.tenant_id, args.action == "disable")
        print(f"{args.action}d tenant {args.tenant_id}")
    elif args.action == "set-quota":
        auth.set_quotas(d, args.tenant_id, _parse_quotas(args.quota))
        print("quotas updated")
    return 0


def cmd_key(args) -> int:
    d = _data_dir(args)
    if args.action == "create":
        token = auth.add_key(d, args.tenant_id, args.role, args.label or "")
        print("API key (shown once; only a salted hash is stored):", file=sys.stderr)
        print(token)
    elif args.action == "revoke":
        auth.revoke_key(d, args.tenant_id, args.key_id)
        print(f"revoked key {args.key_id}")
    return 0


def cmd_shard_of(args) -> int:
    cfg = load_config(args.config)
    shard_names = [s["name"] for s in cfg.router.shards]
    pinned = cfg.router.pins.get(args.tenant_id)
    print(pinned or rendezvous(args.tenant_id, shard_names))
    return 0


def cmd_verify(args) -> int:
    """Offline integrity check of every graph under a data dir. Exit 1 if any fails."""
    from .storage import GraphStorage, SnapshotError
    base = Path(_data_dir(args)) / "tenants"
    bad = 0
    total = 0
    for t in sorted(os.scandir(base), key=lambda e: e.name) if base.is_dir() else []:
        if not names.is_tenant_id(t.name):
            continue
        gdir = Path(t.path) / "graphs"
        for g in sorted(os.scandir(gdir), key=lambda e: e.name) if gdir.is_dir() else []:
            if not (g.name.startswith("g-") and names.is_graph_name(g.name[2:])):
                continue
            total += 1
            st = GraphStorage(Path(g.path))
            try:
                meta = st.read_meta()
                from .encoders import build_encoder
                graph, manifest, degraded, _ = st.load(meta, None)
                nodes = graph.node_count()
                graph.close()
                status = "DEGRADED (older generation)" if degraded else "ok"
                if degraded:
                    bad += 1
                print(f"{t.name}/{g.name[2:]}\t{status}\t{nodes} nodes\tgeneration "
                      f"{manifest.generation if manifest else 0}")
            except (SnapshotError, OSError) as exc:
                bad += 1
                print(f"{t.name}/{g.name[2:]}\tFAILED\t{exc}")
    print(f"{total} graph(s) checked, {bad} problem(s)")
    return 1 if bad else 0


def cmd_replica_sync(args) -> int:
    """Run one replica sync cycle and exit (cron-friendly)."""
    from .auth import TenantRegistry
    from .config import ServerConfig
    from .engine import Engine
    from .encoders import build_encoder
    from .metrics import Metrics
    from .replication import ReplicaSyncer
    cfg = load_config(args.config)
    cfg.role = "replica"
    cfg.validate()
    metrics = Metrics()
    engine = Engine(cfg, build_encoder(cfg.encoder), metrics)
    engine.start()
    try:
        syncer = ReplicaSyncer(cfg, engine, TenantRegistry(cfg.data_dir, cfg.quotas), metrics)
        report = syncer.sync_once()
        print(json.dumps(report, indent=2))
        return 0 if not report["errors"] and not report["refused"] else 1
    finally:
        engine.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="igv-server", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run a storage node (primary or replica)")
    s.add_argument("--config")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--data-dir", dest="data_dir")
    s.add_argument("--role", choices=["primary", "replica"])
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("router", help="run the stateless tenant-sharding router")
    r.add_argument("--config", required=True)
    r.add_argument("--host")
    r.add_argument("--port", type=int)
    r.set_defaults(fn=cmd_router)

    t = sub.add_parser("tenant", help="manage tenants (offline, on the data dir)")
    t.add_argument("action", choices=["create", "list", "show", "disable", "enable", "set-quota"])
    t.add_argument("tenant_id", nargs="?")
    t.add_argument("--quota", action="append", metavar="NAME=VALUE")
    t.add_argument("--data-dir", dest="data_dir")
    t.set_defaults(fn=cmd_tenant)

    k = sub.add_parser("key", help="create or revoke API keys")
    k.add_argument("action", choices=["create", "revoke"])
    k.add_argument("tenant_id")
    k.add_argument("key_id", nargs="?")
    k.add_argument("--role", choices=["rw", "ro"], default="rw")
    k.add_argument("--label")
    k.add_argument("--data-dir", dest="data_dir")
    k.set_defaults(fn=cmd_key)

    sh = sub.add_parser("shard-of", help="which router shard owns a tenant")
    sh.add_argument("tenant_id")
    sh.add_argument("--config", required=True)
    sh.set_defaults(fn=cmd_shard_of)

    v = sub.add_parser("verify", help="offline integrity check of every snapshot in a data dir")
    v.add_argument("--data-dir", dest="data_dir")
    v.set_defaults(fn=cmd_verify)

    rs = sub.add_parser("replica-sync", help="run one replica sync cycle and exit")
    rs.add_argument("--config", required=True)
    rs.set_defaults(fn=cmd_replica_sync)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd in ("tenant", "key") and args.action not in ("list",) and not args.tenant_id:
            raise SystemExit("a tenant id is required")
        if args.cmd == "key" and args.action == "revoke" and not args.key_id:
            raise SystemExit("a key id is required")
        return args.fn(args)
    except (ConfigError, ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":     # pragma: no cover
    sys.exit(main())
