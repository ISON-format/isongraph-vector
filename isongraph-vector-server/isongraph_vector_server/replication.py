"""Snapshot-based read replicas.

What a replica is: a second server process, started with ``role = "replica"``,
with its own data directory, that periodically *pulls* the primary's published
snapshot generations and serves them read-only. What it is not: a hot standby.
There is no write-ahead log, no streaming, no promotion and no failover.

One sync cycle:

 1. fetch the primary's index (every tenant, every graph's current
    generation and the sha256 of its MANIFEST)
 2. for each tenant file / graph whose generation differs from the local one,
    download MANIFEST, graph file and embeddings sidecar into a staging
    directory
 3. verify, in order: the manifest against the index entry; each file's size
    and sha256 against the manifest; the library's sidecar-vs-graph digest
    check; the loaded graph's digest against the manifest's; the manifest's
    digest against the index's
 4. only then install the generation, flip the replica's CURRENT and swap the
    new in-memory graph in. Any failure leaves the replica serving what it had
    ("refused"), and is counted and logged.

Consistency, plainly: a replica serves a *saved generation*, not the primary's
live memory, so a read can be stale by up to
``primary autosave interval + replica sync interval + sync duration``.
Different graphs can be at different primary moments (generations are
per-graph; there is no cross-graph snapshot). If the last fully successful
sync is older than ``max_staleness_seconds`` the replica stops serving
(503 ``replica_stale``) instead of serving unboundedly old data. Revoking a
key or disabling a tenant on the primary reaches replicas on the next sync.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import auth, names
from .config import ServerConfig
from .engine import Engine
from .errors import Unavailable
from .logs import log
from .metrics import Metrics
from .storage import (GRAPH_FILE, MANIFEST_FILE, SIDECAR_FILE, GraphMeta, GraphStorage, Manifest,
                      SnapshotError, atomic_write, new_semantic_graph, sha256_file)

STATE_FILE = "REPLICA_STATE.json"


class ReplicationError(Exception):
    """The primary could not be reached or answered badly (not a verification failure)."""


# -- the index (built by the primary, or read from a directory) -----------------------------------------


def build_index(data_dir: str) -> Dict[str, Any]:
    base = Path(data_dir) / "tenants"
    tenants: List[Dict[str, Any]] = []
    if base.is_dir():
        for entry in sorted(os.scandir(base), key=lambda e: e.name):
            if not (entry.is_dir(follow_symlinks=False) and names.is_tenant_id(entry.name)):
                continue
            tf = Path(entry.path) / "tenant.json"
            try:
                tsha, _ = sha256_file(tf)
            except OSError:
                continue
            graphs = []
            gdir = Path(entry.path) / "graphs"
            if gdir.is_dir():
                for g in sorted(os.scandir(gdir), key=lambda e: e.name):
                    if not (g.is_dir(follow_symlinks=False) and g.name.startswith("g-")
                            and names.is_graph_name(g.name[2:])):
                        continue
                    graphs.append(_graph_entry(Path(g.path), g.name[2:]))
            tenants.append({"tenant_id": entry.name, "tenant_sha256": tsha, "graphs": graphs})
    return {"format": "igv-replication-index", "version": 1, "generated_at": time.time(),
            "tenants": tenants}


def _graph_entry(path: Path, name: str) -> Dict[str, Any]:
    st = GraphStorage(path)
    entry: Dict[str, Any] = {"name": name, "generation": None, "manifest_sha256": None,
                             "graph_digest": None, "meta_sha256": None}
    try:
        entry["meta_sha256"], _ = sha256_file(path / "meta.json")
        cur = st.read_current()
        if cur is not None:
            mpath = st.gen_path(cur) / MANIFEST_FILE
            entry["manifest_sha256"], _ = sha256_file(mpath)
            entry["graph_digest"] = json.loads(mpath.read_text(encoding="utf-8")).get("graph_digest")
            entry["generation"] = cur
    except (OSError, ValueError):
        # Do NOT drop the entry: a replica must not mistake "unreadable right now"
        # for "deleted on the primary".
        entry["error"] = "unreadable"
    return entry


# -- sources -----------------------------------------------------------------------------------------------


class DirSource:
    """Read the primary's data directory directly (shared volume, NFS, rsync target)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def index(self) -> Dict[str, Any]:
        if not (self.root / "tenants").is_dir():
            raise ReplicationError("source directory has no tenants/")
        return build_index(str(self.root))

    def _copy(self, src: Path, dest: Path, max_bytes: int) -> None:
        try:
            if src.stat().st_size > max_bytes:
                raise ReplicationError("file larger than the allowed maximum")
            with open(src, "rb") as r, open(dest, "wb") as w:
                shutil.copyfileobj(r, w, 1 << 20)
        except OSError as exc:
            raise ReplicationError(f"cannot read source file ({type(exc).__name__})") from None

    def fetch_tenant(self, tid: str, dest: Path, max_bytes: int) -> None:
        self._copy(names.tenant_dir(self.root, tid) / "tenant.json", dest, max_bytes)

    def fetch_meta(self, tid: str, g: str, dest: Path, max_bytes: int) -> None:
        self._copy(names.graph_dir(self.root, tid, g) / "meta.json", dest, max_bytes)

    def fetch_file(self, tid: str, g: str, gen: int, fname: str, dest: Path, max_bytes: int) -> None:
        self._copy(names.graph_dir(self.root, tid, g) / f"gen-{gen:06d}" / fname, dest, max_bytes)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class HttpSource:
    """Pull from a primary's ``/_repl/v1`` endpoints with the replication token."""

    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _open(self, path: str):
        req = urllib.request.Request(self.base + path, headers={"Authorization": f"Bearer {self.token}"})
        try:
            return self._opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise ReplicationError(f"primary answered HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise ReplicationError(f"primary unreachable ({type(exc).__name__})") from None

    def index(self) -> Dict[str, Any]:
        with self._open("/_repl/v1/index") as resp:
            data = resp.read(64 * 1024 * 1024)
        try:
            return json.loads(data)
        except ValueError:
            raise ReplicationError("primary sent an unreadable index") from None

    def _download(self, path: str, dest: Path, max_bytes: int) -> None:
        got = 0
        with self._open(path) as resp, open(dest, "wb") as out:
            declared = resp.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise ReplicationError("file larger than the allowed maximum")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if got > max_bytes:
                    raise ReplicationError("file larger than the allowed maximum")
                out.write(chunk)
            if declared is not None and int(declared) != got:
                raise ReplicationError("truncated download")

    def fetch_tenant(self, tid: str, dest: Path, max_bytes: int) -> None:
        self._download(f"/_repl/v1/tenant/{tid}", dest, max_bytes)

    def fetch_meta(self, tid: str, g: str, dest: Path, max_bytes: int) -> None:
        self._download(f"/_repl/v1/meta/{tid}/{g}", dest, max_bytes)

    def fetch_file(self, tid: str, g: str, gen: int, fname: str, dest: Path, max_bytes: int) -> None:
        self._download(f"/_repl/v1/file/{tid}/{g}/{gen:06d}/{fname}", dest, max_bytes)


def make_source(cfg: ServerConfig):
    src = cfg.replica.source
    if src.startswith("dir:"):
        return DirSource(src[4:])
    if src.startswith(("http://", "https://")):
        if not cfg.replication_token:
            raise ValueError("IGV_REPLICATION_TOKEN is required to replicate from an HTTP primary")
        return HttpSource(src, cfg.replication_token, cfg.replica.fetch_timeout_seconds)
    raise ValueError("replica.source must be 'dir:<path>' or an http(s):// URL")


# -- the replica -----------------------------------------------------------------------------------------------


class ReplicaSyncer:
    def __init__(self, cfg: ServerConfig, engine: Engine, registry: auth.TenantRegistry,
                 metrics: Metrics, source=None):
        self.cfg = cfg
        self.engine = engine
        self.registry = registry
        self.metrics = metrics
        self.source = source or make_source(cfg)
        self.data_dir = Path(cfg.data_dir)
        self.last_ok_wall: Optional[float] = None
        self.last_error: Optional[str] = None
        self.syncs_ok = 0
        self.syncs_failed = 0
        self.refused = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load_state()

    # state that survives a restart, so staleness is measured honestly across one
    def _load_state(self) -> None:
        try:
            data = json.loads((self.data_dir / STATE_FILE).read_text(encoding="utf-8"))
            self.last_ok_wall = float(data["last_ok_wall"])
        except (OSError, ValueError, KeyError, TypeError):
            self.last_ok_wall = None

    def _save_state(self) -> None:
        try:
            atomic_write(self.data_dir / STATE_FILE,
                         json.dumps({"last_ok_wall": self.last_ok_wall}).encode("utf-8"))
        except OSError:
            pass

    # -- serving gate ----------------------------------------------------------------------------------

    def staleness_seconds(self) -> Optional[float]:
        return None if self.last_ok_wall is None else max(0.0, time.time() - self.last_ok_wall)

    def is_ready(self) -> bool:
        age = self.staleness_seconds()
        return age is not None and age <= self.cfg.replica.max_staleness_seconds

    def check_fresh(self) -> None:
        age = self.staleness_seconds()
        if age is None:
            raise Unavailable("replica has not completed a sync yet", code="replica_not_synced",
                              headers={"Retry-After": "5"})
        if age > self.cfg.replica.max_staleness_seconds:
            raise Unavailable("replica data is older than its staleness bound; "
                              "use the primary or wait for the replica to resync",
                              code="replica_stale", headers={"Retry-After": "5"})

    def metrics_snapshot(self) -> Dict[str, float]:
        age = self.staleness_seconds()
        return {
            "igv_replica_staleness_seconds": -1 if age is None else age,
            "igv_replica_syncs_ok": self.syncs_ok,
            "igv_replica_syncs_failed": self.syncs_failed,
            "igv_replica_refused_snapshots": self.refused,
        }

    # -- one cycle ----------------------------------------------------------------------------------------

    def sync_once(self) -> Dict[str, Any]:
        report = {"updated": [], "refused": [], "errors": [], "removed": []}
        with self._lock:
            try:
                index = self.source.index()
                if index.get("format") != "igv-replication-index" or index.get("version") != 1:
                    raise ReplicationError("unrecognised index format")
                tenants = index["tenants"]
                if not isinstance(tenants, list):
                    raise ReplicationError("malformed index")
            except (ReplicationError, KeyError, TypeError) as exc:
                self.syncs_failed += 1
                self.last_error = str(exc)[:200]
                self.metrics.inc("igv_replica_sync_failures_total")
                log(logging.ERROR, "replica sync failed: cannot read the primary's index",
                    error=self.last_error)
                report["errors"].append("index")
                return report

            wanted_tenants = set()
            clean = True
            for t in tenants:
                tid = t.get("tenant_id") if isinstance(t, dict) else None
                if not names.is_tenant_id(tid):
                    continue
                wanted_tenants.add(tid)
                clean &= self._sync_tenant(tid, t, report)
            clean &= self._remove_gone(wanted_tenants, tenants, report)
            self.registry.rescan(force=True)

            if clean:
                self.last_ok_wall = time.time()
                now = self.last_ok_wall
                for rt in list(self.engine.tenants.values()):
                    for h in list(rt.graphs.values()):
                        if h.graph is not None:
                            h.synced_at = now
                self.syncs_ok += 1
                self.last_error = None
                self._save_state()
                self.metrics.inc("igv_replica_syncs_total")
            else:
                self.syncs_failed += 1
            return report

    def _fail(self, report, kind: str, tid: str, g: Optional[str], exc: Exception, refused: bool) -> None:
        self.last_error = f"{tid}/{g}: {type(exc).__name__}"
        key = "refused" if refused else "errors"
        report[key].append(f"{tid}/{g}" if g else tid)
        if refused:
            self.refused += 1
            self.metrics.inc("igv_replica_refused_total")
        else:
            self.metrics.inc("igv_replica_sync_failures_total")
        log(logging.ERROR, f"replica {kind}: keeping the previous version", tenant=tid, graph=g,
            reason=str(exc)[:300])

    def _sync_tenant(self, tid: str, entry: Dict[str, Any], report) -> bool:
        ok = True
        tdir = names.tenant_dir(self.data_dir, tid)
        tfile = tdir / "tenant.json"
        want = entry.get("tenant_sha256")
        try:
            have = sha256_file(tfile)[0] if tfile.is_file() else None
        except OSError:
            have = None
        if want and have != want:
            stage = self.data_dir / f".stage-{uuid.uuid4().hex[:12]}"
            try:
                stage.mkdir(parents=True, exist_ok=True)
                tmp = stage / "tenant.json"
                self.source.fetch_tenant(tid, tmp, 1 << 20)
                if sha256_file(tmp)[0] != want:
                    raise SnapshotError("tenant file does not match the index checksum")
                auth.parse_tenant_file(json.loads(tmp.read_text(encoding="utf-8")), tid)
                (tdir / "graphs").mkdir(parents=True, exist_ok=True)
                atomic_write(tfile, tmp.read_bytes())
                report["updated"].append(f"{tid}/tenant.json")
            except (SnapshotError, ValueError) as exc:
                self._fail(report, "refused a tenant file", tid, None, exc, refused=True)
                ok = False
            except ReplicationError as exc:
                self._fail(report, "tenant fetch failed", tid, None, exc, refused=False)
                ok = False
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        if not tfile.is_file():
            return False
        rt = self.engine.runtime(tid, self.cfg.quotas)
        gdir = tdir / "graphs"
        wanted = set()
        for g in entry.get("graphs", []):
            name = g.get("name") if isinstance(g, dict) else None
            if not names.is_graph_name(name):
                continue
            wanted.add(name)
            if "error" in g:
                ok = False
                report["errors"].append(f"{tid}/{name}")
                continue
            ok &= self._sync_graph(rt, gdir, tid, name, g, report)
        # graphs the primary no longer has
        for name in list(rt.graphs):
            if name not in wanted:
                self.engine.drop_graph_local(rt, name)
                report["removed"].append(f"{tid}/{name}")
        return ok

    def _remove_gone(self, wanted_tenants, tenants, report) -> bool:
        base = self.data_dir / "tenants"
        if not base.is_dir():
            return True
        for entry in os.scandir(base):
            if entry.is_dir(follow_symlinks=False) and names.is_tenant_id(entry.name) \
                    and entry.name not in wanted_tenants:
                rt = self.engine.tenants.get(entry.name)
                if rt is not None:
                    for name in list(rt.graphs):
                        self.engine.drop_graph_local(rt, name)
                shutil.rmtree(entry.path, ignore_errors=True)
                report["removed"].append(entry.name)
        return True

    def _local_state(self, gpath: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads((gpath / "REPLICA.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _sync_graph(self, rt, gdir: Path, tid: str, name: str, g: Dict[str, Any], report) -> bool:
        gpath = gdir / names.graph_dirname(name)
        wanted = (g.get("generation"), g.get("manifest_sha256"), g.get("meta_sha256"))
        state = self._local_state(gpath)
        handle = rt.graphs.get(name)
        if (state is not None and handle is not None and handle.graph is not None
                and (state.get("generation"), state.get("manifest_sha256"),
                     state.get("meta_sha256")) == wanted):
            return True
        stage_root = gdir / f".tmp-sync-{uuid.uuid4().hex[:12]}"
        try:
            stage_root.mkdir(parents=True, exist_ok=True)
            meta_path = stage_root / "meta.json"
            self.source.fetch_meta(tid, name, meta_path, 1 << 20)
            if sha256_file(meta_path)[0] != g.get("meta_sha256"):
                raise SnapshotError("meta.json does not match the index checksum")
            meta = GraphMeta.from_json(json.loads(meta_path.read_text(encoding="utf-8")), name)

            gen = g.get("generation")
            if gen is None:
                graph, manifest, stage_gen = new_semantic_graph(meta, self.engine.encoder), None, None
            else:
                if not isinstance(gen, int):
                    raise SnapshotError("bad generation in index")
                stage_gen = stage_root / f"gen-{gen:06d}"
                stage_gen.mkdir()
                self.source.fetch_file(tid, name, gen, MANIFEST_FILE, stage_gen / MANIFEST_FILE, 1 << 20)
                if sha256_file(stage_gen / MANIFEST_FILE)[0] != g.get("manifest_sha256"):
                    raise SnapshotError("MANIFEST does not match the index checksum")
                m = Manifest.from_json(json.loads((stage_gen / MANIFEST_FILE).read_text(encoding="utf-8")))
                if m.generation != gen:
                    raise SnapshotError("manifest generation does not match the index")
                if g.get("graph_digest") and m.graph_digest != g["graph_digest"]:
                    raise SnapshotError("manifest digest does not match the index")
                cap = self.cfg.replica.max_file_bytes
                for fname in (GRAPH_FILE, SIDECAR_FILE):
                    self.source.fetch_file(tid, name, gen, fname, stage_gen / fname,
                                           min(cap, m.files[fname]["size"]))
                storage_probe = GraphStorage(stage_root, self.cfg.keep_generations)
                graph, manifest, caught = storage_probe.load_generation(stage_gen, meta, self.engine.encoder)
                for w in caught:
                    log(logging.WARNING, "library warning while loading a replicated snapshot",
                        tenant=tid, graph=name, warning=w)
        except (SnapshotError, ValueError, KeyError, TypeError) as exc:
            shutil.rmtree(stage_root, ignore_errors=True)
            self._fail(report, "refused a snapshot (verification failed)", tid, name, exc, refused=True)
            return False
        except ReplicationError as exc:
            shutil.rmtree(stage_root, ignore_errors=True)
            self._fail(report, "snapshot fetch failed", tid, name, exc, refused=False)
            return False
        except Exception as exc:      # pragma: no cover - defensive
            shutil.rmtree(stage_root, ignore_errors=True)
            self._fail(report, "snapshot sync error", tid, name, exc, refused=False)
            return False

        # Verified. Install it and swap it in.
        try:
            gpath.mkdir(parents=True, exist_ok=True)
            atomic_write(gpath / "meta.json", meta_path.read_bytes())
            storage = GraphStorage(gpath, self.cfg.keep_generations)
            if stage_gen is not None:
                storage.install_generation(stage_gen, gen)
            else:
                try:
                    (gpath / "CURRENT").unlink()
                except OSError:
                    pass
            atomic_write(gpath / "REPLICA.json", json.dumps(
                {"generation": gen, "manifest_sha256": g.get("manifest_sha256"),
                 "meta_sha256": g.get("meta_sha256")}).encode("utf-8"))
            self.engine.swap_graph(rt, name, meta, storage, graph, manifest)
            report["updated"].append(f"{tid}/{name}@{gen}")
            self.metrics.inc("igv_replica_installs_total")
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        return True

    # -- background thread ------------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="igv-replica-sync", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync_once()
            except Exception as exc:   # pragma: no cover - defensive
                self.syncs_failed += 1
                log(logging.ERROR, "replica sync crashed", error=str(exc)[:200])
            if self._stop.wait(self.cfg.replica.sync_interval_seconds):
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
