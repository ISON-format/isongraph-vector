"""In-memory state: tenants, their graphs, quotas, and the save/load lifecycle.

Isolation is structural. There is no process-wide map keyed by a string that
mixes tenant and graph names: a :class:`TenantRuntime` owns its own
``graphs`` dict, handlers receive the runtime of the *authenticated* tenant
only, and the only way to reach a graph is ``tenant.graphs[name]``. A request
cannot name another tenant's runtime because no request field selects one.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from isongraph_vector import EmbeddingEncoder, SemanticGraph

from . import names
from .config import Quotas, ServerConfig
from .errors import (Conflict, DeadlineExceeded, NotFound, QuotaExceeded, Unavailable)
from .limits import Gate, LockTimeout, RWLock, TokenBucket
from .logs import log
from .metrics import Metrics
from .storage import (GraphMeta, GraphStorage, SnapshotError, acquire_dir_lock, new_semantic_graph,
                      serialize)


@dataclass
class OpContext:
    """Deadline bookkeeping for one request."""
    deadline: float
    lock_timeout: float

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def check(self) -> None:
        if self.remaining() <= 0:
            raise DeadlineExceeded("request deadline exceeded")

    def lock_wait(self) -> float:
        left = self.remaining()
        if left <= 0:
            raise DeadlineExceeded("request deadline exceeded")
        return min(self.lock_timeout, left)


class GraphHandle:
    """One graph of one tenant: the library object plus its lock and bookkeeping."""

    def __init__(self, name: str, meta: Optional[GraphMeta], storage: Optional[GraphStorage],
                 graph: Optional[SemanticGraph]):
        self.name = name
        self.meta = meta
        self.storage = storage
        self.graph = graph
        self.lock = RWLock()
        self.save_lock = threading.Lock()
        self.version = 0              # bumped (under the write lock) by every mutation
        self.saved_version = 0
        self.generation = 0
        self.manifest = None
        self.degraded = False
        self.quarantined: Optional[str] = None
        self.deleted = False
        self.last_save_ts: Optional[float] = None
        self.last_save_error: Optional[str] = None
        self._cache_lock = threading.Lock()
        self._sorted_nodes = None     # (version, [keys])
        self._sorted_edges = None
        self._digest_cache = None     # (version, digest)
        self.synced_at: Optional[float] = None   # replicas: wall time of the last verified sync

    @property
    def dirty(self) -> bool:
        return self.version != self.saved_version

    def bump(self) -> None:
        """Record a mutation. Call with the write lock held."""
        self.version += 1


class TenantRuntime:
    def __init__(self, tenant_id: str, quotas: Quotas):
        self.tenant_id = tenant_id
        self.graphs: Dict[str, GraphHandle] = {}
        self.glock = threading.Lock()          # guards the graphs dict and creating-state
        self._creating: set = set()
        self.quota_lock = threading.Lock()
        self.used_nodes = 0
        self.used_edges = 0
        self.bucket = TokenBucket(quotas.rate_per_second, quotas.rate_burst)
        self._rate_cfg = (quotas.rate_per_second, quotas.rate_burst)
        self.gate = Gate(quotas.max_inflight)
        self.requests = 0
        self.rate_limited = 0

    def apply_quotas(self, quotas: Quotas) -> None:
        cfg = (quotas.rate_per_second, quotas.rate_burst)
        if cfg != self._rate_cfg:
            self.bucket.reconfigure(*cfg)
            self._rate_cfg = cfg
        if self.gate.size != quotas.max_inflight:
            self.gate = Gate(quotas.max_inflight)

    # quota accounting: reserve before applying, reconcile after.
    def reserve(self, quotas: Quotas, nodes: int = 0, edges: int = 0) -> None:
        with self.quota_lock:
            if nodes > 0 and self.used_nodes + nodes > quotas.max_nodes:
                raise QuotaExceeded(
                    f"node quota of {quotas.max_nodes} would be exceeded "
                    f"({self.used_nodes} used, {nodes} requested)")
            if edges > 0 and self.used_edges + edges > quotas.max_edges:
                raise QuotaExceeded(
                    f"edge quota of {quotas.max_edges} would be exceeded "
                    f"({self.used_edges} used, {edges} requested)")
            self.used_nodes += nodes
            self.used_edges += edges

    def adjust(self, nodes: int = 0, edges: int = 0) -> None:
        with self.quota_lock:
            self.used_nodes = max(0, self.used_nodes + nodes)
            self.used_edges = max(0, self.used_edges + edges)


class Engine:
    def __init__(self, cfg: ServerConfig, encoder: Optional[EmbeddingEncoder], metrics: Metrics):
        self.cfg = cfg
        self.encoder = encoder
        self.metrics = metrics
        self.data_dir = Path(cfg.data_dir)
        self.tenants: Dict[str, TenantRuntime] = {}
        self._tenants_lock = threading.Lock()
        self.ready = False
        self._stop = threading.Event()
        self._autosave_thread: Optional[threading.Thread] = None
        self._dir_lock = None
        self.startup_report: Dict[str, Any] = {"loaded": 0, "quarantined": 0, "degraded": 0}

    # -- tenants ---------------------------------------------------------------------

    def runtime(self, tenant_id: str, quotas: Quotas) -> TenantRuntime:
        with self._tenants_lock:
            rt = self.tenants.get(tenant_id)
            if rt is None:
                rt = self.tenants[tenant_id] = TenantRuntime(tenant_id, quotas)
        rt.apply_quotas(quotas)
        return rt

    def _graphs_parent(self, tenant_id: str) -> Path:
        return names.tenant_dir(self.data_dir, tenant_id) / "graphs"

    # -- startup -----------------------------------------------------------------------

    def start(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tenants").mkdir(exist_ok=True)
        self._dir_lock = acquire_dir_lock(self.data_dir)
        self.load_all()
        self.ready = True
        if self.cfg.autosave_interval_seconds > 0 and self.cfg.role == "primary":
            self._autosave_thread = threading.Thread(
                target=self._autosave_loop, name="igv-autosave", daemon=True)
            self._autosave_thread.start()

    def load_all(self) -> None:
        base = self.data_dir / "tenants"
        for entry in sorted(os.scandir(base), key=lambda e: e.name):
            if not (entry.is_dir(follow_symlinks=False) and names.is_tenant_id(entry.name)):
                continue
            tid = entry.name
            parent = self._graphs_parent(tid)
            if not parent.is_dir():
                continue
            rt = self.runtime(tid, self.cfg.quotas)
            for g in sorted(os.scandir(parent), key=lambda e: e.name):
                if g.name.startswith((".tmp-", ".trash-")):
                    shutil.rmtree(g.path, ignore_errors=True)
                    continue
                if not (g.is_dir(follow_symlinks=False) and g.name.startswith("g-")
                        and names.is_graph_name(g.name[2:])):
                    continue
                handle = self._load_handle(tid, g.name[2:], Path(g.path))
                rt.graphs[handle.name] = handle
                if handle.graph is not None:
                    rt.used_nodes += handle.graph.node_count()
                    rt.used_edges += handle.graph.edge_count()
        self._update_gauges()

    def _load_handle(self, tid: str, name: str, path: Path) -> GraphHandle:
        storage = GraphStorage(path, self.cfg.keep_generations)
        storage.clean_tmp()
        try:
            meta = storage.read_meta()
            graph, manifest, degraded, caught = storage.load(meta, self.encoder)
        except (SnapshotError, OSError, ValueError) as exc:
            handle = GraphHandle(name, None, storage, None)
            handle.quarantined = f"{type(exc).__name__}"
            self.startup_report["quarantined"] += 1
            log(logging.ERROR, "graph failed verification and is quarantined",
                tenant=tid, graph=name, reason=str(exc)[:300])
            if self.cfg.strict_startup:
                raise SnapshotError(f"{tid}/{name}: {exc}") from None
            return handle
        handle = GraphHandle(name, meta, storage, graph)
        if manifest is not None:
            handle.generation = manifest.generation
            handle.manifest = manifest
        handle.degraded = degraded
        self.startup_report["loaded"] += 1
        if degraded:
            self.startup_report["degraded"] += 1
            log(logging.ERROR, "current generation failed verification; serving an older one",
                tenant=tid, graph=name, generation=handle.generation)
        for w in caught:
            log(logging.WARNING, "library warning while loading", tenant=tid, graph=name, warning=w)
        return handle

    def _update_gauges(self) -> None:
        q = deg = 0
        for rt in list(self.tenants.values()):
            for h in list(rt.graphs.values()):
                q += h.quarantined is not None
                deg += bool(h.degraded)
        self.metrics.gauge("igv_graphs_quarantined", q)
        self.metrics.gauge("igv_graphs_degraded", deg)

    # -- graph lifecycle ------------------------------------------------------------------

    def get_handle(self, rt: TenantRuntime, name: str, allow_quarantined: bool = False) -> GraphHandle:
        with rt.glock:
            handle = rt.graphs.get(name)
        if handle is None or handle.deleted:
            raise NotFound("graph not found")
        if handle.quarantined is not None and not allow_quarantined:
            raise Unavailable("graph failed integrity verification and is unavailable",
                              code="graph_quarantined")
        return handle

    def create_graph(self, rt: TenantRuntime, quotas: Quotas, meta: GraphMeta) -> GraphHandle:
        with rt.glock:
            if meta.name in rt.graphs or meta.name in rt._creating:
                raise Conflict("a graph with that name already exists", code="already_exists")
            if len(rt.graphs) + len(rt._creating) >= quotas.max_graphs:
                raise QuotaExceeded(f"graph quota of {quotas.max_graphs} reached")
            rt._creating.add(meta.name)
        try:
            storage = GraphStorage.create(self._graphs_parent(rt.tenant_id), meta,
                                          self.cfg.keep_generations)
            graph = new_semantic_graph(meta, self.encoder)
            handle = GraphHandle(meta.name, meta, storage, graph)
            with rt.glock:
                rt.graphs[meta.name] = handle
        finally:
            with rt.glock:
                rt._creating.discard(meta.name)
        return handle

    def delete_graph(self, rt: TenantRuntime, name: str, ctx: OpContext) -> None:
        with rt.glock:
            handle = rt.graphs.get(name)
            if handle is None:
                raise NotFound("graph not found")
            del rt.graphs[name]
        # Wait for in-flight operations, then keep newcomers out.
        try:
            handle.lock.acquire_write(ctx.lock_wait())
        except LockTimeout:
            with rt.glock:
                rt.graphs[name] = handle
            raise Unavailable("graph is busy; retry", code="busy", headers={"Retry-After": "1"}) from None
        try:
            handle.deleted = True
            nodes = edges = 0
            if handle.graph is not None:
                nodes, edges = handle.graph.node_count(), handle.graph.edge_count()
        finally:
            handle.lock.release_write()
        with handle.save_lock:
            if handle.storage is not None:
                handle.storage.remove()
        if handle.graph is not None:
            handle.graph.close()
        rt.adjust(nodes=-nodes, edges=-edges)
        self._update_gauges()

    # -- saving --------------------------------------------------------------------------------

    def save_graph(self, handle: GraphHandle, ctx: Optional[OpContext] = None, force: bool = False) -> bool:
        """Snapshot a dirty graph. Returns True if a generation was written."""
        if handle.graph is None or handle.storage is None:
            return False
        timeout = ctx.lock_wait() if ctx else self.cfg.lock_timeout_seconds
        with handle.save_lock:
            if handle.deleted:
                return False
            if not (handle.dirty or force):
                return False
            try:
                try:
                    handle.lock.acquire_read(timeout)
                except LockTimeout:
                    raise Unavailable("graph is busy; retry", code="busy",
                                      headers={"Retry-After": "1"}) from None
                try:
                    version = handle.version
                    image = serialize(handle.graph)
                finally:
                    handle.lock.release_read()
                manifest = handle.storage.write_generation(image)
            except Unavailable:
                raise
            except Exception as exc:
                handle.last_save_error = type(exc).__name__
                self.metrics.inc("igv_snapshot_failures_total")
                log(logging.ERROR, "snapshot failed", graph=handle.name, error=str(exc)[:300])
                raise
            handle.saved_version = version
            handle.generation = manifest.generation
            handle.manifest = manifest
            handle.degraded = False
            handle.last_save_ts = time.time()
            handle.last_save_error = None
            self.metrics.inc("igv_snapshots_total")
            return True

    def save_all(self) -> int:
        n = 0
        for rt in list(self.tenants.values()):
            for handle in list(rt.graphs.values()):
                if handle.graph is not None and handle.dirty and not handle.deleted:
                    try:
                        if self.save_graph(handle):
                            n += 1
                    except Exception:
                        pass    # already logged and counted
        return n

    def _autosave_loop(self) -> None:
        while not self._stop.wait(self.cfg.autosave_interval_seconds):
            try:
                self.save_all()
            except Exception as exc:   # pragma: no cover - defensive
                log(logging.ERROR, "autosave loop error", error=str(exc)[:200])

    # -- replica hook ------------------------------------------------------------------------------

    def swap_graph(self, rt: TenantRuntime, name: str, meta: GraphMeta, storage: GraphStorage,
                   graph: SemanticGraph, manifest) -> None:
        """Atomically replace (or create) a graph with a verified one. Replica sync only."""
        with rt.glock:
            handle = rt.graphs.get(name)
            if handle is None or handle.quarantined is not None:
                handle = GraphHandle(name, meta, storage, graph)
                rt.graphs[name] = handle
                old = None
                old_counts = (0, 0)
            else:
                old = handle.graph
                old_counts = (old.node_count(), old.edge_count()) if old is not None else (0, 0)
        if old is not None:
            handle.lock.acquire_write(self.cfg.lock_timeout_seconds * 3)
            try:
                handle.graph, handle.meta, handle.storage = graph, meta, storage
                handle.bump()
                handle.saved_version = handle.version
            finally:
                handle.lock.release_write()
            old.close()
        handle.generation = manifest.generation if manifest is not None else 0
        handle.manifest = manifest
        handle.degraded = False
        handle.quarantined = None
        handle.synced_at = time.time()
        rt.adjust(nodes=graph.node_count() - old_counts[0], edges=graph.edge_count() - old_counts[1])
        self._update_gauges()

    def drop_graph_local(self, rt: TenantRuntime, name: str) -> None:
        """Replica: the primary no longer has this graph."""
        with rt.glock:
            handle = rt.graphs.pop(name, None)
        if handle is None:
            return
        handle.lock.acquire_write(self.cfg.lock_timeout_seconds * 3)
        try:
            handle.deleted = True
        finally:
            handle.lock.release_write()
        if handle.graph is not None:
            rt.adjust(nodes=-handle.graph.node_count(), edges=-handle.graph.edge_count())
            handle.graph.close()
        if handle.storage is not None and handle.storage.dir.exists():
            handle.storage.remove()
        self._update_gauges()

    # -- shutdown --------------------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop autosave, write a final snapshot of every dirty graph, release resources."""
        self._stop.set()
        if self._autosave_thread is not None:
            self._autosave_thread.join(timeout=30)
        if self.cfg.role == "primary":
            self.save_all()
        for rt in list(self.tenants.values()):
            for handle in list(rt.graphs.values()):
                if handle.graph is not None:
                    handle.graph.close()
        if self._dir_lock is not None:
            try:
                self._dir_lock.close()
            except OSError:
                pass
            self._dir_lock = None
