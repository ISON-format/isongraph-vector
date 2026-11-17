"""Graph operations: everything a handler does to a graph, after validation.

Rules every function here follows:

* Encoder calls happen *before* a graph lock is taken (a slow embedding
  endpoint must not block readers of the graph).
* A batch is validated completely, under the write lock, before the first
  mutation: quota, dimension, endpoint existence and id-twin conflicts are
  checked up front so a rejected batch changes nothing.
* Quota is reserved atomically on the tenant before applying and reconciled
  after, so two concurrent writers (even into different graphs of one tenant)
  cannot together exceed the limit.
* Null-valued properties are never stored (the ISON text format cannot tell
  null from absent, so the service defines them to be the same).
"""

from __future__ import annotations

import base64
import bisect
import importlib.util
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from isongraph_vector import (UNKNOWN_MODELS, DimensionMismatchError, EmbeddingError,
                              NoEncoderError, SemanticGraph, graph_digest)
from ison_graph import (Direction, DuplicateEdgeError, DuplicateNodeError, EdgeNotFoundError,
                        GraphError, NodeNotFoundError)

from . import names
from .config import Quotas
from .engine import Engine, GraphHandle, OpContext, TenantRuntime
from .errors import (BadRequest, Conflict, NotFound, QuotaExceeded, Unavailable, validation_error)
from .filters import compile_filter
from .limits import LockTimeout
from .storage import GraphMeta

DEFAULT_EMBED_FIELDS = ["name", "description", "content", "text"]
_DIRECTIONS = {"out": Direction.OUT, "in": Direction.IN, "both": Direction.BOTH}


@dataclass
class OpEnv:
    engine: Engine
    rt: TenantRuntime
    quotas: Quotas
    ctx: OpContext

    @property
    def encoder(self):
        return self.engine.encoder


def busy() -> Unavailable:
    return Unavailable("graph is busy; retry", code="busy", headers={"Retry-After": "1"})


@contextmanager
def read_locked(env: OpEnv, handle: GraphHandle):
    try:
        handle.lock.acquire_read(env.ctx.lock_wait())
    except LockTimeout:
        raise busy() from None
    try:
        if handle.deleted or handle.graph is None:
            raise NotFound("graph not found")
        yield handle.graph
    finally:
        handle.lock.release_read()


@contextmanager
def write_locked(env: OpEnv, handle: GraphHandle):
    try:
        handle.lock.acquire_write(env.ctx.lock_wait())
    except LockTimeout:
        raise busy() from None
    try:
        if handle.deleted or handle.graph is None:
            raise NotFound("graph not found")
        yield handle.graph
    finally:
        handle.lock.release_write()


# -- serialization helpers ---------------------------------------------------------------------


def ref_json(ref: Tuple[str, Any]) -> Dict[str, Any]:
    return {"type": ref[0], "id": ref[1]}


def live_props(props: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in props.items() if v is not None}


def node_json(node: Any, include_props: bool = True) -> Dict[str, Any]:
    out = {"type": node.type, "id": node.id}
    if include_props:
        out["properties"] = live_props(node.properties)
    return out


def edge_json(edge: Any) -> Dict[str, Any]:
    return {"rel": edge.rel_type, "source": ref_json(edge.source), "target": ref_json(edge.target),
            "properties": live_props(edge.properties)}


def _score(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def _id_kind(value: Any) -> int:
    return 0 if isinstance(value, int) else 1


def node_sort_key(ref: Tuple[str, Any]):
    return (ref[0], _id_kind(ref[1]), ref[1])


def encode_cursor(key: Sequence[Any]) -> str:
    raw = json.dumps(list(key), separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: Optional[str], arity: int) -> Optional[tuple]:
    """Cursors are plain keyset positions inside the caller's own data; a bad one is a 400."""
    if cursor is None:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError
        return tuple(data)
    except (ValueError, TypeError):
        raise BadRequest("invalid cursor", code="invalid_cursor") from None


def page_limit(env: OpEnv, raw: Optional[str], default: int = 100) -> int:
    if raw is None:
        return min(default, env.engine.cfg.max_page)
    try:
        n = int(raw)
    except ValueError:
        raise BadRequest("limit must be an integer", code="invalid_request") from None
    if n < 1 or n > env.engine.cfg.max_page:
        raise BadRequest(f"limit must be between 1 and {env.engine.cfg.max_page}", code="invalid_request")
    return n


# -- graphs -------------------------------------------------------------------------------------


def _numpy_available() -> bool:
    return importlib.util.find_spec("numpy") is not None


def graph_summary(handle: GraphHandle, graph: SemanticGraph) -> Dict[str, Any]:
    meta = handle.meta
    store = graph.embedding_store
    return {
        "name": handle.name,
        "status": "ok" if not handle.degraded else "degraded",
        "directed": True,
        "mode": meta.mode,
        "dimension": store.dimension if store.dimension is not None else meta.dimension,
        "embed_fields": list(meta.embed_fields),
        "ann": meta.ann,
        "nodes": graph.node_count(),
        "edges": graph.edge_count(),
        "embeddings": store.count(),
        "generation": handle.generation,
        "unsaved_changes": handle.dirty,
        "saved_at": handle.last_save_ts,
    }


def create_graph(env: OpEnv, body: Dict[str, Any]) -> Dict[str, Any]:
    q = env.quotas
    if body.get("directed", True) is not True:
        raise BadRequest("undirected graphs are not supported by the service "
                         "(the ISON file format cannot record directedness); "
                         "use direction 'both' in traversals instead",
                         code="unsupported")
    emb = body.get("embedding") or {}
    mode = emb.get("mode", "server")
    dimension = emb.get("dimension")
    if mode == "server":
        if dimension is not None:
            raise validation_error("$.embedding.dimension",
                                   "not allowed in server mode (it follows the configured encoder)")
        if env.encoder is None:
            raise BadRequest("this server has no embedding encoder configured; "
                             "create the graph with embedding.mode='client'",
                             code="encoder_unavailable")
        try:
            dimension = int(env.encoder.dimension)
        except Exception:
            raise Unavailable("the embedding service is not reachable", code="encoder_failed") from None
    else:
        if dimension is None:
            raise validation_error("$.embedding.dimension", "required in client mode")
    if dimension > q.max_dimension:
        raise QuotaExceeded(f"vector dimension {dimension} exceeds this tenant's limit of {q.max_dimension}")
    ann = body.get("ann")
    if ann and not _numpy_available():
        raise BadRequest("ann='ivf' needs numpy on the server", code="unsupported")
    meta = GraphMeta(
        name=body["name"], mode=mode, dimension=dimension,
        embed_fields=list(body.get("embed_fields") or DEFAULT_EMBED_FIELDS),
        ann=ann, ann_nprobe=body.get("ann_nprobe"))
    handle = env.engine.create_graph(env.rt, q, meta)
    with read_locked(env, handle) as graph:
        return graph_summary(handle, graph)


def list_graphs(env: OpEnv, cursor: Optional[str], limit: int) -> Dict[str, Any]:
    with env.rt.glock:
        handles = sorted(env.rt.graphs.values(), key=lambda h: h.name)
    after = decode_cursor(cursor, 1)
    items = []
    next_cursor = None
    for h in handles:
        if after is not None and h.name <= str(after[0]):
            continue
        if len(items) >= limit:
            next_cursor = encode_cursor([items[-1]["name"]])
            break
        if h.quarantined is not None:
            items.append({"name": h.name, "status": "quarantined"})
            continue
        try:
            with read_locked(env, h) as graph:
                items.append(graph_summary(h, graph))
        except NotFound:
            continue
    return {"items": items, "next_cursor": next_cursor}


def get_graph(env: OpEnv, handle: GraphHandle) -> Dict[str, Any]:
    with read_locked(env, handle) as graph:
        out = graph_summary(handle, graph)
        # The library's graph_digest of the last saved generation: the value a
        # replica verifies. (Not computed from live memory: the file format
        # cannot tell null from absent, so the two can differ by that alone.)
        out["snapshot_digest"] = handle.manifest.graph_digest if handle.manifest else None
        return out


def usage(env: OpEnv) -> Dict[str, Any]:
    rt, q = env.rt, env.quotas
    with rt.glock:
        n_graphs = len(rt.graphs)
    return {
        "quotas": {
            "max_graphs": q.max_graphs, "max_nodes": q.max_nodes, "max_edges": q.max_edges,
            "max_dimension": q.max_dimension, "max_batch_items": q.max_batch_items,
            "max_top_k": q.max_top_k, "max_hops": q.max_hops,
            "rate_per_second": q.rate_per_second, "rate_burst": q.rate_burst,
            "max_inflight": q.max_inflight,
        },
        "usage": {"graphs": n_graphs, "nodes": rt.used_nodes, "edges": rt.used_edges},
        "requests": {"total": rt.requests, "rate_limited": rt.rate_limited},
    }


def delete_graph(env: OpEnv, name: str) -> None:
    env.engine.delete_graph(env.rt, name, env.ctx)


def save_graph(env: OpEnv, handle: GraphHandle) -> Dict[str, Any]:
    written = env.engine.save_graph(handle, env.ctx, force=True)
    return {"saved": bool(written), "generation": handle.generation,
            "graph_digest": handle.manifest.graph_digest if handle.manifest else None}


def maybe_sync_save(env: OpEnv, handle: GraphHandle) -> None:
    """``durability = sync``: the write is on disk before the client hears about it."""
    if env.engine.cfg.durability == "sync":
        try:
            env.engine.save_graph(handle, env.ctx)
        except Unavailable:
            raise
        except Exception:
            raise Unavailable("the change was applied in memory but could not be saved",
                              code="persist_failed") from None


# -- vectors and embedding -----------------------------------------------------------------------


def _check_vector(env: OpEnv, handle: GraphHandle, vec: Sequence[float], path: str) -> None:
    if len(vec) > env.quotas.max_dimension:
        raise validation_error(path, f"vector longer than this tenant's limit of {env.quotas.max_dimension}")
    if handle.meta.mode == "client" and handle.meta.dimension is not None and len(vec) != handle.meta.dimension:
        raise BadRequest(f"{path}: expected {handle.meta.dimension} components, got {len(vec)}",
                         code="dimension_mismatch")


def _encode(env: OpEnv, texts: List[str]) -> List[List[float]]:
    enc = env.encoder
    if enc is None:
        raise Unavailable("no embedding encoder is configured", code="encoder_unavailable")
    env.ctx.check()
    vectors = enc.encode_batch(texts)
    if len(vectors) != len(texts):
        raise Unavailable("the embedding service returned the wrong number of vectors", code="encoder_failed")
    dim = None
    for v in vectors:
        if dim is None:
            dim = len(v)
        if len(v) != dim or any(not math.isfinite(x) for x in v):
            raise Unavailable("the embedding service returned an unusable vector", code="encoder_failed")
    if dim and dim > env.quotas.max_dimension:
        raise QuotaExceeded(f"vector dimension {dim} exceeds this tenant's limit")
    return [list(map(float, v)) for v in vectors]


def _canon_id(node_id: Any) -> str:
    """Spelling-independent form of an id: 7 and "007" are one reference to the library."""
    if isinstance(node_id, str) and node_id.lstrip("-").isdigit():
        return str(int(node_id))
    return str(node_id)


def _twin_conflict(graph: SemanticGraph, node_type: str, node_id: Any) -> bool:
    """The library refuses 7 and "007" as the same node (one reference spells both)."""
    if isinstance(node_id, str) and node_id.lstrip("-").isdigit():
        return graph.has_node(node_type, int(node_id))
    if isinstance(node_id, int):
        return graph.has_node(node_type, str(node_id))
    return False


def _clean_props(props: Dict[str, Any], quotas: Quotas, path: str) -> Dict[str, Any]:
    if len(props) > quotas.max_properties:
        raise validation_error(path, f"at most {quotas.max_properties} properties per node")
    return live_props(props)


# -- node writes ---------------------------------------------------------------------------------------


def _batch_limit(env: OpEnv, n: int, what: str) -> None:
    if n > env.quotas.max_batch_items:
        raise BadRequest(f"at most {env.quotas.max_batch_items} {what} per request",
                         code="batch_too_large")


def upsert_nodes(env: OpEnv, handle: GraphHandle, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create or fully replace nodes. Server mode embeds text; client mode takes vectors."""
    _batch_limit(env, len(items), "nodes")
    meta, q = handle.meta, env.quotas
    seen = set()
    prepared: List[Dict[str, Any]] = []
    to_encode: List[Tuple[int, str]] = []
    graph0 = handle.graph
    for i, it in enumerate(items):
        key = (it["type"], _canon_id(it["id"]))
        if key in seen:
            raise validation_error(f"$.nodes[{i}]", "the same node appears twice in this batch")
        seen.add(key)
        props = _clean_props(it.get("properties") or {}, q, f"$.nodes[{i}].properties")
        vec, text = it.get("vector"), it.get("embed_text")
        if meta.mode == "client":
            if vec is None and text is not None:
                raise validation_error(f"$.nodes[{i}].embed_text",
                                       "a client-mode graph takes 'vector', not text to embed")
            if vec is not None:
                _check_vector(env, handle, vec, f"$.nodes[{i}].vector")
        else:
            if vec is not None:
                raise validation_error(f"$.nodes[{i}].vector",
                                       "a server-mode graph embeds text itself; send 'embed_text' "
                                       "(or embed fields in properties), or use a client-mode graph")
            text = text if text is not None else graph0.embed_text_for(props)
            if text:
                to_encode.append((i, text))
        prepared.append({"type": it["type"], "id": it["id"], "props": props, "vector": vec,
                         "text": text or "", "server": meta.mode == "server"})

    if to_encode:
        for (i, _), vec in zip(to_encode, _encode(env, [t for _, t in to_encode])):
            prepared[i]["vector"] = vec

    created = updated = embedded = 0
    with write_locked(env, handle) as graph:
        store = graph.embedding_store
        pinned = store.dimension
        dims = {len(p["vector"]) for p in prepared if p["vector"] is not None}
        if len(dims) > 1:
            raise BadRequest("vectors in one batch have different lengths", code="dimension_mismatch")
        if dims and pinned is not None and next(iter(dims)) != pinned:
            raise Conflict(f"vectors have {next(iter(dims))} dimensions but this graph holds {pinned}; "
                           f"if the encoder changed, POST .../reembed", code="dimension_mismatch")
        new = 0
        for p in prepared:
            exists = graph.has_node(p["type"], p["id"])
            p["exists"] = exists
            if not exists:
                if _twin_conflict(graph, p["type"], p["id"]):
                    raise Conflict("a node whose id differs only by integer/string spelling already exists",
                                   code="id_conflict")
                new += 1
        env.rt.reserve(q, nodes=new)
        applied_new = 0
        try:
            for p in prepared:
                ref = (p["type"], p["id"])
                if p["exists"]:
                    node = graph.get_node(*ref)
                    node.properties.clear()
                    node.properties.update(p["props"])
                    updated += 1
                else:
                    node = graph.add_node(p["type"], p["id"])
                    node.properties.update(p["props"])
                    created += 1
                    applied_new += 1
                if p["vector"] is not None:
                    store.add(ref, p["text"], vector=p["vector"])
                    embedded += 1
                elif p["server"] and p["exists"]:
                    store.remove(ref)       # new properties leave nothing to embed
        finally:
            if applied_new != new:
                env.rt.adjust(nodes=applied_new - new)
            handle.bump()
    maybe_sync_save(env, handle)
    return {"created": created, "updated": updated, "embedded": embedded}


def patch_node(env: OpEnv, handle: GraphHandle, node_type: str, node_id: Any,
               body: Dict[str, Any]) -> Dict[str, Any]:
    """Merge properties into an existing node (optimistic: the embed text is computed
    outside the lock, and the write fails with 409 if the node changed meanwhile)."""
    meta, q = handle.meta, env.quotas
    set_props = _clean_props({k: v for k, v in (body.get("properties") or {}).items()}, q, "$.properties")
    null_keys = {k for k, v in (body.get("properties") or {}).items() if v is None}
    remove = set(body.get("remove") or []) | null_keys
    vec, text = body.get("vector"), body.get("embed_text")

    with read_locked(env, handle) as graph:
        try:
            node = graph.get_node(node_type, node_id)
        except GraphError:
            raise NotFound("node not found") from None
        before = dict(node.properties)
    merged = {k: v for k, v in before.items() if k not in remove and v is not None}
    merged.update(set_props)
    if len(merged) > q.max_properties:
        raise validation_error("$.properties", f"at most {q.max_properties} properties per node")

    new_vector: Optional[List[float]] = None
    new_text = ""
    touches_embedding = (text is not None or vec is not None
                         or any(f in set_props or f in remove for f in meta.embed_fields))
    if meta.mode == "client":
        if vec is None and text is not None:
            raise validation_error("$.embed_text", "a client-mode graph takes 'vector', not text to embed")
        if vec is not None:
            _check_vector(env, handle, vec, "$.vector")
            new_vector, new_text = vec, text or ""
    else:
        if vec is not None:
            raise validation_error("$.vector", "a server-mode graph embeds text itself")
        if touches_embedding:
            new_text = text if text is not None else handle.graph.embed_text_for(merged)
            if new_text:
                new_vector = _encode(env, [new_text])[0]

    with write_locked(env, handle) as graph:
        try:
            node = graph.get_node(node_type, node_id)
        except GraphError:
            raise NotFound("node not found") from None
        if dict(node.properties) != before:
            raise Conflict("the node changed while this update was being prepared; retry",
                           code="concurrent_update")
        store = graph.embedding_store
        if new_vector is not None and store.dimension is not None and len(new_vector) != store.dimension:
            raise Conflict(f"vector has {len(new_vector)} dimensions but this graph holds {store.dimension}",
                           code="dimension_mismatch")
        for k in remove:
            node.properties.pop(k, None)
        node.properties.update(set_props)
        ref = (node_type, node_id)
        if new_vector is not None:
            store.add(ref, new_text, vector=new_vector)
        elif meta.mode == "server" and touches_embedding:
            store.remove(ref)
        handle.bump()
        out = node_json(node)
    maybe_sync_save(env, handle)
    return out


def delete_nodes(env: OpEnv, handle: GraphHandle, refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    _batch_limit(env, len(refs), "nodes")
    deleted = missing = 0
    with write_locked(env, handle) as graph:
        n0, e0 = graph.node_count(), graph.edge_count()
        try:
            for r in refs:
                if graph.has_node(r["type"], r["id"]):
                    graph.remove_node(r["type"], r["id"])
                    deleted += 1
                else:
                    missing += 1
        finally:
            env.rt.adjust(nodes=graph.node_count() - n0, edges=graph.edge_count() - e0)
            handle.bump()
    maybe_sync_save(env, handle)
    return {"deleted": deleted, "missing": missing}


def get_node(env: OpEnv, handle: GraphHandle, node_type: str, node_id: Any,
             include_vector: bool = False) -> Dict[str, Any]:
    with read_locked(env, handle) as graph:
        try:
            node = graph.get_node(node_type, node_id)
        except GraphError:
            raise NotFound("node not found") from None
        out = node_json(node)
        rec = graph.embedding_store.get((node_type, node_id))
        out["embedded"] = rec is not None
        if rec is not None:
            out["embedding"] = {"model": rec.model, "dimension": len(rec.vector), "text": rec.text}
            if include_vector:
                out["embedding"]["vector"] = rec.vector
        return out


def _sorted_nodes(handle: GraphHandle, graph: SemanticGraph) -> List[tuple]:
    with handle._cache_lock:
        cached = handle._sorted_nodes
        if cached is None or cached[0] != handle.version:
            keys = sorted((node_sort_key(n.ref) for n in graph.nodes()))
            cached = handle._sorted_nodes = (handle.version, keys)
        return cached[1]


def list_nodes(env: OpEnv, handle: GraphHandle, node_type: Optional[str], cursor: Optional[str],
               limit: int) -> Dict[str, Any]:
    after = decode_cursor(cursor, 3)
    with read_locked(env, handle) as graph:
        keys = _sorted_nodes(handle, graph)
        if node_type is not None:
            start = bisect.bisect_left(keys, (node_type,))
        else:
            start = 0
        if after is not None:
            try:
                start = max(start, bisect.bisect_right(keys, (after[0], int(after[1]), after[2])))
            except (TypeError, ValueError, IndexError):
                raise BadRequest("invalid cursor", code="invalid_cursor") from None
        items = []
        i = start
        while i < len(keys) and len(items) < limit:
            key = keys[i]
            if node_type is not None and key[0] != node_type:
                break
            try:
                items.append(node_json(graph.get_node(key[0], key[2])))
            except GraphError:
                pass
            i += 1
        more = i < len(keys) and (node_type is None or keys[i][0] == node_type)
        next_cursor = encode_cursor(node_sort_key((items[-1]["type"], items[-1]["id"]))) \
            if (more and items) else None
        return {"items": items, "next_cursor": next_cursor}


# -- edges ---------------------------------------------------------------------------------------------------


def upsert_edges(env: OpEnv, handle: GraphHandle, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    _batch_limit(env, len(items), "edges")
    q = env.quotas
    seen = set()
    prepared = []
    for i, it in enumerate(items):
        s = (it["source"]["type"], it["source"]["id"])
        t = (it["target"]["type"], it["target"]["id"])
        key = (it["rel"], s, t)
        if key in seen:
            raise validation_error(f"$.edges[{i}]", "the same edge appears twice in this batch")
        seen.add(key)
        props = _clean_props(it.get("properties") or {}, q, f"$.edges[{i}].properties")
        prepared.append((it["rel"], s, t, props))
    created = updated = 0
    with write_locked(env, handle) as graph:
        new = 0
        for i, (rel, s, t, _p) in enumerate(prepared):
            if not graph.has_node(*s) or not graph.has_node(*t):
                raise NotFound(f"edge {i}: source or target node does not exist")
            if not graph.has_edge(rel, s, t):
                new += 1
        env.rt.reserve(q, edges=new)
        applied = 0
        try:
            for rel, s, t, props in prepared:
                if graph.has_edge(rel, s, t):
                    edge = graph.get_edge(rel, s, t)
                    edge.properties.clear()
                    edge.properties.update(props)
                    updated += 1
                else:
                    edge = graph.add_edge(rel, s, t)
                    edge.properties.update(props)
                    created += 1
                    applied += 1
        except (DuplicateEdgeError, NodeNotFoundError) as exc:     # pragma: no cover - pre-checked
            raise Conflict(str(exc)[:120]) from None
        finally:
            if applied != new:
                env.rt.adjust(edges=applied - new)
            handle.bump()
    maybe_sync_save(env, handle)
    return {"created": created, "updated": updated}


def delete_edges(env: OpEnv, handle: GraphHandle, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    _batch_limit(env, len(items), "edges")
    deleted = missing = 0
    with write_locked(env, handle) as graph:
        e0 = graph.edge_count()
        try:
            for it in items:
                s = (it["source"]["type"], it["source"]["id"])
                t = (it["target"]["type"], it["target"]["id"])
                try:
                    graph.remove_edge(it["rel"], s, t)
                    deleted += 1
                except EdgeNotFoundError:
                    missing += 1
        finally:
            env.rt.adjust(edges=graph.edge_count() - e0)
            handle.bump()
    maybe_sync_save(env, handle)
    return {"deleted": deleted, "missing": missing}


def _edge_key(e: Any) -> tuple:
    return (e.rel_type, node_sort_key(e.source), node_sort_key(e.target))


def list_edges(env: OpEnv, handle: GraphHandle, rel: Optional[str], cursor: Optional[str],
               limit: int) -> Dict[str, Any]:
    after = decode_cursor(cursor, 1)
    with read_locked(env, handle) as graph:
        with handle._cache_lock:
            cached = handle._sorted_edges
            if cached is None or cached[0] != handle.version:
                edges = sorted(graph.edges(), key=_edge_key)
                cached = handle._sorted_edges = (handle.version, edges, [_edge_key(e) for e in edges])
        _, edges, keys = cached
        start = 0
        if rel is not None:
            start = bisect.bisect_left(keys, (rel,))
        if after is not None:
            try:
                pos = int(after[0])
            except (TypeError, ValueError):
                raise BadRequest("invalid cursor", code="invalid_cursor") from None
            start = max(start, pos)
        items = []
        i = start
        while i < len(edges) and len(items) < limit:
            if rel is not None and edges[i].rel_type != rel:
                break
            items.append(edge_json(edges[i]))
            i += 1
        more = i < len(edges) and (rel is None or edges[i].rel_type == rel)
        return {"items": items, "next_cursor": encode_cursor([i]) if more else None}


# -- search ----------------------------------------------------------------------------------------------------


def _cap(env_value: int, got: Optional[int], default: int, path: str, what: str) -> int:
    if got is None:
        return min(default, env_value)
    value = got
    if value > env_value:
        raise validation_error(path, f"must be <= {env_value} ({what} limit for this tenant)")
    return value


def resolve_query(env: OpEnv, handle: GraphHandle, body: Dict[str, Any]) -> List[float]:
    """Turn ``query`` (text) or ``vector`` into a query vector, encoding outside any lock."""
    text, vec = body.get("query"), body.get("vector")
    if (text is None) == (vec is None):
        raise BadRequest("send exactly one of 'query' (text) or 'vector'", code="invalid_request")
    meta = handle.meta
    if vec is not None:
        if len(vec) > env.quotas.max_dimension:
            raise validation_error("$.vector", "vector longer than this tenant's limit")
        return vec
    if meta.mode != "server":
        raise BadRequest("this graph takes caller-supplied vectors; send 'vector', not 'query'",
                         code="text_query_unsupported")
    if not text.strip():
        raise validation_error("$.query", "must not be empty")
    enc = env.encoder
    if enc is None:
        raise Unavailable("no embedding encoder is configured", code="encoder_unavailable")
    with read_locked(env, handle) as graph:
        models = graph.embedding_store.models()
        pinned = graph.embedding_store.dimension
    current = getattr(enc, "model_name", None)
    foreign = sorted(m for m in models if m not in UNKNOWN_MODELS and m != current)
    if foreign:
        raise Conflict("the stored vectors were made by a different embedding model than the one "
                       "the server now uses; POST .../reembed first", code="model_mismatch")
    vector = _encode(env, [text])[0]
    if pinned is not None and len(vector) != pinned:
        raise Conflict("the stored vectors have a different dimension than the current encoder; "
                       "POST .../reembed first", code="dimension_mismatch")
    return vector


def _search_errors(fn):
    try:
        return fn()
    except DimensionMismatchError as exc:
        raise BadRequest("query vector has the wrong number of dimensions for this graph",
                         code="dimension_mismatch") from None
    except NoEncoderError:
        raise BadRequest("this graph has no encoder", code="text_query_unsupported") from None
    except EmbeddingError:
        raise Unavailable("search failed", code="search_failed") from None


def _hit_json(graph: SemanticGraph, ref, score: float, include_props: bool, text: Optional[str] = None):
    out = {"type": ref[0], "id": ref[1], "score": _score(score)}
    if include_props:
        try:
            out["properties"] = live_props(graph.get_node(*ref).properties)
        except GraphError:
            out["properties"] = {}
    if text is not None:
        out["text"] = text
    return out


def search(env: OpEnv, handle: GraphHandle, body: Dict[str, Any]) -> Dict[str, Any]:
    q = env.quotas
    top_k = _cap(q.max_top_k, body.get("top_k"), 10, "$.top_k", "top_k")
    vec = resolve_query(env, handle, body)
    pred, types = compile_filter(body.get("filter"))
    env.ctx.check()
    with read_locked(env, handle) as graph:
        results = _search_errors(lambda: graph.similarity_search(
            vec, top_k=top_k, node_type=body.get("node_type"), threshold=body.get("threshold", 0.0),
            node_filter=pred, node_types=types, exact=body.get("exact", False)))
        inc, txt = body.get("include_properties", True), body.get("include_text", False)
        hits = [_hit_json(graph, r.node_ref, r.score, inc, r.text if txt else None) for r in results]
    return {"results": hits, "count": len(hits)}


def similar(env: OpEnv, handle: GraphHandle, body: Dict[str, Any]) -> Dict[str, Any]:
    top_k = _cap(env.quotas.max_top_k, body.get("top_k"), 10, "$.top_k", "top_k")
    ref = (body["node"]["type"], body["node"]["id"])
    with read_locked(env, handle) as graph:
        if not graph.has_node(*ref):
            raise NotFound("node not found")
        embedded = graph.embedding_store.get(ref) is not None
        results = _search_errors(lambda: graph.similar_to_node(
            ref, top_k=top_k, node_type=body.get("node_type"), threshold=body.get("threshold", 0.0)))
        inc = body.get("include_properties", True)
        hits = [_hit_json(graph, r.node_ref, r.score, inc) for r in results]
    return {"results": hits, "count": len(hits), "node_embedded": embedded}


def _traversal_args(env: OpEnv, body: Dict[str, Any], *, max_hops: int, seeds: int, decay: float,
                    threshold: float) -> Dict[str, Any]:
    q = env.quotas
    pred, types = compile_filter(body.get("filter"))
    return {
        "rel_type": body.get("rel_type"),
        "max_hops": _cap(q.max_hops, body.get("max_hops"), max_hops, "$.max_hops", "hop"),
        "top_k_seeds": _cap(q.max_top_k, body.get("top_k_seeds"), seeds, "$.top_k_seeds", "seed"),
        "direction": _DIRECTIONS[body.get("direction", "out")],
        "decay": body.get("decay", decay),
        "threshold": body.get("threshold", threshold),
        "weight_field": body.get("weight_field"),
        "node_filter": pred,
        "node_types": types,
        "strict_filter": body.get("strict_filter", False),
    }


def multi_hop(env: OpEnv, handle: GraphHandle, body: Dict[str, Any]) -> Dict[str, Any]:
    args = _traversal_args(env, body, max_hops=2, seeds=5, decay=0.8, threshold=0.3)
    top = _cap(env.quotas.max_top_k, body.get("top_k_results"), 10, "$.top_k_results", "result")
    vec = resolve_query(env, handle, body)
    env.ctx.check()
    with read_locked(env, handle) as graph:
        results = _search_errors(lambda: graph.semantic_multi_hop(
            vec, top_k_results=top, blend=body.get("blend", 0.0), **args))
        inc = body.get("include_properties", True)
        out = []
        for r in results:
            item = _hit_json(graph, r.node_ref, r.score, inc)
            item["hops"] = r.hop_count
            item["path"] = [ref_json(p) for p in r.path]
            out.append(item)
    return {"results": out, "count": len(out)}


def path(env: OpEnv, handle: GraphHandle, body: Dict[str, Any]) -> Dict[str, Any]:
    args = _traversal_args(env, body, max_hops=5, seeds=3, decay=0.9, threshold=0.0)
    args.pop("weight_field")       # semantic_path scores by hop count only
    target = (body["target"]["type"], body["target"]["id"])
    vec = resolve_query(env, handle, body)
    env.ctx.check()
    with read_locked(env, handle) as graph:
        if not graph.has_node(*target):
            raise NotFound("target node not found")
        result = _search_errors(lambda: graph.semantic_path(vec, target, **args))
        if result is None:
            return {"found": False, "result": None}
        item = _hit_json(graph, result.node_ref, result.score, body.get("include_properties", True))
        item["hops"] = result.hop_count
        item["path"] = [ref_json(p) for p in result.path]
    return {"found": True, "result": item}


def subgraph(env: OpEnv, handle: GraphHandle, body: Dict[str, Any]) -> Dict[str, Any]:
    args = _traversal_args(env, body, max_hops=2, seeds=5, decay=0.8, threshold=0.3)
    top = _cap(env.quotas.max_top_k, body.get("top_k_results"), 10, "$.top_k_results", "result")
    vec = resolve_query(env, handle, body)
    env.ctx.check()
    with read_locked(env, handle) as graph:
        sub = _search_errors(lambda: graph.semantic_subgraph(
            vec, top_k_results=top, blend=body.get("blend", 0.0), connect=body.get("connect", False),
            connect_max_hops=body.get("connect_max_hops", 2), **args))
        try:
            nodes = [node_json(n, body.get("include_properties", True)) for n in sub.nodes()]
            edges = [edge_json(e) for e in sub.edges()]
        finally:
            sub.close()
    return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}


# -- model lifecycle ---------------------------------------------------------------------------------------


def _server_mode_only(handle: GraphHandle) -> None:
    if handle.meta.mode != "server":
        raise BadRequest("embedding lifecycle only applies to server-mode graphs "
                         "(a client-mode graph's vectors belong to the caller)", code="unsupported")


def stale(env: OpEnv, handle: GraphHandle, include_unknown: bool, limit: int, offset: int) -> Dict[str, Any]:
    _server_mode_only(handle)
    if env.encoder is None:
        raise Unavailable("no embedding encoder is configured", code="encoder_unavailable")
    with read_locked(env, handle) as graph:
        models = graph.embedding_store.models()
        refs = graph.stale_embeddings(include_unknown=include_unknown)
    page = refs[offset:offset + limit]
    more = offset + limit < len(refs)
    return {
        "encoder_model": getattr(env.encoder, "model_name", None),
        "stored_models": models,
        "stale_count": len(refs),
        "stale": [ref_json(r) for r in page],
        "next_cursor": encode_cursor([offset + limit]) if more else None,
    }


def reembed(env: OpEnv, handle: GraphHandle, include_unknown: bool) -> Dict[str, Any]:
    """Re-encode stale vectors. Holds the graph's write lock for the duration
    (it calls the encoder), so concurrent reads and writes of *this graph* wait."""
    _server_mode_only(handle)
    if env.encoder is None:
        raise Unavailable("no embedding encoder is configured", code="encoder_unavailable")
    with write_locked(env, handle) as graph:
        try:
            count = graph.reembed_stale(include_unknown=include_unknown)
        finally:
            handle.bump()
    maybe_sync_save(env, handle)
    return {"reembedded": count}
