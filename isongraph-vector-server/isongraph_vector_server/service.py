"""The request pipeline: routing, authentication, limits, validation, handlers.

Order of operations for a tenant request (each step can end the request):

 1. split and validate the path (decoded once, segment-wise; ``.``/``..``,
    separators and control characters are rejected)
 2. match a route (404 / 405)
 3. authenticate: exactly one ``Authorization: Bearer <key>`` header; failures
    are throttled per peer address; the tenant is whatever the *verified key*
    says and nothing else - no header, query parameter or body field selects it
 4. role check (read-only key, read-only replica, replica too stale)
 5. per-tenant rate limit, then global and per-tenant concurrency gates
 6. query-string and JSON body validation against the route's schema
 7. the handler, which only ever sees the authenticated tenant's runtime
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import names, ops, schemas
from .auth import Principal, TenantRegistry
from .config import ServerConfig
from .engine import Engine, OpContext
from .errors import (ApiError, BadRequest, Forbidden, MethodNotAllowed, NotFound, Overloaded,
                     PayloadTooLarge, Unauthorized, Unavailable, UnsupportedMediaType,
                     validation_error)
from .httpd import Request, Response
from .limits import Gate, SlidingWindow, rate_limited
from .logs import log
from .metrics import Metrics
from .replication import build_index
from .names import GENERATION_FILES
from .validation import validate

QUERY_MAX_PARAMS = 16


# -- request parsing ----------------------------------------------------------------------------------


def split_path(target: str) -> Tuple[List[str], Dict[str, str]]:
    path, _, qs = target.partition("?")
    if "#" in path or not path.startswith("/") or len(path) > 2048:
        raise BadRequest("invalid request path", code="invalid_path")
    segs: List[str] = []
    if path != "/":
        for raw in path[1:].split("/"):
            if raw == "":
                raise NotFound("no such resource")
            try:
                seg = urllib.parse.unquote(raw, errors="strict")
            except UnicodeDecodeError:
                raise BadRequest("invalid request path", code="invalid_path") from None
            if (seg in (".", "..") or "/" in seg or "\\" in seg
                    or any(ord(c) < 32 or ord(c) == 127 for c in seg) or len(seg) > 256):
                raise BadRequest("invalid request path", code="invalid_path")
            segs.append(seg)
    query: Dict[str, str] = {}
    if qs:
        try:
            pairs = urllib.parse.parse_qsl(qs, keep_blank_values=True, errors="strict",
                                           max_num_fields=QUERY_MAX_PARAMS)
        except (UnicodeDecodeError, ValueError):
            raise BadRequest("invalid query string", code="invalid_query") from None
        for k, v in pairs:
            if k in query:
                raise BadRequest(f"query parameter {k[:40]!r} given twice", code="invalid_query")
            query[k] = v
    return segs, query


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate key")
        out[k] = v
    return out


def _reject_constant(name):
    raise ValueError("non-finite number")


def parse_json_body(req: Request, optional: bool) -> Any:
    if not req.body:
        if optional:
            return {}
        raise BadRequest("a JSON body is required", code="invalid_json")
    ctype = (req.header("content-type") or "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise UnsupportedMediaType("Content-Type must be application/json")
    try:
        text = req.body.decode("utf-8")
        data = json.loads(text, object_pairs_hook=_no_dupes, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise BadRequest("the body is not valid JSON (duplicate keys, NaN/Infinity and "
                         "invalid UTF-8 are rejected)", code="invalid_json") from None
    return data


def _token_matches(presented: Optional[str], expected: Optional[str]) -> bool:
    if not expected or presented is None:
        return False
    a = hashlib.sha256(presented.encode("utf-8", "replace")).digest()
    b = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(a, b)


def bearer_token(req: Request) -> Optional[str]:
    vals = req.header_all("authorization")
    if len(vals) != 1:
        return None
    v = vals[0]
    if len(v) < 8 or v[:7].lower() != "bearer ":
        return None
    return v[7:].strip()


# -- routes ----------------------------------------------------------------------------------------------


@dataclass
class Route:
    method: str
    pattern: str
    name: str
    fn: Callable
    auth: str = "tenant"            # none | tenant | metrics | repl
    write: bool = False
    schema: Optional[dict] = None
    query: Tuple[str, ...] = ()
    body_optional: bool = False
    segs: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.segs = [s for s in self.pattern.split("/") if s]


@dataclass
class Call:
    req: Request
    principal: Optional[Principal]
    env: Optional[ops.OpEnv]
    params: Dict[str, str]
    query: Dict[str, str]
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    request_id: str = ""


class App:
    def __init__(self, cfg: ServerConfig, engine: Engine, registry: TenantRegistry,
                 metrics: Metrics, replica: Any = None):
        self.cfg = cfg
        self.engine = engine
        self.registry = registry
        self.metrics = metrics
        self.replica = replica               # ReplicaSyncer or None
        self.global_gate = Gate(cfg.max_concurrent_requests)
        self.auth_failures = SlidingWindow(cfg.failed_auth_per_minute)
        self.routes: List[Route] = []
        self._build_routes()

    # ---- route table ---------------------------------------------------------------------------------

    def _build_routes(self) -> None:
        R = self._add
        R("GET", "/healthz", "healthz", self.h_healthz, auth="none")
        R("GET", "/readyz", "readyz", self.h_readyz, auth="none")
        R("GET", "/metrics", "metrics", self.h_metrics, auth="metrics")
        R("GET", "/v1/usage", "usage", self.h_usage)
        R("GET", "/v1/graphs", "graphs.list", self.h_graphs_list, query=("limit", "cursor"))
        R("POST", "/v1/graphs", "graphs.create", self.h_graph_create, write=True, schema=schemas.CREATE_GRAPH)
        g = "/v1/graphs/{g}"
        R("GET", g, "graphs.get", self.h_graph_get)
        R("DELETE", g, "graphs.delete", self.h_graph_delete, write=True)
        R("POST", g + "/save", "graphs.save", self.h_graph_save, write=True, body_optional=True,
          schema={"type": "object"})
        R("GET", g + "/nodes", "nodes.list", self.h_nodes_list, query=("type", "limit", "cursor"))
        R("POST", g + "/nodes", "nodes.upsert", self.h_nodes_upsert, write=True, schema=schemas.UPSERT_NODES)
        R("POST", g + "/nodes/delete", "nodes.delete_batch", self.h_nodes_delete, write=True,
          schema=schemas.DELETE_NODES)
        n = g + "/nodes/{t}/{id}"
        R("GET", n, "node.get", self.h_node_get, query=("include_vector",))
        R("PUT", n, "node.put", self.h_node_put, write=True, schema=schemas.NODE_BODY)
        R("PATCH", n, "node.patch", self.h_node_patch, write=True, schema=schemas.PATCH_NODE)
        R("DELETE", n, "node.delete", self.h_node_delete, write=True)
        R("GET", g + "/edges", "edges.list", self.h_edges_list, query=("rel", "limit", "cursor"))
        R("POST", g + "/edges", "edges.upsert", self.h_edges_upsert, write=True, schema=schemas.UPSERT_EDGES)
        R("POST", g + "/edges/delete", "edges.delete_batch", self.h_edges_delete, write=True,
          schema=schemas.DELETE_EDGES)
        R("POST", g + "/search", "search", self.h_search, schema=schemas.SEARCH)
        R("POST", g + "/multi_hop", "multi_hop", self.h_multi_hop, schema=schemas.MULTI_HOP)
        R("POST", g + "/path", "path", self.h_path, schema=schemas.PATH)
        R("POST", g + "/subgraph", "subgraph", self.h_subgraph, schema=schemas.SUBGRAPH)
        R("POST", g + "/similar", "similar", self.h_similar, schema=schemas.SIMILAR)
        R("GET", g + "/stale", "stale", self.h_stale, query=("include_unknown", "limit", "cursor"))
        R("POST", g + "/reembed", "reembed", self.h_reembed, write=True, schema=schemas.REEMBED,
          body_optional=True)
        if self.cfg.role == "primary":
            R("GET", "/_repl/v1/index", "repl.index", self.h_repl_index, auth="repl")
            R("GET", "/_repl/v1/tenant/{t}", "repl.tenant", self.h_repl_tenant, auth="repl")
            R("GET", "/_repl/v1/meta/{t}/{g}", "repl.meta", self.h_repl_meta, auth="repl")
            R("GET", "/_repl/v1/file/{t}/{g}/{gen}/{f}", "repl.file", self.h_repl_file, auth="repl")

    def _add(self, method, pattern, name, fn, **kw) -> None:
        self.routes.append(Route(method, pattern, name, fn, **kw))

    def _match(self, method: str, segs: List[str]):
        allowed = []
        for route in self.routes:
            if len(route.segs) != len(segs):
                continue
            params: Dict[str, str] = {}
            ok = True
            for pat, seg in zip(route.segs, segs):
                if pat.startswith("{"):
                    params[pat[1:-1]] = seg
                elif pat != seg:
                    ok = False
                    break
            if not ok:
                continue
            if route.method == method:
                return route, params, None
            allowed.append(route.method)
        return None, None, allowed

    # ---- entry point -------------------------------------------------------------------------------------

    def handle(self, req: Request) -> Response:
        t0 = time.monotonic()
        rid = req.header("x-request-id")
        if rid is None or not names.REQUEST_ID_RE.match(rid):
            rid = secrets.token_hex(8)
        info: Dict[str, Any] = {"route": "unmatched", "tenant": None}
        try:
            resp = self._dispatch(req, rid, info)
        except ApiError as exc:
            resp = Response.json(exc.status, exc.to_body(rid), exc.headers)
        except Exception:
            logging.getLogger("igv").exception("unhandled error", extra={"fields": {"request_id": rid}})
            resp = Response.json(500, ApiError("an internal error occurred").to_body(rid))
        resp.headers["X-Request-Id"] = rid
        elapsed = time.monotonic() - t0
        self.metrics.inc("igv_requests_total", route=info["route"], status=f"{resp.status // 100}xx")
        self.metrics.observe(info["route"], elapsed)
        if resp.status >= 500 or self.cfg.log_level.upper() == "DEBUG" or \
                logging.getLogger("igv").isEnabledFor(logging.INFO):
            log(logging.WARNING if resp.status >= 500 else logging.INFO, "request",
                method=req.method, route=info["route"], status=resp.status,
                ms=round(elapsed * 1000, 2), tenant=info["tenant"], request_id=rid)
        return resp

    # ---- pipeline --------------------------------------------------------------------------------------------

    def _client_key(self, req: Request) -> str:
        if req.peer in self.cfg.trusted_proxies:
            xff = req.header_all("x-forwarded-for")
            if len(xff) == 1:
                last = xff[0].split(",")[-1].strip()
                if 0 < len(last) <= 64:
                    return last
        return req.peer

    def _dispatch(self, req: Request, rid: str, info: Dict[str, Any]) -> Response:
        segs, query = split_path(req.target)
        route, params, allowed = self._match(req.method, segs)
        if route is None:
            if allowed:
                raise MethodNotAllowed("method not allowed for this resource",
                                       headers={"Allow": ", ".join(sorted(set(allowed)))})
            raise NotFound("no such resource")
        info["route"] = route.name

        principal: Optional[Principal] = None
        if route.auth == "tenant":
            principal = self._authenticate(req)
            info["tenant"] = principal.tenant_id
        elif route.auth in ("metrics", "repl"):
            expected = self.cfg.metrics_token if route.auth == "metrics" else self.cfg.replication_token
            if not expected:
                raise NotFound("no such resource")           # not configured: the endpoint does not exist
            client = self._client_key(req)
            if self.auth_failures.blocked(client):
                raise ApiError("too many failed authentication attempts", status=429,
                               code="rate_limited", headers={"Retry-After": "30"})
            if not _token_matches(bearer_token(req), expected):
                self.auth_failures.record(client)
                self.metrics.inc("igv_auth_failures_total")
                raise self._unauth()

        if route.auth == "tenant" and not self.engine.ready:
            raise Unavailable("server is starting", code="not_ready", headers={"Retry-After": "2"})

        extra = set(query) - set(route.query)
        if extra:
            raise BadRequest(f"unknown query parameter {sorted(extra)[0][:40]!r}", code="invalid_query")

        if principal is None:
            call = Call(req, None, None, params, query, None, request_id=rid)
            return route.fn(call)

        cfg = self.cfg
        if route.write:
            if cfg.role == "replica":
                raise Forbidden("this node is a read-only replica; send writes to the primary",
                                code="read_only_replica")
            if principal.role != "rw":
                raise Forbidden("this API key is read-only", code="read_only_key")
        if cfg.role == "replica" and self.replica is not None:
            self.replica.check_fresh()

        rt = self.engine.runtime(principal.tenant_id, principal.quotas)
        rt.requests += 1
        wait = rt.bucket.take(1 + len(req.body) // 8192)
        if wait:
            rt.rate_limited += 1
            self.metrics.inc("igv_rate_limited_total")
            raise rate_limited(wait)

        tenant_busy = Overloaded("this tenant has too many requests in flight", code="tenant_overloaded",
                                 headers={"Retry-After": "1"})
        with self.global_gate.enter(cfg.queue_wait_seconds), \
                rt.gate.enter(cfg.queue_wait_seconds, tenant_busy):
            body = None
            if route.schema is not None:
                raw = parse_json_body(req, route.body_optional)
                body = validate(raw, route.schema)
            ctx = OpContext(deadline=time.monotonic() + cfg.request_deadline_seconds,
                            lock_timeout=cfg.lock_timeout_seconds)
            env = ops.OpEnv(self.engine, rt, principal.quotas, ctx)
            call = Call(req, principal, env, params, query, body, request_id=rid)
            return route.fn(call)

    def _unauth(self) -> Unauthorized:
        return Unauthorized("missing or invalid API key", headers={"WWW-Authenticate": 'Bearer realm="igv"'})

    def _authenticate(self, req: Request) -> Principal:
        client = self._client_key(req)
        if self.auth_failures.blocked(client):
            self.metrics.inc("igv_auth_throttled_total")
            raise ApiError("too many failed authentication attempts", status=429, code="rate_limited",
                           headers={"Retry-After": "30"})
        token = bearer_token(req)
        principal = self.registry.authenticate(token) if token else None
        if principal is None:
            self.auth_failures.record(client)
            self.metrics.inc("igv_auth_failures_total")
            log(logging.WARNING, "authentication failed", peer=client)
            raise self._unauth()
        return principal

    # ---- response helpers -----------------------------------------------------------------------------------------

    def _ok(self, call: Call, obj: Any, status: int = 200) -> Response:
        data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(data) > self.cfg.max_response_bytes:
            raise ApiError("the response would be too large; lower limit/top_k", status=413,
                           code="response_too_large")
        headers = {"Content-Type": "application/json"}
        headers.update(call.headers)
        return Response(status, headers, data)

    def _handle(self, call: Call, name: str):
        h = self.engine.get_handle(call.env.rt, name)
        if self.cfg.role == "replica":
            call.headers["X-Snapshot-Generation"] = str(h.generation)
            if h.synced_at is not None:
                call.headers["X-Snapshot-Age-Seconds"] = str(int(max(0, time.time() - h.synced_at)))
        return h

    @staticmethod
    def _graph_name(call: Call) -> str:
        name = call.params["g"]
        if not names.is_graph_name(name):
            raise validation_error("path.graph", "invalid graph name")
        return name

    @staticmethod
    def _node_params(call: Call):
        t, raw = call.params["t"], call.params["id"]
        if not names.TYPE_RE.match(t):
            raise validation_error("path.type", "invalid node type")
        try:
            return t, names.normalize_id(raw)
        except ValueError as exc:
            raise validation_error("path.id", str(exc)) from None

    @staticmethod
    def _bool_q(call: Call, key: str) -> bool:
        v = call.query.get(key)
        if v is None:
            return False
        if v.lower() in ("1", "true"):
            return True
        if v.lower() in ("0", "false"):
            return False
        raise BadRequest(f"{key} must be true or false", code="invalid_query")

    # ---- unauthenticated -------------------------------------------------------------------------------------------

    def h_healthz(self, call: Call) -> Response:
        return Response.json(200, {"status": "ok"})

    def h_readyz(self, call: Call) -> Response:
        ready = self.engine.ready and (self.replica is None or self.replica.is_ready())
        return Response.json(200 if ready else 503,
                             {"status": "ready" if ready else "not_ready", "role": self.cfg.role})

    def h_metrics(self, call: Call) -> Response:
        self.metrics.gauge("igv_in_flight_requests", self.global_gate.in_flight)
        self.metrics.gauge("igv_ready", 1 if self.engine.ready else 0)
        if self.replica is not None:
            for k, v in self.replica.metrics_snapshot().items():
                self.metrics.gauge(k, v)
        return Response(200, {"Content-Type": "text/plain; version=0.0.4"},
                        self.metrics.render().encode("utf-8"))

    # ---- tenant: graphs --------------------------------------------------------------------------------------------

    def h_usage(self, call: Call) -> Response:
        return self._ok(call, ops.usage(call.env))

    def h_graphs_list(self, call: Call) -> Response:
        limit = ops.page_limit(call.env, call.query.get("limit"))
        return self._ok(call, ops.list_graphs(call.env, call.query.get("cursor"), limit))

    def h_graph_create(self, call: Call) -> Response:
        return self._ok(call, ops.create_graph(call.env, call.body), 201)

    def h_graph_get(self, call: Call) -> Response:
        return self._ok(call, ops.get_graph(call.env, self._handle(call, self._graph_name(call))))

    def h_graph_delete(self, call: Call) -> Response:
        ops.delete_graph(call.env, self._graph_name(call))
        return self._ok(call, {"deleted": True})

    def h_graph_save(self, call: Call) -> Response:
        return self._ok(call, ops.save_graph(call.env, self._handle(call, self._graph_name(call))))

    # ---- tenant: nodes ----------------------------------------------------------------------------------------------

    def h_nodes_list(self, call: Call) -> Response:
        t = call.query.get("type")
        if t is not None and not names.TYPE_RE.match(t):
            raise BadRequest("invalid node type", code="invalid_query")
        limit = ops.page_limit(call.env, call.query.get("limit"))
        return self._ok(call, ops.list_nodes(call.env, self._handle(call, self._graph_name(call)),
                                             t, call.query.get("cursor"), limit))

    def h_nodes_upsert(self, call: Call) -> Response:
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.upsert_nodes(call.env, h, call.body["nodes"]))

    def h_nodes_delete(self, call: Call) -> Response:
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.delete_nodes(call.env, h, call.body["nodes"]))

    def h_node_get(self, call: Call) -> Response:
        t, i = self._node_params(call)
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.get_node(call.env, h, t, i, self._bool_q(call, "include_vector")))

    def h_node_put(self, call: Call) -> Response:
        t, i = self._node_params(call)
        h = self._handle(call, self._graph_name(call))
        item = {"type": t, "id": i, **call.body}
        return self._ok(call, ops.upsert_nodes(call.env, h, [item]))

    def h_node_patch(self, call: Call) -> Response:
        t, i = self._node_params(call)
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.patch_node(call.env, h, t, i, call.body))

    def h_node_delete(self, call: Call) -> Response:
        t, i = self._node_params(call)
        h = self._handle(call, self._graph_name(call))
        out = ops.delete_nodes(call.env, h, [{"type": t, "id": i}])
        if not out["deleted"]:
            raise NotFound("node not found")
        return self._ok(call, out)

    # ---- tenant: edges ----------------------------------------------------------------------------------------------

    def h_edges_list(self, call: Call) -> Response:
        rel = call.query.get("rel")
        if rel is not None and not names.TYPE_RE.match(rel):
            raise BadRequest("invalid relationship type", code="invalid_query")
        limit = ops.page_limit(call.env, call.query.get("limit"))
        return self._ok(call, ops.list_edges(call.env, self._handle(call, self._graph_name(call)),
                                             rel, call.query.get("cursor"), limit))

    def h_edges_upsert(self, call: Call) -> Response:
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.upsert_edges(call.env, h, call.body["edges"]))

    def h_edges_delete(self, call: Call) -> Response:
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.delete_edges(call.env, h, call.body["edges"]))

    # ---- tenant: retrieval -------------------------------------------------------------------------------------------

    def h_search(self, call: Call) -> Response:
        return self._ok(call, ops.search(call.env, self._handle(call, self._graph_name(call)), call.body))

    def h_multi_hop(self, call: Call) -> Response:
        return self._ok(call, ops.multi_hop(call.env, self._handle(call, self._graph_name(call)), call.body))

    def h_path(self, call: Call) -> Response:
        return self._ok(call, ops.path(call.env, self._handle(call, self._graph_name(call)), call.body))

    def h_subgraph(self, call: Call) -> Response:
        return self._ok(call, ops.subgraph(call.env, self._handle(call, self._graph_name(call)), call.body))

    def h_similar(self, call: Call) -> Response:
        return self._ok(call, ops.similar(call.env, self._handle(call, self._graph_name(call)), call.body))

    def h_stale(self, call: Call) -> Response:
        h = self._handle(call, self._graph_name(call))
        limit = ops.page_limit(call.env, call.query.get("limit"))
        cur = ops.decode_cursor(call.query.get("cursor"), 1)
        try:
            offset = int(cur[0]) if cur else 0
        except (TypeError, ValueError):
            raise BadRequest("invalid cursor", code="invalid_cursor") from None
        return self._ok(call, ops.stale(call.env, h, self._bool_q(call, "include_unknown"), limit, max(0, offset)))

    def h_reembed(self, call: Call) -> Response:
        h = self._handle(call, self._graph_name(call))
        return self._ok(call, ops.reembed(call.env, h, call.body.get("include_unknown", False)))

    # ---- replication source (primary only; separate credential) ---------------------------------------------------

    def _repl_ids(self, call: Call, with_graph: bool = True):
        t = call.params["t"]
        if not names.is_tenant_id(t):
            raise NotFound("no such resource")
        g = call.params.get("g")
        if with_graph and (g is None or not names.is_graph_name(g)):
            raise NotFound("no such resource")
        return t, g

    def h_repl_index(self, call: Call) -> Response:
        return self._ok(call, build_index(self.cfg.data_dir))

    def _stream(self, path: Path) -> Response:
        try:
            size = path.stat().st_size
            handle = open(path, "rb")
        except OSError:
            raise NotFound("no such resource") from None

        def chunks():
            with handle:
                while True:
                    data = handle.read(1 << 20)
                    if not data:
                        return
                    yield data
        return Response(200, {"Content-Type": "application/octet-stream"}, stream=chunks(), length=size)

    def h_repl_tenant(self, call: Call) -> Response:
        t, _ = self._repl_ids(call, with_graph=False)
        return self._stream(names.tenant_dir(self.cfg.data_dir, t) / "tenant.json")

    def h_repl_meta(self, call: Call) -> Response:
        t, g = self._repl_ids(call)
        return self._stream(names.graph_dir(self.cfg.data_dir, t, g) / "meta.json")

    def h_repl_file(self, call: Call) -> Response:
        t, g = self._repl_ids(call)
        gen, fname = call.params["gen"], call.params["f"]
        if not names.GENERATION_RE.match(gen) or fname not in GENERATION_FILES:
            raise NotFound("no such resource")
        return self._stream(names.graph_dir(self.cfg.data_dir, t, g) / f"gen-{gen}" / fname)
