"""A stateless router that hash-shards tenants across backend nodes.

It holds no data and no credentials. The tenant id is read from the API key
(``igv_<tenant>.<key-id>.<secret>``), mapped to a shard by rendezvous
(highest-random-weight) hashing, and the request is forwarded unchanged,
``Authorization`` header included. **The backend authenticates**: the router
only uses the tenant id as a routing hint, so forging a key that names another
tenant gets the request sent to that tenant's shard, where the key fails
verification. The router cannot widen anyone's access.

What it does not do, deliberately and honestly:

* No resharding. Rendezvous hashing means adding or removing a shard moves only
  the tenants whose winner changed, but *moving their data is manual* (copy the
  tenant directory to the new owner's data dir, or pin the tenant with
  ``router.pins`` while you do).
* No failover. If a shard's primary is down, its tenants get 502/503. Reads can
  fall back from a replica to the primary, never the other way round.
* No retries of writes, ever. Reads are retried once on the next candidate.
* No cross-tenant or cross-shard queries, joins or listings.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import secrets
import threading
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from . import names
from .auth import tenant_id_from_key
from .config import ServerConfig
from .errors import ApiError, BadGateway, NotFound, Unauthorized
from .httpd import Request, Response
from .logs import log
from .service import bearer_token, split_path

READ_SUFFIXES = {"search", "multi_hop", "path", "subgraph", "similar"}
FORWARD_REQUEST_HEADERS = ("authorization", "content-type", "accept", "x-request-id")
FORWARD_RESPONSE_HEADERS = ("content-type", "retry-after", "x-request-id", "www-authenticate", "allow",
                            "x-snapshot-generation", "x-snapshot-age-seconds")
#: Backend answers that mean "this replica cannot serve you right now, ask the primary".
REPLICA_FALLBACK_CODES = (b"replica_stale", b"replica_not_synced", b"not_ready")


def rendezvous(tenant_id: str, shard_names: List[str]) -> str:
    best, best_score = "", -1
    for name in shard_names:
        score = int.from_bytes(hashlib.sha256(f"{name}\x00{tenant_id}".encode()).digest()[:8], "big")
        if score > best_score:
            best, best_score = name, score
    return best


class _Backend:
    def __init__(self, shard: str, kind: str, url: str, timeout: float):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"bad backend URL for shard {shard}")
        self.shard, self.kind, self.timeout = shard, kind, timeout
        self.scheme, self.host = parts.scheme, parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")
        self._idle: List[Any] = []
        self._idle_lock = threading.Lock()
        self.max_idle = 64

    def label(self) -> str:
        return f"{self.shard}/{self.kind}"

    def _new(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def request(self, method: str, target: str, headers: Dict[str, str], body: bytes,
                reuse: bool = False) -> Tuple[int, List[Tuple[str, str]], bytes]:
        """One upstream round trip.

        ``reuse`` (reads only) draws from a pool of keep-alive connections; a pooled connection
        that turns out to be dead is replaced by a fresh one and the read repeated. Writes always
        use a fresh connection and are never repeated: after a failure the router cannot know
        whether the backend applied the change.
        """
        conn = None
        if reuse:
            with self._idle_lock:
                conn = self._idle.pop() if self._idle else None
        fresh = conn is None
        if conn is None:
            conn = self._new()
        try:
            conn.request(method, self.prefix + target, body=body or None, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except (OSError, http.client.HTTPException):
            conn.close()
            if not fresh:
                return self.request(method, target, headers, body, reuse=False)
            raise
        if reuse and not resp.will_close:
            with self._idle_lock:
                if len(self._idle) < self.max_idle:
                    self._idle.append(conn)
                    conn = None
        if conn is not None:
            conn.close()
        return resp.status, resp.getheaders(), data


class Router:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        rc = cfg.router
        if not rc.shards:
            raise ValueError("router needs at least one shard")
        self.shard_names = [s["name"] for s in rc.shards]
        self.primaries: Dict[str, _Backend] = {}
        self.replicas: Dict[str, List[_Backend]] = {}
        for s in rc.shards:
            self.primaries[s["name"]] = _Backend(s["name"], "primary", s["primary"], rc.upstream_timeout_seconds)
            self.replicas[s["name"]] = [_Backend(s["name"], "replica", u, rc.upstream_timeout_seconds)
                                        for u in s.get("replicas", [])]
        self._rr = 0
        self._rr_lock = threading.Lock()

    def shard_for(self, tenant_id: str) -> str:
        pinned = self.cfg.router.pins.get(tenant_id)
        return pinned if pinned else rendezvous(tenant_id, self.shard_names)

    def _next_replica(self, shard: str) -> Optional[_Backend]:
        reps = self.replicas[shard]
        if not reps:
            return None
        with self._rr_lock:
            self._rr += 1
            return reps[self._rr % len(reps)]

    @staticmethod
    def _is_read(method: str, segs: List[str]) -> bool:
        if method == "GET":
            return True
        return (method == "POST" and len(segs) == 4 and segs[0] == "v1" and segs[1] == "graphs"
                and segs[3] in READ_SUFFIXES)

    def handle(self, req: Request) -> Response:
        rid = req.header("x-request-id")
        if rid is None or not names.REQUEST_ID_RE.match(rid):
            rid = secrets.token_hex(8)
        try:
            resp = self._route(req, rid)
        except ApiError as exc:
            resp = Response.json(exc.status, exc.to_body(rid), exc.headers)
        except Exception:
            logging.getLogger("igv").exception("router error", extra={"fields": {"request_id": rid}})
            resp = Response.json(500, ApiError("an internal error occurred").to_body(rid))
        resp.headers["X-Request-Id"] = rid
        return resp

    def _route(self, req: Request, rid: str) -> Response:
        segs, _ = split_path(req.target)
        if segs == ["healthz"] and req.method == "GET":
            return Response.json(200, {"status": "ok", "role": "router"})
        if not segs or segs[0] != "v1":
            raise NotFound("no such resource")       # /metrics, /_repl, ... are never proxied
        token = bearer_token(req)
        tid = tenant_id_from_key(token) if token else None
        if tid is None:
            raise Unauthorized("missing or invalid API key",
                               headers={"WWW-Authenticate": 'Bearer realm="igv"'})
        shard = self.shard_for(tid)

        read = self._is_read(req.method, segs)
        headers = {"Content-Length": str(len(req.body)), "X-Request-Id": rid,
                   "X-Forwarded-For": req.peer, "Connection": "keep-alive" if read else "close"}
        for name in FORWARD_REQUEST_HEADERS:
            if name == "x-request-id":
                continue
            vals = req.header_all(name)
            if len(vals) > 1 and name == "authorization":
                # Ambiguous credentials: let the backend say 401 for the whole request.
                headers["Authorization"] = "Bearer invalid"
            elif vals:
                headers[name.title()] = vals[0]

        forced_primary = (req.header("x-igv-consistency") or "").lower() == "primary"
        candidates: List[_Backend] = []
        if read and not forced_primary and self.cfg.router.read_preference == "replica_preferred":
            rep = self._next_replica(shard)
            if rep is not None:
                candidates.append(rep)
        candidates.append(self.primaries[shard])

        last_err: Optional[Exception] = None
        for i, backend in enumerate(candidates):
            last = i == len(candidates) - 1
            try:
                status, hdrs, data = backend.request(req.method, req.target, headers, req.body, reuse=read)
            except (OSError, http.client.HTTPException) as exc:
                last_err = exc
                log(logging.WARNING, "backend unreachable", shard=shard, kind=backend.kind,
                    error=type(exc).__name__)
                if last or not read:
                    break
                continue
            if (not last and backend.kind == "replica" and status == 503
                    and any(code in data for code in REPLICA_FALLBACK_CODES)):
                continue
            return self._relay(status, hdrs, data, backend)
        raise BadGateway("the backend for this tenant is unavailable", code="backend_unavailable",
                         headers={"Retry-After": "2"})

    def _relay(self, status: int, hdrs: List[Tuple[str, str]], data: bytes, backend: _Backend) -> Response:
        out: Dict[str, str] = {}
        for k, v in hdrs:
            if k.lower() in FORWARD_RESPONSE_HEADERS:
                out[k] = v
        if self.cfg.router.expose_backend_header:
            out["X-Igv-Backend"] = backend.label()
        return Response(status, out, data)
