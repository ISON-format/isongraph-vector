"""A small, bounded HTTP/1.1 server on the standard library.

What it does to stay correct under load and abuse:

* A fixed worker pool (``max_workers``) and a bounded accept queue
  (``accept_queue``). A connection that arrives when both are full is answered
  ``503`` immediately and closed; there is no unbounded thread-per-connection.
* Every socket operation has a timeout (``socket_timeout_seconds``), body reads
  also have a total deadline, so a slow-loris client cannot hold a worker.
* The request line and headers are bounded; ``Transfer-Encoding`` (chunked
  bodies) is refused so there is a single, unambiguous framing (Content-Length);
  duplicate or malformed Content-Length is refused; the body is read only after
  the declared length passes the size limit.
* Graceful shutdown: stop accepting, let in-flight requests finish, close idle
  keep-alive connections.

It is *not* a hardened internet-facing server: no TLS, no HTTP/2, no
protection against many slow connections beyond the limits above. Run it behind a
TLS-terminating reverse proxy (see README).
"""

from __future__ import annotations

import io
import json
import logging
import select
import socket
import socketserver
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import ServerConfig
from .errors import ApiError
from .logs import log


@dataclass
class Request:
    method: str
    target: str                                  # raw request-target (path + query)
    headers: List[Tuple[str, str]]               # (lowercased name, value), in arrival order
    body: bytes
    peer: str

    def header_all(self, name: str) -> List[str]:
        name = name.lower()
        return [v for k, v in self.headers if k == name]

    def header(self, name: str) -> Optional[str]:
        vals = self.header_all(name)
        return vals[0] if vals else None


@dataclass
class Response:
    status: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    stream: Optional[Iterator[bytes]] = None
    length: Optional[int] = None                 # required with ``stream``

    @staticmethod
    def json(status: int, obj: Any, headers: Optional[Dict[str, str]] = None) -> "Response":
        data = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        return Response(status, h, data)


_REASONS = {
    200: "OK", 201: "Created", 204: "No Content", 400: "Bad Request", 401: "Unauthorized",
    403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed", 409: "Conflict",
    411: "Length Required", 413: "Payload Too Large", 414: "URI Too Long",
    415: "Unsupported Media Type", 429: "Too Many Requests", 431: "Request Header Fields Too Large",
    500: "Internal Server Error", 501: "Not Implemented", 502: "Bad Gateway",
    503: "Service Unavailable", 504: "Gateway Timeout",
}

BODY_METHODS = ("POST", "PUT", "PATCH")
MAX_TARGET = 4096
BODY_TOTAL_SECONDS = 30.0


class _DeadlineReader(io.RawIOBase):
    """Socket reader with a *total* deadline on top of the per-operation timeout.

    A per-``recv`` timeout alone lets a client that drips one byte every few seconds hold a
    worker for as long as it likes. ``deadline`` (absolute, monotonic) caps the whole read.
    ``nonblocking`` lets the handler ask "is anything buffered or waiting?" without blocking.
    """

    def __init__(self, sock: socket.socket, per_op: float):
        self.sock = sock
        self.per_op = per_op
        self.deadline: Optional[float] = None
        self.nonblocking = False

    def readable(self) -> bool:
        return True

    def readinto(self, buf):
        if self.nonblocking:
            self.sock.setblocking(False)
            try:
                return self.sock.recv_into(buf)
            except BlockingIOError:
                return None
        timeout = self.per_op
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("request read deadline exceeded")
            timeout = min(timeout, left)
        self.sock.settimeout(timeout)
        return self.sock.recv_into(buf)


HEADER_TOTAL_SECONDS = 15.0


def make_handler(app: Any, cfg: ServerConfig):
    """Build a request-handler class bound to ``app`` (anything with ``handle(Request) -> Response``)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = cfg.socket_timeout_seconds
        server_version = "igv"
        sys_version = ""
        _served = 0

        # -- plumbing -----------------------------------------------------------

        def setup(self):
            self.connection = self.request
            self.connection.settimeout(self.timeout)
            self._raw = _DeadlineReader(self.connection, self.timeout)
            self.rfile = io.BufferedReader(self._raw, 65536)
            self.wfile = self.connection.makefile("wb", 0)

        def _await_next_request(self) -> bool:
            """Wait (politely, in short slices) for the next request to start.

            Returns False when the connection should end: peer closed, idle too long, or the
            server is shutting down. Polling in slices is what lets a graceful shutdown
            close idle keep-alive connections promptly on every platform.
            """
            end = time.monotonic() + cfg.socket_timeout_seconds
            self._raw.deadline = None
            while True:
                if self.server.stopping:
                    return False
                self._raw.nonblocking = True
                try:
                    have = self.rfile.peek(1)
                except OSError:
                    return False
                finally:
                    self._raw.nonblocking = False
                if have:
                    return True
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0.25)
                except (OSError, ValueError):
                    return False
                if readable:
                    return True            # data, or EOF (readline will then return b"")
                if time.monotonic() > end:
                    return False

        def log_message(self, *args, **kwargs):   # we log ourselves, structured
            pass

        def version_string(self) -> str:
            return "igv"

        def send_error(self, code, message=None, explain=None):
            # Raised by the stdlib parser (bad request line, header too long, ...).
            self.request_version = "HTTP/1.1"      # never answer in HTTP/0.9 (no status line)
            err = ApiError(message or _REASONS.get(code, "error"), status=code,
                           code="bad_request" if code < 500 else "server_error")
            self._send(Response.json(code, err.to_body("-")), close=True)

        def handle_one_request(self):
            self.request_version = "HTTP/1.1"      # defaults, so error replies work before parse_request
            self.command = None
            self.requestline = ""
            self.close_connection = True if self.close_connection else False
            try:
                if not self._await_next_request():
                    self.close_connection = True
                    return
                self._raw.deadline = time.monotonic() + HEADER_TOTAL_SECONDS
                self.raw_requestline = self.rfile.readline(MAX_TARGET + 64)
                if not self.raw_requestline:
                    self.close_connection = True
                    return
                if len(self.raw_requestline) > MAX_TARGET + 63:
                    self.send_error(414)
                    return
                if not self.parse_request():
                    return
                self.server.mark_busy(self.connection, True)
                try:
                    self._serve()
                finally:
                    self.server.mark_busy(self.connection, False)
                self.wfile.flush()
            except (socket.timeout, TimeoutError):
                self.close_connection = True
            except (ConnectionError, OSError):
                self.close_connection = True

        # -- one request ------------------------------------------------------------

        def _serve(self) -> None:
            self._served += 1
            if self.server.stopping or self._served >= cfg.max_requests_per_connection:
                self.close_connection = True
            headers = [(k.lower(), v) for k, v in self.headers.items()]
            if sum(len(k) + len(v) + 4 for k, v in headers) > cfg.max_header_bytes:
                self.send_error(431, "headers too large")
                return
            if any(k == "transfer-encoding" for k, _ in headers):
                self._send(Response.json(400, ApiError(
                    "Transfer-Encoding is not supported; send Content-Length",
                    code="unsupported_transfer_encoding", status=400).to_body("-")), close=True)
                return
            lengths = [v for k, v in headers if k == "content-length"]
            body = b""
            if len(lengths) > 1 or (lengths and not lengths[0].strip().isdigit()):
                self._send(Response.json(400, ApiError(
                    "invalid Content-Length", code="invalid_request", status=400).to_body("-")), close=True)
                return
            length = int(lengths[0]) if lengths else 0
            if self.command in BODY_METHODS and not lengths:
                self._send(Response.json(411, ApiError(
                    "Content-Length is required", code="length_required", status=411).to_body("-")), close=True)
                return
            if length > cfg.max_body_bytes:
                self._send(Response.json(413, ApiError(
                    f"body larger than {cfg.max_body_bytes} bytes", code="payload_too_large",
                    status=413).to_body("-")), close=True)
                self._drain(length)
                return
            if length:
                self._raw.deadline = time.monotonic() + BODY_TOTAL_SECONDS
                body = self._read_body(length)
                if body is None:
                    self.close_connection = True
                    return
            req = Request(self.command, self.path, headers, body, self.client_address[0])
            try:
                resp = app.handle(req)
            except Exception:       # backstop: App.handle should never raise
                logging.getLogger("igv").exception("unhandled error in app.handle")
                resp = Response.json(500, ApiError("internal error").to_body("-"))
            self._send(resp, close=self.close_connection)

        def _drain(self, length: int) -> None:
            """After refusing a body, read (and discard) a bounded amount of it, briefly.

            Closing a socket that still has unread request data makes the OS send a reset,
            and the client can lose the response we just wrote; a short lingering read lets
            an honest-but-oversize client actually see the 413.
            """
            remaining = min(length, 4 * 1024 * 1024)
            end = time.monotonic() + 2.0
            try:
                while remaining > 0 and time.monotonic() < end:
                    data = self.rfile.read(min(remaining, 65536))
                    if not data:
                        break
                    remaining -= len(data)
            except OSError:
                pass

        def _read_body(self, length: int) -> Optional[bytes]:
            chunks = []
            remaining = length
            end = time.monotonic() + BODY_TOTAL_SECONDS
            while remaining:
                if time.monotonic() > end:
                    return None
                data = self.rfile.read(min(remaining, 65536))
                if not data:
                    return None
                chunks.append(data)
                remaining -= len(data)
            return b"".join(chunks)

        def _send(self, resp: Response, close: bool = False) -> None:
            if resp.stream is not None:
                length = resp.length
            else:
                length = len(resp.body)
            self.send_response_only(resp.status, _REASONS.get(resp.status, "Status"))
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in resp.headers.items():
                self.send_header(k, v)
            if close or self.close_connection:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            if resp.stream is not None:
                sent = 0
                for chunk in resp.stream:
                    self.wfile.write(chunk)
                    sent += len(chunk)
                if sent != length:                 # the file changed under us: do not lie
                    self.close_connection = True
            elif resp.body:
                self.wfile.write(resp.body)
            if (close or self.close_connection) and resp.status >= 400 and resp.stream is None:
                self._linger()

        def _linger(self, max_bytes: int = 1 << 17, budget: float = 0.25) -> None:
            """Soak up whatever the client is still sending before closing (see _drain)."""
            end = time.monotonic() + budget
            got = 0
            try:
                self.connection.settimeout(0.05)
                while got < max_bytes and time.monotonic() < end:
                    data = self.connection.recv(65536)
                    if not data:
                        break
                    got += len(data)
            except OSError:
                pass

        def _method(self):
            # The stdlib dispatches on do_<METHOD>; route them all to _serve via
            # handle_one_request. Unknown methods get a 501 from parse_request.
            pass

    for m in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        setattr(Handler, f"do_{m}", Handler._method)
    return Handler


class BoundedHTTPServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, address, app: Any, cfg: ServerConfig):
        self.cfg = cfg
        self.stopping = False
        self.request_queue_size = cfg.accept_queue
        self._pool = ThreadPoolExecutor(max_workers=cfg.max_workers, thread_name_prefix="igv-http")
        self._slots = threading.BoundedSemaphore(cfg.max_workers + cfg.accept_queue)
        self._conn_lock = threading.Lock()
        self._conns: Dict[socket.socket, bool] = {}     # socket -> currently in a request
        super().__init__(address, make_handler(app, cfg), bind_and_activate=True)

    # connection tracking, for shutdown --------------------------------------------------
    def mark_busy(self, sock: socket.socket, busy: bool) -> None:
        with self._conn_lock:
            if sock in self._conns:
                self._conns[sock] = busy

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:     # overloaded: tell the client, cheaply, and drop the connection
                request.settimeout(1.0)
                body = b'{"error":{"code":"overloaded","message":"server is at capacity","request_id":"-"}}'
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                                b"Retry-After: 1\r\nConnection: close\r\nContent-Length: "
                                + str(len(body)).encode() + b"\r\n\r\n" + body)
                request.settimeout(0.2)
                request.recv(65536)           # take the client's request off the wire: closing with
            except OSError:                   # unread data would reset the connection under the 503
                pass
            self.shutdown_request(request)
            return
        with self._conn_lock:
            self._conns[request] = False
        self._pool.submit(self._run, request, client_address)

    def _run(self, request, client_address) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            with self._conn_lock:
                self._conns.pop(request, None)
            self.shutdown_request(request)
            self._slots.release()

    def handle_error(self, request, client_address):   # keep stdlib's traceback print off stderr
        logging.getLogger("igv").debug("connection error", exc_info=True)

    def stop(self, grace: float = 10.0) -> None:
        """Stop accepting, let in-flight requests finish, drop idle connections."""
        self.stopping = True
        self.shutdown()                 # ends serve_forever()
        self.server_close()             # closes the listening socket
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            with self._conn_lock:
                idle = [s for s, busy in self._conns.items() if not busy]
                alive = bool(self._conns)
            for s in idle:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if not alive:
                break
            time.sleep(0.05)
        self._pool.shutdown(wait=False, cancel_futures=True)
