"""Shared fixtures. Binds ``isongraph_vector`` from the checkout when it is not installed
(the library's directory name, ``isongraph_vector-py``, is not importable as-is)."""

import importlib.util
import json
import sys
import http.client
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SERVER_ROOT = HERE.parent
REPO_ROOT = SERVER_ROOT.parent

if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


def _bind_library() -> None:
    if "isongraph_vector" in sys.modules or importlib.util.find_spec("isongraph_vector") is not None:
        return
    pkg_dir = REPO_ROOT / "isongraph_vector-py"
    spec = importlib.util.spec_from_file_location(
        "isongraph_vector", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["isongraph_vector"] = module
    spec.loader.exec_module(module)


_bind_library()

from isongraph_vector_server import auth  # noqa: E402
from isongraph_vector_server.config import ServerConfig  # noqa: E402
from isongraph_vector_server.server import RouterServer, Server  # noqa: E402

DIM = 16


class Client:
    """Minimal HTTP client; one connection per call so tests are independent."""

    def __init__(self, url: str, key: str = None):
        self.host, self.port = url.replace("http://", "").split(":")
        self.port = int(self.port)
        self.key = key

    def request(self, method, path, body=None, key="__default__", headers=None, raw=None, timeout=20):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        hdrs = {}
        k = self.key if key == "__default__" else key
        if k:
            hdrs["Authorization"] = f"Bearer {k}"
        data = None
        if raw is not None:
            data = raw
            hdrs["Content-Type"] = "application/json"
        elif body is not None:
            data = json.dumps(body).encode()
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        try:
            conn.request(method, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            payload = resp.read()
            try:
                parsed = json.loads(payload) if payload else None
            except ValueError:
                parsed = payload.decode("utf-8", "replace")
            return resp.status, parsed, dict(resp.getheaders())
        finally:
            conn.close()

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body, **kw)

    def put(self, path, body=None, **kw):
        return self.request("PUT", path, body, **kw)

    def patch(self, path, body=None, **kw):
        return self.request("PATCH", path, body, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)


class Harness:
    """A running node plus helpers to create tenants and clients."""

    def __init__(self, tmp_path: Path, encoder=None, **overrides):
        self.data_dir = str(tmp_path / "data")
        cfg = ServerConfig(data_dir=self.data_dir, host="127.0.0.1", port=0, log_level="CRITICAL")
        cfg.autosave_interval_seconds = 0
        cfg.registry_recheck_seconds = 0.0
        cfg.encoder.kind = "mock"
        cfg.encoder.dimension = DIM
        cfg.quotas.rate_per_second = 10_000
        cfg.quotas.rate_burst = 10_000
        cfg.quotas.max_inflight = 64
        cfg.max_concurrent_requests = 64
        cfg.failed_auth_per_minute = 10_000
        for k, v in overrides.items():
            if "." in k:
                a, b = k.split(".")
                setattr(getattr(cfg, a), b, v)
            else:
                setattr(cfg, k, v)
        self.cfg = cfg
        self.clients = []
        self._encoder = encoder
        self.server = Server(cfg, encoder=encoder, configure_logging=False).start()

    def tenant(self, tid, role="rw", quotas=None):
        try:
            auth.create_tenant(self.data_dir, tid, quotas)
        except FileExistsError:
            pass
        key = auth.add_key(self.data_dir, tid, role)
        c = Client(self.server.url, key)
        self.clients.append(c)
        return c

    def restart(self, graceful=True, **overrides):
        """Stop the node (gracefully: final snapshot) and start a new one on the same data dir."""
        if graceful:
            self.server.stop()
        else:
            self.server.httpd.stop()
            self.server.engine._stop.set()
            self.server.engine._dir_lock.close()      # as if the process had died: no final save
            self.server.engine._dir_lock = None
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        self.server = Server(self.cfg, encoder=self._encoder, configure_logging=False).start()
        for c in self.clients:
            c.host, c.port = self.server.address[0], self.server.address[1]
        return self.server

    def anon(self):
        return Client(self.server.url, None)

    def stop(self):
        self.server.stop()


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(**overrides):
        h = Harness(tmp_path / f"h{len(made)}", **overrides)
        made.append(h)
        return h

    yield make
    for h in made:
        try:
            h.stop()
        except Exception:
            pass


@pytest.fixture
def replica_of(harness):
    """Factory: a replica node pulling from ``source`` (a ``dir:`` path or a primary URL)."""
    def make(source, **overrides):
        opts = {"role": "replica", "replica.source": source, "replica.sync_interval_seconds": 3600,
                "replica.max_staleness_seconds": 3600, "replica.autostart": False}
        opts.update(overrides)
        h = harness(**opts)
        return h
    return make


@pytest.fixture
def node(harness):
    return harness()


@pytest.fixture
def alice(node):
    return node.tenant("alice")


def create_graph(client, name="kb", **body):
    status, resp, _ = client.post("/v1/graphs", {"name": name, **body})
    assert status == 201, (status, resp)
    return resp


def seed(client, graph="kb"):
    """A small graph: three people, two edges."""
    s, r, _ = client.post(f"/v1/graphs/{graph}/nodes", {"nodes": [
        {"type": "person", "id": 1, "properties": {"name": "Alice", "description": "builds payment APIs", "age": 31}},
        {"type": "person", "id": 2, "properties": {"name": "Bob", "description": "trains machine learning models", "age": 45}},
        {"type": "person", "id": "carol", "properties": {"name": "Carol", "description": "plans product roadmaps", "age": 38}},
    ]})
    assert s == 200, r
    s, r, _ = client.post(f"/v1/graphs/{graph}/edges", {"edges": [
        {"rel": "KNOWS", "source": {"type": "person", "id": 1}, "target": {"type": "person", "id": 2}, "properties": {"weight": 0.9}},
        {"rel": "KNOWS", "source": {"type": "person", "id": 2}, "target": {"type": "person", "id": "carol"}},
    ]})
    assert s == 200, r
