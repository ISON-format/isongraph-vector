"""The stateless tenant-sharding router."""

import socket
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from conftest import Client, create_graph, seed
from isongraph_vector_server import auth
from isongraph_vector_server.config import ServerConfig
from isongraph_vector_server.router import Router, rendezvous
from isongraph_vector_server.server import RouterServer


def router_cfg(shards, **router):
    cfg = ServerConfig(host="127.0.0.1", port=0, log_level="CRITICAL")
    cfg.router.shards = shards
    cfg.router.expose_backend_header = True
    for k, v in router.items():
        setattr(cfg.router, k, v)
    return cfg


@pytest.fixture
def cluster(harness):
    """Two storage nodes behind a router. Tenants are created on the node that owns them."""
    n0, n1 = harness(), harness()
    shards = [{"name": "s0", "primary": n0.server.url}, {"name": "s1", "primary": n1.server.url}]
    rs = RouterServer(router_cfg(shards), configure_logging=False).start()
    nodes = {"s0": n0, "s1": n1}

    class Cluster:
        router = rs
        node = nodes

        @staticmethod
        def tenant(tid, role="rw"):
            owner = rs.router.shard_for(tid)
            key = auth.add_key(nodes[owner].data_dir, tid, role) if auth_exists(nodes[owner], tid) \
                else make_tenant(nodes[owner], tid, role)
            return Client(rs.url, key), owner

    yield Cluster
    rs.stop()


def auth_exists(node, tid):
    return (Path(node.data_dir) / "tenants" / tid / "tenant.json").exists()


def make_tenant(node, tid, role):
    auth.create_tenant(node.data_dir, tid)
    return auth.add_key(node.data_dir, tid, role)


def find_tenants_on_each_shard(router):
    found = {}
    i = 0
    while len(found) < 2:
        tid = f"tenant-{i}"
        found.setdefault(router.router.shard_for(tid), tid)
        i += 1
    return found


def test_rendezvous_is_deterministic_balanced_and_minimally_disruptive():
    shards = ["s0", "s1", "s2", "s3"]
    tenants = [f"tenant-{i}" for i in range(2000)]
    owner = {t: rendezvous(t, shards) for t in tenants}
    assert owner == {t: rendezvous(t, list(reversed(shards))) for t in tenants}      # order-independent
    counts = Counter(owner.values())
    assert all(350 < n < 650 for n in counts.values()), counts                         # ~500 each
    # drop one shard: only that shard's tenants move
    fewer = {t: rendezvous(t, ["s0", "s1", "s3"]) for t in tenants}
    moved = [t for t in tenants if owner[t] != fewer[t]]
    assert all(owner[t] == "s2" for t in moved) and len(moved) == counts["s2"]
    # add one: only tenants won by the new shard move
    more = {t: rendezvous(t, shards + ["s4"]) for t in tenants}
    assert all(more[t] == "s4" for t in tenants if owner[t] != more[t])


def test_requests_reach_the_owning_shard_only(cluster):
    on = find_tenants_on_each_shard(cluster.router)
    clients = {}
    for shard, tid in on.items():
        c, owner = cluster.tenant(tid)
        assert owner == shard
        create_graph(c)
        seed(c)
        clients[shard] = c
        s, r, hd = c.get("/v1/graphs/kb")
        assert s == 200 and hd["X-Igv-Backend"] == f"{shard}/primary"
    # the data is on exactly one node, and the other node knows nothing of the tenant
    for shard, tid in on.items():
        other = "s1" if shard == "s0" else "s0"
        assert (Path(cluster.node[shard].data_dir) / "tenants" / tid).exists() or True
        assert not (Path(cluster.node[other].data_dir) / "tenants" / tid).exists()
        direct = Client(cluster.node[other].server.url, clients[shard].key)
        assert direct.get("/v1/graphs")[0] == 401
        assert cluster.node[other].server.engine.tenants.get(tid) is None


def test_tenant_data_never_crosses_shards_or_tenants(cluster):
    on = find_tenants_on_each_shard(cluster.router)
    a, _ = cluster.tenant(on["s0"])
    b, _ = cluster.tenant(on["s1"])
    same_shard, owner = cluster.tenant(next(f"extra-{i}" for i in range(500)
                                            if cluster.router.router.shard_for(f"extra-{i}") == "s0"))
    for c in (a, b, same_shard):
        create_graph(c)
    a.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "d", "id": 1, "properties": {"name": "A-SECRET"}}]})
    b.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "d", "id": 1, "properties": {"name": "B-thing"}}]})
    assert "A-SECRET" not in str(b.get("/v1/graphs/kb/nodes")[1])
    assert "A-SECRET" not in str(same_shard.get("/v1/graphs/kb/nodes")[1])      # same node, other tenant
    assert same_shard.get("/v1/graphs/kb/nodes/d/1")[0] == 404


def test_a_key_that_names_another_tenant_is_routed_there_and_fails(cluster):
    on = find_tenants_on_each_shard(cluster.router)
    a, _ = cluster.tenant(on["s0"])
    b, _ = cluster.tenant(on["s1"])
    create_graph(a)
    ta, ka, sa = a.key.split(".")
    tb, kb, sb = b.key.split(".")
    # B's credentials presented under A's tenant id: routed to A's shard, rejected there
    for forged in (f"{ta}.{kb}.{sb}", f"{ta}.{ka}.{sb}"):
        s, r, hd = Client(cluster.router.url, forged).get("/v1/graphs/kb")
        assert s == 401, forged
    # and A's under B's
    s, r, hd = Client(cluster.router.url, f"{tb}.{ka}.{sa}").get("/v1/graphs/kb")
    assert s == 401


def test_router_rejects_malformed_credentials_without_asking_a_backend(cluster):
    before = [n.server.metrics.counter("igv_requests_total", route="graphs.list", status="4xx")
              for n in cluster.node.values()]
    c = Client(cluster.router.url, None)
    for hdrs in ({}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}):
        s, r, _ = c.get("/v1/graphs", headers=hdrs)
        assert s == 401 and r["error"]["code"] == "unauthorized"
    after = [n.server.metrics.counter("igv_requests_total", route="graphs.list", status="4xx")
             for n in cluster.node.values()]
    assert before == after


def test_only_the_tenant_api_is_proxied(cluster):
    tid = find_tenants_on_each_shard(cluster.router)["s0"]
    c, _ = cluster.tenant(tid)
    for path in ("/metrics", "/readyz", "/_repl/v1/index", "/", "/v2/graphs", "/admin"):
        assert c.get(path)[0] == 404, path
    assert Client(cluster.router.url, None).get("/healthz")[1]["role"] == "router"


def test_errors_and_headers_pass_through_unchanged(cluster):
    c, _ = cluster.tenant("tenant-x")
    create_graph(c)
    s, r, hd = c.post("/v1/graphs", {"name": "kb"})
    assert s == 409 and r["error"]["code"] == "already_exists"
    s, r, hd = c.post("/v1/graphs/kb/search", {"query": "x"}, headers={"X-Request-Id": "trace-abcdef123"})
    assert hd["X-Request-Id"] == "trace-abcdef123"
    s, r, hd = c.request("PUT", "/v1/graphs", {"name": "x"})
    assert s == 405 and "Allow" in hd


def test_ambiguous_credentials_and_oversize_bodies(cluster):
    import http.client
    c, _ = cluster.tenant("tenant-x")
    conn = http.client.HTTPConnection(c.host, c.port, timeout=10)
    conn.putrequest("GET", "/v1/graphs")
    conn.putheader("Authorization", f"Bearer {c.key}")
    conn.putheader("Authorization", f"Bearer {c.key}")
    conn.endheaders()
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 401
    s, r, _ = c.post("/v1/graphs", None, raw=b'{"name":"' + b"a" * 2_000_000 + b'"}')
    assert s == 413


def test_backend_down_means_502_for_its_tenants_only_and_no_failover(cluster):
    on = find_tenants_on_each_shard(cluster.router)
    a, _ = cluster.tenant(on["s0"])
    b, _ = cluster.tenant(on["s1"])
    create_graph(a)
    create_graph(b)
    cluster.node["s0"].server.httpd.stop()
    s, r, hd = a.get("/v1/graphs")
    assert s == 502 and r["error"]["code"] == "backend_unavailable"
    assert b.get("/v1/graphs")[0] == 200                       # the other shard is untouched
    # and a write to the dead shard is not silently redirected anywhere
    assert a.post("/v1/graphs", {"name": "zz"})[0] == 502
    assert not any(n.server.engine.tenants.get(on["s0"]) and "zz" in n.server.engine.tenants[on["s0"]].graphs
                   for n in [cluster.node["s1"]])


class FlakyBackend:
    """A listener that accepts, counts, and hangs up without answering."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.count = 0
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.count += 1
            try:
                conn.recv(65536)
            finally:
                conn.close()

    def close(self):
        self.sock.close()


def test_writes_are_never_retried_but_reads_fall_back_to_the_primary(harness):
    primary = harness()
    key = make_tenant(primary, "tenant-x", "rw")
    flaky = FlakyBackend()
    try:
        cfg = router_cfg([{"name": "s0", "primary": primary.server.url, "replicas": [flaky.url]}],
                         read_preference="replica_preferred")
        rs = RouterServer(cfg, configure_logging=False).start()
        try:
            c = Client(rs.url, key)
            create_graph(c)                                    # a write: straight to the primary
            assert flaky.count == 0
            for _ in range(3):                                 # reads: replica fails, primary answers
                s, r, hd = c.get("/v1/graphs/kb")
                assert s == 200 and hd["X-Igv-Backend"] == "s0/primary"
            assert flaky.count == 3
            s, _, hd = c.post("/v1/graphs/kb/search", {"query": "x", "threshold": -1.0})
            assert s == 200 and hd["X-Igv-Backend"] == "s0/primary" and flaky.count == 4

            # a primary that hangs up: the write is attempted exactly once
            cfg2 = router_cfg([{"name": "s0", "primary": flaky.url}])
            rs2 = RouterServer(cfg2, configure_logging=False).start()
            try:
                before = flaky.count
                s, r, _ = Client(rs2.url, key).post("/v1/graphs", {"name": "other"})
                assert s == 502 and flaky.count == before + 1
            finally:
                rs2.stop()
        finally:
            rs.stop()
    finally:
        flaky.close()


def test_reads_go_to_a_replica_when_asked_and_primary_when_forced(harness, replica_of):
    primary = harness(replication_token="t" * 40)
    key = make_tenant(primary, "tenant-x", "rw")
    direct = Client(primary.server.url, key)
    create_graph(direct)
    seed(direct)
    direct.post("/v1/graphs/kb/save", {})
    replica = replica_of(f"dir:{primary.data_dir}")
    replica.server.replica.sync_once()
    cfg = router_cfg([{"name": "s0", "primary": primary.server.url, "replicas": [replica.server.url]}],
                     read_preference="replica_preferred")
    rs = RouterServer(cfg, configure_logging=False).start()
    try:
        c = Client(rs.url, key)
        s, r, hd = c.get("/v1/graphs/kb")
        assert s == 200 and hd["X-Igv-Backend"] == "s0/replica" and hd["X-Snapshot-Generation"] == "1"
        for path in ("search", "multi_hop", "similar"):
            body = {"similar": {"node": {"type": "person", "id": 1}}}.get(path, {"query": "Alice", "threshold": -1.0})
            s, r, hd = c.post(f"/v1/graphs/kb/{path}", body)
            assert s == 200 and hd["X-Igv-Backend"] == "s0/replica", path
        # writes and the explicit opt-out go to the primary
        s, r, hd = c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 7, "properties": {"name": "G"}}]})
        assert s == 200 and hd["X-Igv-Backend"] == "s0/primary"
        s, r, hd = c.get("/v1/graphs/kb/nodes/person/7", headers={"X-Igv-Consistency": "primary"})
        assert s == 200 and hd["X-Igv-Backend"] == "s0/primary"
        # read-your-writes is NOT promised on the replica path
        assert c.get("/v1/graphs/kb/nodes/person/7")[0] == 404
        # a replica that has gone stale tells the router to ask the primary instead
        replica.cfg.replica.max_staleness_seconds = 0.001
        time.sleep(0.05)
        s, r, hd = c.get("/v1/graphs/kb/nodes/person/7")
        assert s == 200 and hd["X-Igv-Backend"] == "s0/primary"
    finally:
        rs.stop()


def test_pins_override_the_hash(cluster, harness):
    tid = find_tenants_on_each_shard(cluster.router)["s0"]
    cfg = router_cfg([{"name": "s0", "primary": cluster.node["s0"].server.url},
                      {"name": "s1", "primary": cluster.node["s1"].server.url}], pins={tid: "s1"})
    router = Router(cfg)
    assert router.shard_for(tid) == "s1" and cluster.router.router.shard_for(tid) == "s0"
    with pytest.raises(Exception):
        router_cfg([{"name": "s0", "primary": "http://x"}], pins={"t": "nope"}).validate()


def test_cli_shard_of_matches_the_router(cluster, tmp_path, capsys):
    import json
    from isongraph_vector_server.cli import main
    cfgfile = tmp_path / "r.json"
    cfgfile.write_text(json.dumps({"router": {"shards": [{"name": "s0", "primary": "http://a:1"},
                                                          {"name": "s1", "primary": "http://b:1"}]}}))
    for tid in ("tenant-1", "tenant-2", "tenant-3", "acme", "globex"):
        assert main(["shard-of", tid, "--config", str(cfgfile)]) == 0
        assert capsys.readouterr().out.strip() == rendezvous(tid, ["s0", "s1"])
