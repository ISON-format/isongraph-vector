"""Quotas, protocol limits, bounded concurrency, deadlines, and the locks underneath them."""

import json
import socket
import threading
import time

import pytest

from conftest import DIM, Client, create_graph, seed
from isongraph_vector import EmbeddingError, MockEncoder
from isongraph_vector_server.limits import LockTimeout, RWLock, SlidingWindow, TokenBucket


class SlowEncoder(MockEncoder):
    def __init__(self, delay, dim=DIM):
        super().__init__(dim)
        self.delay = delay

    def encode(self, text):
        time.sleep(self.delay)
        return super().encode(text)

    def encode_batch(self, texts):
        time.sleep(self.delay)
        return [MockEncoder.encode(self, t) for t in texts]


class BrokenEncoder(MockEncoder):
    def encode(self, text):
        raise EmbeddingError("upstream https://secret.internal/embed returned 500: token=hunter2")

    def encode_batch(self, texts):
        raise EmbeddingError("upstream https://secret.internal/embed returned 500: token=hunter2")


# ---- quotas ---------------------------------------------------------------------------------------------------


def nodes(start, n):
    return {"nodes": [{"type": "n", "id": i, "properties": {"name": f"n{i}"}} for i in range(start, start + n)]}


def test_node_quota_is_exact_atomic_and_freed_by_deletes(harness):
    h = harness()
    c = h.tenant("alice", quotas={"max_nodes": 10})
    create_graph(c, "a")
    create_graph(c, "b")
    assert c.post("/v1/graphs/a/nodes", nodes(0, 6))[0] == 200
    s, r, _ = c.post("/v1/graphs/b/nodes", nodes(0, 5))               # 6 + 5 > 10: whole batch refused
    assert s == 403 and r["error"]["code"] == "quota_exceeded"
    assert c.get("/v1/graphs/b")[1]["nodes"] == 0
    assert c.post("/v1/graphs/b/nodes", nodes(0, 4))[0] == 200         # exactly at the limit
    assert c.post("/v1/graphs/b/nodes", nodes(100, 1))[0] == 403
    # updating existing nodes costs nothing
    assert c.post("/v1/graphs/a/nodes", nodes(0, 6))[0] == 200
    assert c.get("/v1/usage")[1]["usage"]["nodes"] == 10
    assert c.post("/v1/graphs/a/nodes/delete", {"nodes": [{"type": "n", "id": 0}, {"type": "n", "id": 1}]})[0] == 200
    assert c.get("/v1/usage")[1]["usage"]["nodes"] == 8
    assert c.post("/v1/graphs/b/nodes", nodes(200, 2))[0] == 200
    # deleting a whole graph frees its share
    assert c.delete("/v1/graphs/a")[0] == 200
    assert c.get("/v1/usage")[1]["usage"]["nodes"] == 6


def test_concurrent_writers_into_different_graphs_cannot_exceed_the_node_quota(harness):
    h = harness()
    c = h.tenant("alice", quotas={"max_nodes": 100})
    for g in ("a", "b", "c", "d"):
        create_graph(c, g)
    results = []

    def writer(graph, base):
        for k in range(12):
            s, _, _ = c.post(f"/v1/graphs/{graph}/nodes", nodes(base + k * 5, 5))
            results.append(s)

    threads = [threading.Thread(target=writer, args=(g, i * 1000)) for i, g in enumerate("abcd")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert set(results) <= {200, 403}
    total = sum(c.get(f"/v1/graphs/{g}")[1]["nodes"] for g in "abcd")
    assert total <= 100 and total == c.get("/v1/usage")[1]["usage"]["nodes"]
    assert total == 100 and results.count(200) == 20                  # all 20 batches that fit were admitted


def test_edge_graph_dimension_and_batch_quotas(harness):
    h = harness()
    c = h.tenant("alice", quotas={"max_edges": 3, "max_graphs": 2, "max_dimension": 8, "max_batch_items": 5})
    s, r, _ = c.post("/v1/graphs", {"name": "kb"})                      # encoder dimension 16 > 8
    assert s == 403 and "dimension" in r["error"]["message"]
    create_graph(c, "vec", embedding={"mode": "client", "dimension": 8})
    assert c.post("/v1/graphs", {"name": "big", "embedding": {"mode": "client", "dimension": 9}})[0] == 403
    create_graph(c, "vec2", embedding={"mode": "client", "dimension": 4})
    assert c.post("/v1/graphs", {"name": "third", "embedding": {"mode": "client", "dimension": 4}})[0] == 403  # graphs
    assert c.post("/v1/graphs/vec/nodes", {"nodes": [{"type": "n", "id": i} for i in range(6)]})[1]["error"]["code"] == "batch_too_large"
    assert c.post("/v1/graphs/vec/nodes", {"nodes": [{"type": "n", "id": i} for i in range(5)]})[0] == 200
    assert c.post("/v1/graphs/vec/nodes", {"nodes": [{"type": "n", "id": 9, "vector": [0.1] * 9}]})[0] == 400
    ring = [{"rel": "R", "source": {"type": "n", "id": i}, "target": {"type": "n", "id": (i + 1) % 5}} for i in range(5)]
    assert c.post("/v1/graphs/vec/edges", {"edges": ring[:3]})[0] == 200
    s, r, _ = c.post("/v1/graphs/vec/edges", {"edges": ring[3:]})
    assert s == 403 and r["error"]["code"] == "quota_exceeded"
    assert c.get("/v1/usage")[1]["usage"]["edges"] == 3


def test_query_parameter_caps_follow_the_tenants_quota(harness):
    h = harness()
    c = h.tenant("alice", quotas={"max_top_k": 5, "max_hops": 2})
    create_graph(c)
    seed(c)
    assert c.post("/v1/graphs/kb/search", {"query": "x", "top_k": 6})[0] == 400
    assert c.post("/v1/graphs/kb/search", {"query": "x", "top_k": 5})[0] == 200
    assert c.post("/v1/graphs/kb/multi_hop", {"query": "x", "max_hops": 3})[0] == 400
    assert c.post("/v1/graphs/kb/multi_hop", {"query": "x", "top_k_seeds": 6})[0] == 400
    assert c.post("/v1/graphs/kb/search", {"query": "x", "top_k": 10 ** 9})[0] == 400


def test_oversized_vectors_and_property_counts_are_refused(harness):
    h = harness()
    c = h.tenant("alice", quotas={"max_properties": 3, "max_dimension": 32})
    create_graph(c)
    s, r, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": 1, "properties": {f"p{i}": i for i in range(4)}}]})
    assert s == 400
    assert c.post("/v1/graphs/kb/search", {"vector": [0.1] * 33})[0] == 400
    assert c.post("/v1/graphs/kb/search", {"vector": [float("nan")]})[0] == 400


def test_rejected_batches_leave_no_trace(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 7, "properties": {"name": "seven"}}]})
    before = c.get("/v1/graphs/kb")[1]
    bad = {"nodes": [{"type": "person", "id": 50, "properties": {"name": "ok"}},
                     {"type": "person", "id": "007"}]}          # "007" collides with the existing 7: whole batch refused
    s, r, _ = c.post("/v1/graphs/kb/nodes", bad)
    assert s == 409 and c.get("/v1/graphs/kb/nodes/person/50")[0] == 404
    twins = {"nodes": [{"type": "person", "id": 60}, {"type": "person", "id": "60"}]}      # same node twice, spelled two ways
    assert c.post("/v1/graphs/kb/nodes", twins)[0] == 400
    after = c.get("/v1/graphs/kb")[1]
    assert (after["nodes"], after["embeddings"]) == (before["nodes"], before["embeddings"])
    assert c.get("/v1/usage")[1]["usage"]["nodes"] == 4


# ---- protocol limits ---------------------------------------------------------------------------------------------


def raw_exchange(node, data: bytes, read=True, timeout=5):
    host, port = node.server.address
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.sendall(data)
        if not read:
            return sock
        out = b""
        while True:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                break
            if not chunk:
                break
            out += chunk
        return out
    finally:
        if read:
            sock.close()


def status_of(raw: bytes) -> int:
    return int(raw.split(b" ", 2)[1])


def test_body_size_limit_and_framing_rules(harness):
    h = harness(max_body_bytes=2000)
    c = h.tenant("alice")
    create_graph(c)
    big = json.dumps({"query": "x" * 5000}).encode()
    s, r, _ = c.post("/v1/graphs/kb/search", None, raw=big)
    assert s == 413 and r["error"]["code"] == "payload_too_large"
    auth = f"Authorization: Bearer {c.key}\r\n"
    # chunked bodies are refused: one unambiguous framing
    chunked = (f"POST /v1/graphs/kb/search HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\n"
               "Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n").encode()
    assert status_of(raw_exchange(h, chunked)) == 400
    # POST without Content-Length
    nolen = f"POST /v1/graphs/kb/search HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\n\r\n".encode()
    assert status_of(raw_exchange(h, nolen)) == 411
    # conflicting / malformed lengths (request-smuggling shapes)
    two = (f"POST /v1/graphs/kb/search HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\n"
           "Content-Length: 2\r\nContent-Length: 40\r\n\r\n{}").encode()
    assert status_of(raw_exchange(h, two)) == 400
    neg = (f"POST /v1/graphs/kb/search HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\n"
           "Content-Length: -5\r\n\r\n").encode()
    assert status_of(raw_exchange(h, neg)) == 400
    both = (f"POST /v1/graphs/kb/search HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\n"
            "Content-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}").encode()
    assert status_of(raw_exchange(h, both)) == 400
    # absurd request line / header block
    assert status_of(raw_exchange(h, b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\nHost: x\r\n\r\n")) in (400, 414)
    hdr = f"GET /v1/graphs HTTP/1.1\r\nHost: x\r\n{auth}X-Big: {'a' * 60000}\r\n\r\n".encode()
    assert status_of(raw_exchange(h, hdr)) in (400, 431)
    # malformed request line / unknown method
    assert status_of(raw_exchange(h, b"GARBAGE\r\n\r\n")) == 400
    assert status_of(raw_exchange(h, f"TRACE /v1/graphs HTTP/1.1\r\nHost: x\r\n{auth}\r\n".encode())) in (405, 501)
    # the server is still fine afterwards
    assert c.get("/v1/graphs")[0] == 200


def test_slow_clients_cannot_hold_a_worker_forever(harness):
    h = harness(socket_timeout_seconds=0.4, max_workers=2, accept_queue=2)
    c = h.tenant("alice")
    stalled = [raw_exchange(h, b"GET /v1/gra", read=False) for _ in range(2)]       # half a request line, then silence
    t0 = time.time()
    time.sleep(0.9)
    for s in stalled:
        s.settimeout(2)
        assert s.recv(10) == b""                                                      # the server hung up on them
        s.close()
    assert c.get("/v1/graphs")[0] == 200
    # a slow-drip body is cut off too (headers promise 100 bytes, client sends 3 and stops)
    auth = f"Authorization: Bearer {c.key}\r\n"
    sock = raw_exchange(h, (f"POST /v1/graphs HTTP/1.1\r\nHost: x\r\n{auth}Content-Type: application/json\r\n"
                            "Content-Length: 100\r\n\r\n{\"n").encode(), read=False)
    sock.settimeout(3)
    data = b""
    try:
        data = sock.recv(100)
    except socket.timeout:
        pass
    sock.close()
    assert data == b""                                  # closed without a response, worker freed
    assert c.get("/v1/graphs")[0] == 200


def test_connections_beyond_the_pool_get_an_immediate_503(harness):
    h = harness(max_workers=2, accept_queue=1, socket_timeout_seconds=5)
    c = h.tenant("alice")
    holders = [raw_exchange(h, b"", read=False) for _ in range(3)]       # 2 workers busy + 1 queued
    time.sleep(0.3)
    t0 = time.time()
    out = raw_exchange(h, f"GET /v1/graphs HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {c.key}\r\n\r\n".encode(),
                       timeout=3)
    assert status_of(out) == 503 and b"overloaded" in out and time.time() - t0 < 2
    for s in holders:
        s.close()
    time.sleep(0.3)
    assert c.get("/v1/graphs")[0] == 200


# ---- concurrency gates, deadlines, lock timeouts -----------------------------------------------------------------


def test_tenant_inflight_cap_sheds_load_without_affecting_other_tenants(harness):
    h = harness(encoder=SlowEncoder(0.6), queue_wait_seconds=0.05)
    a = h.tenant("alice", quotas={"max_inflight": 1})
    b = h.tenant("bob")
    create_graph(a)
    create_graph(b)
    out = {}

    def slow():
        out["slow"] = a.post("/v1/graphs/kb/search", {"query": "slow one"})[0]

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.2)
    s, r, hd = a.get("/v1/usage")
    assert s == 503 and r["error"]["code"] == "tenant_overloaded" and hd["Retry-After"] == "1"
    assert b.get("/v1/usage")[0] == 200                      # another tenant is not shed
    t.join()
    assert out["slow"] == 200
    assert a.get("/v1/usage")[0] == 200


def test_global_concurrency_cap_sheds_load(harness):
    h = harness(encoder=SlowEncoder(0.6), queue_wait_seconds=0.05, max_concurrent_requests=1)
    a, b = h.tenant("alice"), h.tenant("bob")
    create_graph(a)
    t = threading.Thread(target=lambda: a.post("/v1/graphs/kb/search", {"query": "slow"}))
    t.start()
    time.sleep(0.2)
    s, r, _ = b.get("/v1/usage")
    assert s == 503 and r["error"]["code"] == "overloaded"
    t.join()
    assert b.get("/v1/usage")[0] == 200


def test_request_deadline_is_enforced_between_phases(harness):
    h = harness(encoder=SlowEncoder(0.5), request_deadline_seconds=0.2)
    c = h.tenant("alice")
    s, r, _ = c.post("/v1/graphs", {"name": "kb"})
    assert s == 201
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": "takes too long"})
    assert s == 504 and r["error"]["code"] == "deadline_exceeded"
    s, r, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": 1, "properties": {"name": "x"}}]})
    assert s == 504
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 0           # the late write did not happen


def test_a_busy_graph_answers_503_instead_of_queueing_forever(harness):
    h = harness(lock_timeout_seconds=0.3)
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    handle = h.server.engine.tenants["alice"].graphs["kb"]
    handle.lock.acquire_write(5)                              # someone is mid-write (or mid-reembed)
    try:
        t0 = time.time()
        s, r, hd = c.post("/v1/graphs/kb/search", {"query": "x"})
        assert s == 503 and r["error"]["code"] == "busy" and hd["Retry-After"] == "1"
        assert time.time() - t0 < 2
        assert c.post("/v1/graphs/kb/nodes", nodes(900, 1))[0] == 503
        assert c.get("/v1/graphs/kb/nodes/person/1")[0] == 503
    finally:
        handle.lock.release_write()
    assert c.post("/v1/graphs/kb/search", {"query": "x"})[0] == 200
    handle.lock.acquire_read(5)                               # a long reader blocks writers, not readers
    try:
        assert c.post("/v1/graphs/kb/search", {"query": "x"})[0] == 200
        assert c.post("/v1/graphs/kb/nodes", nodes(900, 1))[0] == 503
    finally:
        handle.lock.release_read()


def test_encoder_failures_are_generic_to_the_client(harness):
    h = harness(encoder=BrokenEncoder(DIM))
    c = h.tenant("alice")
    create_graph(c)
    for method, path, body in [("POST", "/v1/graphs/kb/search", {"query": "x"}),
                               ("POST", "/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": 1, "properties": {"name": "x"}}]})]:
        s, r, _ = c.request(method, path, body)
        assert s == 503 and r["error"]["code"] == "encoder_failed"
        assert "secret.internal" not in json.dumps(r) and "hunter2" not in json.dumps(r)
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 0


def test_response_size_cap(harness):
    h = harness(max_response_bytes=2000)
    c = h.tenant("alice")
    create_graph(c)
    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i, "properties": {"name": "x" * 200}} for i in range(50)]})
    s, r, _ = c.get("/v1/graphs/kb/nodes?limit=100")
    assert s == 413 and r["error"]["code"] == "response_too_large"
    assert c.get("/v1/graphs/kb/nodes?limit=3")[0] == 200


def test_mixed_concurrent_load_keeps_graph_store_and_quota_consistent(harness):
    h = harness()
    c = h.tenant("alice", quotas={"max_nodes": 100000})
    create_graph(c)
    errors = []
    stop = time.time() + 3

    def worker(wid):
        n = 0
        while time.time() < stop:
            try:
                i = wid * 100000 + n
                n += 1
                for s in (
                    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i, "properties": {"name": f"w{wid} {i}"}}]})[0],
                    c.post("/v1/graphs/kb/search", {"query": f"w{wid}", "top_k": 5, "threshold": -1.0})[0],
                    c.patch(f"/v1/graphs/kb/nodes/n/{i}", {"properties": {"name": f"w{wid} patched {i}"}})[0],
                    c.post("/v1/graphs/kb/edges", {"edges": [{"rel": "R", "source": {"type": "n", "id": i},
                                                              "target": {"type": "n", "id": i}}]})[0],
                    c.post("/v1/graphs/kb/multi_hop", {"query": f"w{wid}", "threshold": -1.0})[0],
                ):
                    if s != 200:
                        errors.append(s)
                if n % 3 == 0:
                    if c.delete(f"/v1/graphs/kb/nodes/n/{i}")[0] != 200:
                        errors.append("delete")
            except Exception as exc:                       # noqa: BLE001
                errors.append(repr(exc))

    def saver():
        while time.time() < stop:
            if c.post("/v1/graphs/kb/save", {})[0] != 200:
                errors.append("save")
            time.sleep(0.2)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)] + [threading.Thread(target=saver)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:5]
    g = c.get("/v1/graphs/kb")[1]
    usage = c.get("/v1/usage")[1]["usage"]
    assert g["nodes"] == g["embeddings"] == usage["nodes"] and g["edges"] == usage["edges"]
    assert g["nodes"] > 50


# ---- the building blocks ---------------------------------------------------------------------------------------------


def test_token_bucket_with_a_fake_clock():
    now = [0.0]
    b = TokenBucket(rate=2, burst=3, clock=lambda: now[0])
    assert [b.take() for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = b.take()
    assert wait == pytest.approx(0.5)
    now[0] += 0.5
    assert b.take() == 0.0 and b.take() > 0
    now[0] += 100
    assert [b.take() for _ in range(3)] == [0.0, 0.0, 0.0] and b.take() > 0      # capped at burst
    assert TokenBucket(1, 2, clock=lambda: 0.0).take(cost=50) == 0.0             # oversized cost is clamped to the burst


def test_sliding_window():
    now = [0.0]
    w = SlidingWindow(limit=3, window=10, clock=lambda: now[0])
    for _ in range(3):
        assert not w.blocked("ip")
        w.record("ip")
    assert w.blocked("ip") and not w.blocked("other")
    now[0] = 11
    assert not w.blocked("ip")


def test_rwlock_semantics():
    lock = RWLock()
    lock.acquire_read()
    lock.acquire_read()                                       # readers share
    with pytest.raises(LockTimeout):
        lock.acquire_write(0.05)                              # a writer waits for them
    lock.release_read()
    lock.release_read()
    lock.acquire_write(0.05)
    with pytest.raises(LockTimeout):
        lock.acquire_read(0.05)
    lock.release_write()
    # writer preference: a waiting writer stops new readers (no writer starvation)
    lock.acquire_read()
    got = []
    w = threading.Thread(target=lambda: (lock.acquire_write(2), got.append("w"), lock.release_write()))
    w.start()
    time.sleep(0.1)
    with pytest.raises(LockTimeout):
        lock.acquire_read(0.1)                                # queued behind the waiting writer
    lock.release_read()
    w.join()
    assert got == ["w"]
    lock.acquire_read(0.1)
    lock.release_read()


def test_rwlock_under_contention_never_overlaps_writers():
    lock = RWLock()
    state = {"w": 0, "r": 0, "bad": 0}
    guard = threading.Lock()

    def reader():
        for _ in range(300):
            with lock.read(5):
                with guard:
                    state["r"] += 1
                    if state["w"]:
                        state["bad"] += 1
                with guard:
                    state["r"] -= 1

    def writer():
        for _ in range(100):
            with lock.write(5):
                with guard:
                    state["w"] += 1
                    if state["w"] > 1 or state["r"]:
                        state["bad"] += 1
                time.sleep(0.0005)
                with guard:
                    state["w"] -= 1

    ts = [threading.Thread(target=reader) for _ in range(6)] + [threading.Thread(target=writer) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert state["bad"] == 0


def test_graceful_stop_is_prompt_even_with_idle_keepalive_connections(harness):
    """Regression: stop() used to wait out the whole socket timeout (10 s) for every idle
    keep-alive connection, because a thread blocked in recv() is not woken on all platforms."""
    import http.client
    h = harness()
    c = h.tenant("alice")
    conns = []
    for _ in range(3):
        conn = http.client.HTTPConnection(*h.server.address, timeout=5)
        conn.request("GET", "/healthz")
        conn.getresponse().read()
        conns.append(conn)                                   # left open and idle
    t0 = time.time()
    h.server.stop()
    assert time.time() - t0 < 3
    h.restart()                                              # fixture teardown wants a live server


def test_pipelined_requests_on_one_connection_are_both_answered(harness):
    h = harness()
    c = h.tenant("alice")
    req = f"GET /v1/usage HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {c.key}\r\n\r\n".encode()
    out = raw_exchange(h, req * 2 + b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", timeout=3)
    assert out.count(b"HTTP/1.1 200") == 3


def test_a_client_dripping_bytes_is_cut_off_by_the_total_deadline(harness, monkeypatch):
    from isongraph_vector_server import httpd
    monkeypatch.setattr(httpd, "HEADER_TOTAL_SECONDS", 1.0)
    h = harness(socket_timeout_seconds=5)                    # each byte arrives well inside the per-recv timeout
    host, port = h.server.address
    sock = socket.create_connection((host, port), timeout=5)
    t0 = time.time()
    closed = False
    try:
        for ch in b"GET /healthz HTTP/1.1\r\nHost: slow\r\n":
            sock.sendall(bytes([ch]))
            time.sleep(0.15)
            sock.settimeout(0.01)
            try:
                if sock.recv(10) == b"":
                    closed = True
                    break
            except socket.timeout:
                pass
            except OSError:
                closed = True
                break
    except OSError:
        closed = True
    finally:
        sock.close()
    assert closed and time.time() - t0 < 4
