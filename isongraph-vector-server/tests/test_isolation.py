"""Adversarial tenant-isolation tests.

The question every test asks: can tenant B (or an anonymous caller) read,
write, enumerate or *infer* anything about tenant A? Tenants share ids, graph
names and one process on purpose, so any bleed shows up as wrong content.
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest

from conftest import DIM, Client, create_graph, seed
from isongraph_vector_server import auth

SECRET_A = "A-ONLY-SECRET-TEXT"


@pytest.fixture
def two(harness):
    h = harness()
    a, b = h.tenant("tenant-a"), h.tenant("tenant-b")
    # identical graph name and identical node ids, different content
    create_graph(a, "kb")
    create_graph(b, "kb")
    a.post("/v1/graphs/kb/nodes", {"nodes": [
        {"type": "doc", "id": 1, "properties": {"name": "alpha", "text": SECRET_A, "owner": "a"}},
        {"type": "doc", "id": 2, "properties": {"name": "beta", "text": "also A"}}]})
    b.post("/v1/graphs/kb/nodes", {"nodes": [
        {"type": "doc", "id": 1, "properties": {"name": "gamma", "text": "B content", "owner": "b"}}]})
    a.post("/v1/graphs/kb/edges", {"edges": [{"rel": "R", "source": {"type": "doc", "id": 1},
                                              "target": {"type": "doc", "id": 2}}]})
    return h, a, b


def tree(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


# ---- same ids / same names in two tenants ---------------------------------------------------------------


def test_same_graph_name_and_ids_are_separate_universes(two):
    h, a, b = two
    assert a.get("/v1/graphs/kb/nodes/doc/1")[1]["properties"]["owner"] == "a"
    assert b.get("/v1/graphs/kb/nodes/doc/1")[1]["properties"]["owner"] == "b"
    assert b.get("/v1/graphs/kb/nodes/doc/2")[0] == 404          # exists only for A
    assert [x["id"] for x in b.get("/v1/graphs/kb/nodes")[1]["items"]] == [1]
    assert b.get("/v1/graphs/kb/edges")[1]["items"] == []
    s, r, _ = b.post("/v1/graphs/kb/search", {"query": SECRET_A, "top_k": 50, "threshold": -1.0})
    assert SECRET_A not in json.dumps(r) and [x["id"] for x in r["results"]] == [1]
    assert r["results"][0]["properties"]["owner"] == "b"
    s, r, _ = b.post("/v1/graphs/kb/similar", {"node": {"type": "doc", "id": 1}, "top_k": 50, "threshold": -1.0})
    assert r["results"] == []
    s, r, _ = b.post("/v1/graphs/kb/multi_hop", {"query": "alpha", "threshold": -1.0, "max_hops": 3})
    assert all(x["properties"].get("owner") == "b" for x in r["results"])
    s, r, _ = b.post("/v1/graphs/kb/subgraph", {"query": "alpha", "threshold": -1.0, "max_hops": 3})
    assert SECRET_A not in json.dumps(r)
    # B deleting "its" kb must not touch A's
    assert b.delete("/v1/graphs/kb")[0] == 200
    assert a.get("/v1/graphs/kb")[1]["nodes"] == 2
    assert a.get("/v1/graphs/kb/nodes/doc/1")[1]["properties"]["text"] == SECRET_A


def test_writes_to_same_ids_do_not_cross(two):
    h, a, b = two
    b.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "doc", "id": 2, "properties": {"name": "B-two"}}]})
    b.patch("/v1/graphs/kb/nodes/doc/1", {"properties": {"name": "B-changed"}})
    assert a.get("/v1/graphs/kb/nodes/doc/2")[1]["properties"]["name"] == "beta"
    assert a.get("/v1/graphs/kb/nodes/doc/1")[1]["properties"]["name"] == "alpha"
    s, r, _ = b.post("/v1/graphs/kb/nodes/delete", {"nodes": [{"type": "doc", "id": 2}]})
    assert r["deleted"] == 1
    assert a.get("/v1/graphs/kb/nodes/doc/2")[0] == 200


def test_unknown_to_me_looks_identical_whether_or_not_it_exists_for_someone_else(harness):
    h = harness()
    a, b = h.tenant("tenant-a"), h.tenant("tenant-b")
    create_graph(a, "only-a")
    s1, r1, _ = b.get("/v1/graphs/only-a")           # exists, for A
    s2, r2, _ = b.get("/v1/graphs/never-existed")    # exists for nobody
    assert (s1, r1["error"]["code"], r1["error"]["message"]) == (s2, r2["error"]["code"], r2["error"]["message"])
    for path, method, body in [("/v1/graphs/{}", "DELETE", None), ("/v1/graphs/{}/nodes", "GET", None),
                               ("/v1/graphs/{}/search", "POST", {"query": "x"}),
                               ("/v1/graphs/{}/save", "POST", {}),
                               ("/v1/graphs/{}/nodes/doc/1", "GET", None)]:
        x = b.request(method, path.format("only-a"), body)
        y = b.request(method, path.format("never-existed"), body)
        assert x[0] == y[0] == 404
        assert x[1]["error"]["code"] == y[1]["error"]["code"] and x[1]["error"]["message"] == y[1]["error"]["message"]
    assert a.get("/v1/graphs/only-a")[0] == 200       # and B's DELETE did nothing


def test_graph_listing_and_usage_are_own_only(two):
    h, a, b = two
    create_graph(a, "a-secret-graph-name")
    names = [g["name"] for g in b.get("/v1/graphs")[1]["items"]]
    assert names == ["kb"]
    assert "a-secret-graph-name" not in json.dumps(b.get("/v1/graphs")[1])
    assert b.get("/v1/usage")[1]["usage"] == {"graphs": 1, "nodes": 1, "edges": 0}
    assert a.get("/v1/usage")[1]["usage"]["graphs"] == 2


# ---- batch endpoints ---------------------------------------------------------------------------------------


def test_edge_batch_cannot_reach_other_tenants_nodes(two):
    h, a, b = two
    # B names node doc/2 (exists only in A's kb) as an edge endpoint
    s, r, _ = b.post("/v1/graphs/kb/edges", {"edges": [
        {"rel": "R", "source": {"type": "doc", "id": 1}, "target": {"type": "doc", "id": 2}}]})
    assert s == 404
    assert b.get("/v1/graphs/kb/edges")[1]["items"] == []
    assert a.get("/v1/graphs/kb/edges")[1]["items"][0]["properties"] == {}
    # and the same edge delete removes nothing of A's
    s, r, _ = b.post("/v1/graphs/kb/edges/delete", {"edges": [
        {"rel": "R", "source": {"type": "doc", "id": 1}, "target": {"type": "doc", "id": 2}}]})
    assert r == {"deleted": 0, "missing": 1}
    assert len(a.get("/v1/graphs/kb/edges")[1]["items"]) == 1


def test_node_batch_delete_reports_the_same_for_foreign_and_nonexistent(two):
    h, a, b = two
    s, r, _ = b.post("/v1/graphs/kb/nodes/delete", {"nodes": [{"type": "doc", "id": 2}, {"type": "doc", "id": 77}]})
    assert r == {"deleted": 0, "missing": 2}
    assert a.get("/v1/graphs/kb")[1]["nodes"] == 2


def test_upsert_batch_with_colliding_ids_only_touches_own_graph(two):
    h, a, b = two
    b.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "doc", "id": i, "properties": {"name": f"b{i}"}}
                                             for i in range(1, 30)]})
    assert a.get("/v1/graphs/kb")[1]["nodes"] == 2
    assert a.get("/v1/graphs/kb/nodes/doc/1")[1]["properties"]["owner"] == "a"


def test_cursors_from_another_tenant_do_not_enumerate_it(two):
    h, a, b = two
    s, r, _ = a.get("/v1/graphs/kb/nodes?limit=1")
    cursor_from_a = r["next_cursor"]
    assert cursor_from_a
    s, r, _ = b.get(f"/v1/graphs/kb/nodes?cursor={cursor_from_a}")
    assert s == 200 and r["items"] == [] and SECRET_A not in json.dumps(r)
    s, r, _ = b.get("/v1/graphs/kb/nodes?cursor=" + "AAAA")
    assert s == 400


# ---- credentials and headers -----------------------------------------------------------------------------------


def test_no_header_selects_a_tenant(two):
    h, a, b = two
    tricks = {
        "X-Tenant": "tenant-a", "X-Tenant-Id": "tenant-a", "X-Tenant-ID": "tenant-a",
        "X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1", "X-Original-URL": "/v1/graphs/kb",
        "X-Forwarded-Host": "tenant-a.example", "Host": "tenant-a.example", "Forwarded": "for=tenant-a",
        "X-Igv-Tenant": "tenant-a", "X-Auth-Tenant": "tenant-a", "X-HTTP-Method-Override": "DELETE",
        "Cookie": "tenant=tenant-a; session=tenant-a",
    }
    for k, v in tricks.items():
        s, r, _ = b.get("/v1/graphs/kb/nodes/doc/2", headers={k: v})
        assert s == 404, (k, r)
        s, r, _ = b.get("/v1/graphs/kb/nodes", headers={k: v})
        assert [x["id"] for x in r["items"]] == [1], k
    # and the same tricks do not authenticate an anonymous caller as A
    for k, v in tricks.items():
        assert h.anon().get("/v1/graphs", headers={k: v})[0] == 401


def test_query_and_body_cannot_name_a_tenant(two):
    h, a, b = two
    for q in ("tenant=tenant-a", "tenant_id=tenant-a", "as=tenant-a", "api_key=" + a.key, "key=" + a.key,
              "access_token=" + a.key):
        assert b.get("/v1/graphs?" + q)[0] == 400, q
        assert h.anon().get("/v1/graphs?" + q)[0] in (400, 401), q
    s, r, _ = b.post("/v1/graphs/kb/search", {"query": "x", "tenant": "tenant-a"})
    assert s == 400
    s, r, _ = b.post("/v1/graphs", {"name": "zz", "tenant": "tenant-a"})
    assert s == 400
    assert a.get("/v1/graphs")[1]["items"][0]["name"] == "kb"
    # the key in the URL does not authenticate either
    assert h.anon().get("/v1/graphs", key=None)[0] == 401


def test_key_forgery_variants_all_fail(two):
    h, a, b = two
    ta, ka, sa = a.key.split(".")[0], a.key.split(".")[1], a.key.split(".")[2]
    tb, kb, sb = b.key.split(".")[0], b.key.split(".")[1], b.key.split(".")[2]
    forged = [
        f"{ta}.{kb}.{sb}",                 # B's key id+secret under A's tenant prefix
        f"{ta}.{ka}.{sb}",                 # A's key id, B's secret
        f"{tb}.{ka}.{sa}",                 # A's key id+secret under B's prefix
        f"{ta}.{ka}.{sa[:-1]}",            # truncated
        f"{ta}.{ka}.{sa}x",                # extended
        a.key.upper(), a.key.replace("igv_", "IGV_"),
        a.key + "\x00", "Bearer " + a.key, a.key.replace(".", ","), a.key + "\t.",
        f"igv_tenant-a.{'0' * 16}.{sa}", f"igv_nobody.{ka}.{sa}", "igv_", "", "null", "undefined",
        "A" * 5000, "{}", "' OR '1'='1",
    ]
    for f in forged:
        s, r, _ = h.anon().get("/v1/graphs", headers={"Authorization": f"Bearer {f}"})
        assert s == 401, repr(f[:60])
        assert r["error"]["code"] == "unauthorized"
    for scheme in ("Basic", "Token", "Digest", "bearer-ish"):
        assert h.anon().get("/v1/graphs", headers={"Authorization": f"{scheme} {a.key}"})[0] == 401
    # sanity: the real keys work, lowercase scheme too
    assert a.get("/v1/graphs")[0] == 200
    assert h.anon().get("/v1/graphs", headers={"Authorization": f"bearer {a.key}"})[0] == 200


def test_duplicate_authorization_headers_are_rejected(two):
    import http.client
    h, a, b = two
    for first, second in ((a.key, b.key), (b.key, a.key), (a.key, "garbage")):
        conn = http.client.HTTPConnection(a.host, a.port, timeout=10)
        conn.putrequest("GET", "/v1/graphs/kb/nodes")
        conn.putheader("Authorization", f"Bearer {first}")
        conn.putheader("Authorization", f"Bearer {second}")
        conn.endheaders()
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 401
        conn.close()
    # comma-joined credentials are one malformed token, not two
    s, _, _ = h.anon().get("/v1/graphs", headers={"Authorization": f"Bearer {a.key}, Bearer {b.key}"})
    assert s == 401


def test_unknown_tenant_and_wrong_secret_are_indistinguishable(two):
    h, a, b = two
    sa = a.key.split(".")[2]
    wrong_secret = a.key[:-3] + ("AAA" if not a.key.endswith("AAA") else "BBB")
    unknown_tenant = f"igv_zzz-nobody.{a.key.split('.')[1]}.{sa}"
    unknown_key_id = f"igv_tenant-a.{'f' * 16}.{sa}"
    outs = []
    for k in (wrong_secret, unknown_tenant, unknown_key_id, "garbage"):
        s, r, hd = h.anon().get("/v1/graphs", headers={"Authorization": f"Bearer {k}"})
        r["error"].pop("request_id")
        outs.append((s, json.dumps(r, sort_keys=True), hd.get("WWW-Authenticate")))
    assert len(set(outs)) == 1


def test_revoked_key_and_disabled_tenant_stop_working(harness):
    h = harness()
    c = h.tenant("tenant-a")
    assert c.get("/v1/graphs")[0] == 200
    key_id = c.key.split(".")[1]
    auth.revoke_key(h.data_dir, "tenant-a", key_id)
    assert c.get("/v1/graphs")[0] == 401
    c2 = h.tenant("tenant-a")                     # a new key for the same tenant
    assert c2.get("/v1/graphs")[0] == 200
    auth.set_disabled(h.data_dir, "tenant-a", True)
    assert c2.get("/v1/graphs")[0] == 401
    auth.set_disabled(h.data_dir, "tenant-a", False)
    assert c2.get("/v1/graphs")[0] == 200


def test_key_created_after_the_server_started_works_at_once(harness):
    """Regression: a freshly minted key used to be rejected until the registry's
    recheck interval passed (observed failing before the fix)."""
    h = harness(registry_recheck_seconds=60.0)
    first = h.tenant("tenant-a")
    assert first.get("/v1/graphs")[0] == 200
    second = h.tenant("tenant-a")                 # new key, tenant already cached
    assert second.get("/v1/graphs")[0] == 200
    brand_new = h.tenant("tenant-c")              # new tenant
    assert brand_new.get("/v1/graphs")[0] == 200


def test_read_only_key_cannot_write(harness):
    h = harness()
    rw = h.tenant("tenant-a")
    create_graph(rw, "kb")
    seed(rw)
    ro = Client(h.server.url, auth.add_key(h.data_dir, "tenant-a", "ro"))
    assert ro.post("/v1/graphs/kb/search", {"query": "x", "threshold": -1.0})[0] == 200
    assert ro.get("/v1/graphs/kb/nodes")[0] == 200
    for method, path, body in [("POST", "/v1/graphs", {"name": "n2"}), ("DELETE", "/v1/graphs/kb", None),
                               ("POST", "/v1/graphs/kb/nodes", {"nodes": [{"type": "a", "id": 1}]}),
                               ("PUT", "/v1/graphs/kb/nodes/a/1", {}), ("PATCH", "/v1/graphs/kb/nodes/person/1", {}),
                               ("DELETE", "/v1/graphs/kb/nodes/person/1", None),
                               ("POST", "/v1/graphs/kb/edges", {"edges": [{"rel": "R", "source": {"type": "a", "id": 1},
                                                                          "target": {"type": "a", "id": 1}}]}),
                               ("POST", "/v1/graphs/kb/save", {}), ("POST", "/v1/graphs/kb/reembed", {})]:
        s, r, _ = ro.request(method, path, body)
        assert s == 403 and r["error"]["code"] == "read_only_key", (method, path)
    assert rw.get("/v1/graphs/kb")[1]["nodes"] == 3


def test_tenant_key_forged_with_valid_shape_for_other_tenant_dir(two):
    """A key is only valid for the tenant whose tenant.json holds its hash."""
    h, a, b = two
    # copy B's hash record into A's file would be filesystem access; instead check the
    # protocol-level fact: the same key id existing in two tenants authenticates only its own.
    _, _, secret_b = b.key.split(".")
    kid_b = b.key.split(".")[1]
    assert h.anon().get("/v1/graphs", headers={"Authorization": f"Bearer igv_tenant-a.{kid_b}.{secret_b}"})[0] == 401


# ---- names, paths, filesystem -----------------------------------------------------------------------------------


CRAFTED = ["..", "../x", "..%2fx", "%2e%2e", "%2e%2e%2f", "a%2fb", "a/b", "a\\b", "a%5cb", "%00", "a%00b",
           "CON", "NUL", "aux.", "a.", "a ", " a", "a%20", "a::$DATA", "..\\..", "....//",
           "x" * 64, "x" * 300, "UPPER", "café", "a;b", "a:b", "a|b", "a*b", "a?b", "~", ".", "%252e%252e",
           "%c0%ae%c0%ae", "%ef%bc%8e", "‮", "a\nb", "a%0ab", "%"]


def test_crafted_graph_names_cannot_escape_or_create_anything(two):
    h, a, b = two
    root = Path(h.data_dir).parent
    before = tree(root)
    for name in CRAFTED:
        # create via body
        s, r, _ = b.post("/v1/graphs", {"name": name.replace("%2f", "/").replace("%5c", "\\")})
        assert s in (400, 404), (name, s)
        # address via path (the server decodes once, then validates)
        for method in ("GET", "DELETE", "POST"):
            body = {"query": "x"} if method == "POST" else None
            from urllib.parse import quote
            s, r, _ = b.request(method, f"/v1/graphs/{quote(name, safe='%')}"
                                + ("/search" if method == "POST" else ""), body)
            assert s in (400, 404), (name, method, s)
    assert tree(root) == before
    assert a.get("/v1/graphs/kb")[0] == 200 and b.get("/v1/graphs/kb")[0] == 200


def test_crafted_node_types_and_ids_in_path_and_body(two):
    h, a, b = two
    root = Path(h.data_dir).parent
    before = tree(root)
    for bad in ["..", "%2e%2e", "a%2fb", "a:b", "a|b", "a%00", "a%20b", "x" * 200, "%"]:
        for path in (f"/v1/graphs/kb/nodes/{bad}/1", f"/v1/graphs/kb/nodes/doc/{bad}"):
            for method in ("GET", "DELETE", "PUT", "PATCH"):
                s, _, _ = b.request(method, path, {} if method in ("PUT", "PATCH") else None)
                assert s in (400, 404), (path, method, s)
    assert tree(root) == before
    assert a.get("/v1/graphs/kb")[1]["nodes"] == 2


def test_path_oddities_are_rejected(two):
    h, a, b = two
    import http.client
    for target in ["/v1//graphs", "/v1/graphs/", "/v1/./graphs", "/v1/graphs/../graphs",
                   "/v1/graphs/%2e%2e/%2e%2e/etc", "/v1/graphs/kb/../../x", "/v1/graphs#frag",
                   "http://evil/v1/graphs", "*"]:
        conn = http.client.HTTPConnection(b.host, b.port, timeout=10)
        conn.putrequest("GET", target, skip_host=False)
        conn.putheader("Authorization", f"Bearer {b.key}")
        conn.endheaders()
        resp = conn.getresponse()
        data = resp.read()
        assert resp.status in (400, 404, 405), (target, resp.status)
        assert b"Traceback" not in data
        conn.close()


def test_traversal_in_replication_routes(harness):
    h = harness()
    a = h.tenant("tenant-a")
    # no replication token configured: the endpoints do not exist
    for p in ("/_repl/v1/index", "/_repl/v1/tenant/tenant-a", "/_repl/v1/file/tenant-a/kb/000001/graph.isong"):
        assert a.get(p)[0] == 404
        assert h.anon().get(p)[0] == 404


def test_replication_endpoints_reject_tenant_keys_and_traversal(harness):
    token = "r" * 40
    h = harness(replication_token=token)
    a = h.tenant("tenant-a")
    create_graph(a, "kb")
    seed(a)
    a.post("/v1/graphs/kb/save", {})
    assert a.get("/_repl/v1/index")[0] == 401           # a tenant key is not a replication token
    assert h.anon().get("/_repl/v1/index")[0] == 401
    repl = Client(h.server.url, token)
    s, idx, _ = repl.get("/_repl/v1/index")
    assert s == 200 and idx["tenants"][0]["tenant_id"] == "tenant-a"
    assert repl.get("/_repl/v1/file/tenant-a/kb/000001/graph.isong")[0] == 200
    # the replication token is not a tenant key
    assert repl.get("/v1/graphs")[0] == 401
    for bad in ("/_repl/v1/file/tenant-a/kb/000001/..%2f..%2ftenant.json", "/_repl/v1/file/tenant-a/kb/000001/tenant.json",
                "/_repl/v1/file/tenant-a/kb/../000001/graph.isong", "/_repl/v1/file/tenant-a/KB/000001/graph.isong",
                "/_repl/v1/file/..%2f/kb/000001/graph.isong", "/_repl/v1/file/tenant-a/kb/1/graph.isong",
                "/_repl/v1/file/tenant-a/kb/000001/MANIFEST.json.bak", "/_repl/v1/tenant/..",
                "/_repl/v1/meta/tenant-a/..%2f..", "/_repl/v1/tenant/tenant-a%00"):
        assert repl.get(bad)[0] in (400, 404), bad


# ---- metrics, errors, quotas: no side channels ------------------------------------------------------------------


def test_metrics_endpoint_is_protected_and_aggregate(harness):
    h = harness()
    a = h.tenant("tenant-a")
    assert a.get("/metrics")[0] == 404                  # not configured: does not exist
    token = "m" * 40
    h2 = harness(metrics_token=token)
    t1 = h2.tenant("tenant-one")
    create_graph(t1, "secretgraph")
    assert t1.get("/metrics")[0] == 401                 # a tenant key is not the metrics token
    assert h2.anon().get("/metrics")[0] == 401
    s, text, _ = Client(h2.server.url, token).get("/metrics")
    assert s == 200 and "igv_requests_total" in text
    assert "tenant-one" not in text and "secretgraph" not in text


def test_quota_errors_report_only_own_usage(harness):
    h = harness()
    a = h.tenant("tenant-a")
    b = h.tenant("tenant-b", quotas={"max_nodes": 5})
    create_graph(a, "kb")
    a.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i} for i in range(100)]})
    create_graph(b, "kb")
    s, r, _ = b.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i} for i in range(10)]})
    assert s == 403 and r["error"]["code"] == "quota_exceeded"
    assert "100" not in r["error"]["message"]          # A's usage never appears in B's error
    assert "0 used" in r["error"]["message"]
    assert b.get("/v1/graphs/kb")[1]["nodes"] == 0     # rejected batch applied nothing


def test_rate_limits_are_per_tenant(harness):
    h = harness()
    a = h.tenant("tenant-a", quotas={"rate_per_second": 0.5, "rate_burst": 3})
    b = h.tenant("tenant-b")
    codes = [a.get("/v1/usage")[0] for _ in range(10)]
    assert 429 in codes and codes[0] == 200
    assert all(b.get("/v1/usage")[0] == 200 for _ in range(20))     # B is unaffected
    s, r, hd = a.get("/v1/usage")
    assert s == 429 and "Retry-After" in hd and r["error"]["code"] == "rate_limited"


def test_failed_auth_throttle_blocks_the_peer_not_the_tenant(harness):
    h = harness(failed_auth_per_minute=5)
    a = h.tenant("tenant-a")
    for _ in range(5):
        assert h.anon().get("/v1/graphs", headers={"Authorization": "Bearer nope"})[0] == 401
    # the 6th bad attempt from this address is throttled ...
    assert h.anon().get("/v1/graphs", headers={"Authorization": "Bearer nope"})[0] == 429
    # ... and so is a good key from the same address (that is the cost of throttling by address)
    assert a.get("/v1/graphs")[0] == 429


def test_logs_never_contain_keys(harness, caplog):
    import logging
    from isongraph_vector_server.logs import setup_logging
    import io
    stream = io.StringIO()
    setup_logging("DEBUG", stream)
    h = harness()
    a = h.tenant("tenant-a")
    create_graph(a, "kb")
    a.post("/v1/graphs/kb/search", {"query": "x"})
    h.anon().get("/v1/graphs", headers={"Authorization": f"Bearer {a.key[:-4]}AAAA"})
    h.anon().get("/v1/graphs", headers={"Authorization": f"Bearer {a.key}x"})
    secret = a.key.split(".")[2]
    out = stream.getvalue()
    assert out and secret not in out and a.key not in out
    assert secret[:12] not in out
    from isongraph_vector_server.logs import redact
    assert secret not in redact(f"oops {a.key} and Authorization: Bearer {a.key}")


def test_secret_is_stored_only_as_a_salted_hash(harness):
    h = harness()
    a = h.tenant("tenant-a")
    secret = a.key.split(".")[2]
    raw = (Path(h.data_dir) / "tenants" / "tenant-a" / "tenant.json").read_text()
    assert secret not in raw and a.key not in raw
    doc = json.loads(raw)
    rec = doc["keys"][0]
    assert rec["algo"] == "hmac-sha256" and len(rec["salt"]) == 32 and len(rec["hash"]) == 64
    b = h.tenant("tenant-a")                                   # a second key gets a different salt
    doc = json.loads((Path(h.data_dir) / "tenants" / "tenant-a" / "tenant.json").read_text())
    assert doc["keys"][0]["salt"] != doc["keys"][1]["salt"]
    # describe_tenant (CLI) never prints salts or hashes
    assert "hash" not in json.dumps(auth.describe_tenant(h.data_dir, "tenant-a"))


# ---- concurrency ----------------------------------------------------------------------------------------------------


def test_concurrent_tenants_with_identical_ids_never_see_each_other(harness):
    h = harness()
    tenants = [h.tenant(f"tenant-{i}") for i in range(4)]
    for i, c in enumerate(tenants):
        create_graph(c, "kb")
    errors = []

    def work(i, c):
        try:
            marker = f"MARKER{i}"
            for rnd in range(15):
                s, r, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [
                    {"type": "doc", "id": n, "properties": {"name": f"{marker}-{n}", "owner": marker}}
                    for n in range(20)]})
                assert s == 200, r
                s, r, _ = c.post("/v1/graphs/kb/search", {"query": f"{marker}-3", "top_k": 20, "threshold": -1.0})
                assert s == 200
                owners = {x["properties"]["owner"] for x in r["results"]}
                assert owners == {marker}, owners
                s, r, _ = c.get("/v1/graphs/kb/nodes/doc/5")
                assert r["properties"]["owner"] == marker
        except AssertionError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i, c)) for i, c in enumerate(tenants)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:3]


def test_operator_tokens_are_throttled_like_keys(harness):
    h = harness(metrics_token="m" * 40, replication_token="r" * 40, failed_auth_per_minute=4)
    anon = h.anon()
    for path in ("/metrics", "/_repl/v1/index", "/metrics", "/_repl/v1/index"):
        assert anon.get(path, headers={"Authorization": "Bearer wrong"})[0] == 401
    assert anon.get("/metrics", headers={"Authorization": "Bearer " + "m" * 40})[0] == 429     # peer is throttled
    assert anon.get("/_repl/v1/index", headers={"Authorization": "Bearer " + "r" * 40})[0] == 429
