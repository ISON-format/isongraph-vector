"""Embedding model lifecycle, ANN opt-in, traversal options, CLI and config."""

import io
import json
import os

import pytest

from conftest import DIM, create_graph, seed
from isongraph_vector import MockEncoder
from isongraph_vector_server import auth
from isongraph_vector_server.cli import main
from isongraph_vector_server.config import ConfigError, load_config
from isongraph_vector_server.logs import setup_logging


class ModelB(MockEncoder):
    """Same dimension, different model: vectors from the two are not comparable."""

    def __init__(self, dim=DIM, name="model-b"):
        super().__init__(dim)
        self.model_name = name

    def encode(self, text):
        return super().encode("b:" + text)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


# ---- stale / reembed ----------------------------------------------------------------------------------------


def test_changing_the_model_is_detected_and_repaired_by_reembed(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/save", {})
    s, r, _ = c.get("/v1/graphs/kb/stale")
    assert r["stale_count"] == 0 and r["stored_models"] == {f"mock-{DIM}": 3}

    h._encoder = ModelB()                                    # operator swaps the embedding model
    h.restart()
    s, r, _ = c.get("/v1/graphs/kb/stale")
    assert s == 200 and r["stale_count"] == 3 and r["encoder_model"] == "model-b"
    assert len(r["stale"]) == 3
    # text queries would compare vectors from different spaces: refused, not silently wrong
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": "Bob", "threshold": -1.0})
    assert s == 409 and r["error"]["code"] == "model_mismatch"
    # a vector query is the caller's responsibility and still works
    assert c.post("/v1/graphs/kb/search", {"vector": [0.1] * DIM, "threshold": -1.0})[0] == 200

    s, r, _ = c.post("/v1/graphs/kb/reembed", {})
    assert s == 200 and r["reembedded"] == 3
    assert c.get("/v1/graphs/kb/stale")[1]["stale_count"] == 0
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": "Bob trains machine learning models", "top_k": 1})
    assert s == 200 and r["results"][0]["id"] == 2
    assert c.post("/v1/graphs/kb/reembed", {})[1]["reembedded"] == 0


def test_changing_the_dimension_blocks_writes_until_reembed(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    h._encoder = ModelB(dim=32, name="model-32")
    h.restart()
    s, r, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 9, "properties": {"name": "New"}}]})
    assert s == 409 and r["error"]["code"] == "dimension_mismatch"
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 3                   # nothing half-applied
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": "x", "threshold": -1.0})
    assert s == 409 and r["error"]["code"] in ("model_mismatch", "dimension_mismatch")
    assert c.post("/v1/graphs/kb/reembed", {})[1]["reembedded"] == 3
    assert c.get("/v1/graphs/kb")[1]["dimension"] == 32
    assert c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 9, "properties": {"name": "New"}}]})[0] == 200
    assert c.post("/v1/graphs/kb/search", {"query": "New", "top_k": 1, "threshold": -1.0})[0] == 200


def test_client_mode_graphs_have_no_model_lifecycle(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c, "v", embedding={"mode": "client", "dimension": 3})
    assert c.get("/v1/graphs/v/stale")[0] == 400
    assert c.post("/v1/graphs/v/reembed", {})[0] == 400


def test_server_without_an_encoder_still_serves_client_mode_graphs(harness):
    h = harness(**{"encoder.kind": "none"})
    c = h.tenant("alice")
    s, r, _ = c.post("/v1/graphs", {"name": "kb"})
    assert s == 400 and r["error"]["code"] == "encoder_unavailable"
    create_graph(c, "v", embedding={"mode": "client", "dimension": 2})
    c.post("/v1/graphs/v/nodes", {"nodes": [{"type": "d", "id": 1, "vector": [1, 0]}, {"type": "d", "id": 2, "vector": [0, 1]}]})
    assert c.post("/v1/graphs/v/search", {"vector": [1, 0.1], "top_k": 1})[1]["results"][0]["id"] == 1
    h.restart()
    assert c.post("/v1/graphs/v/search", {"vector": [1, 0.1], "top_k": 1})[1]["results"][0]["id"] == 1


# ---- traversal options -----------------------------------------------------------------------------------------------


def chain(c):
    create_graph(c)
    c.post("/v1/graphs/kb/nodes", {"nodes": [
        {"type": "t", "id": i, "properties": {"name": f"node {i}", "grp": "x" if i % 2 == 0 else "y"}} for i in range(5)]})
    c.post("/v1/graphs/kb/edges", {"edges": [
        {"rel": "NEXT", "source": {"type": "t", "id": i}, "target": {"type": "t", "id": i + 1}} for i in range(4)]})


def test_direction_and_strict_filter(harness):
    h = harness()
    c = h.tenant("alice")
    chain(c)
    q = {"query": "node 4", "rel_type": "NEXT", "max_hops": 4, "threshold": 0.9, "top_k_seeds": 1}
    out = c.post("/v1/graphs/kb/multi_hop", q)[1]["results"]
    assert [x["id"] for x in out] == [4]                                    # edges point away from node 4
    out = c.post("/v1/graphs/kb/multi_hop", {**q, "direction": "in", "top_k_results": 10})[1]["results"]
    assert sorted(x["id"] for x in out) == [0, 1, 2, 3, 4]
    flt = {"where": [{"field": "grp", "op": "eq", "value": "x"}]}
    out = c.post("/v1/graphs/kb/multi_hop", {**q, "direction": "in", "filter": flt, "top_k_results": 10})[1]["results"]
    assert sorted(x["id"] for x in out) == [0, 2, 4]                       # only passing nodes returned ...
    assert any(1 in [p["id"] for p in x["path"]] for x in out)             # ... but paths may run through others
    out = c.post("/v1/graphs/kb/multi_hop", {**q, "direction": "in", "filter": flt, "strict_filter": True,
                                              "top_k_results": 10})[1]["results"]
    assert [x["id"] for x in out] == [4]                                    # strict: failing nodes are walls
    s, r, _ = c.post("/v1/graphs/kb/path", {"query": "node 4", "target": {"type": "t", "id": 0}, "direction": "in",
                                            "threshold": 0.9})
    assert r["found"] and [p["id"] for p in r["result"]["path"]] == [4, 3, 2, 1, 0]
    s, r, _ = c.post("/v1/graphs/kb/path", {"query": "node 4", "target": {"type": "t", "id": 0}, "direction": "in",
                                            "threshold": 0.9, "filter": flt, "strict_filter": True})
    assert r["found"] is False
    assert c.post("/v1/graphs/kb/path", {"query": "x", "target": {"type": "t", "id": 99}})[0] == 404


def test_filter_semantics(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    c.post("/v1/graphs/kb/nodes", {"nodes": [
        {"type": "t", "id": 1, "properties": {"name": "a b", "n": 5, "s": "apple", "flag": True}},
        {"type": "t", "id": 2, "properties": {"name": "a b", "n": 7.5, "s": "banana", "flag": False}},
        {"type": "t", "id": 3, "properties": {"name": "a b"}}]})

    def ids(where):
        s, r, _ = c.post("/v1/graphs/kb/search", {"query": "a b", "top_k": 10, "threshold": -1.0,
                                                  "filter": {"where": where}})
        assert s == 200, r
        return sorted(x["id"] for x in r["results"])

    assert ids([{"field": "n", "op": "gt", "value": 5}]) == [2]
    assert ids([{"field": "n", "op": "gte", "value": 5}]) == [1, 2]
    assert ids([{"field": "n", "op": "lt", "value": "x"}]) == []              # a number never orders against a string
    assert ids([{"field": "flag", "op": "eq", "value": 1}]) == []             # True is not 1
    assert ids([{"field": "flag", "op": "eq", "value": True}]) == [1]
    assert ids([{"field": "s", "op": "prefix", "value": "ba"}]) == [2]
    assert ids([{"field": "s", "op": "contains", "value": "pp"}]) == [1]
    assert ids([{"field": "s", "op": "in", "value": ["apple", "kiwi"]}]) == [1]
    assert ids([{"field": "s", "op": "nin", "value": ["apple"]}]) == [2, 3]
    assert ids([{"field": "s", "op": "ne", "value": "apple"}]) == [2, 3]      # missing counts as not-equal
    assert ids([{"field": "s", "op": "exists"}]) == [1, 2]
    assert ids([{"field": "s", "op": "exists", "value": False}]) == [3]
    assert ids([{"field": "n", "op": "gte", "value": 5}, {"field": "s", "op": "prefix", "value": "a"}]) == [1]
    for bad in ([{"field": "n", "op": "gt", "value": [1]}], [{"field": "n", "op": "in", "value": 3}],
                [{"field": "n", "op": "regex", "value": "x"}], [{"field": "n", "op": "gt"}],
                [{"field": "n", "op": "in", "value": list(range(65))}], [{"field": "bad name", "op": "eq", "value": 1}],
                [{"field": "s", "op": "prefix", "value": 3}]):
        s, r, _ = c.post("/v1/graphs/kb/search", {"query": "a", "filter": {"where": bad}})
        assert s == 400, bad
    # a filter cannot execute code: it is data
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": "a", "filter": {"where": [
        {"field": "s", "op": "eq", "value": "__import__('os').system('echo pwned')"}]}})
    assert s == 200 and r["results"] == []


def test_undirected_graphs_are_refused_with_an_explanation(harness):
    h = harness()
    c = h.tenant("alice")
    s, r, _ = c.post("/v1/graphs", {"name": "kb", "directed": False})
    assert s == 400 and r["error"]["code"] == "unsupported" and "direction" in r["error"]["message"]


def test_ann_is_opt_in_and_matches_exact_search(harness):
    pytest.importorskip("numpy")
    h = harness()
    c = h.tenant("alice")
    g = create_graph(c, "fast", ann="ivf", ann_nprobe=4)
    assert g["ann"] == "ivf"
    assert create_graph(c, "plain")["ann"] is None
    for name in ("fast", "plain"):
        c.post(f"/v1/graphs/{name}/nodes", {"nodes": [{"type": "d", "id": i, "properties": {"name": f"doc {i}"}}
                                                      for i in range(60)]})
    q = {"query": "doc 17", "top_k": 5, "threshold": -1.0}
    assert c.post("/v1/graphs/fast/search", q)[1]["results"] == c.post("/v1/graphs/plain/search", q)[1]["results"]
    assert c.post("/v1/graphs/fast/search", {**q, "exact": True})[0] == 200
    h.restart()                                               # the setting persists; the index is rebuilt lazily
    assert c.get("/v1/graphs/fast")[1]["ann"] == "ivf"
    assert c.post("/v1/graphs/fast/search", q)[0] == 200


def test_empty_graph_and_unembedded_nodes_are_handled(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    assert c.post("/v1/graphs/kb/search", {"query": "anything"})[1] == {"results": [], "count": 0}
    assert c.post("/v1/graphs/kb/search", {"vector": [0.5] * 7})[0] == 200          # nothing pinned yet
    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "t", "id": 1, "properties": {"color": "red"}}]})  # no embed field
    s, r, _ = c.get("/v1/graphs/kb/nodes/t/1")
    assert r["embedded"] is False
    s, r, _ = c.post("/v1/graphs/kb/similar", {"node": {"type": "t", "id": 1}})
    assert s == 200 and r["node_embedded"] is False and r["results"] == []
    # PUT with an embedding, then PUT again with nothing to embed drops the stale vector
    c.put("/v1/graphs/kb/nodes/t/2", {"properties": {"name": "has text"}})
    assert c.get("/v1/graphs/kb/nodes/t/2")[1]["embedded"] is True
    c.put("/v1/graphs/kb/nodes/t/2", {"properties": {"color": "blue"}})
    assert c.get("/v1/graphs/kb/nodes/t/2")[1]["embedded"] is False
    assert c.get("/v1/graphs/kb")[1]["embeddings"] == 0
    # explicit embed_text overrides the fields
    c.put("/v1/graphs/kb/nodes/t/3", {"properties": {"name": "ignored"}, "embed_text": "custom text"})
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": "custom text", "top_k": 1})
    assert r["results"][0]["id"] == 3 and r["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)


# ---- CLI --------------------------------------------------------------------------------------------------------------


def test_cli_tenant_and_key_management(tmp_path, capsys):
    d = str(tmp_path / "data")
    assert main(["tenant", "create", "acme", "--data-dir", d, "--quota", "max_nodes=50", "--quota", "rate_burst=5"]) == 0
    assert main(["tenant", "create", "acme", "--data-dir", d]) == 2                       # already exists
    assert main(["tenant", "create", "Bad_Name", "--data-dir", d]) == 2
    assert main(["tenant", "create", "x", "--data-dir", d]) == 2                          # too short
    assert main(["tenant", "create", "q", "--data-dir", d, "--quota", "bogus=1"]) == 2
    capsys.readouterr()
    assert main(["key", "create", "acme", "--data-dir", d, "--role", "ro", "--label", "ci"]) == 0
    captured = capsys.readouterr()
    token = captured.out.strip()
    assert token.startswith("igv_acme.") and "shown once" in captured.err
    assert token not in json.dumps(auth.describe_tenant(d, "acme"))
    reg = auth.TenantRegistry(d, load_config().quotas, 0)
    p = reg.authenticate(token)
    assert p.tenant_id == "acme" and p.role == "ro" and p.quotas.max_nodes == 50 and p.quotas.rate_burst == 5
    key_id = token.split(".")[1]
    assert main(["tenant", "list", "--data-dir", d]) == 0
    assert "acme\tactive\t1 key(s)" in capsys.readouterr().out
    assert main(["key", "revoke", "acme", key_id, "--data-dir", d]) == 0
    assert reg.authenticate(token) is None
    assert main(["key", "revoke", "acme", "0" * 16, "--data-dir", d]) == 2
    assert main(["tenant", "disable", "acme", "--data-dir", d]) == 0
    token2 = auth.add_key(d, "acme")
    assert reg.authenticate(token2) is None
    assert main(["tenant", "enable", "acme", "--data-dir", d]) == 0
    assert reg.authenticate(token2) is not None
    assert main(["tenant", "set-quota", "acme", "--data-dir", d, "--quota", "max_nodes=7"]) == 0
    assert reg.authenticate(token2).quotas.max_nodes == 7
    assert main(["tenant", "show", "acme", "--data-dir", d]) == 0
    assert "hash" not in capsys.readouterr().out


def test_key_files_are_private_on_posix(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permission bits")
    d = str(tmp_path / "data")
    auth.create_tenant(d, "acme")
    mode = (tmp_path / "data" / "tenants" / "acme" / "tenant.json").stat().st_mode & 0o777
    assert mode == 0o600


# ---- config and logs ------------------------------------------------------------------------------------------------------


def test_config_file_env_overrides_and_validation(tmp_path):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"port": 9000, "data_dir": "/x", "quotas": {"max_nodes": 5},
                             "encoder": {"kind": "mock", "dimension": 8}}))
    cfg = load_config(str(f), env={"IGV_PORT": "9100", "IGV_METRICS_TOKEN": "m" * 30})
    assert (cfg.port, cfg.data_dir, cfg.quotas.max_nodes, cfg.encoder.dimension) == (9100, "/x", 5, 8)
    assert cfg.metrics_token == "m" * 30 and "m" * 30 not in repr(cfg)             # secrets never in repr
    for bad in ({"metrics_token": "x" * 40}, {"replication_token": "x" * 40}, {"nope": 1}, {"quotas": {"nope": 1}},
                {"quotas": {"max_nodes": -1}}, {"role": "leader"}, {"durability": "maybe"},
                {"encoder": {"kind": "mock"}}, {"encoder": {"kind": "http", "url": "http://x"}},
                {"role": "replica"}, {"keep_generations": 1}, {"port": 99999}, {"max_workers": 0},
                {"router": {"shards": [{"name": "a"}]}}):
        f.write_text(json.dumps(bad))
        with pytest.raises(ConfigError):
            load_config(str(f), env={})
    with pytest.raises(ConfigError):
        load_config(None, env={"IGV_METRICS_TOKEN": "short"})
    with pytest.raises(ConfigError):
        load_config(None, env={"IGV_PORT": "abc"})


def test_logs_are_json_lines_with_route_templates_and_no_payloads(harness):
    stream = io.StringIO()
    setup_logging("INFO", stream)
    h = harness()
    c = h.tenant("alice")
    create_graph(c, "kb")
    c.post("/v1/graphs/kb/search", {"query": "my very private query text"})
    c.get("/v1/graphs/kb/nodes/doc/some-private-id")
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    reqs = [rec for rec in lines if rec["msg"] == "request"]
    assert reqs and all({"ts", "level", "route", "status", "ms", "request_id"} <= set(r) for r in reqs)
    assert {r["route"] for r in reqs} >= {"graphs.create", "search", "node.get"}
    text = stream.getvalue()
    assert "my very private query text" not in text and "some-private-id" not in text
    assert all(r["tenant"] == "alice" for r in reqs)


def test_internal_errors_become_generic_500s_with_a_request_id(harness, monkeypatch):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    from isongraph_vector_server import ops

    def boom(*a, **k):
        raise RuntimeError("secret internal detail /srv/x.py")

    monkeypatch.setattr(ops, "list_nodes", boom)
    s, r, hd = c.get("/v1/graphs/kb/nodes")
    assert s == 500 and r["error"]["code"] == "internal_error"
    assert "secret" not in json.dumps(r) and "/srv" not in json.dumps(r) and "RuntimeError" not in json.dumps(r)
    assert r["error"]["request_id"] == hd["X-Request-Id"]
    monkeypatch.undo()
    assert c.get("/v1/graphs/kb/nodes")[0] == 200


def test_the_shipped_example_config_is_valid():
    from pathlib import Path
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.example.json"), env={})
    assert cfg.encoder.kind == "http" and cfg.quotas.max_nodes == 100000 and cfg.role == "primary"
