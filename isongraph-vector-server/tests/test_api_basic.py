"""Happy-path API behaviour and the error model."""

import json

import pytest

from conftest import DIM, create_graph, seed


def test_health_and_ready_need_no_key(node):
    anon = node.anon()
    assert anon.get("/healthz")[:2] == (200, {"status": "ok"})
    s, body, _ = anon.get("/readyz")
    assert s == 200 and body == {"status": "ready", "role": "primary"}


def test_graph_lifecycle(alice):
    g = create_graph(alice, "kb")
    assert g["mode"] == "server" and g["dimension"] == DIM and g["nodes"] == 0
    s, body, _ = alice.post("/v1/graphs", {"name": "kb"})
    assert s == 409 and body["error"]["code"] == "already_exists"
    s, body, _ = alice.get("/v1/graphs")
    assert [x["name"] for x in body["items"]] == ["kb"]
    s, body, _ = alice.get("/v1/graphs/kb")
    assert s == 200 and body["name"] == "kb"
    assert alice.delete("/v1/graphs/kb")[0] == 200
    assert alice.get("/v1/graphs/kb")[0] == 404
    assert alice.delete("/v1/graphs/kb")[0] == 404


def test_search_returns_best_match_first(alice):
    create_graph(alice)
    seed(alice)
    s, r, _ = alice.post("/v1/graphs/kb/search", {"query": "trains machine learning models Bob", "top_k": 3,
                                                   "threshold": -1.0})
    assert s == 200, r
    assert r["count"] == 3
    # MockEncoder is deterministic: the identical text has similarity 1.0.
    s, r2, _ = alice.post("/v1/graphs/kb/search", {"query": "Bob trains machine learning models", "top_k": 1})
    assert s == 200
    s, r3, _ = alice.post("/v1/graphs/kb/search", {"query": "Bob trains machine learning models", "top_k": 3,
                                                   "threshold": -1.0})
    scores = [x["score"] for x in r3["results"]]
    assert scores == sorted(scores, reverse=True)


def test_exact_text_scores_one(alice):
    create_graph(alice)
    seed(alice)
    # The embed text for Bob is "Bob trains machine learning models" (name + description).
    s, r, _ = alice.post("/v1/graphs/kb/search", {"query": "Bob trains machine learning models", "top_k": 1})
    top = r["results"][0]
    assert top["id"] == 2 and top["score"] == pytest.approx(1.0, abs=1e-5)
    assert top["properties"]["name"] == "Bob"


def test_vector_query_and_filters(alice):
    create_graph(alice)
    seed(alice)
    s, r, _ = alice.post("/v1/graphs/kb/search", {
        "query": "anything", "top_k": 10, "threshold": -1.0,
        "filter": {"where": [{"field": "age", "op": "gte", "value": 40}]}})
    assert s == 200
    assert [x["id"] for x in r["results"]] == [2]
    s, r, _ = alice.post("/v1/graphs/kb/search", {
        "query": "anything", "top_k": 10, "threshold": -1.0,
        "filter": {"types": ["person"], "where": [{"field": "name", "op": "in", "value": ["Alice", "Carol"]}]}})
    assert sorted(str(x["id"]) for x in r["results"]) == ["1", "carol"]


def test_similar_path_multihop_subgraph(alice):
    create_graph(alice)
    seed(alice)
    s, r, _ = alice.post("/v1/graphs/kb/similar", {"node": {"type": "person", "id": 1}, "top_k": 5,
                                                   "threshold": -1.0})
    assert s == 200 and r["node_embedded"] and all(x["id"] != 1 for x in r["results"])

    s, r, _ = alice.post("/v1/graphs/kb/multi_hop", {
        "query": "Bob trains machine learning models", "rel_type": "KNOWS", "max_hops": 2, "threshold": 0.5})
    assert s == 200
    by_id = {str(x["id"]): x for x in r["results"]}
    assert by_id["2"]["hops"] == 0 and by_id["carol"]["hops"] == 1
    assert by_id["carol"]["score"] == pytest.approx(by_id["2"]["score"] * 0.8, abs=1e-4)

    s, r, _ = alice.post("/v1/graphs/kb/path", {
        "query": "Alice builds payment APIs", "target": {"type": "person", "id": "carol"}, "threshold": 0.5})
    assert s == 200 and r["found"]
    assert [p["id"] for p in r["result"]["path"]] == [1, 2, "carol"]

    s, r, _ = alice.post("/v1/graphs/kb/subgraph", {"query": "Bob trains machine learning models",
                                                    "max_hops": 1, "threshold": 0.5})
    assert s == 200 and r["node_count"] == 2 and r["edge_count"] == 1


def test_weight_field_changes_scores(alice):
    create_graph(alice)
    seed(alice)
    q = {"query": "Alice builds payment APIs", "rel_type": "KNOWS", "max_hops": 1, "threshold": 0.5}
    s, plain, _ = alice.post("/v1/graphs/kb/multi_hop", q)
    s, weighted, _ = alice.post("/v1/graphs/kb/multi_hop", {**q, "weight_field": "weight"})
    p = {str(x["id"]): x["score"] for x in plain["results"]}
    w = {str(x["id"]): x["score"] for x in weighted["results"]}
    assert w["2"] == pytest.approx(p["2"] * 0.9, abs=1e-4)


def test_node_crud_and_patch_reembeds(alice):
    create_graph(alice)
    seed(alice)
    s, r, _ = alice.get("/v1/graphs/kb/nodes/person/2")
    assert s == 200 and r["embedded"] and r["properties"]["age"] == 45
    s, r, _ = alice.patch("/v1/graphs/kb/nodes/person/2", {"properties": {"description": "now writes poetry"}})
    assert s == 200 and r["properties"]["age"] == 45
    s, r, _ = alice.post("/v1/graphs/kb/search", {"query": "Bob now writes poetry", "top_k": 1})
    assert r["results"][0]["id"] == 2 and r["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)
    s, r, _ = alice.patch("/v1/graphs/kb/nodes/person/2", {"remove": ["age"]})
    assert "age" not in r["properties"]
    assert alice.delete("/v1/graphs/kb/nodes/person/2")[0] == 200
    assert alice.get("/v1/graphs/kb/nodes/person/2")[0] == 404
    # the edge touching it went with it
    s, r, _ = alice.get("/v1/graphs/kb/edges")
    assert r["items"] == [] or all(e["source"]["id"] != 2 and e["target"]["id"] != 2 for e in r["items"])


def test_put_replaces_properties(alice):
    create_graph(alice)
    alice.put("/v1/graphs/kb/nodes/doc/a", {"properties": {"name": "x", "extra": 1}})
    alice.put("/v1/graphs/kb/nodes/doc/a", {"properties": {"name": "y"}})
    s, r, _ = alice.get("/v1/graphs/kb/nodes/doc/a")
    assert r["properties"] == {"name": "y"}


def test_client_mode_graph_takes_vectors(alice):
    create_graph(alice, "vec", embedding={"mode": "client", "dimension": 3})
    s, r, _ = alice.post("/v1/graphs/vec/nodes", {"nodes": [
        {"type": "d", "id": 1, "vector": [1, 0, 0]}, {"type": "d", "id": 2, "vector": [0, 1, 0]}]})
    assert s == 200 and r["embedded"] == 2
    s, r, _ = alice.post("/v1/graphs/vec/search", {"vector": [0.9, 0.1, 0], "top_k": 1})
    assert r["results"][0]["id"] == 1
    # text is refused both ways
    assert alice.post("/v1/graphs/vec/search", {"query": "x"})[1]["error"]["code"] == "text_query_unsupported"
    assert alice.post("/v1/graphs/vec/nodes", {"nodes": [{"type": "d", "id": 3, "embed_text": "x"}]})[0] == 400
    # wrong length
    s, r, _ = alice.post("/v1/graphs/vec/nodes", {"nodes": [{"type": "d", "id": 3, "vector": [1, 2]}]})
    assert s == 400 and r["error"]["code"] == "dimension_mismatch"
    s, r, _ = alice.post("/v1/graphs/vec/search", {"vector": [1, 2]})
    assert s == 400 and r["error"]["code"] == "dimension_mismatch"
    # server-mode graph refuses caller vectors
    create_graph(alice, "txt")
    s, r, _ = alice.post("/v1/graphs/txt/nodes", {"nodes": [{"type": "d", "id": 1, "vector": [1] * DIM}]})
    assert s == 400


def test_pagination(alice):
    create_graph(alice)
    alice.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i, "properties": {"name": f"n{i}"}}
                                                 for i in range(25)]})
    seen, cursor = [], None
    while True:
        s, r, _ = alice.get("/v1/graphs/kb/nodes?limit=10" + (f"&cursor={cursor}" if cursor else ""))
        assert s == 200
        seen += [x["id"] for x in r["items"]]
        cursor = r["next_cursor"]
        if not cursor:
            break
    assert seen == sorted(range(25))
    s, r, _ = alice.get("/v1/graphs/kb/nodes?cursor=%%%")
    assert s == 400 and r["error"]["code"] == "invalid_cursor"
    assert alice.get("/v1/graphs/kb/nodes?limit=0")[0] == 400
    assert alice.get("/v1/graphs/kb/nodes?limit=99999")[0] == 400
    # edges
    alice.post("/v1/graphs/kb/edges", {"edges": [
        {"rel": "R", "source": {"type": "n", "id": i}, "target": {"type": "n", "id": i + 1}} for i in range(12)]})
    got, cursor = 0, None
    while True:
        s, r, _ = alice.get("/v1/graphs/kb/edges?limit=5" + (f"&cursor={cursor}" if cursor else ""))
        got += len(r["items"])
        cursor = r["next_cursor"]
        if not cursor:
            break
    assert got == 12


def test_error_model_is_consistent_and_leaks_nothing(alice):
    s, r, h = alice.get("/v1/graphs/nope")
    assert s == 404
    assert set(r["error"]) >= {"code", "message", "request_id"}
    assert r["error"]["request_id"] == h["X-Request-Id"]
    for bad in ({"name": "UPPER"}, {"name": "../x"}, {}, {"name": "ok", "bogus": 1}):
        s, r, _ = alice.post("/v1/graphs", bad)
        assert s == 400 and r["error"]["code"] == "validation_failed", bad
    text = json.dumps(r)
    assert "Traceback" not in text and ".py" not in text


def test_validation_rejects_bad_bodies(alice):
    create_graph(alice)
    cases = [
        {"nodes": [{"type": "a:b", "id": 1}]},                       # bad type
        {"nodes": [{"type": "a", "id": "has space"}]},
        {"nodes": [{"type": "a", "id": 1.5}]},
        {"nodes": [{"type": "a", "id": True}]},
        {"nodes": [{"type": "a", "id": 1, "properties": {"x": [1]}}]},
        {"nodes": [{"type": "a", "id": 1, "properties": {"x": {"y": 1}}}]},
        {"nodes": [{"type": "a", "id": 1, "properties": {"id": 1}}]},  # reserved column
        {"nodes": [{"type": "a", "id": 1, "properties": {"_x": 1}}]},
        {"nodes": [{"type": "a", "id": 1, "properties": {"x": 1e300}}]},
        {"nodes": [{"type": "a", "id": 1, "properties": {"x": "a b"}}]},
        {"nodes": [{"type": "a", "id": 1}, {"type": "a", "id": "1"}]},   # same node twice
        {"nodes": []},
        {"nodes": [{"type": "a", "id": 1, "unknown": 1}]},
    ]
    for body in cases:
        s, r, _ = alice.post("/v1/graphs/kb/nodes", body)
        assert s == 400, (body, s, r)
    # duplicate JSON keys / NaN / bad JSON / wrong content-type
    assert alice.request("POST", "/v1/graphs/kb/search", raw=b'{"query":"a","query":"b"}')[0] == 400
    assert alice.request("POST", "/v1/graphs/kb/search", raw=b'{"query":"a","vector":[NaN]}')[0] == 400
    assert alice.request("POST", "/v1/graphs/kb/search", raw=b'{not json')[0] == 400
    assert alice.request("POST", "/v1/graphs/kb/search", raw=b'\xff\xfe')[0] == 400
    s, r, _ = alice.request("POST", "/v1/graphs/kb/search", raw=b'{"query":"x"}',
                            headers={"Content-Type": "text/plain"})
    assert s == 415
    # unknown query parameter, wrong method
    assert alice.get("/v1/graphs?bogus=1")[0] == 400
    s, _, h = alice.request("PUT", "/v1/graphs", {"name": "x"})
    assert s == 405 and "POST" in h["Allow"]


def test_ids_5_and_string_5_are_the_same_node(alice):
    create_graph(alice)
    alice.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": 5, "properties": {"name": "five"}}]})
    s, r, _ = alice.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": "5", "properties": {"name": "FIVE"}}]})
    assert s == 200 and r["updated"] == 1
    s, r, _ = alice.get("/v1/graphs/kb/nodes/n/5")
    assert r["properties"]["name"] == "FIVE"
    # "007" is a string id, but the library cannot hold both 7 and "007"
    alice.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": 7}]})
    s, r, _ = alice.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": "007"}]})
    assert s == 409 and r["error"]["code"] == "id_conflict"
    s, r, _ = alice.get("/v1/graphs/kb")
    assert r["nodes"] == 2          # the rejected batch changed nothing


def test_edge_errors(alice):
    create_graph(alice)
    seed(alice)
    s, r, _ = alice.post("/v1/graphs/kb/edges", {"edges": [
        {"rel": "X", "source": {"type": "person", "id": 1}, "target": {"type": "person", "id": 99}}]})
    assert s == 404
    # upsert existing edge updates properties
    s, r, _ = alice.post("/v1/graphs/kb/edges", {"edges": [
        {"rel": "KNOWS", "source": {"type": "person", "id": 1}, "target": {"type": "person", "id": 2},
         "properties": {"weight": 0.1}}]})
    assert r == {"created": 0, "updated": 1}
    s, r, _ = alice.post("/v1/graphs/kb/edges/delete", {"edges": [
        {"rel": "KNOWS", "source": {"type": "person", "id": 1}, "target": {"type": "person", "id": 2}},
        {"rel": "KNOWS", "source": {"type": "person", "id": 1}, "target": {"type": "person", "id": 2}}]})
    assert r == {"deleted": 1, "missing": 1}


def test_usage_endpoint(alice):
    create_graph(alice)
    seed(alice)
    s, r, _ = alice.get("/v1/usage")
    assert r["usage"] == {"graphs": 1, "nodes": 3, "edges": 2}
    assert r["quotas"]["max_nodes"] == 100_000
