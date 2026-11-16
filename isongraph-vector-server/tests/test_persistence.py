"""Snapshots, generations, verification on load, shutdown behaviour."""

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import DIM, Client, create_graph, seed
from isongraph_vector import SemanticGraph, graph_digest
from ison_graph import ISONGraph
from isongraph_vector_server import storage
from isongraph_vector_server.storage import GraphStorage, SnapshotError, semantic_digest


def gdir(h, tenant="alice", graph="kb"):
    return Path(h.data_dir) / "tenants" / tenant / "graphs" / f"g-{graph}"


def searchable(c, text="Bob trains machine learning models"):
    s, r, _ = c.post("/v1/graphs/kb/search", {"query": text, "top_k": 1})
    assert s == 200, r
    return r["results"][0]


def test_state_survives_a_graceful_restart(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    h.restart()                                     # graceful: final snapshot is written
    g = c.get("/v1/graphs/kb")[1]
    assert (g["nodes"], g["edges"], g["embeddings"]) == (3, 2, 3)
    top = searchable(c)
    assert top["id"] == 2 and top["score"] == pytest.approx(1.0, abs=1e-5)
    assert c.get("/v1/graphs/kb/nodes/person/carol")[1]["properties"]["age"] == 38


def test_unsaved_changes_are_lost_on_a_hard_stop_and_saved_ones_are_not(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    assert c.post("/v1/graphs/kb/save", {})[1]["saved"]
    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 99, "properties": {"name": "late"}}]})
    assert c.get("/v1/graphs/kb")[1]["unsaved_changes"] is True
    h.restart(graceful=False)
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 3          # documented loss window
    assert c.get("/v1/graphs/kb/nodes/person/99")[0] == 404


def test_sync_durability_survives_a_hard_stop(harness):
    h = harness(durability="sync")
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 99, "properties": {"name": "durable"}}]})
    c.patch("/v1/graphs/kb/nodes/person/99", {"properties": {"name": "durable2"}})
    c.post("/v1/graphs/kb/nodes/delete", {"nodes": [{"type": "person", "id": 1}]})
    h.restart(graceful=False)
    assert c.get("/v1/graphs/kb/nodes/person/99")[1]["properties"]["name"] == "durable2"
    assert c.get("/v1/graphs/kb/nodes/person/1")[0] == 404
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 3


def test_autosave_writes_dirty_graphs(harness):
    h = harness(autosave_interval_seconds=0.2)
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    deadline = time.time() + 5
    while time.time() < deadline and c.get("/v1/graphs/kb")[1]["unsaved_changes"]:
        time.sleep(0.1)
    g = c.get("/v1/graphs/kb")[1]
    assert not g["unsaved_changes"] and g["generation"] >= 1
    h.restart(graceful=False)
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 3


def test_snapshot_layout_manifest_and_digest(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/save", {})
    d = gdir(h)
    assert (d / "CURRENT").read_text().strip() == "000001"
    gen = d / "gen-000001"
    assert sorted(p.name for p in gen.iterdir()) == ["MANIFEST.json", "graph.isong", "graph.isong.embeddings.json"]
    manifest = json.loads((gen / "MANIFEST.json").read_text())
    sidecar = json.loads((gen / "graph.isong.embeddings.json").read_text())
    assert sidecar["graph_digest"] == manifest["graph_digest"]
    assert manifest["node_count"] == 3 and manifest["embedding_count"] == 3
    # the library, handed only the file pair, agrees (this is what the digest is for)
    from conftest import REPO_ROOT  # noqa: F401
    lib = SemanticGraph.load(gen / "graph.isong", encoder=None, auto_embed=False)
    assert graph_digest(lib) == manifest["graph_digest"]
    assert c.get("/v1/graphs/kb")[1]["snapshot_digest"] == manifest["graph_digest"]
    # unchanged graph: save is a no-op unless forced; the endpoint forces
    assert c.post("/v1/graphs/kb/save", {})[1]["generation"] == 2


def test_generations_are_pruned(harness):
    h = harness(keep_generations=3)
    c = h.tenant("alice")
    create_graph(c)
    for i in range(6):
        c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i, "properties": {"name": f"n{i}"}}]})
        c.post("/v1/graphs/kb/save", {})
    gens = sorted(p.name for p in gdir(h).iterdir() if p.name.startswith("gen-"))
    assert gens == ["gen-000004", "gen-000005", "gen-000006"]


# ---- verification on load ------------------------------------------------------------------------------------


def two_generations(h):
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/save", {})                      # gen 1: 3 nodes
    c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 4, "properties": {"name": "Dan"}}]})
    c.post("/v1/graphs/kb/save", {})                      # gen 2: 4 nodes
    return c


def test_corrupt_current_generation_falls_back_and_is_flagged(harness):
    h = harness()
    c = two_generations(h)
    h.server.stop()
    victim = gdir(h) / "gen-000002" / "graph.isong"
    victim.write_bytes(victim.read_bytes().replace(b"Dan", b"Eve"))     # same length, wrong checksum
    srv = h.restart()
    g = c.get("/v1/graphs/kb")[1]
    assert g["nodes"] == 3 and g["generation"] == 1 and g["status"] == "degraded"
    assert srv.engine.startup_report["degraded"] == 1


def test_sidecar_from_another_save_is_caught_even_with_matching_checksums(harness):
    """Swap in an internally consistent but *different* sidecar and fix the manifest checksum:
    the library's graph-digest check (and ours) must still refuse it."""
    h = harness()
    c = two_generations(h)
    h.server.stop()
    g1, g2 = gdir(h) / "gen-000001", gdir(h) / "gen-000002"
    foreign = (g1 / "graph.isong.embeddings.json").read_bytes()          # belongs to the 3-node graph
    (g2 / "graph.isong.embeddings.json").write_bytes(foreign)
    manifest = json.loads((g2 / "MANIFEST.json").read_text())
    sha, size = storage.sha256_file(g2 / "graph.isong.embeddings.json")
    manifest["files"]["graph.isong.embeddings.json"] = {"sha256": sha, "size": size}
    (g2 / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(SnapshotError, match="sidecar|digest|belong"):
        GraphStorage(gdir(h)).load_generation(g2, GraphStorage(gdir(h)).read_meta(), h.server.encoder)
    h.restart()
    assert c.get("/v1/graphs/kb")[1]["generation"] == 1


def test_all_generations_corrupt_quarantines_only_that_graph(harness):
    h = harness()
    c = two_generations(h)
    create_graph(c, "other")
    c.post("/v1/graphs/other/save", {})
    h.server.stop()
    for gen in ("gen-000001", "gen-000002"):
        (gdir(h) / gen / "graph.isong").write_bytes(b"garbage")
    h.restart()
    s, r, _ = c.get("/v1/graphs/kb")
    assert s == 503 and r["error"]["code"] == "graph_quarantined"
    assert c.post("/v1/graphs/kb/search", {"query": "x"})[0] == 503
    assert c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "a", "id": 1}]})[0] == 503
    assert c.get("/v1/graphs/other")[0] == 200                             # neighbours unaffected
    listing = {g["name"]: g["status"] for g in c.get("/v1/graphs")[1]["items"]}
    assert listing["kb"] == "quarantined" and listing["other"] == "ok"
    assert "garbage" not in json.dumps(c.get("/v1/graphs/kb")[1])
    # not deleted: the files are still there for an operator, and the tenant can drop it
    assert (gdir(h) / "gen-000002").exists()
    assert c.delete("/v1/graphs/kb")[0] == 200
    assert not gdir(h).exists()


def test_strict_startup_refuses_to_start(harness):
    h = harness()
    two_generations(h)
    h.server.stop()
    for gen in ("gen-000001", "gen-000002"):
        (gdir(h) / gen / "MANIFEST.json").write_bytes(b"{}")
    from isongraph_vector_server.server import Server
    h.cfg.strict_startup = True
    with pytest.raises(SnapshotError):
        Server(h.cfg, configure_logging=False).start()
    h.cfg.strict_startup = False
    h.server = Server(h.cfg, configure_logging=False).start()    # fixture teardown needs a live one


def test_leftover_temp_dirs_are_cleaned_on_start(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/save", {})
    h.server.stop()
    (gdir(h) / ".tmp-gen-000002-deadbeef").mkdir()
    (gdir(h) / ".tmp-gen-000002-deadbeef" / "graph.isong").write_text("half written")
    (gdir(h).parent / ".tmp-create-abc").mkdir()
    h.restart()
    assert not (gdir(h) / ".tmp-gen-000002-deadbeef").exists()
    assert not (gdir(h).parent / ".tmp-create-abc").exists()
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 3


def test_missing_current_pointer_uses_newest_verified_generation(harness):
    h = harness()
    c = two_generations(h)
    h.server.stop()
    (gdir(h) / "CURRENT").unlink()
    h.restart()
    assert c.get("/v1/graphs/kb")[1]["nodes"] == 4


def test_two_processes_cannot_share_a_data_dir(harness):
    h = harness()
    from isongraph_vector_server.server import Server
    cfg = h.cfg
    with pytest.raises(SnapshotError, match="in use"):
        Server(cfg, configure_logging=False).start()


def test_graceful_shutdown_finishes_in_flight_requests_and_saves(harness):
    import threading
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    results = []

    def writer():
        for i in range(30):
            try:
                s, _, _ = c.post("/v1/graphs/kb/nodes",
                                 {"nodes": [{"type": "n", "id": i, "properties": {"name": f"n{i}"}}]})
            except OSError:          # the server went away mid-request: not acknowledged
                return
            results.append((i, s))

    t = threading.Thread(target=writer)
    t.start()
    time.sleep(0.15)
    h.server.stop()
    t.join()
    acked = [i for i, s in results if s == 200]
    assert acked, "the writer should have got some acknowledged writes in before shutdown"
    h.restart()
    # every write acknowledged before shutdown is present afterwards
    have = {x["id"] for x in c.get("/v1/graphs/kb/nodes?limit=1000")[1]["items"]}
    assert set(acked) <= have


# ---- the format's quirks -------------------------------------------------------------------------------------------


def test_library_roundtrip_turns_missing_properties_into_null():
    """Characterisation of the library (not of this service): why `semantic_digest` exists."""
    g = ISONGraph()
    g.add_node("t", 1, name="a")
    g.add_node("t", 2)
    back = ISONGraph.from_ison(g.to_ison(), "x")
    assert back.get_node("t", 2).properties == {"name": None}
    assert graph_digest(g) != graph_digest(back)
    assert semantic_digest(g) == semantic_digest(back)


def test_heterogeneous_nodes_snapshot_and_reload_cleanly(harness):
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    c.post("/v1/graphs/kb/nodes", {"nodes": [
        {"type": "t", "id": 1, "properties": {"name": "a", "only_one": 1}},
        {"type": "t", "id": 2, "properties": {"name": "b"}},
        {"type": "t", "id": 3, "properties": {}},
        {"type": "u", "id": "x", "properties": {"other": 2.5, "flag": True}}]})
    c.post("/v1/graphs/kb/edges", {"edges": [
        {"rel": "R", "source": {"type": "t", "id": 1}, "target": {"type": "t", "id": 2}, "properties": {"w": 1}},
        {"rel": "R", "source": {"type": "t", "id": 2}, "target": {"type": "t", "id": 3}}]})
    before = {n["id"] if n["type"] == "t" else "x": n["properties"] for n in
              c.get("/v1/graphs/kb/nodes?limit=100")[1]["items"]}
    h.restart()
    after = {n["id"] if n["type"] == "t" else "x": n["properties"] for n in
             c.get("/v1/graphs/kb/nodes?limit=100")[1]["items"]}
    assert before == after                           # nulls are never shown
    edges = c.get("/v1/graphs/kb/edges")[1]["items"]
    assert sorted(e["properties"].get("w", -1) for e in edges) == [-1, 1]
    assert c.post("/v1/graphs/kb/save", {})[0] == 200  # re-saving a reloaded graph still verifies


def test_every_value_the_api_accepts_survives_a_snapshot_round_trip():
    """Fuzz the validator against the real ISON round trip: whatever validation lets
    through must come back identical (this is how the unsafe characters were found)."""
    from isongraph_vector_server import validation
    from isongraph_vector_server.errors import ApiError
    rnd = random.Random(1234)
    alphabet = list("ab1 \"'\\\n\r\t#|:,;{}[]~$@/-.=_") + ["é", "日", "\U0001f600", "\x00", "\x01",
                                                          "\x7f", " ", " ", "\u0085", "﻿", "　",
                                                          "​", "‎", "⁠", "\ud800"]
    values = [0, 1, -1, 2 ** 53 - 1, 0.5, -0.0, 1.0, 100.0, 0.1, 1e15, 123456789.125, 1e-4, True, False, ""]
    for _ in range(4000):
        values.append("".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 6))))
    accepted = 0
    g = SemanticGraph(encoder=None, auto_embed=False)
    i = 0
    for v in values:
        try:
            validation.scalar_property_value(v, "$")
        except ApiError:
            continue
        accepted += 1
        i += 1
        node = g.add_node("t", i)
        node.properties["v"] = v
        node.properties["w"] = v
    assert accepted > 2000 and i == accepted
    img = storage.serialize(g)
    back = ISONGraph.from_ison(img.graph_text, "x")
    assert semantic_digest(back) == img.semantic
    # explicit: strict equality of every value including type (1.0 stays a float, "1" stays a string)
    for node in g.nodes():
        got = back.get_node("t", node.id).properties["v"]
        want = node.properties["v"]
        assert got == want and type(got) is type(want), (want, got)


def test_unsafe_unicode_whitespace_is_rejected_up_front(harness):
    """Regression for a round-trip bug found by fuzzing: U+00A0, U+2028, U+0085 and friends
    are stripped by the ISON text format, so the stored value would silently change."""
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    for ch in (" ", " ", "\u0085", "﻿", "　"):
        s, r, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "t", "id": 1, "properties": {"v": f"a{ch}"}}]})
        assert s == 400, repr(ch)
    assert c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "t", "id": 1, "properties": {"v": "café 日\U0001f600"}}]})[0] == 200


def test_save_refuses_to_publish_a_lossy_snapshot(harness):
    """Defence in depth: if memory ever holds something the format cannot carry, the save fails
    loudly and the previous generation stays current."""
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    seed(c)
    c.post("/v1/graphs/kb/save", {})
    handle = h.server.engine.tenants["alice"].graphs["kb"]
    handle.graph.get_node("person", 1).properties["zzlast"] = "x "   # bypasses validation; last column, so the format strips it
    handle.bump()
    s, r, _ = c.post("/v1/graphs/kb/save", {})
    assert s >= 500 and "round trip" not in json.dumps(r)                   # generic error, no internals
    assert (gdir(h) / "CURRENT").read_text().strip() == "000001"
    assert not [p for p in gdir(h).iterdir() if p.name.startswith(".tmp-gen")]
    assert h.server.metrics.counter("igv_snapshot_failures_total") == 1


def test_snapshot_does_not_block_readers_and_is_consistent_under_writes(harness):
    import threading
    h = harness()
    c = h.tenant("alice")
    create_graph(c)
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            s, r, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [
                {"type": "n", "id": i * 2, "properties": {"name": f"a{i}"}},
                {"type": "n", "id": i * 2 + 1, "properties": {"name": f"b{i}"}}]})
            if s != 200:
                errors.append(r)
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(8):
            assert c.post("/v1/graphs/kb/save", {})[0] == 200
            time.sleep(0.02)
    finally:
        stop.set()
        t.join()
    assert not errors
    h.server.stop()
    # every generation on disk is internally consistent AND holds whole batches (even count of nodes)
    st = GraphStorage(gdir(h))
    meta = st.read_meta()
    for gen in st.generations():
        graph, manifest, _ = st.load_generation(st.gen_path(gen), meta, h.server.encoder)
        assert graph.node_count() % 2 == 0
        assert graph.embedding_store.count() == graph.node_count()
        graph.close()
    h.restart()


def test_cli_verify_reports_problems(harness, tmp_path, capsys):
    from isongraph_vector_server.cli import main
    h = harness()
    c = two_generations(h)
    h.server.stop()
    assert main(["verify", "--data-dir", h.data_dir]) == 0
    (gdir(h) / "gen-000002" / "graph.isong").write_bytes(b"x")
    assert main(["verify", "--data-dir", h.data_dir]) == 1
    out = capsys.readouterr().out
    assert "DEGRADED" in out
    h.restart()
