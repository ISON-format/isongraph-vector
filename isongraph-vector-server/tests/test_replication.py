"""Snapshot read replicas: what they serve, and when they refuse to."""

import json
import shutil
import time
from pathlib import Path

import pytest

from conftest import Client, create_graph, seed
from isongraph_vector_server import auth, storage

TOKEN = "t" * 40


def gen_dir(h, tenant, graph, gen):
    return Path(h.data_dir) / "tenants" / tenant / "graphs" / f"g-{graph}" / f"gen-{gen:06d}"


@pytest.fixture
def primary(harness):
    h = harness(replication_token=TOKEN)
    a, b = h.tenant("tenant-a"), h.tenant("tenant-b")
    create_graph(a, "kb")
    seed(a)
    create_graph(b, "kb")
    b.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 1, "properties": {"name": "Zed"}}]})
    for c in (a, b):
        assert c.post("/v1/graphs/kb/save", {})[0] == 200
    return h, a, b


def as_replica(replica_h, client):
    return Client(replica_h.server.url, client.key)


def sync(replica_h):
    return replica_h.server.replica.sync_once()


@pytest.fixture(params=["dir", "http"])
def source(request, primary):
    h, _, _ = primary
    return f"dir:{h.data_dir}" if request.param == "dir" else h.server.url


def test_replica_serves_the_primarys_snapshot_read_only(primary, replica_of, source):
    h, a, b = primary
    r = replica_of(source, **({"replication_token": TOKEN} if source.startswith("http") else {}))
    report = sync(r)
    assert not report["errors"] and not report["refused"], report
    ra, rb = as_replica(r, a), as_replica(r, b)
    # same answers as the primary
    q = {"query": "Bob trains machine learning models", "top_k": 3, "threshold": -1.0}
    assert ra.post("/v1/graphs/kb/search", q)[1] == a.post("/v1/graphs/kb/search", q)[1]
    s, g, hd = ra.get("/v1/graphs/kb")
    assert g["nodes"] == 3 and g["snapshot_digest"] == a.get("/v1/graphs/kb")[1]["snapshot_digest"]
    assert hd["X-Snapshot-Generation"] == "1" and int(hd["X-Snapshot-Age-Seconds"]) < 30
    # tenant isolation holds on the replica too
    assert [x["id"] for x in rb.get("/v1/graphs/kb/nodes")[1]["items"]] == [1]
    assert rb.get("/v1/graphs/kb/nodes/person/2")[0] == 404
    # writes of every kind are refused with a stable code
    for method, path, body in [("POST", "/v1/graphs", {"name": "x"}), ("DELETE", "/v1/graphs/kb", None),
                               ("POST", "/v1/graphs/kb/nodes", {"nodes": [{"type": "a", "id": 1}]}),
                               ("PATCH", "/v1/graphs/kb/nodes/person/1", {}), ("POST", "/v1/graphs/kb/save", {}),
                               ("POST", "/v1/graphs/kb/reembed", {})]:
        s, resp, _ = ra.request(method, path, body)
        assert s == 403 and resp["error"]["code"] == "read_only_replica", (method, path)
    # a replica does not itself serve replication
    assert Client(r.server.url, TOKEN).get("/_repl/v1/index")[0] == 404


def test_incremental_sync_only_moves_what_changed(primary, replica_of):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}")
    first = sync(r)
    assert sorted(first["updated"]) == ["tenant-a/kb@1", "tenant-a/tenant.json", "tenant-b/kb@1", "tenant-b/tenant.json"]
    assert sync(r)["updated"] == []                                  # nothing changed
    a.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 4, "properties": {"name": "Dan"}}]})
    a.post("/v1/graphs/kb/save", {})
    second = sync(r)
    assert second["updated"] == ["tenant-a/kb@2"]
    assert as_replica(r, a).get("/v1/graphs/kb")[1]["nodes"] == 4
    assert as_replica(r, b).get("/v1/graphs/kb")[1]["generation"] == 1


def test_replica_lags_by_design_until_the_primary_saves_and_it_syncs(primary, replica_of):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}")
    sync(r)
    a.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 5, "properties": {"name": "Eve"}}]})
    ra = as_replica(r, a)
    sync(r)
    assert ra.get("/v1/graphs/kb/nodes/person/5")[0] == 404          # unsaved on the primary: invisible
    a.post("/v1/graphs/kb/save", {})
    assert ra.get("/v1/graphs/kb/nodes/person/5")[0] == 404          # saved but not yet synced
    sync(r)
    assert ra.get("/v1/graphs/kb/nodes/person/5")[0] == 200


# ---- refusing bad snapshots --------------------------------------------------------------------------------------


def make_gen2(h, a):
    a.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 4, "properties": {"name": "Dan"}}]})
    assert a.post("/v1/graphs/kb/save", {})[0] == 200
    return gen_dir(h, "tenant-a", "kb", 2)


def rewrite_manifest(g2, **file_updates):
    m = json.loads((g2 / "MANIFEST.json").read_text())
    for fname in ("graph.isong", "graph.isong.embeddings.json"):
        sha, size = storage.sha256_file(g2 / fname)
        m["files"][fname] = {"sha256": sha, "size": size}
    m.update(file_updates)
    (g2 / "MANIFEST.json").write_text(json.dumps(m))


def tamper_checksum(g2):
    (g2 / "graph.isong").write_bytes((g2 / "graph.isong").read_bytes().replace(b"Dan", b"Eve"))


def tamper_graph_consistent_checksums(g2):
    """The attacker fixes the manifest checksums too: only the digest binding catches it."""
    (g2 / "graph.isong").write_bytes((g2 / "graph.isong").read_bytes().replace(b"Dan", b"Eve"))
    rewrite_manifest(g2)


def swap_in_older_sidecar(g2, h):
    older = gen_dir(h, "tenant-a", "kb", 1) / "graph.isong.embeddings.json"
    (g2 / "graph.isong.embeddings.json").write_bytes(older.read_bytes())
    rewrite_manifest(g2)


def lie_about_digest_in_manifest(g2):
    m = json.loads((g2 / "MANIFEST.json").read_text())
    m["graph_digest"] = "sha256:" + "0" * 64
    (g2 / "MANIFEST.json").write_text(json.dumps(m))


def truncate_sidecar(g2):
    p = g2 / "graph.isong.embeddings.json"
    p.write_bytes(p.read_bytes()[:-50])


def strip_digest_from_sidecar(g2):
    p = g2 / "graph.isong.embeddings.json"
    payload = json.loads(p.read_text())
    payload.pop("graph_digest")
    payload.pop("graph_digest_basis")
    p.write_text(json.dumps(payload))
    rewrite_manifest(g2)


@pytest.mark.parametrize("tamper", [
    "tamper_checksum", "tamper_graph_consistent_checksums", "swap_in_older_sidecar",
    "lie_about_digest_in_manifest", "truncate_sidecar", "strip_digest_from_sidecar"])
def test_replica_refuses_a_snapshot_that_fails_verification(primary, replica_of, tamper):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}")
    sync(r)
    ra = as_replica(r, a)
    before = ra.get("/v1/graphs/kb")[1]
    last_ok = r.server.replica.last_ok_wall
    g2 = make_gen2(h, a)
    fn = globals()[tamper]
    fn(g2, h) if tamper == "swap_in_older_sidecar" else fn(g2)

    report = sync(r)
    assert report["refused"] == ["tenant-a/kb"], report
    assert report["updated"] == []
    # still serving the last verified generation, byte for byte
    now = ra.get("/v1/graphs/kb")[1]
    assert (now["generation"], now["nodes"], now["snapshot_digest"]) == (
        before["generation"], before["nodes"], before["snapshot_digest"])
    assert ra.get("/v1/graphs/kb/nodes/person/4")[0] == 404
    assert r.server.replica.refused == 1
    assert "tenant-a/kb" in r.server.replica.last_error
    assert r.server.replica.last_ok_wall == last_ok                      # a refused cycle is not a fresh one
    assert r.server.metrics.counter("igv_replica_refused_total") == 1
    # nothing of the bad generation was installed
    rgraph = Path(r.data_dir) / "tenants" / "tenant-a" / "graphs" / "g-kb"
    assert not (rgraph / "gen-000002").exists()
    assert (rgraph / "CURRENT").read_text().strip() == "000001"
    assert not [p for p in rgraph.parent.iterdir() if p.name.startswith(".tmp")]
    # other tenants still sync fine in the same cycle
    b.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 9, "properties": {"name": "Ok"}}]})
    b.post("/v1/graphs/kb/save", {})
    report = sync(r)
    assert report["refused"] == ["tenant-a/kb"] and report["updated"] == ["tenant-b/kb@2"]
    # and once the primary publishes a good generation, the replica catches up
    a.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "person", "id": 6, "properties": {"name": "Fay"}}]})
    a.post("/v1/graphs/kb/save", {})
    report = sync(r)
    assert report["refused"] == [] and ra.get("/v1/graphs/kb/nodes/person/6")[0] == 200


def test_replica_refuses_over_http_when_the_transport_corrupts_a_file(primary, replica_of):
    h, a, b = primary
    r = replica_of(h.server.url, replication_token=TOKEN)
    sync(r)
    g2 = make_gen2(h, a)
    tamper_checksum(g2)
    report = sync(r)
    assert report["refused"] == ["tenant-a/kb"]


def test_staleness_bound_makes_the_replica_stop_serving(primary, replica_of, tmp_path):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}", **{"replica.max_staleness_seconds": 1.5})
    sync(r)
    ra = as_replica(r, a)
    assert ra.get("/v1/graphs/kb")[0] == 200
    assert r.anon().get("/readyz")[0] == 200
    # the primary disappears: syncs fail, the clock keeps running
    shutil.move(h.data_dir + "/tenants", h.data_dir + "/tenants.gone")
    try:
        assert sync(r)["errors"] == ["index"]
        time.sleep(1.7)
        s, resp, hd = ra.get("/v1/graphs/kb")
        assert s == 503 and resp["error"]["code"] == "replica_stale" and "Retry-After" in hd
        assert ra.post("/v1/graphs/kb/search", {"query": "x"})[0] == 503
        assert r.anon().get("/readyz")[0] == 503
        assert r.anon().get("/healthz")[0] == 200                        # alive, just not serving
    finally:
        shutil.move(h.data_dir + "/tenants.gone", h.data_dir + "/tenants")
    assert not sync(r)["errors"]
    assert ra.get("/v1/graphs/kb")[0] == 200
    assert r.anon().get("/readyz")[0] == 200


def test_a_replica_that_never_synced_serves_nothing(primary, replica_of, tmp_path):
    h, a, b = primary
    r = replica_of(f"dir:{tmp_path / 'does-not-exist'}")
    assert r.anon().get("/readyz")[0] == 503
    s, resp, _ = as_replica(r, a).get("/v1/graphs")
    assert s == 401 or (s == 503 and resp["error"]["code"] == "replica_not_synced")


def test_replica_restart_keeps_serving_verified_data_and_remembers_staleness(primary, replica_of):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}", **{"replica.max_staleness_seconds": 3600})
    sync(r)
    stamp = r.server.replica.last_ok_wall
    shutil.move(h.data_dir + "/tenants", h.data_dir + "/tenants.gone")        # primary unreachable during the restart
    try:
        r.restart()
        assert r.server.replica.last_ok_wall == pytest.approx(stamp, abs=1e-3)
        ra = as_replica(r, a)
        assert ra.get("/v1/graphs/kb")[1]["nodes"] == 3
    finally:
        shutil.move(h.data_dir + "/tenants.gone", h.data_dir + "/tenants")


def test_revoking_a_key_reaches_the_replica_on_the_next_sync(primary, replica_of):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}")
    sync(r)
    ra = as_replica(r, a)
    assert ra.get("/v1/graphs/kb")[0] == 200
    auth.revoke_key(h.data_dir, "tenant-a", a.key.split(".")[1])
    assert a.get("/v1/graphs")[0] == 401                  # primary: immediately
    assert ra.get("/v1/graphs/kb")[0] == 200              # replica: documented lag until the next sync
    sync(r)
    assert ra.get("/v1/graphs/kb")[0] == 401


def test_deletions_propagate(primary, replica_of):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}")
    sync(r)
    create_graph(a, "temp")
    a.post("/v1/graphs/temp/save", {})
    sync(r)
    ra = as_replica(r, a)
    assert ra.get("/v1/graphs/temp")[0] == 200
    a.delete("/v1/graphs/temp")
    report = sync(r)
    assert report["removed"] == ["tenant-a/temp"]
    assert ra.get("/v1/graphs/temp")[0] == 404
    assert not (Path(r.data_dir) / "tenants" / "tenant-a" / "graphs" / "g-temp").exists()


def test_an_unreadable_graph_on_the_primary_is_not_mistaken_for_a_deleted_one(primary, replica_of):
    h, a, b = primary
    r = replica_of(f"dir:{h.data_dir}")
    sync(r)
    (Path(h.data_dir) / "tenants" / "tenant-a" / "graphs" / "g-kb" / "meta.json").unlink()
    report = sync(r)
    assert report["removed"] == [] and "tenant-a/kb" in report["errors"]
    assert as_replica(r, a).get("/v1/graphs/kb")[1]["nodes"] == 3        # replica keeps what it verified


def test_empty_never_saved_graph_replicates(primary, replica_of):
    h, a, b = primary
    create_graph(a, "fresh")                               # created, never saved: no generations
    r = replica_of(f"dir:{h.data_dir}")
    report = sync(r)
    assert "tenant-a/fresh@None" in report["updated"]
    s, g, _ = as_replica(r, a).get("/v1/graphs/fresh")
    assert s == 200 and g["nodes"] == 0


def test_http_source_requires_the_right_token(primary, replica_of):
    h, a, b = primary
    r = replica_of(h.server.url, replication_token="x" * 40)
    report = sync(r)
    assert report["errors"] == ["index"]
    assert as_replica(r, a).get("/v1/graphs")[0] in (401, 503)
