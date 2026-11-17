"""Crash safety: SIGKILL a process in the middle of a save, then load what is on disk.

Two kinds of test. The deterministic ones stall a child *inside* the save at a named
point (storage.fault_hook, a test seam) and kill it there; the stress ones kill a child
that is saving continuously at random moments. In both, the invariant is the same:
whatever is on disk must verify, with no repair step, and must be one whole generation.
"""

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import DIM, Client, SERVER_ROOT, create_graph, seed
from isongraph_vector import MockEncoder
from isongraph_vector_server.storage import GraphStorage

CHILD = Path(__file__).resolve().parent / "crash_child.py"
pytestmark = pytest.mark.slow


def spawn(*args):
    return subprocess.Popen([sys.executable, str(CHILD), *map(str, args)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)


def read_until(proc, prefix, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError("child exited early: " + proc.stderr.read()[-2000:])
            continue
        if line.startswith(prefix):
            return line.strip()
    raise AssertionError("timed out waiting for " + prefix)


def load(path):
    st = GraphStorage(path)
    graph, manifest, degraded, _ = st.load(st.read_meta(), MockEncoder(16))
    return st, graph, manifest, degraded


@pytest.mark.parametrize("point,expect_nodes,expect_gen", [
    ("graph_written", 1500, 1),          # killed with a half-built temp dir
    ("files_written", 1500, 1),          # temp dir complete, not yet renamed
    ("generation_renamed", 1500, 1),     # gen-2 is on disk and complete, CURRENT still says 1
    ("current_tmp_written", 1500, 1),    # CURRENT.tmp written, replace not done
    ("current_flipped", 3000, 2),        # CURRENT says 2
])
def test_kill_at_every_step_of_a_save(tmp_path, point, expect_nodes, expect_gen):
    path = tmp_path / "g-kb"
    proc = spawn("point", path, point)
    try:
        read_until(proc, "GEN1")
        read_until(proc, "AT " + point)
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    # No repair step is run: load exactly what the dead process left behind.
    st, graph, manifest, degraded = load(path)
    assert not degraded, "CURRENT must always name a complete, verifying generation"
    assert graph.node_count() == expect_nodes and manifest.generation == expect_gen
    assert graph.embedding_store.count() == expect_nodes
    assert st.verify_generation(st.gen_path(expect_gen)).graph_digest == manifest.graph_digest
    graph.close()
    # startup would now remove the debris; the next save still works and supersedes it
    st.clean_tmp()
    assert not [p for p in path.iterdir() if p.name.startswith(".tmp-")]


def test_next_save_after_a_crash_continues_cleanly(tmp_path):
    path = tmp_path / "g-kb"
    proc = spawn("point", path, "generation_renamed")
    read_until(proc, "GEN1")
    read_until(proc, "AT generation_renamed")
    proc.kill()
    proc.wait(timeout=30)
    # gen-000002 exists on disk but CURRENT says 1; a new save must not collide with it
    st, graph, manifest, degraded = load(path)
    assert (path / "gen-000002").exists() and manifest.generation == 1
    from crash_child import fill
    fill(graph, 5000, 5010)
    from isongraph_vector_server.storage import serialize
    m = st.write_generation(serialize(graph))
    assert m.generation == 3
    _, g2, m2, deg = load(path)
    assert not deg and m2.generation == 3 and g2.node_count() == 1510


def test_random_kills_during_continuous_saving(tmp_path):
    path = tmp_path / "g-kb"
    rnd = random.Random(7)
    last_nodes = 0
    killed_mid_save = 0
    for round_no in range(10):
        proc = spawn("loop", path, 400)
        try:
            read_until(proc, "READY")
            read_until(proc, "SAVED")                              # at least one save lands
            time.sleep(rnd.uniform(0.0, 0.35))                     # then die at a random moment
        finally:
            proc.kill()
            proc.wait(timeout=30)
        had_tmp = any(p.name.startswith(".tmp-gen") for p in path.iterdir())
        killed_mid_save += had_tmp
        st, graph, manifest, degraded = load(path)
        assert not degraded, f"round {round_no}: CURRENT pointed at something that does not verify"
        assert graph.node_count() == manifest.node_count
        assert graph.node_count() % 400 == 0 and graph.node_count() >= last_nodes   # whole generations only
        assert graph.embedding_store.count() == graph.node_count()
        last_nodes = graph.node_count()
        graph.close()
    assert last_nodes >= 400 * 10 or last_nodes > 0
    # (killed_mid_save is informational: it shows the kills did land inside saves)
    print("kills that left a partial temp directory:", killed_mid_save)


def test_sigkill_of_a_real_server_loses_no_acknowledged_write_in_sync_mode(tmp_path):
    from isongraph_vector_server import auth
    data = tmp_path / "data"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "data_dir": str(data), "host": "127.0.0.1", "port": 0, "durability": "sync", "log_level": "CRITICAL",
        "autosave_interval_seconds": 0,
        "encoder": {"kind": "mock", "dimension": DIM},
        "quotas": {"rate_per_second": 100000, "rate_burst": 100000}}))
    auth.create_tenant(str(data), "alice")
    key = auth.add_key(str(data), "alice")
    # port 0 is not discoverable from outside; pick a free one
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cfg.write_text(cfg.read_text().replace('"port": 0', f'"port": {port}'))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SERVER_ROOT), str(Path(SERVER_ROOT).parent),
                                                      os.environ.get("PYTHONPATH", "")]))
    boot = ("import sys; sys.path.insert(0, %r); import conftest; "
            "from isongraph_vector_server.cli import main; sys.exit(main(['serve', '--config', %r]))"
            % (str(Path(__file__).resolve().parent), str(cfg)))

    def start():
        p = subprocess.Popen([sys.executable, "-c", boot], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
        c = Client(f"http://127.0.0.1:{port}", key)
        for _ in range(100):
            try:
                if c.get("/readyz", key=None)[0] == 200:
                    return p, c
            except OSError:
                time.sleep(0.1)
        p.kill()
        raise AssertionError("server did not start: " + p.stderr.read().decode()[-2000:])

    proc, c = start()
    try:
        create_graph(c)
        acked = []
        deadline = time.time() + 1.5
        i = 0
        while time.time() < deadline:
            s, _, _ = c.post("/v1/graphs/kb/nodes", {"nodes": [{"type": "n", "id": i, "properties": {"name": f"n{i}"}}]})
            if s == 200:
                acked.append(i)
            i += 1
        proc.kill()                                  # SIGKILL / TerminateProcess: no shutdown hook runs
        proc.wait(timeout=30)
        assert len(acked) > 5
        proc, c = start()
        have = {x["id"] for x in c.get("/v1/graphs/kb/nodes?limit=1000")[1]["items"]}
        missing = [i for i in acked if i not in have]
        assert not missing, f"acknowledged writes lost: {missing[:5]}"
        assert c.get("/v1/graphs/kb")[1]["status"] == "ok"
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_sigterm_style_shutdown_saves_unsaved_changes(tmp_path):
    """The real CLI process, snapshot durability: a termination signal makes it stop accepting,
    finish, write a final snapshot and exit 0 (SIGTERM on POSIX, CTRL_BREAK on Windows)."""
    from isongraph_vector_server import auth
    import socket
    data = tmp_path / "data"
    auth.create_tenant(str(data), "alice")
    key = auth.add_key(str(data), "alice")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "data_dir": str(data), "host": "127.0.0.1", "port": port, "log_level": "CRITICAL",
        "autosave_interval_seconds": 0, "encoder": {"kind": "mock", "dimension": DIM}}))
    env = dict(os.environ)
    boot = ("import sys; sys.path.insert(0, %r); import conftest; "
            "from isongraph_vector_server.cli import main; sys.exit(main(['serve', '--config', %r]))"
            % (str(Path(__file__).resolve().parent), str(cfg)))
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-c", boot], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, creationflags=flags)
    c = Client(f"http://127.0.0.1:{port}", key)
    try:
        for _ in range(100):
            try:
                if c.get("/readyz", key=None)[0] == 200:
                    break
            except OSError:
                time.sleep(0.1)
        create_graph(c)
        seed(c)
        assert c.get("/v1/graphs/kb")[1]["unsaved_changes"] is True
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    st = GraphStorage(data / "tenants" / "alice" / "graphs" / "g-kb")
    graph, manifest, degraded, _ = st.load(st.read_meta(), MockEncoder(16))
    assert not degraded and graph.node_count() == 3 and graph.edge_count() == 2


def test_library_in_place_save_can_leave_a_mismatched_pair(tmp_path):
    """Why generations exist: the library's own save() is two atomic renames, not one. Killed between
    them, the graph file and its sidecar come from different saves. The library *detects* that on load
    (digest check) but has no older copy to fall back to; the service's generations do."""
    from isongraph_vector import GraphEmbeddingMismatchError, SemanticGraph
    from unittest import mock
    enc = MockEncoder(8)
    g = SemanticGraph(encoder=enc, auto_embed=False)
    for i in range(5):
        g.add_node("n", i)
        g.embedding_store.add(("n", i), f"n{i}", vector=enc.encode(f"n{i}"))
    path = tmp_path / "plain.isong"
    g.save(path)
    g.add_node("n", 99)
    g.embedding_store.add(("n", 99), "n99", vector=enc.encode("n99"))
    with mock.patch.object(type(g.embedding_store), "save_embeddings", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            g.save(path)                              # "dies" after the graph file, before the sidecar
    with pytest.raises(GraphEmbeddingMismatchError):
        SemanticGraph.load(path, encoder=enc)         # detected, but there is nothing to roll back to
