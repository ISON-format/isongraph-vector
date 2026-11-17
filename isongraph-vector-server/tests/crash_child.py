"""Child process for the crash tests: saves a graph and gets killed by the parent.

usage:
  crash_child.py point  <graphdir> <fault-point>     publish gen 1 cleanly, then stall *inside* the
                                                     save of gen 2 at <fault-point> (parent SIGKILLs)
  crash_child.py loop   <graphdir> <nodes-per-round> keep mutating and saving forever
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conftest  # noqa: F401,E402  (binds isongraph_vector from the checkout)

from isongraph_vector import MockEncoder  # noqa: E402
from isongraph_vector_server import storage  # noqa: E402
from isongraph_vector_server.storage import GraphMeta, GraphStorage, new_semantic_graph, serialize  # noqa: E402

ENC = MockEncoder(16)


def fill(graph, start, stop):
    for i in range(start, stop):
        node = graph.add_node("n", i)
        node.properties["name"] = f"node {i}"
        graph.embedding_store.add(("n", i), f"node {i}", vector=ENC.encode(f"node {i}"))


def open_or_create(path: Path):
    meta = GraphMeta(name=path.name[2:], mode="server", dimension=16,
                     embed_fields=["name"], ann=None, ann_nprobe=None)
    if (path / "meta.json").exists():
        st = GraphStorage(path)
        st.clean_tmp()
        graph, manifest, degraded, _ = st.load(st.read_meta(), ENC)
        assert not degraded
        return st, graph
    st = GraphStorage.create(path.parent, meta)
    return st, new_semantic_graph(meta, ENC)


def main():
    mode, path = sys.argv[1], Path(sys.argv[2])
    st, graph = open_or_create(path)
    if mode == "point":
        point = sys.argv[3]
        fill(graph, 0, 1500)
        st.write_generation(serialize(graph))          # generation 1, clean
        print("GEN1", flush=True)
        fill(graph, 1500, 3000)

        def hook(name):
            if name == point:
                print("AT " + name, flush=True)
                time.sleep(600)
        storage.fault_hook = hook
        st.write_generation(serialize(graph))          # stalls inside, parent kills us
    else:
        per_round = int(sys.argv[3])
        base = graph.node_count()
        print("READY", flush=True)
        while True:
            fill(graph, base, base + per_round)
            base += per_round
            st.write_generation(serialize(graph))
            print(f"SAVED {base}", flush=True)


if __name__ == "__main__":
    main()
