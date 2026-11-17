"""Python half of the graph-file cross-port check (see graph_files.py).

    graph_py.py write    <dir>
    graph_py.py read     <dir> <observation.json>
    graph_py.py mismatch <dir>

`write` saves the shared fixture graph as <dir>/graph.ison plus its
embeddings sidecar and binary snapshot. `read` loads <dir>/graph.ison with
on_mismatch="error" and writes what it found (and the load source, snapshot or
json, with the reason a snapshot was skipped) to <observation.json> for the
orchestrator to judge.
`mismatch` loads a graph whose file was edited after the save and exits 0 only
if the load raises GraphEmbeddingMismatchError.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401  - binds `isongraph_vector` to the port directory

from isongraph_vector import (  # noqa: E402
    GraphEmbeddingMismatchError,
    MockEncoder,
    SemanticGraph,
)

DIM = 384
BS = chr(92)  # a backslash, spelled so no source-level escaping can mangle it
DOC_TEXT = "Résumé of 日本語 engineers, \"quoted\" and back" + BS + "slash"
QUERIES = [
    "Bob analyses ML models",
    "Carol plans roadmaps for über-teams",
    DOC_TEXT,
]

# Keep in step with the other graph_* helpers: the point of the check is that
# every port writes (and expects) the same graph. Python ids keep their type,
# so 1 and 2 are integers here.
NODES = [
    ("person", 1, dict(text="Alice is a software engineer", age=30, score=0.5,
                       active=True, nickname=None)),
    ("person", 2, dict(text="Bob analyses ML models", age=41, score=2.75,
                       active=False, nickname=None)),
    ("person", "carol", dict(text="Carol plans roadmaps for über-teams", age=35,
                             score=-1.25, active=True, nickname="Caz")),
    ("doc", "résumé", dict(text=DOC_TEXT)),
    ("topic", "ml", dict(text="machine learning models", weight=0.125)),
]
EDGES = [
    ("KNOWS", ("person", 1), ("person", 2), dict(since=2020, weight=0.75)),
    ("WROTE", ("person", "carol"), ("doc", "résumé"), dict(note="first draft")),
    ("ABOUT", ("doc", "résumé"), ("topic", "ml"), {}),
]


def graph_path(directory: str) -> Path:
    return Path(directory) / "graph.ison"


def write(directory: str) -> None:
    graph = SemanticGraph("fixture", encoder=MockEncoder(DIM))
    for node_type, node_id, props in NODES:
        graph.add_node(node_type, node_id, _embed_text=props["text"], **props)
    for rel, source, target, props in EDGES:
        graph.add_edge(rel, source, target, **props)
    Path(directory).mkdir(parents=True, exist_ok=True)
    graph.save(graph_path(directory), snapshot=True)
    graph.close()


def observe(graph: SemanticGraph, encoder: MockEncoder, report) -> dict:
    nodes, missing, worst = [], [], 0.0
    for node in graph.nodes():
        nodes.append({"type": node.type, "id": node.id, "props": dict(node.properties)})
        record = graph.get_embedding((node.type, node.id))
        if record is None:
            missing.append([node.type, node.id])
            continue
        expected = encoder.encode(node.properties["text"])
        if len(expected) != len(record.vector):
            worst = 1e9
        else:
            worst = max(worst, max(abs(float(a) - float(b))
                                   for a, b in zip(record.vector, expected)))
    edges = []
    for rel in graph.edge_types():
        for edge in graph.edges(rel):
            edges.append({"rel": edge.rel_type, "source": list(edge.source),
                          "target": list(edge.target), "props": dict(edge.properties)})
    searches = []
    for query in QUERIES:
        hits = graph.similarity_search(query, top_k=1)
        top = hits[0] if hits else None
        searches.append({"query": query,
                         "top": list(top.node_ref) if top else None,
                         "score": float(top.score) if top else None})
    return {"port": "python", "nodes": nodes, "edges": edges,
            "embedding_count": graph.embedding_store.count(),
            "missing_embeddings": missing, "max_vec_diff": worst, "searches": searches,
            "source": report.source, "snapshot_skipped": report.snapshot_skipped}


def read(directory: str, out: str) -> None:
    encoder = MockEncoder(DIM)
    graph, report = SemanticGraph.load_with_report(
        graph_path(directory), encoder=encoder, on_mismatch="error")
    observation = observe(graph, encoder, report)
    graph.close()
    Path(out).write_text(json.dumps(observation, ensure_ascii=False), encoding="utf-8")


def mismatch(directory: str) -> int:
    try:
        SemanticGraph.load(graph_path(directory), encoder=MockEncoder(DIM),
                           on_mismatch="error").close()
    except GraphEmbeddingMismatchError:
        return 0
    except Exception as exc:  # noqa: BLE001 - report whatever came instead
        print(f"expected GraphEmbeddingMismatchError, got {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print("expected GraphEmbeddingMismatchError, but the load succeeded", file=sys.stderr)
    return 1


def main(argv: list) -> int:
    if len(argv) == 3 and argv[1] == "write":
        write(argv[2])
    elif len(argv) == 4 and argv[1] == "read":
        read(argv[2], argv[3])
    elif len(argv) == 3 and argv[1] == "mismatch":
        return mismatch(argv[2])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
