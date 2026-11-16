#!/usr/bin/env python3
"""Cross-port check of graph files, embeddings sidecars and the graph digest.

The SQLite and payload checks (node_write.mjs, py_read.py and friends) prove the
ports share one embeddings format. This proves the same for what
`SemanticGraph.save` writes: a graph file plus a `<graph>.embeddings.json`
sidecar carrying a `graph_digest` that `load` verifies.

Every port has a small standalone helper (graph_py.py, graph_node.mjs for the
JavaScript and TypeScript ports, rust/, csharp/ and cpp/) that can

    write    <dir>                save the fixture graph + sidecar into <dir>
    read     <dir> <out.json>     load <dir>/graph.ison with the mismatch mode
                                  set to ERROR and dump what it found
    mismatch <dir>                load a graph edited after the save and exit 0
                                  only if the port raised its mismatch error

This script builds the helpers outside the repository, has every port write the
fixture, and then has every port read every port's output (a full writers x
readers matrix). For each pair it judges the observation the reader produced:

  * the sidecar digest verified (the load did not raise in ERROR mode);
  * node and edge counts, every node's properties, every edge's properties;
  * every stored embedding equals the reader's own MockEncoder(text) within 1e-6;
  * a text search finds the expected node for three different queries.

A second matrix tampers with one property in each written graph file (age 30
becomes 31) and requires every reader to raise its graph/embeddings mismatch
error rather than load the stale vectors.

Allowances, each visible in the output rather than silent:

  * Rust, C# and C++ node ids are strings, so they write the fixture's integer
    ids 1 and 2 as "1" and "2". Ids are compared by their string form unless
    both the writer and the reader keep integer ids (Python, JS, TS), in which
    case the integer type must survive the trip.
  * The C# and C++ base libraries drop null-valued properties when they read a
    file, so for those readers a null property may be absent.

Usage: python graph_files.py [--ports py,js,ts,rs,cs,cpp] [--workdir DIR]
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ALL_PORTS = ["py", "js", "ts", "rs", "cs", "cpp"]
LABEL = {"py": "Python", "js": "JS", "ts": "TS", "rs": "Rust", "cs": "C#", "cpp": "C++"}

# Ports whose node ids keep their type (an integer id stays an integer).
TYPED_IDS = {"py", "js", "ts"}
# Readers whose base ISON library drops null-valued properties on load.
DROPS_NULLS = {"cs", "cpp"}

# Pairs that legitimately cannot work: (writer, reader) -> reason. Skipped
# visibly. Empty today - every pair is expected to pass.
KNOWN_SKIPS: dict = {}

TOLERANCE = 1e-6

# What every helper writes. Keep in step with graph_py.py and friends.
BS = chr(92)
DOC_TEXT = 'Résumé of 日本語 engineers, "quoted" and back' + BS + "slash"
EXPECTED_NODES = {
    ("person", "1"): {"text": "Alice is a software engineer", "age": 30, "score": 0.5,
                      "active": True, "nickname": None},
    ("person", "2"): {"text": "Bob analyses ML models", "age": 41, "score": 2.75,
                      "active": False, "nickname": None},
    ("person", "carol"): {"text": "Carol plans roadmaps for über-teams", "age": 35,
                          "score": -1.25, "active": True, "nickname": "Caz"},
    ("doc", "résumé"): {"text": DOC_TEXT},
    ("topic", "ml"): {"text": "machine learning models", "weight": 0.125},
}
# Ids that are integers in the typed-id ports.
INT_IDS = {("person", "1"), ("person", "2")}
EXPECTED_EDGES = {
    ("KNOWS", ("person", "1"), ("person", "2")): {"since": 2020, "weight": 0.75},
    ("WROTE", ("person", "carol"), ("doc", "résumé")): {"note": "first draft"},
    ("ABOUT", ("doc", "résumé"), ("topic", "ml")): {},
}
EXPECTED_SEARCHES = [
    ("Bob analyses ML models", ("person", "2")),
    ("Carol plans roadmaps for über-teams", ("person", "carol")),
    (DOC_TEXT, ("doc", "résumé")),
]


class Fatal(Exception):
    pass


# -- running things -----------------------------------------------------------

def run(cmd, cwd=None, env=None, check=True):
    full_env = dict(os.environ, PYTHONUTF8="1", DOTNET_NOLOGO="1",
                    DOTNET_CLI_TELEMETRY_OPTOUT="1", DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1")
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=full_env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError as exc:
        raise Fatal(f"cannot run {cmd[0]}: {exc}") from exc
    output = proc.stdout.decode("utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise Fatal(f"{' '.join(str(c) for c in cmd[:3])} ... exited {proc.returncode}\n{output.strip()}")
    return proc.returncode, output


def need(tool, why):
    if shutil.which(tool) is None:
        raise Fatal(f"'{tool}' is not on PATH (needed for {why})")


# -- building the helpers (all outside the repository) --------------------------

def build_py(wd):
    return [sys.executable, str(HERE / "graph_py.py")]


def build_js(wd):
    need("node", "the JavaScript port")
    if not (REPO / "isongraph-vector-js" / "node_modules" / "ison-graph-js").exists():
        raise Fatal("isongraph-vector-js has no node_modules: run "
                    "`npm --prefix isongraph-vector-js install` first")
    return ["node", str(HERE / "graph_node.mjs"), "js"]


def build_ts(wd):
    need("node", "the TypeScript port")
    ts = REPO / "isongraph-vector-ts"
    tsc = ts / "node_modules" / "typescript" / "bin" / "tsc"
    if not tsc.exists():
        raise Fatal("isongraph-vector-ts has no node_modules: run "
                    "`npm --prefix isongraph-vector-ts install` first")
    out = wd / "ts"
    run(["node", tsc, "-p", "tsconfig.build.json", "--outDir", out / "dist"], cwd=ts)
    # The compiled files import 'ison-graph-ts'; node finds it by walking up
    # from the output directory, so link the port's node_modules next to it.
    link = out / "node_modules"
    if not link.exists():
        if os.name == "nt":
            run(["cmd", "/c", "mklink", "/J", link, ts / "node_modules"])
        else:
            os.symlink(ts / "node_modules", link, target_is_directory=True)
    return ["node", str(HERE / "graph_node.mjs"), f"ts={out / 'dist'}"]


def build_rs(wd):
    need("cargo", "the Rust port")
    target = Path(os.environ.get("CROSS_PORT_RUST_TARGET_DIR") or wd / "rust-target")
    run(["cargo", "build", "--release", "--quiet",
         "--manifest-path", HERE / "rust" / "Cargo.toml", "--target-dir", target])
    exe = target / "release" / ("cross_port_graph.exe" if os.name == "nt" else "cross_port_graph")
    if not exe.exists():
        raise Fatal(f"cargo built no {exe}")
    return [str(exe)]


def build_cs(wd):
    need("dotnet", "the C# port")
    obj, out = wd / "cs-obj", wd / "cs-out"
    # Trailing separators matter: MSBuild concatenates these with file names.
    run(["dotnet", "build", HERE / "csharp" / "CrossPortGraph.csproj", "-c", "Release",
         "--nologo", "-v", "q", f"-p:BaseIntermediateOutputPath={obj.as_posix()}/",
         f"-p:OutDir={out.as_posix()}/"])
    return ["dotnet", str(out / "CrossPortGraph.dll")]


def build_cpp(wd):
    need("cmake", "the C++ port")
    build = wd / "cpp-build"
    run(["cmake", "-S", HERE / "cpp", "-B", build, "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", build, "--config", "Release"])
    name = "cross_port_graph.exe" if os.name == "nt" else "cross_port_graph"
    for candidate in (build / "Release" / name, build / name):
        if candidate.exists():
            return [str(candidate)]
    raise Fatal(f"cmake built no {name} under {build}")


BUILDERS = {"py": build_py, "js": build_js, "ts": build_ts,
            "rs": build_rs, "cs": build_cs, "cpp": build_cpp}


# -- judging an observation -----------------------------------------------------

def same(a, b):
    """Equality that keeps bool, int and float apart (Python's == does not)."""
    return type(a) is type(b) and a == b


def props_problem(expected, actual, drops_nulls):
    if drops_nulls:
        expected = {k: v for k, v in expected.items() if v is not None}
        actual = {k: v for k, v in actual.items() if v is not None}
    if set(expected) != set(actual):
        return f"properties {sorted(actual)} != {sorted(expected)}"
    for key in expected:
        if not same(expected[key], actual[key]):
            return f"property {key!r} is {actual[key]!r}, expected {expected[key]!r}"
    return None


def judge(writer, reader, obs):
    """Return a list of problems with what `reader` loaded from `writer`'s files."""
    problems = []
    strict_ids = writer in TYPED_IDS and reader in TYPED_IDS
    drops_nulls = reader in DROPS_NULLS

    def key_of(ref):
        return (ref[0], str(ref[1]))

    def id_problem(ref, where):
        if strict_ids and key_of(ref) in INT_IDS and not (
                isinstance(ref[1], int) and not isinstance(ref[1], bool)):
            return f"{where}: id {ref[1]!r} should have stayed an integer"
        if strict_ids and key_of(ref) not in INT_IDS and not isinstance(ref[1], str):
            return f"{where}: id {ref[1]!r} should be a string"
        return None

    nodes = {}
    for node in obs["nodes"]:
        ref = (node["type"], node["id"])
        problem = id_problem(ref, "node")
        if problem:
            problems.append(problem)
        nodes[key_of(ref)] = node["props"]
    if len(obs["nodes"]) != len(EXPECTED_NODES) or set(nodes) != set(EXPECTED_NODES):
        problems.append(f"nodes {sorted(nodes)} != {sorted(EXPECTED_NODES)}")
    else:
        for ref, expected in EXPECTED_NODES.items():
            problem = props_problem(expected, nodes[ref], drops_nulls)
            if problem:
                problems.append(f"node {ref}: {problem}")

    edges = {}
    for edge in obs["edges"]:
        for end in (edge["source"], edge["target"]):
            problem = id_problem(end, "edge endpoint")
            if problem:
                problems.append(problem)
        edges[(edge["rel"], key_of(edge["source"]), key_of(edge["target"]))] = edge["props"]
    if len(obs["edges"]) != len(EXPECTED_EDGES) or set(edges) != set(EXPECTED_EDGES):
        problems.append(f"edges {sorted(edges)} != {sorted(EXPECTED_EDGES)}")
    else:
        for key, expected in EXPECTED_EDGES.items():
            problem = props_problem(expected, edges[key], drops_nulls)
            if problem:
                problems.append(f"edge {key}: {problem}")

    if obs["embedding_count"] != len(EXPECTED_NODES):
        problems.append(f"{obs['embedding_count']} embeddings, expected {len(EXPECTED_NODES)}")
    if obs["missing_embeddings"]:
        problems.append(f"nodes without an embedding: {obs['missing_embeddings']}")
    if not obs["max_vec_diff"] <= TOLERANCE:
        problems.append(f"stored vectors differ from the reader's MockEncoder by "
                        f"{obs['max_vec_diff']:.3g} (> {TOLERANCE:g})")

    searches = {s["query"]: s for s in obs["searches"]}
    for query, expected in EXPECTED_SEARCHES:
        found = searches.get(query)
        top = key_of(found["top"]) if found and found["top"] else None
        if top != expected:
            problems.append(f"search {query[:24]!r} found {top}, expected {expected}")
    return problems


# -- the matrix -------------------------------------------------------------------

def tamper(source_dir, target_dir):
    """Copy a written graph and change one property (age 30 -> 31) in the file."""
    shutil.copytree(source_dir, target_dir)
    path = target_dir / "graph.ison"
    data = path.read_bytes()
    edited, count = re.subn(rb"(?<![\w.])30(?![\w.])", b"31", data, count=1)
    if count != 1:
        raise Fatal(f"could not find the value to tamper with in {path}")
    path.write_bytes(edited)


def table(title, ports, cells):
    cell = max(len(LABEL[p]) for p in ports) + 2
    label = max(len(LABEL[p]) for p in ports) + len(" writes") + 2
    lines = [title, "  " + "reader ->".ljust(label) + "".join(LABEL[p].ljust(cell) for p in ports)]
    for writer in ports:
        row = "".join(cells[(writer, reader)].ljust(cell) for reader in ports)
        lines.append("  " + (LABEL[writer] + " writes").ljust(label) + row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ports", default=",".join(ALL_PORTS),
                        help="comma-separated subset of: " + ", ".join(ALL_PORTS))
    parser.add_argument("--workdir", help="keep build output and graphs here "
                        "(default: a temporary directory, removed afterwards)")
    args = parser.parse_args()

    ports = [p.strip() for p in args.ports.split(",") if p.strip()]
    unknown = [p for p in ports if p not in BUILDERS]
    if unknown or not ports:
        parser.error(f"unknown port(s) {unknown}; choose from {ALL_PORTS}")

    owns_workdir = args.workdir is None
    wd = Path(args.workdir or tempfile.mkdtemp(prefix="cross_port_graph_")).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    if wd == REPO or REPO in wd.parents:
        parser.error("--workdir must be outside the repository")

    try:
        return execute(ports, wd)
    finally:
        if owns_workdir:
            shutil.rmtree(wd, ignore_errors=True)


def execute(ports, wd):
    start = time.time()
    print(f"building helpers for: {', '.join(LABEL[p] for p in ports)}  (work dir {wd})")
    helpers, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = {pool.submit(BUILDERS[p], wd): p for p in ports}
        for future in concurrent.futures.as_completed(futures):
            port = futures[future]
            try:
                helpers[port] = future.result()
            except Fatal as exc:
                failures.append(f"build {LABEL[port]}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 2

    written, tampered, problems = {}, {}, []
    for writer in ports:
        out = wd / "graphs" / writer
        code, output = run(helpers[writer] + ["write", out], check=False)
        if code != 0 or not (out / "graph.ison").exists() or \
                not (out / "graph.ison.embeddings.json").exists():
            problems.append(f"{LABEL[writer]} could not write the fixture graph (exit {code}): {output.strip()}")
            continue
        written[writer] = out
        tampered[writer] = wd / "tampered" / writer
        tamper(out, tampered[writer])

    roundtrip, detect, details = {}, {}, []
    for writer in ports:
        for reader in ports:
            pair = (writer, reader)
            skip = KNOWN_SKIPS.get(pair)
            if skip:
                roundtrip[pair] = detect[pair] = "SKIP"
                details.append(f"SKIP {LABEL[writer]} -> {LABEL[reader]}: {skip}")
                continue
            if writer not in written:
                roundtrip[pair] = detect[pair] = "FAIL"
                continue

            obs_path = wd / "obs" / f"{writer}_{reader}.json"
            obs_path.parent.mkdir(parents=True, exist_ok=True)
            code, output = run(helpers[reader] + ["read", written[writer], obs_path], check=False)
            if code != 0 or not obs_path.exists():
                roundtrip[pair] = "FAIL"
                details.append(f"FAIL {LABEL[writer]} -> {LABEL[reader]} (load): exit {code}: {output.strip()}")
            else:
                found = judge(writer, reader, json.loads(obs_path.read_text(encoding="utf-8")))
                roundtrip[pair] = "FAIL" if found else "PASS"
                details.extend(f"FAIL {LABEL[writer]} -> {LABEL[reader]}: {p}" for p in found)

            code, output = run(helpers[reader] + ["mismatch", tampered[writer]], check=False)
            detect[pair] = "PASS" if code == 0 else "FAIL"
            if code != 0:
                details.append(f"FAIL {LABEL[writer]} -> {LABEL[reader]} (tampered file): {output.strip()}")

    print()
    print(table("Round trip: reader loads the writer's graph + sidecar in ERROR mode and "
                "everything matches", ports, roundtrip))
    print()
    print(table("Tamper: reader must RAISE its mismatch error for an edited graph file", ports, detect))
    print()
    for line in problems + details:
        print(line)
    if problems or details:
        print()

    failed = problems or any(v == "FAIL" for v in list(roundtrip.values()) + list(detect.values()))
    cells = len(roundtrip) + len(detect)
    passed = sum(v == "PASS" for v in list(roundtrip.values()) + list(detect.values()))
    skipped = sum(v == "SKIP" for v in list(roundtrip.values()) + list(detect.values()))
    print(f"{'FAIL' if failed else 'PASS'}: {passed}/{cells} checks passed"
          f"{f', {skipped} skipped' if skipped else ''} in {time.time() - start:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
