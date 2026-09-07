"""Read a store and payload written by another port and verify the vectors.

MockEncoder is byte-identical across all six ports, so this compares floats
exactly. A tolerance would hide the drift this exists to catch.

The payload is checked by importing it into a store rather than by comparing
the JSON text, because the text legitimately differs between ports: every
vector is float32, but Rust serialises the shortest form that round-trips as
f32 (0.33590567) while Python widens to float64 first (0.33590567111968994).
Those are the same number. Importing narrows back to float32, which is the
operation a consumer actually performs, and after that the comparison is exact.

Usage: python py_read.py <db-path> <json-path>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401  - binds `isongraph_vector` to the port directory

from isongraph_vector import EmbeddingStore, MockEncoder  # noqa: E402

failures = 0


def check(label: str, actual, text: str, encoder: MockEncoder) -> None:
    global failures
    expected = encoder.encode(text)
    same = list(actual) == list(expected)
    print(f"  {'ok  ' if same else 'FAIL'}  {label}")
    if not same:
        failures += 1
        print(f"        got      {list(actual)[:3]}")
        print(f"        expected {list(expected)[:3]}")


def main() -> int:
    global failures
    if len(sys.argv) != 3:
        print("usage: python py_read.py <db-path> <json-path>", file=sys.stderr)
        return 2
    db_path, json_path = sys.argv[1], sys.argv[2]
    encoder = MockEncoder(384)

    store = EmbeddingStore(db_path=db_path, encoder=encoder)
    print(f"python opened {db_path}, {store.count()} embeddings")
    for record in store.export_embeddings()["embeddings"]:
        check(f"db     {record['type']}:{record['id']}", record["vector"],
              record["text"], encoder)
    store.close()

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    header_ok = (payload.get("format") == "ison-embeddings"
                 and payload.get("version") == 1)
    print(f"  {'ok  ' if header_ok else 'FAIL'}  payload header: "
          f"{payload.get('format')} v{payload.get('version')}")
    if not header_ok:
        failures += 1

    imported = EmbeddingStore(db_path=":memory:", encoder=encoder)
    count = imported.import_embeddings(payload)
    if count != len(payload["embeddings"]):
        print(f"  FAIL  imported {count} of {len(payload['embeddings'])} rows")
        failures += 1
    for record in imported.export_embeddings()["embeddings"]:
        check(f"json   {record['type']}:{record['id']}", record["vector"],
              record["text"], encoder)
    imported.close()

    print(f"{'FAILED' if failures else 'ok'} - {failures} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
