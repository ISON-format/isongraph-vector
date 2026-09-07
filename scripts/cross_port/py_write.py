"""Write a SQLite store and a portable JSON payload from the Python port.

See node_write.mjs for why these scripts exist: the six ports claim to share
one SQLite schema and one payload format, and CI should prove it rather than
take it on trust.

Usage: python py_write.py <db-path> <json-path>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401  - binds `isongraph_vector` to the port directory

from isongraph_vector import EmbeddingStore, MockEncoder  # noqa: E402

# Matches FIXTURES in node_write.mjs: a string id, an integer id, and
# non-ASCII text, which are the three things a shared format gets wrong.
FIXTURES = [
    (("person", "1"), "Alice is a software engineer"),
    (("person", 2), "Bob analyses ML models"),
    (("doc", "r\u00e9sum\u00e9"), "Carol plans roadmaps for \u00fcber-teams"),
]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python py_write.py <db-path> <json-path>", file=sys.stderr)
        return 2
    db_path, json_path = sys.argv[1], sys.argv[2]

    store = EmbeddingStore(db_path=db_path, encoder=MockEncoder(384))
    store.add_batch(FIXTURES)
    Path(json_path).write_text(
        json.dumps(store.export_embeddings(), indent=2), encoding="utf-8"
    )
    store.close()

    print(f"python wrote {len(FIXTURES)} embeddings to {db_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
