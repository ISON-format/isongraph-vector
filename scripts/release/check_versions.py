"""Check that every port declares the same version, and that it matches a tag.

Nine files carry the version, in six syntaxes. A release where one of them
lags is not hypothetical - it is the default outcome of bumping by hand - and
the registries will happily accept a 1.0.1 crate alongside a 1.0.0 wheel. The
release workflow runs this before it builds anything; run it locally after a
bump for the same reason.

Usage:
    python scripts/release/check_versions.py            # print and check agreement
    python scripts/release/check_versions.py 1.0.0      # also require this version
    python scripts/release/check_versions.py v1.0.0     # a leading v is fine
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (label, path, extractor)
SOURCES = [
    ("pyproject.toml", "pyproject.toml",
     lambda s: re.search(r'^version\s*=\s*"([^"]+)"', s, re.M)),
    ("Python __version__", "isongraph_vector-py/__init__.py",
     lambda s: re.search(r'^__version__\s*=\s*"([^"]+)"', s, re.M)),
    ("Cargo.toml", "isongraph-vector-rs/Cargo.toml",
     lambda s: re.search(r'^\[package\][^\[]*?^version\s*=\s*"([^"]+)"', s, re.M | re.S)),
    ("TypeScript package.json", "isongraph-vector-ts/package.json",
     lambda s: re.match(r"(.*)", json.loads(s)["version"])),
    ("JavaScript package.json", "isongraph-vector-js/package.json",
     lambda s: re.match(r"(.*)", json.loads(s)["version"])),
    ("IsonGraph.Vector.csproj",
     "isongraph-vector-csharp/src/IsonGraph.Vector/IsonGraph.Vector.csproj",
     lambda s: re.search(r"<Version>([^<]+)</Version>", s)),
    ("IsonGraph.Vector.Sqlite.csproj",
     "isongraph-vector-csharp/src/IsonGraph.Vector.Sqlite/IsonGraph.Vector.Sqlite.csproj",
     lambda s: re.search(r"<Version>([^<]+)</Version>", s)),
    ("CMakeLists.txt", "isongraph-vector-cpp/CMakeLists.txt",
     lambda s: re.search(r"^project\([^)]*?VERSION\s+([0-9][^\s)]*)", s, re.M | re.S)),
    ("C++ VERSION constant", "isongraph-vector-cpp/include/isongraph_vector.hpp",
     lambda s: re.search(r'constexpr const char\* VERSION\s*=\s*"([^"]+)"', s)),
]


def main() -> int:
    expected = sys.argv[1].lstrip("v") if len(sys.argv) > 1 else None

    found = {}
    for label, rel, extract in SOURCES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        m = extract(text)
        found[label] = m.group(1) if m else None

    width = max(len(k) for k in found)
    for label, version in found.items():
        print(f"  {label:<{width}}  {version or 'NOT FOUND'}")

    problems = []
    versions = {v for v in found.values() if v}
    missing = [k for k, v in found.items() if not v]
    if missing:
        problems.append("could not read a version from: " + ", ".join(missing))
    if len(versions) > 1:
        problems.append("versions disagree: " + ", ".join(sorted(versions)))

    if expected is not None:
        if versions != {expected}:
            problems.append(f"expected {expected} everywhere")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        if not re.search(rf"^## \[{re.escape(expected)}\]", changelog, re.M):
            problems.append(f"CHANGELOG.md has no '## [{expected}]' section")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    only = next(iter(versions))
    print(f"ok: {len(found)} sources agree on {only}"
          + (f", CHANGELOG has a [{only}] section" if expected else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
