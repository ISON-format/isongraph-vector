"""Names, identifiers and the only code that turns a name into a path.

Nothing a client sends is ever used as a path component directly. Tenant ids
come from the authenticated key (and were validated when the key was made),
graph names are matched against a whitelist regular expression, and the
directory is derived from the *validated* name with a fixed prefix. The
whitelist is lowercase-only on purpose: case-insensitive filesystems (Windows,
macOS) would otherwise map "Foo" and "foo" to one directory, i.e. let two
names alias each other.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}[a-z0-9]$")
GRAPH_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
# Node types, relationship types and property names. These become ISON block
# names / field names, which the format cannot quote, so keep them tame.
TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
PROP_RE = TYPE_RE
# String node ids. ASCII on purpose: ids appear in URLs, logs and ISON
# references. No whitespace, ':' or '|' (the library forbids them too).
ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@+=~-]{0,127}$")
CANON_INT_RE = re.compile(r"^-?(0|[1-9][0-9]{0,15})$")
GENERATION_RE = re.compile(r"^[0-9]{6,12}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")

MAX_INT_ID = 2 ** 53 - 1

# Files a snapshot generation may contain; the replication endpoints serve
# nothing else.
GENERATION_FILES = ("graph.isong", "graph.isong.embeddings.json", "MANIFEST.json")


def is_tenant_id(value: object) -> bool:
    return isinstance(value, str) and TENANT_RE.match(value) is not None


def is_graph_name(value: object) -> bool:
    return isinstance(value, str) and GRAPH_RE.match(value) is not None


def normalize_id(value: object):
    """Return the canonical node id (int or str) or raise ``ValueError``.

    A string that spells a canonical integer ("5", "-12") *is* that integer:
    the underlying library treats 5 and "5" as the same node (it refuses the
    second as a twin), so the API says so up front instead of erroring later.
    bool is not an id. Leading zeros keep a string a string ("007").
    """
    if isinstance(value, bool):
        raise ValueError("id must be an integer or a string")
    if isinstance(value, int):
        if abs(value) > MAX_INT_ID:
            raise ValueError("integer id out of range")
        return value
    if isinstance(value, str):
        if CANON_INT_RE.match(value):
            number = int(value)
            if abs(number) > MAX_INT_ID:
                raise ValueError("integer id out of range")
            return number
        if ID_RE.match(value):
            return value
        raise ValueError("id must match [A-Za-z0-9_][A-Za-z0-9_.@+=~-]{0,127}")
    raise ValueError("id must be an integer or a string")


def graph_dirname(name: str) -> str:
    if not is_graph_name(name):
        raise ValueError("invalid graph name")
    return "g-" + name


def contained(base: Path, candidate: Path) -> Path:
    """Return ``candidate`` resolved, or raise if it is not inside ``base``.

    Defence in depth: the whitelist above already rules out separators and
    dots, so this should never fire. It exists so that a future change which
    loosens a regex fails closed instead of opening a traversal.
    """
    base_r = Path(os.path.realpath(base))
    cand_r = Path(os.path.realpath(candidate))
    try:
        cand_r.relative_to(base_r)
    except ValueError:
        raise PermissionError("path escapes its base directory") from None
    return cand_r


def tenant_dir(data_dir: Union[str, Path], tenant_id: str) -> Path:
    if not is_tenant_id(tenant_id):
        raise ValueError("invalid tenant id")
    base = Path(data_dir) / "tenants"
    return contained(base, base / tenant_id)


def graph_dir(data_dir: Union[str, Path], tenant_id: str, name: str) -> Path:
    base = tenant_dir(data_dir, tenant_id) / "graphs"
    return contained(base, base / graph_dirname(name))
