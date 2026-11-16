"""Explicit request validation: a small JSON-Schema-subset interpreter.

Every endpoint that takes a body declares a schema (see ``schemas.py``); the
body is validated against it *before* any handler code runs. Objects reject
unknown keys by default, so a typo or a smuggled field is an error instead of
being silently ignored. Supported keywords: ``type`` (object, array, string,
integer, number, boolean, null, or a list of them), ``properties``,
``required``, ``additionalProperties`` (default False), ``items``,
``minItems``/``maxItems``, ``minLength``/``maxLength``, ``pattern``,
``enum``, ``minimum``/``maximum``, and ``kind`` for the domain types
(``id``, ``typename``, ``propname``, ``props``, ``vector``, ``text``).
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List

from . import names
from .errors import validation_error

MAX_DEPTH = 8
MAX_VECTOR = 8192
MAX_STR = 1_000_000

# Unicode whitespace/format characters that the ISON text format strips or
# folds on a save/load round trip (found by fuzzing the round trip). A string
# containing them would not come back identical, so they are refused up front.
_ROUNDTRIP_UNSAFE = set("\u0085          "
                        "       　﻿")


def _check_text(value: str, path: str) -> None:
    for ch in value:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            raise validation_error(path, "lone surrogate in string")
        if ch in _ROUNDTRIP_UNSAFE:
            raise validation_error(path, "string contains an unsupported whitespace character")


def _type_ok(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "null":
        return value is None
    raise AssertionError(f"unknown schema type {name}")


def _kind_id(value, path, schema):
    try:
        return names.normalize_id(value)
    except ValueError as exc:
        raise validation_error(path, str(exc)) from None


def _kind_typename(value, path, schema):
    if not isinstance(value, str) or not names.TYPE_RE.match(value):
        raise validation_error(path, "must match [A-Za-z_][A-Za-z0-9_]{0,63}")
    return value


def _kind_graphname(value, path, schema):
    if not isinstance(value, str) or not names.GRAPH_RE.match(value):
        raise validation_error(path, "must match [a-z0-9][a-z0-9_-]{0,62}")
    return value


def scalar_property_value(value: Any, path: str, max_chars: int = 20_000) -> Any:
    """A property value: null, bool, int, finite float or string - nothing nested.

    Floats must round-trip through the ISON text format, which writes a float
    with an exponent ("1e+20") as a string; those are refused rather than
    silently changing type on the next restart.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > names.MAX_INT_ID:
            raise validation_error(path, "integer out of range (|n| <= 2^53-1)")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise validation_error(path, "number must be finite")
        if "e" in repr(value).lower():
            raise validation_error(path, "number too large or small to store exactly; send it as a string")
        return value
    if isinstance(value, str):
        if len(value) > max_chars:
            raise validation_error(path, f"string longer than {max_chars} characters")
        _check_text(value, path)
        return value
    raise validation_error(path, "property values must be null, boolean, number or string")


def _kind_props(value, path, schema):
    if not isinstance(value, dict):
        raise validation_error(path, "must be an object")
    limit = schema.get("maxProperties", 64)
    if len(value) > limit:
        raise validation_error(path, f"at most {limit} properties")
    out = {}
    for key, item in value.items():
        if not isinstance(key, str) or not names.PROP_RE.match(key):
            raise validation_error(path, f"invalid property name {key[:40]!r}")
        if key.startswith("_"):
            raise validation_error(path, "property names may not start with '_' (reserved)")
        if key in schema.get("reserved", ()):
            raise validation_error(path, f"property name {key!r} is reserved (it is a column of the ISON format)")
        out[key] = scalar_property_value(item, f"{path}.{key}", schema.get("maxChars", 20_000))
    return out


def _kind_vector(value, path, schema):
    if not isinstance(value, list) or not value:
        raise validation_error(path, "must be a non-empty array of numbers")
    if len(value) > MAX_VECTOR:
        raise validation_error(path, f"at most {MAX_VECTOR} components")
    out: List[float] = []
    for i, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise validation_error(f"{path}[{i}]", "must be a number")
        f = float(item)
        if not math.isfinite(f):
            raise validation_error(f"{path}[{i}]", "must be finite")
        out.append(f)
    return out


def _kind_text(value, path, schema):
    if not isinstance(value, str):
        raise validation_error(path, "must be a string")
    limit = schema.get("maxLength", 20_000)
    if len(value) > limit:
        raise validation_error(path, f"longer than {limit} characters")
    _check_text(value, path)
    return value


KINDS: Dict[str, Callable[[Any, str, dict], Any]] = {
    "id": _kind_id,
    "typename": _kind_typename,
    "propname": _kind_typename,
    "graphname": _kind_graphname,
    "props": _kind_props,
    "vector": _kind_vector,
    "text": _kind_text,
}


def validate(value: Any, schema: Dict[str, Any], path: str = "$", depth: int = 0) -> Any:
    """Validate ``value`` against ``schema``; return the normalised value.

    Raises ``BadRequest`` (code ``validation_failed``) naming the first
    offending path.
    """
    if depth > MAX_DEPTH:
        raise validation_error(path, "nested too deeply")

    kind = schema.get("kind")
    if kind is not None:
        if value is None and schema.get("nullable"):
            return None
        return KINDS[kind](value, path, schema)

    types = schema.get("type")
    if types is not None:
        tlist = types if isinstance(types, list) else [types]
        if not any(_type_ok(value, t) for t in tlist):
            raise validation_error(path, "must be " + " or ".join(tlist))

    if "enum" in schema and value not in schema["enum"]:
        raise validation_error(path, "must be one of " + ", ".join(repr(e) for e in schema["enum"]))

    if isinstance(value, str):
        if len(value) > schema.get("maxLength", MAX_STR):
            raise validation_error(path, f"longer than {schema.get('maxLength', MAX_STR)} characters")
        if len(value) < schema.get("minLength", 0):
            raise validation_error(path, f"shorter than {schema['minLength']} characters")
        if "pattern" in schema and not re.match(schema["pattern"], value):
            raise validation_error(path, "does not match the required pattern")
        _check_text(value, path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise validation_error(path, "must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise validation_error(path, f"must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise validation_error(path, f"must be <= {schema['maximum']}")

    if isinstance(value, list):
        if len(value) > schema.get("maxItems", 100_000):
            raise validation_error(path, f"at most {schema.get('maxItems', 100_000)} items")
        if len(value) < schema.get("minItems", 0):
            raise validation_error(path, f"at least {schema['minItems']} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            return [validate(item, item_schema, f"{path}[{i}]", depth + 1)
                    for i, item in enumerate(value)]
        return value

    if isinstance(value, dict) and (
            "properties" in schema or "required" in schema or "additionalProperties" in schema
            or types == "object"):
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", False)
        out = {}
        for key in schema.get("required", ()):
            if key not in value:
                raise validation_error(path, f"missing required field '{key}'")
        for key, item in value.items():
            if key in props:
                out[key] = validate(item, props[key], f"{path}.{key}", depth + 1)
            elif extra is False:
                raise validation_error(path, f"unknown field {str(key)[:40]!r}")
            elif isinstance(extra, dict):
                out[key] = validate(item, extra, f"{path}.{key}", depth + 1)
            else:
                out[key] = item
        for key, sub in props.items():
            if key not in out and "default" in sub:
                out[key] = sub["default"]
        return out

    return value
