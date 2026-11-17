"""Declarative node filters.

The library takes ``node_filter`` as a Python callback. Over HTTP that has to
be data, never code, so a filter is a list of conditions (ANDed) plus an
optional list of allowed node types:

    {"types": ["person"], "where": [{"field": "age", "op": "gte", "value": 30},
                                    {"field": "team", "op": "in", "value": ["a", "b"]}]}

Semantics: a missing field fails every condition except ``ne``, ``nin`` and
``exists`` with ``false``. Values of different types never compare equal and
never order (no exception, just "no match"); booleans are not numbers.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

from . import validation
from .errors import validation_error

MAX_IN = 64


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _comparable(a: Any, b: Any) -> bool:
    if _num(a) and _num(b):
        return True
    return isinstance(a, str) and isinstance(b, str)


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _num(a) and _num(b):
        return a == b
    return type(a) is type(b) and a == b


def _make_test(op: str, value: Any) -> Callable[[bool, Any], bool]:
    """Return ``test(present, actual)``."""
    if op == "exists":
        want = bool(value)
        return lambda present, actual: present == want
    if op == "eq":
        return lambda present, actual: present and _eq(actual, value)
    if op == "ne":
        return lambda present, actual: (not present) or not _eq(actual, value)
    if op in ("lt", "lte", "gt", "gte"):
        cmp = {"lt": lambda a, b: a < b, "lte": lambda a, b: a <= b,
               "gt": lambda a, b: a > b, "gte": lambda a, b: a >= b}[op]
        return lambda present, actual: present and _comparable(actual, value) and cmp(actual, value)
    if op == "in":
        return lambda present, actual: present and any(_eq(actual, v) for v in value)
    if op == "nin":
        return lambda present, actual: (not present) or not any(_eq(actual, v) for v in value)
    if op == "prefix":
        return lambda present, actual: present and isinstance(actual, str) and actual.startswith(value)
    if op == "contains":
        return lambda present, actual: present and isinstance(actual, str) and value in actual
    raise AssertionError(op)


def compile_filter(spec: Optional[dict], path: str = "$.filter"
                   ) -> Tuple[Optional[Callable[[Any], bool]], Optional[List[str]]]:
    """Return ``(node_predicate or None, node_types or None)`` for the library."""
    if not spec:
        return None, None
    types = spec.get("types")
    tests: List[Tuple[str, Callable[[bool, Any], bool]]] = []
    for i, cond in enumerate(spec.get("where", [])):
        where = f"{path}.where[{i}]"
        op = cond["op"]
        value = cond.get("value")
        if op == "exists":
            if "value" in cond and not isinstance(value, bool):
                raise validation_error(where + ".value", "must be a boolean")
            value = True if "value" not in cond else value
        elif op in ("in", "nin"):
            if not isinstance(value, list) or len(value) > MAX_IN:
                raise validation_error(where + ".value", f"must be an array of at most {MAX_IN} scalars")
            for j, item in enumerate(value):
                validation.scalar_property_value(item, f"{where}.value[{j}]")
        else:
            if "value" not in cond:
                raise validation_error(where, "missing 'value'")
            validation.scalar_property_value(value, where + ".value")
            if op in ("lt", "lte", "gt", "gte") and not (_num(value) or isinstance(value, str)):
                raise validation_error(where + ".value", "ordering needs a number or a string")
            if op in ("prefix", "contains") and not isinstance(value, str):
                raise validation_error(where + ".value", "must be a string")
        tests.append((cond["field"], _make_test(op, value)))

    if not tests:
        return None, (list(types) if types is not None else None)

    def predicate(node: Any) -> bool:
        props = node.properties
        for field_name, test in tests:
            actual = props.get(field_name)
            # null and absent are the same thing (see ops.py)
            if not test(actual is not None, actual):
                return False
        return True

    return predicate, (list(types) if types is not None else None)
