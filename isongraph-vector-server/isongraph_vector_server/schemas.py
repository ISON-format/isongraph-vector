"""Request body schemas (see ``validation.py`` for the dialect).

Numeric caps here are hard ceilings; the per-tenant quotas (``max_top_k``,
``max_hops``, ``max_batch_items`` ...) are enforced by the handlers on top.
"""

from __future__ import annotations

from typing import Any, Dict

NAME = {"kind": "graphname"}
TYPENAME = {"kind": "typename"}
PROPNAME = {"kind": "propname"}
ID = {"kind": "id"}
PROPS = {"kind": "props", "reserved": ["id"]}
EDGE_PROPS = {"kind": "props", "reserved": ["source", "target"]}
VECTOR = {"kind": "vector"}
EMBED_TEXT = {"kind": "text", "maxLength": 20_000}
QUERY_TEXT = {"kind": "text", "maxLength": 8_000}

NODE_REF: Dict[str, Any] = {
    "type": "object",
    "required": ["type", "id"],
    "properties": {"type": TYPENAME, "id": ID},
}

FILTER_OPS = ["eq", "ne", "lt", "lte", "gt", "gte", "in", "nin", "exists", "prefix", "contains"]

CONDITION: Dict[str, Any] = {
    "type": "object",
    "required": ["field", "op"],
    "properties": {
        "field": PROPNAME,
        "op": {"type": "string", "enum": FILTER_OPS},
        # any scalar, or (for in/nin) a short list of scalars; checked in filters.py
        "value": {},
    },
}

FILTER: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "types": {"type": "array", "items": TYPENAME, "maxItems": 16},
        "where": {"type": "array", "items": CONDITION, "maxItems": 16},
    },
}

DIRECTION = {"type": "string", "enum": ["out", "in", "both"], "default": "out"}

CREATE_GRAPH = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": NAME,
        "directed": {"type": "boolean", "default": True},
        "embedding": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["server", "client"], "default": "server"},
                "dimension": {"type": "integer", "minimum": 1, "maximum": 8192},
            },
        },
        "embed_fields": {"type": "array", "items": PROPNAME, "maxItems": 8, "minItems": 1},
        "ann": {"type": ["string", "null"], "enum": ["ivf", None]},
        "ann_nprobe": {"type": "integer", "minimum": 1, "maximum": 100_000},
    },
}

NODE_BODY = {
    "type": "object",
    "properties": {
        "properties": PROPS,
        "embed_text": EMBED_TEXT,
        "vector": VECTOR,
    },
}

NODE_UPSERT = {
    "type": "object",
    "required": ["type", "id"],
    "properties": {
        "type": TYPENAME,
        "id": ID,
        "properties": PROPS,
        "embed_text": EMBED_TEXT,
        "vector": VECTOR,
    },
}

UPSERT_NODES = {
    "type": "object",
    "required": ["nodes"],
    "properties": {"nodes": {"type": "array", "items": NODE_UPSERT, "minItems": 1, "maxItems": 10_000}},
}

PATCH_NODE = {
    "type": "object",
    "properties": {
        "properties": PROPS,
        "remove": {"type": "array", "items": PROPNAME, "maxItems": 64},
        "embed_text": EMBED_TEXT,
        "vector": VECTOR,
    },
}

DELETE_NODES = {
    "type": "object",
    "required": ["nodes"],
    "properties": {"nodes": {"type": "array", "items": NODE_REF, "minItems": 1, "maxItems": 10_000}},
}

EDGE = {
    "type": "object",
    "required": ["rel", "source", "target"],
    "properties": {
        "rel": TYPENAME,
        "source": NODE_REF,
        "target": NODE_REF,
        "properties": EDGE_PROPS,
    },
}

EDGE_REF = {
    "type": "object",
    "required": ["rel", "source", "target"],
    "properties": {"rel": TYPENAME, "source": NODE_REF, "target": NODE_REF},
}

UPSERT_EDGES = {
    "type": "object",
    "required": ["edges"],
    "properties": {"edges": {"type": "array", "items": EDGE, "minItems": 1, "maxItems": 10_000}},
}

DELETE_EDGES = {
    "type": "object",
    "required": ["edges"],
    "properties": {"edges": {"type": "array", "items": EDGE_REF, "minItems": 1, "maxItems": 10_000}},
}

_QUERY = {
    "query": QUERY_TEXT,
    "vector": VECTOR,
}

SEARCH = {
    "type": "object",
    "properties": {
        **_QUERY,
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 10},
        "node_type": TYPENAME,
        "threshold": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": 0.0},
        "filter": FILTER,
        "exact": {"type": "boolean", "default": False},
        "include_properties": {"type": "boolean", "default": True},
        "include_text": {"type": "boolean", "default": False},
    },
}

_TRAVERSAL = {
    **_QUERY,
    "rel_type": TYPENAME,
    "max_hops": {"type": "integer", "minimum": 0, "maximum": 50},
    "top_k_seeds": {"type": "integer", "minimum": 1, "maximum": 10_000},
    "direction": DIRECTION,
    "decay": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "threshold": {"type": "number", "minimum": -1.0, "maximum": 1.0},
    "weight_field": PROPNAME,
    "filter": FILTER,
    "strict_filter": {"type": "boolean", "default": False},
    "include_properties": {"type": "boolean", "default": True},
}

MULTI_HOP = {
    "type": "object",
    "properties": {
        **_TRAVERSAL,
        "top_k_results": {"type": "integer", "minimum": 1, "maximum": 10_000},
        "blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

PATH = {
    "type": "object",
    "required": ["target"],
    "properties": {**_TRAVERSAL, "target": NODE_REF},
}

SUBGRAPH = {
    "type": "object",
    "properties": {
        **_TRAVERSAL,
        "top_k_results": {"type": "integer", "minimum": 1, "maximum": 10_000},
        "blend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "connect": {"type": "boolean", "default": False},
        "connect_max_hops": {"type": "integer", "minimum": 1, "maximum": 3},
    },
}

SIMILAR = {
    "type": "object",
    "required": ["node"],
    "properties": {
        "node": NODE_REF,
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 10},
        "node_type": TYPENAME,
        "threshold": {"type": "number", "minimum": -1.0, "maximum": 1.0, "default": 0.0},
        "include_properties": {"type": "boolean", "default": True},
    },
}

REEMBED = {
    "type": "object",
    "properties": {"include_unknown": {"type": "boolean", "default": False}},
}
