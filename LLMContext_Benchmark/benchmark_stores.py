#!/usr/bin/env python3
"""
ISONGraph Store Benchmark
=========================

Compares ISONGraph against other graph stores for:
- CRUD operations (nodes, edges)
- Traversal (1-hop, multi-hop)
- Path finding (shortest path, all paths)
- Graph analysis (cycles, connectivity)
- Serialization (save/load)

Stores Tested:
- ISONGraph (in-memory, Python)
- NetworkX (in-memory, Python)
- igraph (in-memory, Python/C)
- Neo4j (database, via neo4j-driver) [optional]

Usage:
    python benchmark_stores.py                  # Quick benchmark (1K nodes)
    python benchmark_stores.py --full           # Full benchmark (10K nodes)
    python benchmark_stores.py --scale 50000    # Custom scale
    python benchmark_stores.py --stores isongraph networkx  # Specific stores
"""

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_NODE_COUNT = 1000
FULL_NODE_COUNT = 10000
EDGE_MULTIPLIER = 5  # edges = nodes * multiplier
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
TRAVERSAL_ITERATIONS = 50
PATH_ITERATIONS = 20

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark operation."""
    operation: str
    store: str
    iterations: int
    total_time_ms: float
    min_time_us: float
    max_time_us: float
    avg_time_us: float
    median_time_us: float
    std_dev_us: float
    ops_per_sec: float
    memory_mb: float = 0.0

@dataclass
class StoreMetrics:
    """Aggregated metrics for a store."""
    store: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    # New metrics
    memory_after_load_mb: float = 0.0
    serialized_size_bytes: int = 0
    token_count: int = 0
    # Context window metrics
    context_4k_pct: float = 0.0    # % of 4K context used
    context_8k_pct: float = 0.0    # % of 8K context used
    context_32k_pct: float = 0.0   # % of 32K context used
    context_128k_pct: float = 0.0  # % of 128K context used
    nodes_per_1k_tokens: float = 0.0  # How many nodes fit in 1K tokens


# =============================================================================
# Abstract Store Interface
# =============================================================================

class GraphStore(ABC):
    """Abstract interface for graph stores."""

    name: str = "abstract"

    @abstractmethod
    def setup(self) -> None:
        """Initialize the store."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
        pass

    # CRUD Operations
    @abstractmethod
    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        pass

    @abstractmethod
    def get_node(self, node_type: str, node_id: int) -> Any:
        pass

    @abstractmethod
    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        pass

    @abstractmethod
    def delete_node(self, node_type: str, node_id: int) -> None:
        pass

    @abstractmethod
    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        pass

    @abstractmethod
    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        pass

    @abstractmethod
    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        pass

    # Traversal
    @abstractmethod
    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        pass

    @abstractmethod
    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        pass

    # Path Finding
    @abstractmethod
    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        pass

    @abstractmethod
    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        pass

    # Analysis
    @abstractmethod
    def has_cycle(self) -> bool:
        pass

    @abstractmethod
    def node_count(self) -> int:
        pass

    @abstractmethod
    def edge_count(self) -> int:
        pass

    # Serialization
    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


# =============================================================================
# ISONGraph Store Implementation
# =============================================================================

class ISONGraphStore(GraphStore):
    """ISONGraph store implementation."""

    name = "ISONGraph"

    def __init__(self):
        self.graph = None

    def setup(self) -> None:
        from ison_graph import ISONGraph
        self.graph = ISONGraph(name="benchmark")

    def teardown(self) -> None:
        self.graph = None

    def clear(self) -> None:
        from ison_graph import ISONGraph
        self.graph = ISONGraph(name="benchmark")

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        self.graph.add_node(node_type, node_id, **properties)

    def get_node(self, node_type: str, node_id: int) -> Any:
        return self.graph.get_node(node_type, node_id)

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        self.graph.update_node(node_type, node_id, **properties)

    def delete_node(self, node_type: str, node_id: int) -> None:
        self.graph.remove_node(node_type, node_id)

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        self.graph.add_edge(rel_type, source, target, **properties)

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        return self.graph.get_edge(rel_type, source, target)

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        self.graph.remove_edge(rel_type, source, target)

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        from ison_graph import Direction
        return self.graph.neighbors((node_type, node_id), rel_type, Direction.OUT)

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        from ison_graph import Direction
        return self.graph.multi_hop((node_type, node_id), rel_type, hops, Direction.OUT)

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        path = self.graph.shortest_path(source, target)
        return path.nodes if path else None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        return self.graph.path_exists(source, target)

    def has_cycle(self) -> bool:
        return self.graph.has_cycle()

    def node_count(self) -> int:
        return self.graph.node_count()

    def edge_count(self) -> int:
        return self.graph.edge_count()

    def save(self, path: str) -> None:
        self.graph.save(path)

    def load(self, path: str) -> None:
        from ison_graph import ISONGraph
        self.graph = ISONGraph.load(path)


# =============================================================================
# NetworkX Store Implementation
# =============================================================================

class NetworkXStore(GraphStore):
    """NetworkX store implementation."""

    name = "NetworkX"

    def __init__(self):
        self.graph = None
        self._node_types = {}  # Track node types separately

    def setup(self) -> None:
        import networkx as nx
        self.graph = nx.DiGraph()
        self._node_types = {}

    def teardown(self) -> None:
        self.graph = None
        self._node_types = {}

    def clear(self) -> None:
        self.graph.clear()
        self._node_types = {}

    def _node_key(self, node_type: str, node_id: int) -> str:
        return f"{node_type}:{node_id}"

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        key = self._node_key(node_type, node_id)
        self.graph.add_node(key, type=node_type, id=node_id, **properties)
        self._node_types[key] = node_type

    def get_node(self, node_type: str, node_id: int) -> Any:
        key = self._node_key(node_type, node_id)
        return self.graph.nodes.get(key)

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        key = self._node_key(node_type, node_id)
        self.graph.nodes[key].update(properties)

    def delete_node(self, node_type: str, node_id: int) -> None:
        key = self._node_key(node_type, node_id)
        self.graph.remove_node(key)
        del self._node_types[key]

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        src_key = self._node_key(source[0], source[1])
        tgt_key = self._node_key(target[0], target[1])
        self.graph.add_edge(src_key, tgt_key, rel_type=rel_type, **properties)

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        src_key = self._node_key(source[0], source[1])
        tgt_key = self._node_key(target[0], target[1])
        return self.graph.edges.get((src_key, tgt_key))

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        src_key = self._node_key(source[0], source[1])
        tgt_key = self._node_key(target[0], target[1])
        self.graph.remove_edge(src_key, tgt_key)

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        import networkx as nx
        key = self._node_key(node_type, node_id)
        neighbors = list(self.graph.successors(key))
        if rel_type:
            neighbors = [n for n in neighbors
                        if self.graph.edges[key, n].get('rel_type') == rel_type]
        return neighbors

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        import networkx as nx
        key = self._node_key(node_type, node_id)
        # BFS to specified depth
        visited = set()
        current = {key}
        for _ in range(hops):
            next_level = set()
            for node in current:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        if rel_type is None or self.graph.edges[node, neighbor].get('rel_type') == rel_type:
                            next_level.add(neighbor)
            visited.update(current)
            current = next_level
        return list(current)

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        import networkx as nx
        src_key = self._node_key(source[0], source[1])
        tgt_key = self._node_key(target[0], target[1])
        try:
            return nx.shortest_path(self.graph, src_key, tgt_key)
        except nx.NetworkXNoPath:
            return None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        import networkx as nx
        src_key = self._node_key(source[0], source[1])
        tgt_key = self._node_key(target[0], target[1])
        return nx.has_path(self.graph, src_key, tgt_key)

    def has_cycle(self) -> bool:
        import networkx as nx
        try:
            nx.find_cycle(self.graph)
            return True
        except nx.NetworkXNoCycle:
            return False

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def save(self, path: str) -> None:
        import networkx as nx
        nx.write_gml(self.graph, path)

    def load(self, path: str) -> None:
        import networkx as nx
        self.graph = nx.read_gml(path)


# =============================================================================
# igraph Store Implementation
# =============================================================================

class IGraphStore(GraphStore):
    """igraph store implementation."""

    name = "igraph"

    def __init__(self):
        self.graph = None
        self._node_map = {}  # (type, id) -> vertex index
        self._reverse_map = {}  # vertex index -> (type, id)

    def setup(self) -> None:
        import igraph as ig
        self.graph = ig.Graph(directed=True)
        self._node_map = {}
        self._reverse_map = {}

    def teardown(self) -> None:
        self.graph = None
        self._node_map = {}
        self._reverse_map = {}

    def clear(self) -> None:
        import igraph as ig
        self.graph = ig.Graph(directed=True)
        self._node_map = {}
        self._reverse_map = {}

    def _get_or_create_vertex(self, node_type: str, node_id: int) -> int:
        key = (node_type, node_id)
        if key not in self._node_map:
            idx = self.graph.vcount()
            self.graph.add_vertex(name=f"{node_type}:{node_id}")
            self._node_map[key] = idx
            self._reverse_map[idx] = key
        return self._node_map[key]

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        idx = self._get_or_create_vertex(node_type, node_id)
        self.graph.vs[idx]["type"] = node_type
        self.graph.vs[idx]["node_id"] = node_id
        for k, v in properties.items():
            self.graph.vs[idx][k] = v

    def get_node(self, node_type: str, node_id: int) -> Any:
        key = (node_type, node_id)
        if key in self._node_map:
            return self.graph.vs[self._node_map[key]].attributes()
        return None

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        key = (node_type, node_id)
        if key in self._node_map:
            idx = self._node_map[key]
            for k, v in properties.items():
                self.graph.vs[idx][k] = v

    def delete_node(self, node_type: str, node_id: int) -> None:
        key = (node_type, node_id)
        if key in self._node_map:
            idx = self._node_map[key]
            self.graph.delete_vertices(idx)
            # Rebuild maps after deletion
            self._rebuild_maps()

    def _rebuild_maps(self) -> None:
        self._node_map = {}
        self._reverse_map = {}
        for idx, v in enumerate(self.graph.vs):
            if "type" in v.attributes() and "node_id" in v.attributes():
                key = (v["type"], v["node_id"])
                self._node_map[key] = idx
                self._reverse_map[idx] = key

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        src_idx = self._get_or_create_vertex(source[0], source[1])
        tgt_idx = self._get_or_create_vertex(target[0], target[1])
        self.graph.add_edge(src_idx, tgt_idx, rel_type=rel_type, **properties)

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        src_idx = self._node_map.get(source)
        tgt_idx = self._node_map.get(target)
        if src_idx is not None and tgt_idx is not None:
            eid = self.graph.get_eid(src_idx, tgt_idx, error=False)
            if eid >= 0:
                return self.graph.es[eid].attributes()
        return None

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        src_idx = self._node_map.get(source)
        tgt_idx = self._node_map.get(target)
        if src_idx is not None and tgt_idx is not None:
            eid = self.graph.get_eid(src_idx, tgt_idx, error=False)
            if eid >= 0:
                self.graph.delete_edges(eid)

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        key = (node_type, node_id)
        if key not in self._node_map:
            return []
        idx = self._node_map[key]
        neighbor_indices = self.graph.neighbors(idx, mode="out")
        return [self._reverse_map.get(n) for n in neighbor_indices if n in self._reverse_map]

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        key = (node_type, node_id)
        if key not in self._node_map:
            return []
        idx = self._node_map[key]
        # BFS to specified depth
        visited = set()
        current = {idx}
        for _ in range(hops):
            next_level = set()
            for node in current:
                for neighbor in self.graph.neighbors(node, mode="out"):
                    if neighbor not in visited:
                        next_level.add(neighbor)
            visited.update(current)
            current = next_level
        return [self._reverse_map.get(n) for n in current if n in self._reverse_map]

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        src_idx = self._node_map.get(source)
        tgt_idx = self._node_map.get(target)
        if src_idx is None or tgt_idx is None:
            return None
        paths = self.graph.get_shortest_paths(src_idx, to=tgt_idx, mode="out")
        if paths and paths[0]:
            return [self._reverse_map.get(n) for n in paths[0]]
        return None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        path = self.shortest_path(source, target)
        return path is not None and len(path) > 0

    def has_cycle(self) -> bool:
        return not self.graph.is_dag()

    def node_count(self) -> int:
        return self.graph.vcount()

    def edge_count(self) -> int:
        return self.graph.ecount()

    def save(self, path: str) -> None:
        self.graph.write_graphml(path)

    def load(self, path: str) -> None:
        import igraph as ig
        self.graph = ig.Graph.Read_GraphML(path)
        self._rebuild_maps()


# =============================================================================
# RDFLib Store Implementation
# =============================================================================

class RDFLibStore(GraphStore):
    """RDFLib RDF graph store implementation."""

    name = "RDFLib"

    def __init__(self):
        self.graph = None
        self._ns = None

    def setup(self) -> None:
        from rdflib import Graph, Namespace
        self.graph = Graph()
        self._ns = Namespace("http://example.org/")

    def teardown(self) -> None:
        self.graph = None

    def clear(self) -> None:
        from rdflib import Graph, Namespace
        self.graph = Graph()
        self._ns = Namespace("http://example.org/")

    def _node_uri(self, node_type: str, node_id: int):
        from rdflib import URIRef
        return URIRef(f"http://example.org/{node_type}/{node_id}")

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        from rdflib import Literal, RDF
        uri = self._node_uri(node_type, node_id)
        self.graph.add((uri, RDF.type, self._ns[node_type]))
        for k, v in properties.items():
            self.graph.add((uri, self._ns[k], Literal(v)))

    def get_node(self, node_type: str, node_id: int) -> Any:
        uri = self._node_uri(node_type, node_id)
        props = {}
        for p, o in self.graph.predicate_objects(uri):
            props[str(p).split('/')[-1]] = str(o)
        return props if props else None

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        from rdflib import Literal
        uri = self._node_uri(node_type, node_id)
        for k, v in properties.items():
            # Remove old value
            for old_val in list(self.graph.objects(uri, self._ns[k])):
                self.graph.remove((uri, self._ns[k], old_val))
            # Add new value
            self.graph.add((uri, self._ns[k], Literal(v)))

    def delete_node(self, node_type: str, node_id: int) -> None:
        uri = self._node_uri(node_type, node_id)
        self.graph.remove((uri, None, None))
        self.graph.remove((None, None, uri))

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        src_uri = self._node_uri(source[0], source[1])
        tgt_uri = self._node_uri(target[0], target[1])
        self.graph.add((src_uri, self._ns[rel_type], tgt_uri))

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        src_uri = self._node_uri(source[0], source[1])
        tgt_uri = self._node_uri(target[0], target[1])
        if (src_uri, self._ns[rel_type], tgt_uri) in self.graph:
            return {"rel_type": rel_type}
        return None

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        src_uri = self._node_uri(source[0], source[1])
        tgt_uri = self._node_uri(target[0], target[1])
        self.graph.remove((src_uri, self._ns[rel_type], tgt_uri))

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        uri = self._node_uri(node_type, node_id)
        pred = self._ns[rel_type] if rel_type else None
        neighbors = []
        for o in self.graph.objects(uri, pred):
            if str(o).startswith("http://example.org/"):
                parts = str(o).split('/')
                neighbors.append((parts[-2], int(parts[-1])))
        return neighbors

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        visited = set()
        current = {(node_type, node_id)}
        for _ in range(hops):
            next_level = set()
            for nt, nid in current:
                for neighbor in self.neighbors(nt, nid, rel_type):
                    if neighbor not in visited:
                        next_level.add(neighbor)
            visited.update(current)
            current = next_level
        return list(current)

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        from collections import deque
        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in self.neighbors(node[0], node[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        return self.shortest_path(source, target) is not None

    def has_cycle(self) -> bool:
        # Simple DFS-based cycle detection
        visited = set()
        rec_stack = set()

        def get_all_nodes():
            from rdflib import RDF
            nodes = set()
            for s in self.graph.subjects(RDF.type, None):
                parts = str(s).split('/')
                if len(parts) >= 2:
                    try:
                        nodes.add((parts[-2], int(parts[-1])))
                    except:
                        pass
            return nodes

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.neighbors(node[0], node[1]):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in get_all_nodes():
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def node_count(self) -> int:
        from rdflib import RDF
        return len(set(self.graph.subjects(RDF.type, None)))

    def edge_count(self) -> int:
        from rdflib import RDF
        count = 0
        for s, p, o in self.graph:
            if p != RDF.type and str(o).startswith("http://example.org/"):
                count += 1
        return count

    def save(self, path: str) -> None:
        self.graph.serialize(destination=path, format='turtle')

    def load(self, path: str) -> None:
        from rdflib import Graph, Namespace
        self.graph = Graph()
        self.graph.parse(path, format='turtle')
        self._ns = Namespace("http://example.org/")


# =============================================================================
# Kuzu Embedded Graph Database Store Implementation
# =============================================================================

class KuzuStore(GraphStore):
    """Kuzu embedded graph database store implementation."""

    name = "Kuzu"

    def __init__(self):
        self.db = None
        self.conn = None
        self._temp_dir = None

    def setup(self) -> None:
        import kuzu
        import tempfile
        self._temp_dir = tempfile.mkdtemp()
        self.db = kuzu.Database(self._temp_dir)
        self.conn = kuzu.Connection(self.db)
        # Create schema
        self.conn.execute("CREATE NODE TABLE person(id INT64, name STRING, age INT64, PRIMARY KEY(id))")
        self.conn.execute("CREATE REL TABLE KNOWS(FROM person TO person, since INT64)")

    def teardown(self) -> None:
        import shutil
        self.conn = None
        self.db = None
        if self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir)
            except:
                pass

    def clear(self) -> None:
        self.teardown()
        self.setup()

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        name = properties.get('name', f'Node_{node_id}')
        age = properties.get('age', 0)
        try:
            self.conn.execute(f"CREATE (p:person {{id: {node_id}, name: '{name}', age: {age}}})")
        except:
            pass

    def get_node(self, node_type: str, node_id: int) -> Any:
        result = self.conn.execute(f"MATCH (p:person {{id: {node_id}}}) RETURN p.name, p.age")
        while result.has_next():
            row = result.get_next()
            return {"name": row[0], "age": row[1]}
        return None

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        for k, v in properties.items():
            if isinstance(v, str):
                self.conn.execute(f"MATCH (p:person {{id: {node_id}}}) SET p.{k} = '{v}'")
            else:
                self.conn.execute(f"MATCH (p:person {{id: {node_id}}}) SET p.{k} = {v}")

    def delete_node(self, node_type: str, node_id: int) -> None:
        self.conn.execute(f"MATCH (p:person {{id: {node_id}}}) DELETE p")

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        since = properties.get('since', 2020)
        try:
            self.conn.execute(f"""
                MATCH (a:person {{id: {source[1]}}}), (b:person {{id: {target[1]}}})
                CREATE (a)-[:KNOWS {{since: {since}}}]->(b)
            """)
        except:
            pass

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        result = self.conn.execute(f"""
            MATCH (a:person {{id: {source[1]}}})-[r:KNOWS]->(b:person {{id: {target[1]}}})
            RETURN r.since
        """)
        while result.has_next():
            row = result.get_next()
            return {"since": row[0]}
        return None

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        self.conn.execute(f"""
            MATCH (a:person {{id: {source[1]}}})-[r:KNOWS]->(b:person {{id: {target[1]}}})
            DELETE r
        """)

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        result = self.conn.execute(f"""
            MATCH (a:person {{id: {node_id}}})-[:KNOWS]->(b:person)
            RETURN b.id
        """)
        neighbors = []
        while result.has_next():
            row = result.get_next()
            neighbors.append(("person", row[0]))
        return neighbors

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        result = self.conn.execute(f"""
            MATCH (a:person {{id: {node_id}}})-[:KNOWS*{hops}]->(b:person)
            RETURN DISTINCT b.id
        """)
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(("person", row[0]))
        return nodes

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        result = self.conn.execute(f"""
            MATCH p = shortestPath((a:person {{id: {source[1]}}})-[:KNOWS*]->(b:person {{id: {target[1]}}}))
            RETURN nodes(p)
        """)
        while result.has_next():
            row = result.get_next()
            return row[0]
        return None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        return self.shortest_path(source, target) is not None

    def has_cycle(self) -> bool:
        # Check if any node can reach itself
        result = self.conn.execute("""
            MATCH (a:person)-[:KNOWS*1..10]->(a)
            RETURN COUNT(*) > 0
        """)
        while result.has_next():
            return result.get_next()[0]
        return False

    def node_count(self) -> int:
        result = self.conn.execute("MATCH (p:person) RETURN COUNT(*)")
        while result.has_next():
            return result.get_next()[0]
        return 0

    def edge_count(self) -> int:
        result = self.conn.execute("MATCH ()-[r:KNOWS]->() RETURN COUNT(*)")
        while result.has_next():
            return result.get_next()[0]
        return 0

    def save(self, path: str) -> None:
        # Kuzu uses its own storage, we'll export to CSV
        result = self.conn.execute("MATCH (p:person) RETURN p.id, p.name, p.age")
        with open(path, 'w') as f:
            f.write("id,name,age\n")
            while result.has_next():
                row = result.get_next()
                f.write(f"{row[0]},{row[1]},{row[2]}\n")

    def load(self, path: str) -> None:
        # For benchmark purposes, just re-setup
        self.setup()


# =============================================================================
# SQLite Graph Store Implementation
# =============================================================================

class SQLiteGraphStore(GraphStore):
    """SQLite-based graph store implementation."""

    name = "SQLite"

    def __init__(self):
        self.conn = None
        self._db_path = None

    def setup(self) -> None:
        import sqlite3
        import tempfile
        self._db_path = tempfile.mktemp(suffix='.db')
        self.conn = sqlite3.connect(self._db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_type TEXT,
                node_id INTEGER,
                name TEXT,
                age INTEGER,
                PRIMARY KEY (node_type, node_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                rel_type TEXT,
                src_type TEXT,
                src_id INTEGER,
                tgt_type TEXT,
                tgt_id INTEGER,
                since INTEGER,
                PRIMARY KEY (rel_type, src_type, src_id, tgt_type, tgt_id)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_type, src_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(tgt_type, tgt_id)")
        self.conn.commit()

    def teardown(self) -> None:
        import os
        if self.conn:
            self.conn.close()
        if self._db_path and os.path.exists(self._db_path):
            os.remove(self._db_path)

    def clear(self) -> None:
        self.conn.execute("DELETE FROM nodes")
        self.conn.execute("DELETE FROM edges")
        self.conn.commit()

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        name = properties.get('name', '')
        age = properties.get('age', 0)
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?)",
            (node_type, node_id, name, age)
        )

    def get_node(self, node_type: str, node_id: int) -> Any:
        cur = self.conn.execute(
            "SELECT name, age FROM nodes WHERE node_type=? AND node_id=?",
            (node_type, node_id)
        )
        row = cur.fetchone()
        if row:
            return {"name": row[0], "age": row[1]}
        return None

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        for k, v in properties.items():
            self.conn.execute(
                f"UPDATE nodes SET {k}=? WHERE node_type=? AND node_id=?",
                (v, node_type, node_id)
            )

    def delete_node(self, node_type: str, node_id: int) -> None:
        self.conn.execute(
            "DELETE FROM nodes WHERE node_type=? AND node_id=?",
            (node_type, node_id)
        )
        self.conn.execute(
            "DELETE FROM edges WHERE (src_type=? AND src_id=?) OR (tgt_type=? AND tgt_id=?)",
            (node_type, node_id, node_type, node_id)
        )

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        since = properties.get('since', 2020)
        self.conn.execute(
            "INSERT OR REPLACE INTO edges VALUES (?, ?, ?, ?, ?, ?)",
            (rel_type, source[0], source[1], target[0], target[1], since)
        )

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        cur = self.conn.execute(
            "SELECT since FROM edges WHERE rel_type=? AND src_type=? AND src_id=? AND tgt_type=? AND tgt_id=?",
            (rel_type, source[0], source[1], target[0], target[1])
        )
        row = cur.fetchone()
        if row:
            return {"since": row[0]}
        return None

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        self.conn.execute(
            "DELETE FROM edges WHERE rel_type=? AND src_type=? AND src_id=? AND tgt_type=? AND tgt_id=?",
            (rel_type, source[0], source[1], target[0], target[1])
        )

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        if rel_type:
            cur = self.conn.execute(
                "SELECT tgt_type, tgt_id FROM edges WHERE src_type=? AND src_id=? AND rel_type=?",
                (node_type, node_id, rel_type)
            )
        else:
            cur = self.conn.execute(
                "SELECT tgt_type, tgt_id FROM edges WHERE src_type=? AND src_id=?",
                (node_type, node_id)
            )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        visited = set()
        current = {(node_type, node_id)}
        for _ in range(hops):
            next_level = set()
            for nt, nid in current:
                for neighbor in self.neighbors(nt, nid, rel_type):
                    if neighbor not in visited:
                        next_level.add(neighbor)
            visited.update(current)
            current = next_level
        return list(current)

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        from collections import deque
        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in self.neighbors(node[0], node[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        return self.shortest_path(source, target) is not None

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        cur = self.conn.execute("SELECT DISTINCT node_type, node_id FROM nodes")
        all_nodes = cur.fetchall()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self.neighbors(node[0], node[1]):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in all_nodes:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def node_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM nodes")
        return cur.fetchone()[0]

    def edge_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM edges")
        return cur.fetchone()[0]

    def save(self, path: str) -> None:
        import shutil
        self.conn.commit()
        shutil.copy(self._db_path, path)

    def load(self, path: str) -> None:
        import sqlite3
        if self.conn:
            self.conn.close()
        self._db_path = path
        self.conn = sqlite3.connect(path)


# =============================================================================
# DictGraph - Pure Python Baseline
# =============================================================================

class DictGraphStore(GraphStore):
    """Pure Python dict-based graph store (baseline)."""

    name = "DictGraph"

    def __init__(self):
        self._nodes = {}
        self._edges = {}
        self._out_edges = {}
        self._in_edges = {}

    def setup(self) -> None:
        self._nodes = {}
        self._edges = {}
        self._out_edges = {}
        self._in_edges = {}

    def teardown(self) -> None:
        self._nodes = {}
        self._edges = {}
        self._out_edges = {}
        self._in_edges = {}

    def clear(self) -> None:
        self.setup()

    def add_node(self, node_type: str, node_id: int, **properties) -> None:
        key = (node_type, node_id)
        self._nodes[key] = {"type": node_type, "id": node_id, **properties}

    def get_node(self, node_type: str, node_id: int) -> Any:
        return self._nodes.get((node_type, node_id))

    def update_node(self, node_type: str, node_id: int, **properties) -> None:
        key = (node_type, node_id)
        if key in self._nodes:
            self._nodes[key].update(properties)

    def delete_node(self, node_type: str, node_id: int) -> None:
        key = (node_type, node_id)
        self._nodes.pop(key, None)
        # Remove associated edges
        for edge_key in list(self._edges.keys()):
            if edge_key[1] == key or edge_key[2] == key:
                del self._edges[edge_key]

    def add_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int], **properties) -> None:
        key = (rel_type, source, target)
        self._edges[key] = {"rel_type": rel_type, "source": source, "target": target, **properties}
        self._out_edges.setdefault(source, []).append(target)
        self._in_edges.setdefault(target, []).append(source)

    def get_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> Any:
        return self._edges.get((rel_type, source, target))

    def delete_edge(self, rel_type: str, source: Tuple[str, int], target: Tuple[str, int]) -> None:
        key = (rel_type, source, target)
        self._edges.pop(key, None)

    def neighbors(self, node_type: str, node_id: int, rel_type: str = None) -> List:
        return self._out_edges.get((node_type, node_id), [])

    def multi_hop(self, node_type: str, node_id: int, hops: int, rel_type: str = None) -> List:
        visited = set()
        current = {(node_type, node_id)}
        for _ in range(hops):
            next_level = set()
            for node in current:
                for neighbor in self._out_edges.get(node, []):
                    if neighbor not in visited:
                        next_level.add(neighbor)
            visited.update(current)
            current = next_level
        return list(current)

    def shortest_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> Optional[List]:
        from collections import deque
        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in self._out_edges.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def has_path(self, source: Tuple[str, int], target: Tuple[str, int]) -> bool:
        return self.shortest_path(source, target) is not None

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._out_edges.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in self._nodes.keys():
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def save(self, path: str) -> None:
        import json
        data = {
            "nodes": {f"{k[0]}:{k[1]}": v for k, v in self._nodes.items()},
            "edges": {f"{k[0]}:{k[1][0]}:{k[1][1]}:{k[2][0]}:{k[2][1]}": v for k, v in self._edges.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        # Simplified load for benchmark
        self.setup()


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs benchmarks across multiple stores."""

    def __init__(self, node_count: int = DEFAULT_NODE_COUNT, verbose: bool = True):
        self.node_count = node_count
        self.edge_count = node_count * EDGE_MULTIPLIER
        self.verbose = verbose
        self.results: Dict[str, StoreMetrics] = {}

        # Pre-generate test data
        self._generate_test_data()

    def _generate_test_data(self) -> None:
        """Generate random test data."""
        random.seed(42)  # Reproducible

        # Node data
        self.nodes = [
            ("person", i, {"name": f"Person_{i}", "age": random.randint(18, 80)})
            for i in range(self.node_count)
        ]

        # Edge data (random connections)
        self.edges = []
        for _ in range(self.edge_count):
            src_id = random.randint(0, self.node_count - 1)
            tgt_id = random.randint(0, self.node_count - 1)
            if src_id != tgt_id:
                self.edges.append((
                    "KNOWS",
                    ("person", src_id),
                    ("person", tgt_id),
                    {"since": random.randint(2000, 2024)}
                ))

        # Sample nodes for queries
        self.sample_nodes = random.sample(range(self.node_count), min(100, self.node_count))

        # Sample pairs for path queries
        self.sample_pairs = [
            (("person", random.randint(0, self.node_count - 1)),
             ("person", random.randint(0, self.node_count - 1)))
            for _ in range(50)
        ]

    def _time_operation(self, operation: Callable, iterations: int, warmup: int = WARMUP_ITERATIONS) -> List[float]:
        """Time an operation multiple times, returning list of times in microseconds."""
        # Warmup
        for _ in range(warmup):
            operation()

        # Actual timing
        times = []
        for _ in range(iterations):
            gc.disable()
            start = time.perf_counter_ns()
            operation()
            end = time.perf_counter_ns()
            gc.enable()
            times.append((end - start) / 1000)  # Convert to microseconds

        return times

    def _measure_memory(self, operation: Callable) -> float:
        """Measure peak memory usage in MB."""
        gc.collect()
        tracemalloc.start()
        operation()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (1024 * 1024)  # Convert to MB

    def _create_result(self, operation: str, store: str, times: List[float], memory_mb: float = 0.0) -> BenchmarkResult:
        """Create a BenchmarkResult from timing data."""
        return BenchmarkResult(
            operation=operation,
            store=store,
            iterations=len(times),
            total_time_ms=sum(times) / 1000,
            min_time_us=min(times),
            max_time_us=max(times),
            avg_time_us=statistics.mean(times),
            median_time_us=statistics.median(times),
            std_dev_us=statistics.stdev(times) if len(times) > 1 else 0,
            ops_per_sec=1_000_000 / statistics.mean(times) if statistics.mean(times) > 0 else 0,
            memory_mb=memory_mb
        )

    def benchmark_store(self, store: GraphStore) -> StoreMetrics:
        """Run all benchmarks for a single store."""
        metrics = StoreMetrics(store=store.name)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {store.name}")
            print(f"{'='*60}")

        store.setup()

        # 1. Add Nodes
        if self.verbose:
            print(f"  Adding {self.node_count} nodes...")

        def add_all_nodes():
            for node_type, node_id, props in self.nodes:
                store.add_node(node_type, node_id, **props)

        memory = self._measure_memory(add_all_nodes)
        store.clear()

        # Time individual add_node operations with unique IDs
        test_id = [self.node_count + 1000]  # Start beyond existing range
        def add_single_node():
            store.add_node("person", test_id[0], name="Test", age=25)
            test_id[0] += 1

        times = self._time_operation(add_single_node, BENCHMARK_ITERATIONS)
        store.clear()

        # Actually add all nodes for subsequent tests
        add_all_nodes()

        result = self._create_result("add_node", store.name, times, memory)
        metrics.results.append(result)
        if self.verbose:
            print(f"    add_node: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 2. Get Node
        if self.verbose:
            print(f"  Getting nodes...")

        times = self._time_operation(
            lambda: store.get_node("person", random.choice(self.sample_nodes)),
            BENCHMARK_ITERATIONS
        )
        result = self._create_result("get_node", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    get_node: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 3. Update Node
        if self.verbose:
            print(f"  Updating nodes...")

        times = self._time_operation(
            lambda: store.update_node("person", random.choice(self.sample_nodes), age=30),
            BENCHMARK_ITERATIONS
        )
        result = self._create_result("update_node", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    update_node: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 4. Add Edges
        if self.verbose:
            print(f"  Adding {len(self.edges)} edges...")

        def add_all_edges():
            for rel_type, source, target, props in self.edges:
                try:
                    store.add_edge(rel_type, source, target, **props)
                except:
                    pass  # Ignore duplicate edges

        add_all_edges()

        # Time individual add_edge operations with unique targets
        edge_test_id = [self.node_count + 2000]
        def add_single_edge():
            # Add a temporary node and edge
            target_id = edge_test_id[0]
            try:
                store.add_node("person", target_id, name="EdgeTest", age=30)
            except:
                pass
            try:
                store.add_edge("KNOWS", ("person", 0), ("person", target_id), since=2020)
            except:
                pass
            edge_test_id[0] += 1

        times = self._time_operation(add_single_edge, BENCHMARK_ITERATIONS)
        result = self._create_result("add_edge", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    add_edge: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 5. Get Edge
        if self.verbose:
            print(f"  Getting edges...")

        sample_edge = self.edges[0] if self.edges else ("KNOWS", ("person", 0), ("person", 1), {})
        times = self._time_operation(
            lambda: store.get_edge(sample_edge[0], sample_edge[1], sample_edge[2]),
            BENCHMARK_ITERATIONS
        )
        result = self._create_result("get_edge", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    get_edge: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 6. Neighbors (1-hop)
        if self.verbose:
            print(f"  1-hop traversal...")

        times = self._time_operation(
            lambda: store.neighbors("person", random.choice(self.sample_nodes)),
            TRAVERSAL_ITERATIONS
        )
        result = self._create_result("neighbors_1hop", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    neighbors: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 7. Multi-hop (2-hop)
        if self.verbose:
            print(f"  2-hop traversal...")

        times = self._time_operation(
            lambda: store.multi_hop("person", random.choice(self.sample_nodes), 2),
            TRAVERSAL_ITERATIONS
        )
        result = self._create_result("multi_hop_2", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    2-hop: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 8. Multi-hop (3-hop)
        if self.verbose:
            print(f"  3-hop traversal...")

        times = self._time_operation(
            lambda: store.multi_hop("person", random.choice(self.sample_nodes), 3),
            TRAVERSAL_ITERATIONS
        )
        result = self._create_result("multi_hop_3", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    3-hop: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 9. Shortest Path
        if self.verbose:
            print(f"  Shortest path...")

        times = self._time_operation(
            lambda: store.shortest_path(*random.choice(self.sample_pairs)),
            PATH_ITERATIONS
        )
        result = self._create_result("shortest_path", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    shortest_path: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 10. Has Path
        if self.verbose:
            print(f"  Path exists check...")

        times = self._time_operation(
            lambda: store.has_path(*random.choice(self.sample_pairs)),
            PATH_ITERATIONS
        )
        result = self._create_result("has_path", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    has_path: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 11. Has Cycle
        if self.verbose:
            print(f"  Cycle detection...")

        times = self._time_operation(
            lambda: store.has_cycle(),
            PATH_ITERATIONS
        )
        result = self._create_result("has_cycle", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    has_cycle: {result.avg_time_us:.2f} us/op, {result.ops_per_sec:.0f} ops/sec")

        # 12. Serialization (Save)
        if self.verbose:
            print(f"  Serialization...")

        temp_path = f"temp_benchmark_{store.name}.tmp"
        times = self._time_operation(
            lambda: store.save(temp_path),
            10  # Fewer iterations for I/O
        )
        result = self._create_result("save", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    save: {result.avg_time_us:.2f} us/op")

        # 13. Serialization (Load)
        times = self._time_operation(
            lambda: store.load(temp_path),
            10
        )
        result = self._create_result("load", store.name, times)
        metrics.results.append(result)
        if self.verbose:
            print(f"    load: {result.avg_time_us:.2f} us/op")

        # 14. Measure serialized size and token count
        if self.verbose:
            print(f"  Measuring size and tokens...")

        if os.path.exists(temp_path):
            metrics.serialized_size_bytes = os.path.getsize(temp_path)

            # Count tokens using tiktoken
            try:
                import tiktoken
                enc = tiktoken.get_encoding("o200k_base")
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                metrics.token_count = len(enc.encode(content))
            except Exception as e:
                # Fallback: estimate tokens as bytes/4
                metrics.token_count = metrics.serialized_size_bytes // 4

            # Calculate context window utilization
            if metrics.token_count > 0:
                metrics.context_4k_pct = (metrics.token_count / 4096) * 100
                metrics.context_8k_pct = (metrics.token_count / 8192) * 100
                metrics.context_32k_pct = (metrics.token_count / 32768) * 100
                metrics.context_128k_pct = (metrics.token_count / 131072) * 100
                metrics.nodes_per_1k_tokens = (self.node_count / metrics.token_count) * 1000

            if self.verbose:
                print(f"    size: {metrics.serialized_size_bytes:,} bytes")
                print(f"    tokens: {metrics.token_count:,}")
                print(f"    context usage: {metrics.context_4k_pct:.1f}% of 4K, {metrics.context_32k_pct:.1f}% of 32K")
                print(f"    nodes/1K tokens: {metrics.nodes_per_1k_tokens:.1f}")

        # 15. Measure memory after full load
        if self.verbose:
            print(f"  Measuring memory...")

        gc.collect()
        tracemalloc.start()
        store.clear()
        for node_type, node_id, props in self.nodes:
            store.add_node(node_type, node_id, **props)
        for rel_type, source, target, props in self.edges:
            try:
                store.add_edge(rel_type, source, target, **props)
            except:
                pass
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics.memory_after_load_mb = peak / (1024 * 1024)

        if self.verbose:
            print(f"    memory (loaded): {metrics.memory_after_load_mb:.2f} MB")

        # Cleanup
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass

        # Calculate totals
        metrics.total_time_ms = sum(r.total_time_ms for r in metrics.results)
        metrics.peak_memory_mb = max(r.memory_mb for r in metrics.results) if metrics.results else 0

        store.teardown()

        return metrics

    def run(self, stores: List[GraphStore]) -> Dict[str, StoreMetrics]:
        """Run benchmarks for all stores."""
        print(f"\n{'#'*60}")
        print(f"# ISONGraph Store Benchmark")
        print(f"# Nodes: {self.node_count:,}, Edges: {self.edge_count:,}")
        print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        for store in stores:
            try:
                metrics = self.benchmark_store(store)
                self.results[store.name] = metrics
            except ImportError as e:
                print(f"\n  Skipping {store.name}: {e}")
            except Exception as e:
                print(f"\n  Error benchmarking {store.name}: {e}")

        return self.results

    def print_summary(self) -> None:
        """Print summary comparison table."""
        if not self.results:
            print("No results to display.")
            return

        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")

        # Get all operations
        operations = []
        for metrics in self.results.values():
            for r in metrics.results:
                if r.operation not in operations:
                    operations.append(r.operation)

        # Header
        stores = list(self.results.keys())
        header = f"{'Operation':<20}"
        for store in stores:
            header += f" | {store:>15}"
        print(header)
        print("-" * len(header))

        # Data rows (avg time in us)
        for op in operations:
            row = f"{op:<20}"
            times = []
            for store in stores:
                metrics = self.results.get(store)
                if metrics:
                    result = next((r for r in metrics.results if r.operation == op), None)
                    if result:
                        times.append((store, result.avg_time_us))
                        row += f" | {result.avg_time_us:>12.2f} us"
                    else:
                        row += f" | {'N/A':>15}"
                else:
                    row += f" | {'N/A':>15}"

            # Mark winner
            if times:
                winner = min(times, key=lambda x: x[1])[0]
                row += f"  <- {winner}"

            print(row)

        # Totals
        print("-" * len(header))
        row = f"{'TOTAL TIME':<20}"
        for store in stores:
            metrics = self.results.get(store)
            if metrics:
                row += f" | {metrics.total_time_ms:>12.2f} ms"
            else:
                row += f" | {'N/A':>15}"
        print(row)

        # Winner summary
        print(f"\n{'='*80}")
        print("WINNER BY OPERATION")
        print(f"{'='*80}")

        wins = {store: 0 for store in stores}
        for op in operations:
            times = []
            for store in stores:
                metrics = self.results.get(store)
                if metrics:
                    result = next((r for r in metrics.results if r.operation == op), None)
                    if result:
                        times.append((store, result.avg_time_us))
            if times:
                winner = min(times, key=lambda x: x[1])[0]
                wins[winner] += 1

        for store, count in sorted(wins.items(), key=lambda x: -x[1]):
            bar = "#" * count
            print(f"  {store:<15} {bar} ({count}/{len(operations)})")

        # Print efficiency metrics
        print(f"\n{'='*80}")
        print("EFFICIENCY METRICS (Size, Tokens, Memory)")
        print(f"{'='*80}")

        header = f"{'Store':<15} | {'Size (bytes)':<14} | {'Tokens':<10} | {'Memory (MB)':<12} | {'Nodes/1K tok':<12} | {'4K ctx %':<10}"
        print(header)
        print("-" * len(header))

        for store in stores:
            metrics = self.results.get(store)
            if metrics:
                print(f"{store:<15} | {metrics.serialized_size_bytes:>12,} | {metrics.token_count:>10,} | {metrics.memory_after_load_mb:>10.2f} | {metrics.nodes_per_1k_tokens:>12.1f} | {metrics.context_4k_pct:>8.1f}%")

        # Context window capacity
        print(f"\n{'='*80}")
        print("CONTEXT WINDOW CAPACITY (How many nodes fit?)")
        print(f"{'='*80}")

        header = f"{'Store':<15} | {'4K ctx':<12} | {'8K ctx':<12} | {'32K ctx':<12} | {'128K ctx':<12}"
        print(header)
        print("-" * len(header))

        for store in stores:
            metrics = self.results.get(store)
            if metrics and metrics.nodes_per_1k_tokens > 0:
                nodes_4k = int(metrics.nodes_per_1k_tokens * 4)
                nodes_8k = int(metrics.nodes_per_1k_tokens * 8)
                nodes_32k = int(metrics.nodes_per_1k_tokens * 32)
                nodes_128k = int(metrics.nodes_per_1k_tokens * 128)
                print(f"{store:<15} | {nodes_4k:>10,} | {nodes_8k:>10,} | {nodes_32k:>10,} | {nodes_128k:>10,}")

    def save_results(self, path: str) -> None:
        """Save results to JSON file."""
        data = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "node_count": self.node_count,
                "edge_count": self.edge_count,
            },
            "results": {}
        }

        for store_name, metrics in self.results.items():
            data["results"][store_name] = {
                "total_time_ms": metrics.total_time_ms,
                "peak_memory_mb": metrics.peak_memory_mb,
                "memory_after_load_mb": metrics.memory_after_load_mb,
                "serialized_size_bytes": metrics.serialized_size_bytes,
                "token_count": metrics.token_count,
                "nodes_per_1k_tokens": metrics.nodes_per_1k_tokens,
                "context_4k_pct": metrics.context_4k_pct,
                "context_8k_pct": metrics.context_8k_pct,
                "context_32k_pct": metrics.context_32k_pct,
                "context_128k_pct": metrics.context_128k_pct,
                "operations": [
                    {
                        "operation": r.operation,
                        "avg_time_us": r.avg_time_us,
                        "median_time_us": r.median_time_us,
                        "min_time_us": r.min_time_us,
                        "max_time_us": r.max_time_us,
                        "ops_per_sec": r.ops_per_sec,
                    }
                    for r in metrics.results
                ]
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {path}")

    def save_log(self, path: str) -> None:
        """Save detailed log file with all metrics."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("ISONGRAPH STORE BENCHMARK - DETAILED LOG\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Node Count: {self.node_count:,}\n")
            f.write(f"Edge Count: {self.edge_count:,}\n")
            f.write(f"Warmup Iterations: {WARMUP_ITERATIONS}\n")
            f.write(f"Benchmark Iterations: {BENCHMARK_ITERATIONS}\n")
            f.write(f"Traversal Iterations: {TRAVERSAL_ITERATIONS}\n")
            f.write(f"Path Iterations: {PATH_ITERATIONS}\n\n")

            f.write("STORES TESTED\n")
            f.write("-" * 40 + "\n")
            for store_name in self.results.keys():
                f.write(f"  - {store_name}\n")
            f.write("\n")

            # Detailed results per store
            for store_name, metrics in self.results.items():
                f.write("=" * 80 + "\n")
                f.write(f"STORE: {store_name}\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Total Time: {metrics.total_time_ms:.2f} ms\n")
                f.write(f"Peak Memory: {metrics.peak_memory_mb:.2f} MB\n")
                f.write(f"Memory After Load: {metrics.memory_after_load_mb:.2f} MB\n")
                f.write(f"Serialized Size: {metrics.serialized_size_bytes:,} bytes\n")
                f.write(f"Token Count: {metrics.token_count:,}\n")
                f.write(f"Nodes per 1K Tokens: {metrics.nodes_per_1k_tokens:.1f}\n")
                f.write(f"Context Usage (4K): {metrics.context_4k_pct:.1f}%\n")
                f.write(f"Context Usage (32K): {metrics.context_32k_pct:.1f}%\n\n")

                f.write("OPERATION DETAILS\n")
                f.write("-" * 60 + "\n")
                f.write(f"{'Operation':<20} {'Avg (us)':<12} {'Min (us)':<12} {'Max (us)':<12} {'Ops/sec':<15}\n")
                f.write("-" * 60 + "\n")

                for r in metrics.results:
                    f.write(f"{r.operation:<20} {r.avg_time_us:<12.2f} {r.min_time_us:<12.2f} {r.max_time_us:<12.2f} {r.ops_per_sec:<15,.0f}\n")

                f.write("\n")

            # Summary comparison table
            f.write("=" * 80 + "\n")
            f.write("COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Get all operations
            operations = []
            for metrics in self.results.values():
                for r in metrics.results:
                    if r.operation not in operations:
                        operations.append(r.operation)

            stores = list(self.results.keys())

            # Header
            header = f"{'Operation':<20}"
            for store in stores:
                header += f" | {store:>15}"
            header += " | Winner"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Data rows
            wins = {store: 0 for store in stores}
            for op in operations:
                row = f"{op:<20}"
                times = []
                for store in stores:
                    metrics = self.results.get(store)
                    if metrics:
                        result = next((r for r in metrics.results if r.operation == op), None)
                        if result:
                            times.append((store, result.avg_time_us))
                            row += f" | {result.avg_time_us:>12.2f} us"
                        else:
                            row += f" | {'N/A':>15}"
                    else:
                        row += f" | {'N/A':>15}"

                if times:
                    winner = min(times, key=lambda x: x[1])[0]
                    row += f" | {winner}"
                    wins[winner] += 1

                f.write(row + "\n")

            f.write("\n")
            f.write("WINNER SUMMARY\n")
            f.write("-" * 40 + "\n")
            for store, count in sorted(wins.items(), key=lambda x: -x[1]):
                pct = (count / len(operations)) * 100 if operations else 0
                f.write(f"  {store:<15} {count}/{len(operations)} operations ({pct:.1f}%)\n")

        print(f"Log saved to: {path}")


# =============================================================================
# Main
# =============================================================================

def get_available_stores(requested: List[str] = None) -> List[GraphStore]:
    """Get list of available stores."""
    all_stores = {
        "isongraph": ISONGraphStore,
        "networkx": NetworkXStore,
        "igraph": IGraphStore,
        "rdflib": RDFLibStore,
        "kuzu": KuzuStore,
        "sqlite": SQLiteGraphStore,
        "dictgraph": DictGraphStore,
    }

    if requested:
        requested = [s.lower() for s in requested]
        stores = []
        for name in requested:
            if name in all_stores:
                stores.append(all_stores[name]())
            else:
                print(f"Warning: Unknown store '{name}'")
        return stores

    # Try to load all stores
    stores = []

    # ISONGraph
    try:
        from ison_graph import ISONGraph
        stores.append(ISONGraphStore())
        print("  [+] ISONGraph - loaded")
    except ImportError:
        print("  [-] ISONGraph - not installed (pip install -e ../ison-graph)")

    # NetworkX
    try:
        import networkx
        stores.append(NetworkXStore())
        print("  [+] NetworkX - loaded")
    except ImportError:
        print("  [-] NetworkX - not installed (pip install networkx)")

    # igraph
    try:
        import igraph
        stores.append(IGraphStore())
        print("  [+] igraph - loaded")
    except ImportError:
        print("  [-] igraph - not installed (pip install igraph)")

    # RDFLib
    try:
        import rdflib
        stores.append(RDFLibStore())
        print("  [+] RDFLib - loaded")
    except ImportError:
        print("  [-] RDFLib - not installed (pip install rdflib)")

    # Kuzu
    try:
        import kuzu
        stores.append(KuzuStore())
        print("  [+] Kuzu - loaded")
    except ImportError:
        print("  [-] Kuzu - not installed (pip install kuzu)")

    # SQLite (always available in Python)
    stores.append(SQLiteGraphStore())
    print("  [+] SQLite - loaded (built-in)")

    # DictGraph (pure Python baseline, always available)
    stores.append(DictGraphStore())
    print("  [+] DictGraph - loaded (baseline)")

    return stores


def main():
    parser = argparse.ArgumentParser(description="ISONGraph Store Benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (10K nodes)")
    parser.add_argument("--scale", type=int, help="Custom node count")
    parser.add_argument("--stores", nargs="+", help="Specific stores to test")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Determine scale
    if args.scale:
        node_count = args.scale
    elif args.full:
        node_count = FULL_NODE_COUNT
    else:
        node_count = DEFAULT_NODE_COUNT

    # Get stores
    stores = get_available_stores(args.stores)

    if not stores:
        print("Error: No graph stores available for benchmarking.")
        print("Install at least one of: ison-graph, networkx, igraph")
        sys.exit(1)

    # Run benchmark
    runner = BenchmarkRunner(node_count=node_count, verbose=not args.quiet)
    runner.run(stores)
    runner.print_summary()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        runner.save_results(args.output)
        runner.save_log(args.output.replace(".json", ".log"))
    else:
        runner.save_results(f"benchmark_stores_{timestamp}.json")
        runner.save_log(f"benchmark_stores_{timestamp}.log")


if __name__ == "__main__":
    main()
