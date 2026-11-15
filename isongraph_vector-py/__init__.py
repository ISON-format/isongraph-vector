#!/usr/bin/env python3
"""
ISONGraph Vector - Semantic Graph Extension

Extends ISONGraph with embedding-based similarity search and semantic traversal.
Bridges the gap between vector similarity and graph structure.

Usage:
    from isongraph_vector import SemanticGraph, MockEncoder

    # Create semantic graph
    graph = SemanticGraph()

    # Add nodes with auto-embedding
    graph.add_node('person', 1, name='Alice', description='Software engineer')
    graph.add_node('person', 2, name='Bob', description='Data scientist')

    # Semantic similarity search
    similar = graph.similarity_search("engineering roles", top_k=5)

    # Semantic multi-hop: combines embedding similarity with graph traversal
    results = graph.semantic_multi_hop(
        query="Who knows engineers?",
        rel_type='KNOWS',
        max_hops=2
    )

    # Persist graph state *and* embeddings together, then pick up where you
    # left off - including the encoder, which is not stored in the file.
    graph.save("social.isong")
    graph = SemanticGraph.load("social.isong", encoder=graph.encoder)

Dependencies:
    - ison-graph (required)
    - sentence-transformers (required for SentenceTransformerEncoder)
    - numpy (optional; vectorizes similarity search, orders of magnitude faster)

Author: Mahesh Vaikri
Version: 1.0.0
"""

from __future__ import annotations

import os
import sqlite3
import struct
import hashlib
import heapq
import json
import threading
import uuid
import warnings
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any, Dict, List, Optional, Sequence,
    Union, Callable, Iterable, Tuple
)
from pathlib import Path as FilePath

# Import from ison-graph
from ison_graph import (
    ISONGraph, Node, NodeRef, Edge,
    Direction, GraphError
)

try:  # optional: vectorized similarity search
    import numpy as _np
except ImportError:  # pragma: no cover - covered by installs without numpy
    _np = None

try:  # optional: approximate nearest-neighbour search inside SQLite
    import sqlite_vec as _sqlite_vec
except ImportError:  # pragma: no cover - covered by installs without sqlite-vec
    _sqlite_vec = None

#: Virtual table holding the vec0 index, alongside the `embeddings` table.
VEC_TABLE = "vec_embeddings"

__version__ = "1.0.0"
__author__ = "Mahesh Vaikri"

#: Portable embedding sidecar format written next to a saved graph, and by
#: :meth:`EmbeddingStore.export_embeddings`. Every language port reads and
#: writes this same shape, so a store built by one can be used by another.
EMBEDDING_FORMAT = "ison-embeddings"
EMBEDDING_FORMAT_VERSION = 1

#: Suffix appended to a graph path to locate its embeddings.
EMBEDDING_SIDECAR_SUFFIX = ".embeddings.json"

#: Name of the serialization :func:`graph_digest` hashes. Written next to the
#: digest so a reader that computes a different digest (or none) can tell it is
#: not comparable and skip the check instead of reporting a false mismatch.
GRAPH_DIGEST_BASIS = "nodes-edges-json-v1"

#: Model names that carry no information about which encoder made a vector:
#: ``"provided"`` (vector handed in, store has no encoder) and ``"unknown"``
#: (sidecars from ports that do not record a model).
UNKNOWN_MODELS = frozenset({"unknown", "provided"})


# =============================================================================
# Errors
# =============================================================================

class EmbeddingError(Exception):
    """Base class for embedding-specific failures."""


class DimensionMismatchError(EmbeddingError, ValueError):
    """
    A vector was stored or queried with the wrong number of dimensions.

    Cosine similarity over a mismatched pair returns a plausible-looking
    number rather than an error - a 2-dim vector used to score 0.06 against a
    384-dim query in a 384-dim store instead of failing, and an empty vector
    scored 0.0. A store therefore pins its dimension to the first vector it
    sees and refuses anything else.
    """


class NoEncoderError(EmbeddingError, ValueError):
    """Text was handed to a store that has no encoder to embed it with."""


class ModelMismatchError(EmbeddingError):
    """
    A text query was encoded by a different model than the stored vectors.

    Two models of the same dimension give a plausible-looking cosine score
    between vectors that live in unrelated spaces, so the ranking is noise
    rather than an error. Raised only with ``on_model_mismatch="error"``; the
    default is a warning.
    """


class GraphEmbeddingMismatchError(EmbeddingError):
    """
    A graph file and its embeddings sidecar are not from the same save.

    Raised by :meth:`SemanticGraph.load` when the sidecar's recorded
    ``graph_digest`` does not match the graph that was read - the sidecar came
    from a different save, or the graph file was edited afterwards.
    """


# =============================================================================
# File helpers
# =============================================================================

def _atomic_write_text(path: Union[str, FilePath], text: str) -> None:
    """
    Write ``text`` to ``path`` so a reader sees the old file or the new one.

    The text goes to a temp file in the same directory (same filesystem, so
    the rename is not a copy), is flushed to disk, and then ``os.replace``d
    into place. A crash mid-write leaves the previous file intact instead of
    a truncated one. ``newline=''`` leaves line endings alone on every platform, so
    a sidecar's bytes do not depend on the OS that wrote it.
    """
    path = FilePath(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:  # pragma: no cover - fsync unsupported on this fs
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def graph_digest(graph: ISONGraph) -> str:
    """
    Content digest of a graph's nodes and edges: ``"sha256:<hex>"``.

    Hashes a canonical JSON rendering (``GRAPH_DIGEST_BASIS``): the
    ``directed`` flag, every node as ``[type, id-kind, id, properties]`` and
    every edge as ``[rel_type, source, target, properties]``, each list sorted
    and every object key-sorted. The same graph therefore hashes the same
    regardless of insertion order. The graph *name* is excluded (it is derived
    from the file name on load), and so are embeddings.
    """
    def ref_key(ref: NodeRef) -> list:
        return [ref[0], 'i' if isinstance(ref[1], int) and not isinstance(ref[1], bool)
                else 's', str(ref[1])]

    def dump(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, default=str)

    nodes = sorted(dump([ref_key(n.ref), n.properties]) for n in graph.nodes())
    edges = sorted(
        dump([e.rel_type, ref_key(e.source), ref_key(e.target), e.properties])
        for e in graph.edges()
    )
    body = '{"directed":%s,"edges":[%s],"nodes":[%s]}' % (
        'true' if graph.directed else 'false', ','.join(edges), ','.join(nodes))
    return "sha256:" + hashlib.sha256(body.encode('utf-8')).hexdigest()


# =============================================================================
# Embedding Encoder Interface
# =============================================================================

class EmbeddingEncoder(ABC):
    """
    Abstract base class for embedding encoders.

    Encoders may expose two optional pieces of metadata the store records:
    ``model_name`` (stamped on every vector it writes; decides whether stored
    vectors are *stale* for this encoder) and ``version`` (a free-form string,
    e.g. a model revision or weights hash, written to the sidecar).
    """

    #: Optional free-form encoder/model version; ``None`` when unknown.
    version: Optional[str] = None

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embedding vectors."""
        pass


class SentenceTransformerEncoder(EmbeddingEncoder):
    """
    Embedding encoder using sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dimensions) - small, fast, and good
    enough for the graph sizes this package is built for. Any other
    sentence-transformers model works; pass its name to the constructor.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 version: Optional[str] = None):
        """
        Initialize encoder with a sentence-transformers model.

        Args:
            model_name: HuggingFace model name
            version: Optional free-form revision string recorded in the sidecar
        """
        self.version = version
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> List[float]:
        """Encode single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts efficiently."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class MockEncoder(EmbeddingEncoder):
    """
    Deterministic encoder with no dependencies, for tests and offline work.

    The same text produces the same vector in *every* language port
    (Python/TypeScript/JavaScript/Rust/C++), so a store written by one can be
    searched by another: a 32-bit FNV-1a hash over the UTF-8 bytes seeds an
    xorshift32 generator, and the top 24 bits of each state become a value in
    [-1, 1).

    This replaces an earlier scheme that emitted the *low* 16 bits of an LCG.
    An LCG's low bits evolve independently of its high bits, so every vector
    was determined by `seed mod 65536` and distinct texts collided outright -
    280 collisions in 5,000 texts in the exact-arithmetic ports (Rust/C++),
    with colliding texts scoring a perfect 1.0 against each other. The Python
    port had a different flaw: 384 dimensions built from just 24 distinct
    values repeated 16 times.

    These are pseudo-embeddings with no semantic structure. Use them to test
    plumbing, never to judge relevance quality.
    """

    _FNV_OFFSET_BASIS = 0x811C9DC5
    _FNV_PRIME = 0x01000193
    _MASK32 = 0xFFFFFFFF

    def __init__(self, dimension: int = 384, version: Optional[str] = None):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self.model_name = f"mock-{dimension}"
        self.version = version

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> List[float]:
        """Generate a deterministic pseudo-embedding."""
        state = self._FNV_OFFSET_BASIS
        for byte in text.encode('utf-8'):
            state = ((state ^ byte) * self._FNV_PRIME) & self._MASK32
        # xorshift32 has no way out of a zero state; steer off it.
        if state == 0:
            state = 0x9E3779B9

        values: List[float] = []
        for _ in range(self._dimension):
            state ^= (state << 13) & self._MASK32
            state ^= state >> 17
            state ^= (state << 5) & self._MASK32
            state &= self._MASK32
            values.append((state >> 8) / 8388608.0 - 1.0)  # top 24 bits
        return values

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


# =============================================================================
# Embedding Store
# =============================================================================

@dataclass
class EmbeddingRecord:
    """A stored embedding with metadata."""
    id: str
    node_ref: NodeRef
    vector: List[float]
    text: str
    model: str = "all-MiniLM-L6-v2"


@dataclass
class SimilarityResult:
    """Result of similarity search."""
    node_ref: NodeRef
    score: float
    text: str

    def __repr__(self) -> str:
        return f"SimilarityResult(:{self.node_ref[0]}:{self.node_ref[1]}, score={self.score:.4f})"


class _ScanCache:
    """
    Decoded, incrementally maintained snapshot of the whole table.

    Every search reads every row (brute force), so decoding blobs and, with
    numpy, stacking them into one matrix pays for itself immediately on the
    second query: the per-query cost drops from a Python loop over N x d
    floats to a single matrix-vector product.

    With numpy the snapshot is one C-contiguous float32 matrix with spare row
    capacity, the vector norms, a type code per row and a liveness flag. Writes
    patch it in place instead of invalidating it: an insert appends a row, an
    update overwrites one, a delete tombstones one (compacted once a quarter
    of the rows are dead). Rows stay in insertion order, which is the order
    the table's rowids give, so ties break exactly as a rebuild would.

    ``n`` counts used rows including tombstones, ``dead`` the tombstones.
    Without numpy (or with ragged vector lengths) ``matrix`` is None and
    ``vectors``/``types`` hold plain lists; that form is rebuilt, not patched.
    """

    __slots__ = ('refs', 'types', 'texts', 'vectors', 'matrix', 'norms',
                 'n', 'dead', 'alive', 'codes', 'code_of', 'pos')

    def __init__(self, refs, types, texts, vectors, matrix, norms):
        self.refs = refs
        self.types = types
        self.texts = texts
        self.vectors = vectors
        self.matrix = matrix
        self.norms = norms
        self.n = len(refs)
        self.dead = 0
        self.alive = None
        self.codes = None
        self.code_of = None
        self.pos = None


class EmbeddingStore:
    """
    SQLite-backed embedding storage with similarity search.

    Brute-force cosine similarity, vectorized through numpy when it is
    installed and falling back to a pure-Python loop when it is not. There is
    no ANN index: this is built for an application's own knowledge graph, not
    as a vector-database replacement.

    Thread-safe: every operation is serialized on a re-entrant lock, and the
    connection is opened with ``check_same_thread=False`` so a web request
    handled on a different thread than the one that built the store still
    works. Both halves are required. With the flag but no lock, a shared
    ``sqlite3.Connection`` under concurrent use raises
    ``sqlite3.InterfaceError``, hands back ``None`` rows mid-query, and drops
    roughly 0.5% of writes outright - measured at 8 threads.
    """

    def __init__(
        self,
        db_path: Union[str, FilePath] = ":memory:",
        encoder: Optional[EmbeddingEncoder] = None,
        dimension: Optional[int] = None,
        sqlite_vec: bool = False,
        version: Optional[str] = None,
        on_model_mismatch: str = "warn"
    ):
        """
        Initialize embedding store.

        Args:
            db_path: SQLite database path (":memory:" for in-memory)
            encoder: Embedding encoder (default: SentenceTransformerEncoder)
            dimension: Expected vector length. Inferred from the first vector
                       stored (or from an existing database) when omitted.
            sqlite_vec: Answer top-k searches through a sqlite-vec `vec0`
                 virtual table instead of scanning every row from Python.
                 Opt-in, and needs the `sqlite-vec` package (the `vec`
                 extra). Results are the same: vec0 runs an exact
                 brute-force KNN in C, so this is a speed change and not an
                 accuracy trade - measured identical top-10 ordering, scores
                 within 2e-07 of the scan, and about 1.4x faster over
                 20,000 vectors.
            version: Free-form version string for the encoder/model that made
                 these vectors, recorded in the sidecar. Falls back to the
                 encoder's own ``version`` attribute when omitted.
            on_model_mismatch: What a *text* query does when the encoder's
                 ``model_name`` differs from the model that produced the
                 stored vectors: ``"warn"`` (default; one ``UserWarning`` per
                 distinct pair), ``"error"`` (raise
                 :class:`ModelMismatchError`) or ``"ignore"``. Vectors whose
                 model is ``"unknown"``/``"provided"`` never trigger it.
                 Vector queries are never checked.

        Raises:
            EmbeddingError: sqlite_vec=True but the package is not installed
                            or the extension cannot be loaded
            ValueError: on_model_mismatch is not one of the three modes
        """
        if on_model_mismatch not in ("warn", "error", "ignore"):
            raise ValueError(
                f"on_model_mismatch must be 'warn', 'error' or 'ignore', "
                f"got {on_model_mismatch!r}")
        self.db_path = str(db_path)
        self.encoder = encoder
        self._version = version
        self.on_model_mismatch = on_model_mismatch
        self._models_cache: Optional[Dict[str, int]] = None
        self._warned_mismatch: set = set()
        #: Metadata of the most recent imported sidecar (empty before any).
        self.last_import_meta: Dict[str, Any] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._dimension = dimension
        self._cache: Optional[_ScanCache] = None
        self._sqlite_vec = sqlite_vec
        self._vec_ready = False
        with self._lock:
            self._get_conn()
        if self._dimension is None:
            self._adopt_stored_dimension()
        if self._sqlite_vec and self._dimension is not None:
            with self._lock:
                self._ensure_vec_table(self._dimension)

    @property
    def sqlite_vec(self) -> bool:
        """Whether top-k search is answered from the sqlite-vec index."""
        return self._sqlite_vec

    @property
    def version(self) -> Optional[str]:
        """Encoder/model version recorded in the sidecar, or None if unknown."""
        if self._version is not None:
            return self._version
        return getattr(self.encoder, 'version', None) if self.encoder else None

    @version.setter
    def version(self, value: Optional[str]) -> None:
        self._version = value

    def _invalidate(self) -> None:
        """Drop everything derived from the table.

        Writes patch the caches in place (see :class:`_ScanCache`); this is
        the fallback for the ones that cannot, and for a write that failed
        part-way and left the derived state unknown.
        """
        self._cache = None
        self._models_cache = None

    def _models_adjust(self, old: Optional[str], new: Optional[str]) -> None:
        """Move one row between model buckets in the cached ``models()`` counts."""
        counts = self._models_cache
        if counts is None:
            return
        if old is not None:
            left = counts.get(old, 0) - 1
            if left > 0:
                counts[old] = left
            else:
                counts.pop(old, None)
        if new is not None:
            counts[new] = counts.get(new, 0) + 1

    # -- schema ---------------------------------------------------------

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the schema on a freshly opened connection.

        Called from _get_conn rather than once at construction, so a store
        that is closed and then used again re-creates what it needs instead
        of failing with "no such table".
        """
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_id_type TEXT NOT NULL DEFAULT 'str',
                vector BLOB NOT NULL,
                text TEXT NOT NULL,
                model TEXT DEFAULT 'all-MiniLM-L6-v2',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node_type, node_id)
            );

            CREATE INDEX IF NOT EXISTS idx_embeddings_node
                ON embeddings(node_type, node_id);
        """)
        # CREATE TABLE IF NOT EXISTS is a no-op against a database written by
        # an older version, which has no node_id_type column - so add it
        # rather than failing on every read.
        columns = {row['name'] for row in
                   conn.execute("PRAGMA table_info(embeddings)").fetchall()}
        if 'node_id_type' not in columns:
            conn.execute(
                "ALTER TABLE embeddings ADD COLUMN "
                "node_id_type TEXT NOT NULL DEFAULT 'str'"
            )
        conn.commit()

    def _adopt_stored_dimension(self) -> None:
        """Pin the dimension to whatever an existing database already holds."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT LENGTH(vector) AS n FROM embeddings LIMIT 1"
            ).fetchone()
            if row is not None and row['n']:
                self._dimension = row['n'] // 4

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection.

        check_same_thread=False because this store is used from async web
        server contexts (e.g. ison-cache's HTTP API), where a request can
        legitimately be handled on a different thread than the one that
        constructed this object. Matches the pattern already used by
        ison-cache's own sqlite backend. Serialization is provided by
        self._lock - the flag alone only removes sqlite3's guard rail.
        """
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            if self._sqlite_vec:
                self._load_vec_extension(conn)
                self._vec_ready = False   # the vec table belongs to this conn
            self._create_schema(conn)
            self._conn = conn
        return self._conn

    @staticmethod
    def _load_vec_extension(conn: sqlite3.Connection) -> None:
        """Load sqlite-vec into a freshly opened connection.

        Extensions are per-connection, so this runs on every open rather than
        once at construction.
        """
        if _sqlite_vec is None:
            raise EmbeddingError(
                "sqlite_vec=True needs the sqlite-vec package: pip install "
                "'isongraph-vector[vec]' (or pip install sqlite-vec)"
            )
        try:
            conn.enable_load_extension(True)
            _sqlite_vec.load(conn)
        except (AttributeError, sqlite3.OperationalError) as exc:
            # Some Python builds ship a sqlite3 compiled without
            # enable_load_extension; there is no way to use vec0 there.
            raise EmbeddingError(
                f"could not load the sqlite-vec extension: {exc}. "
                "This Python build may not support SQLite extensions; "
                "use sqlite_vec=False for the exact scan."
            ) from exc
        finally:
            try:
                conn.enable_load_extension(False)
            except AttributeError:
                pass

    def _ensure_vec_table(self, dimension: int) -> None:
        """Create the vec0 index (and backfill it) once. Caller holds the lock.

        The vector width is part of the DDL, so the table cannot be created
        until the first vector fixes the dimension.
        """
        if not self._sqlite_vec or self._vec_ready:
            return

        conn = self._get_conn()
        # node_type is a vec0 metadata column so a type-filtered search stays
        # exact - filtering after a top-k fetch would silently drop matches.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0("
            f"    node_type text,"
            f"    embedding float[{dimension}] distance_metric=cosine"
            f")"
        )

        indexed = conn.execute(f"SELECT COUNT(*) FROM {VEC_TABLE}").fetchone()[0]
        stored = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if indexed != stored:
            # Opening an existing database with sqlite_vec=True for the first time,
            # or after rows were written with it off.
            conn.execute(f"DELETE FROM {VEC_TABLE}")
            for row in conn.execute(
                "SELECT rowid, node_type, vector FROM embeddings"
            ).fetchall():
                conn.execute(
                    f"INSERT INTO {VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)",
                    (row['rowid'], row['node_type'], row['vector'])
                )
            conn.commit()

        self._vec_ready = True

    # -- vector plumbing ------------------------------------------------

    @property
    def dimension(self) -> Optional[int]:
        """Vector length this store holds, or None while it is still empty."""
        return self._dimension

    def _vector_to_bytes(self, vector: Sequence[float]) -> bytes:
        """Convert vector to bytes for storage."""
        return struct.pack(f'{len(vector)}f', *vector)

    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """Convert bytes back to vector."""
        n = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{n}f', data))

    def _check_dimension(self, vector: Sequence[float], what: str) -> None:
        """Pin the store's dimension, or reject a vector that disagrees."""
        length = len(vector)
        if length == 0:
            raise DimensionMismatchError(f"{what} is empty")
        if self._dimension is None:
            self._dimension = length
        elif length != self._dimension:
            raise DimensionMismatchError(
                f"{what} has {length} dimensions, store holds {self._dimension}"
            )

    def _restore_node_id(self, node_id: str, node_id_type: str) -> Union[int, str]:
        """
        Reconstruct a node id's original Python type from what was stored.

        node_id is always persisted as TEXT (SQLite has no way to key a
        UNIQUE constraint across mixed types cleanly), so the original
        int-vs-str distinction has to be tracked explicitly via
        node_id_type rather than guessed. A prior version of this method
        guessed by trying int(node_id) and catching ValueError - which
        corrupts any string id that merely looks numeric (e.g. "007",
        a zip code or account number with a leading zero: int("007")
        silently becomes 7, and get_node_by_ref(('type', 7)) then fails
        to find the node that was actually stored under ('type', '007')).
        """
        return int(node_id) if node_id_type == 'int' else node_id

    @staticmethod
    def _node_id_type(node_id: Union[int, str]) -> str:
        return 'int' if isinstance(node_id, int) and not isinstance(node_id, bool) else 'str'

    def _cosine_similarity(self, v1: Sequence[float], v2: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors of equal length."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _encode(self, text: str) -> List[float]:
        if self.encoder is None:
            raise NoEncoderError("No encoder provided and no vector given")
        return self.encoder.encode(text)

    def _resolve_query(self, query: Union[str, Sequence[float]]) -> List[float]:
        """Turn a text or vector query into a validated query vector."""
        if isinstance(query, str):
            if self.encoder is None:
                raise NoEncoderError("No encoder provided for text query")
            self._check_model_compat()
            vector = self.encoder.encode(query)
        else:
            vector = list(query)
        if self._dimension is not None and len(vector) != self._dimension:
            raise DimensionMismatchError(
                f"query vector has {len(vector)} dimensions, "
                f"store holds {self._dimension}"
            )
        if not vector:
            raise DimensionMismatchError("query vector is empty")
        return vector

    def _model_name(self) -> str:
        return getattr(self.encoder, 'model_name', 'unknown') if self.encoder else 'provided'

    # -- model lifecycle --------------------------------------------------

    def models(self) -> Dict[str, int]:
        """
        Which models produced the stored vectors: ``{model_name: count}``.

        ``"unknown"`` / ``"provided"`` appear as-is (vectors imported from a
        port that records no model, or handed in with no encoder attached).
        """
        with self._lock:
            if self._models_cache is None:
                rows = self._get_conn().execute(
                    "SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model"
                ).fetchall()
                self._models_cache = {row['model']: row['n'] for row in rows}
            return dict(self._models_cache)

    def _current_model(self) -> Optional[str]:
        name = getattr(self.encoder, 'model_name', None) if self.encoder else None
        return name if name and name not in UNKNOWN_MODELS else None

    def _check_model_compat(self) -> None:
        """Complain when a text query's encoder is not the stored vectors' model."""
        if self.on_model_mismatch == "ignore":
            return
        current = self._current_model()
        if current is None:
            return
        foreign = {m for m in self.models() if m not in UNKNOWN_MODELS and m != current}
        if not foreign:
            return
        message = (
            f"encoder model {current!r} differs from the model that produced the "
            f"stored vectors ({', '.join(sorted(repr(m) for m in foreign))}); "
            f"scores compare vectors from different spaces. Re-embed with "
            f"reembed_stale() or search with the original encoder."
        )
        if self.on_model_mismatch == "error":
            raise ModelMismatchError(message)
        key = (current, frozenset(foreign))
        if key not in self._warned_mismatch:
            self._warned_mismatch.add(key)
            warnings.warn(message, UserWarning, stacklevel=4)

    def stale(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        include_unknown: bool = False
    ) -> List[NodeRef]:
        """
        Node refs whose stored vector does not belong to the current encoder.

        A vector is stale when its recorded model differs from ``model``
        (default: the encoder's ``model_name``) or its length differs from
        ``dimension`` (default: the encoder's ``dimension``). Vectors whose
        model is ``"unknown"``/``"provided"`` cannot be judged by model and are
        skipped unless ``include_unknown`` is True (a dimension mismatch still
        counts). With no encoder and no arguments nothing can be judged and
        the result is empty.
        """
        if model is None:
            model = self._current_model()
        if dimension is None and self.encoder is not None:
            try:
                dimension = int(self.encoder.dimension)
            except (AttributeError, TypeError):
                dimension = None
        with self._lock:
            rows = self._get_conn().execute("""
                SELECT node_type, node_id, node_id_type, model, LENGTH(vector) AS n
                FROM embeddings ORDER BY rowid
            """).fetchall()
            out: List[NodeRef] = []
            for row in rows:
                by_model = (
                    model is not None and row['model'] != model
                    and (include_unknown or row['model'] not in UNKNOWN_MODELS)
                )
                by_dim = dimension is not None and row['n'] // 4 != dimension
                if by_model or by_dim:
                    out.append((row['node_type'],
                                self._restore_node_id(row['node_id'], row['node_id_type'])))
            return out

    def _reset_dimension(self) -> None:
        """Forget the pinned dimension (and the vec0 table). Caller cleared rows."""
        self._dimension = None
        if self._sqlite_vec:
            conn = self._get_conn()
            conn.execute(f"DROP TABLE IF EXISTS {VEC_TABLE}")
            conn.commit()
            self._vec_ready = False

    # -- writes ---------------------------------------------------------

    def _write_row(
        self,
        conn: sqlite3.Connection,
        node_ref: NodeRef,
        vector: Sequence[float],
        text: str,
        model: str,
        node_id_type: Optional[str] = None
    ) -> str:
        """
        Insert or update one embedding, keeping the vec0 index in step.

        An UPSERT rather than INSERT OR REPLACE: REPLACE deletes and re-inserts
        the row, which assigns a *new* rowid and would orphan the matching
        vec0 entry (and reset created_at). The `id` column is a pure function
        of (node_type, node_id), so a conflict on one implies a conflict on the
        other and this single conflict target catches both.

        Caller holds the lock and commits.
        """
        emb_id = self._embedding_id(node_ref)
        blob = self._vector_to_bytes(vector)
        old_model = None
        if self._models_cache is not None:
            previous = conn.execute(
                "SELECT model FROM embeddings WHERE node_type = ? AND node_id = ?",
                (node_ref[0], str(node_ref[1]))
            ).fetchone()
            old_model = previous['model'] if previous is not None else None
        conn.execute("""
            INSERT INTO embeddings
            (id, node_type, node_id, node_id_type, vector, text, model)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_type, node_id) DO UPDATE SET
                node_id_type = excluded.node_id_type,
                vector = excluded.vector,
                text = excluded.text,
                model = excluded.model
        """, (
            emb_id, node_ref[0], str(node_ref[1]),
            node_id_type or self._node_id_type(node_ref[1]),
            blob, text, model
        ))

        if self._sqlite_vec:
            self._ensure_vec_table(len(vector))
            row = conn.execute(
                "SELECT rowid FROM embeddings WHERE node_type = ? AND node_id = ?",
                (node_ref[0], str(node_ref[1]))
            ).fetchone()
            rowid = row['rowid']
            # vec0 has no upsert; replace the entry at this rowid.
            conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (rowid,))
            conn.execute(
                f"INSERT INTO {VEC_TABLE}(rowid, node_type, embedding) VALUES (?, ?, ?)",
                (rowid, node_ref[0], blob)
            )

        # Patch what is derived from the table rather than dropping it.
        self._models_adjust(old_model, model)
        self._cache_put(node_ref, node_id_type or self._node_id_type(node_ref[1]), blob, text)
        return emb_id

    def add(
        self,
        node_ref: NodeRef,
        text: str,
        vector: Optional[Sequence[float]] = None
    ) -> str:
        """
        Add embedding for a node.

        Args:
            node_ref: Node reference (type, id)
            text: Text to embed (or the source text, if vector is provided)
            vector: Pre-computed vector (optional, will encode if not provided)

        Returns:
            Embedding ID

        Raises:
            NoEncoderError: text given with no encoder and no vector
            DimensionMismatchError: vector length disagrees with the store
        """
        with self._lock:
            if vector is None:
                vector = self._encode(text)
            self._check_dimension(vector, "vector")

            conn = self._get_conn()
            try:
                emb_id = self._write_row(conn, node_ref, vector, text, self._model_name())
                conn.commit()
            except BaseException:
                self._invalidate()
                raise
            return emb_id

    def add_batch(
        self,
        entries: Iterable[Union[Tuple[NodeRef, str], Tuple[NodeRef, str, Sequence[float]]]]
    ) -> int:
        """
        Add many embeddings, encoding the texts in one batch.

        A real transformer encoder is far faster per text on a batch than on
        the same texts one at a time, and this commits once rather than once
        per row.

        Args:
            entries: (node_ref, text) or (node_ref, text, vector) tuples

        Returns:
            Number of embeddings written
        """
        items = [tuple(e) for e in entries]
        if not items:
            return 0

        with self._lock:
            needs_encoding = [i for i, e in enumerate(items) if len(e) < 3 or e[2] is None]
            if needs_encoding:
                if self.encoder is None:
                    raise NoEncoderError("No encoder provided and no vectors given")
                encoded = self.encoder.encode_batch([items[i][1] for i in needs_encoding])
                if len(encoded) != len(needs_encoding):
                    raise EmbeddingError(
                        f"encoder returned {len(encoded)} vectors for "
                        f"{len(needs_encoding)} texts"
                    )
                for slot, vector in zip(needs_encoding, encoded):
                    ref, text = items[slot][0], items[slot][1]
                    items[slot] = (ref, text, vector)

            model = self._model_name()
            for ref, text, vector in items:
                self._check_dimension(vector, "vector")

            conn = self._get_conn()
            try:
                for ref, text, vector in items:
                    self._write_row(conn, ref, vector, text, model)
                conn.commit()
            except BaseException:
                self._invalidate()
                raise
            return len(items)

    @staticmethod
    def _embedding_id(node_ref: NodeRef) -> str:
        return hashlib.md5(f"{node_ref[0]}:{node_ref[1]}".encode()).hexdigest()[:16]

    def remove(self, node_ref: NodeRef) -> bool:
        """Remove embedding for a node."""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT rowid, model FROM embeddings WHERE node_type = ? AND node_id = ?",
                    (node_ref[0], str(node_ref[1]))
                ).fetchone()
                if row is not None and self._sqlite_vec and self._vec_ready:
                    conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (row['rowid'],))
                cursor = conn.execute("""
                    DELETE FROM embeddings
                    WHERE node_type = ? AND node_id = ?
                """, (node_ref[0], str(node_ref[1])))
                conn.commit()
            except BaseException:
                self._invalidate()
                raise
            if cursor.rowcount > 0:
                if row is not None:
                    self._models_adjust(row['model'], None)
                self._cache_drop(node_ref)
            return cursor.rowcount > 0

    def clear(self, reset_dimension: bool = False) -> None:
        """
        Clear all embeddings.

        Args:
            reset_dimension: Also forget the pinned vector length, so the next
                vector may have a different one (switching to an encoder of a
                different dimension). Off by default: a store normally refuses
                vectors of the wrong length even when empty.
        """
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM embeddings")
            if self._sqlite_vec and self._vec_ready:
                conn.execute(f"DELETE FROM {VEC_TABLE}")
            conn.commit()
            self._invalidate()
            if reset_dimension:
                self._reset_dimension()

    # -- reads ----------------------------------------------------------

    def get(self, node_ref: NodeRef) -> Optional[EmbeddingRecord]:
        """Get embedding for a node."""
        with self._lock:
            row = self._get_conn().execute("""
                SELECT id, node_type, node_id, node_id_type, vector, text, model
                FROM embeddings
                WHERE node_type = ? AND node_id = ?
            """, (node_ref[0], str(node_ref[1]))).fetchone()

            if row is None:
                return None

            return EmbeddingRecord(
                id=row['id'],
                node_ref=(row['node_type'],
                          self._restore_node_id(row['node_id'], row['node_id_type'])),
                vector=self._bytes_to_vector(row['vector']),
                text=row['text'],
                model=row['model']
            )

    def count(self) -> int:
        """Count total embeddings."""
        with self._lock:
            return self._get_conn().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    # -- incremental scan cache -------------------------------------------

    _CACHE_CHUNK = 8192

    def _ensure_cache(self) -> _ScanCache:
        """Build (or reuse) the decoded snapshot every search reads. Locked."""
        if self._cache is not None:
            return self._cache

        cursor = self._get_conn().cursor()
        cursor.row_factory = None          # plain tuples: no Row overhead per field
        query = ("SELECT node_type, node_id, node_id_type, vector, text "
                 "FROM embeddings ORDER BY rowid")
        cache = None
        if _np is not None:
            total = cursor.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if total:
                cache = self._build_numpy_cache(cursor.execute(query), total)
        if cache is None:
            cache = self._build_list_cache(cursor.execute(query).fetchall())
        self._cache = cache
        return cache

    def _build_numpy_cache(self, cursor, total: int) -> Optional[_ScanCache]:
        """Stream the table into a spare-capacity float32 matrix.

        Returns None when the stored vectors are not all one length (the
        caller then falls back to the list form, which tolerates that).
        """
        rows = cursor.fetchmany(self._CACHE_CHUNK)
        if not rows:
            return None
        width = len(rows[0][3])
        if width == 0 or width % 4:
            return None
        dim = width // 4
        capacity = total + total // 8 + 16
        matrix = _np.empty((capacity, dim), dtype=_np.float32)
        norms = _np.zeros(capacity, dtype=_np.float32)
        codes = _np.zeros(capacity, dtype=_np.int32)
        alive = _np.zeros(capacity, dtype=bool)
        refs: List[Optional[NodeRef]] = []
        texts: List[Optional[str]] = []
        pos: Dict[Tuple[str, str], int] = {}
        code_of: Dict[str, int] = {}
        filled = 0
        while rows:
            m = len(rows)
            if filled + m > capacity:
                return None
            types, ids, id_types, blobs, row_texts = zip(*rows)      # columns, in C
            if set(map(len, blobs)) != {width}:
                return None
            block = _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(m, dim)
            matrix[filled:filled + m] = block
            norms[filled:filled + m] = _np.linalg.norm(block, axis=1)
            alive[filled:filled + m] = True
            for t in set(types):
                if t not in code_of:
                    code_of[t] = len(code_of)
            codes[filled:filled + m] = [code_of[t] for t in types]
            keys = list(zip(types, ids))
            if 'int' in id_types:
                refs.extend([(t, self._restore_node_id(i, k))
                             for t, i, k in zip(types, ids, id_types)])
            else:
                refs.extend(keys)           # the key tuple doubles as the ref
            pos.update(zip(keys, range(filled, filled + m)))
            texts.extend(row_texts)
            filled += m
            rows = cursor.fetchmany(self._CACHE_CHUNK)
        cache = _ScanCache(refs, None, texts, None, matrix, norms)
        cache.alive = alive
        cache.codes = codes
        cache.code_of = code_of
        cache.pos = pos
        return cache

    def _build_list_cache(self, rows) -> _ScanCache:
        """Pure-Python form: no numpy, or vectors of mixed length."""
        refs = [(r[0], self._restore_node_id(r[1], r[2])) for r in rows]
        return _ScanCache(
            refs, [r[0] for r in rows], [r[4] for r in rows],
            [self._bytes_to_vector(r[3]) for r in rows], None, None)

    @staticmethod
    def _cache_grow(cache: _ScanCache) -> None:
        """Make room for more rows: grow capacity by a quarter (min 1024)."""
        capacity, dim = cache.matrix.shape
        new_capacity = capacity + max(1024, capacity // 4)
        n = cache.n
        matrix = _np.empty((new_capacity, dim), dtype=_np.float32)
        matrix[:n] = cache.matrix[:n]
        norms = _np.zeros(new_capacity, dtype=_np.float32)
        norms[:n] = cache.norms[:n]
        codes = _np.zeros(new_capacity, dtype=_np.int32)
        codes[:n] = cache.codes[:n]
        alive = _np.zeros(new_capacity, dtype=bool)
        alive[:n] = cache.alive[:n]
        cache.matrix, cache.norms, cache.codes, cache.alive = matrix, norms, codes, alive

    def _cache_put(self, node_ref: NodeRef, id_type: str, blob: bytes, text: str) -> None:
        """Mirror one upserted row into the cache. Caller holds the lock."""
        cache = self._cache
        if cache is None:
            return
        matrix = cache.matrix
        if matrix is None or len(blob) != matrix.shape[1] * 4:
            self._cache = None             # list form / new width: rebuild lazily
            return
        node_type = str(node_ref[0])
        id_text = str(node_ref[1])
        key = (node_type, id_text)
        ref = key if id_type != 'int' else (node_type, int(id_text))
        row = _np.frombuffer(blob, dtype=_np.float32)
        i = cache.pos.get(key)
        if i is None:
            if cache.n >= matrix.shape[0]:
                self._cache_grow(cache)
                matrix = cache.matrix
            i = cache.n
            cache.n = i + 1
            cache.refs.append(ref)
            cache.texts.append(text)
            cache.pos[key] = i
            code = cache.code_of.get(node_type)
            if code is None:
                code = cache.code_of[node_type] = len(cache.code_of)
            cache.codes[i] = code
            cache.alive[i] = True
        else:                              # overwrite in place: keeps its position
            cache.refs[i] = ref
            cache.texts[i] = text
        matrix[i] = row
        cache.norms[i] = _np.linalg.norm(row.reshape(1, -1), axis=1)[0]

    def _cache_drop(self, node_ref: NodeRef) -> None:
        """Mirror one deleted row into the cache. Caller holds the lock."""
        cache = self._cache
        if cache is None:
            return
        if cache.matrix is None:
            self._cache = None
            return
        i = cache.pos.pop((str(node_ref[0]), str(node_ref[1])), None)
        if i is None:
            return
        cache.matrix[i] = 0.0
        cache.norms[i] = 0.0
        cache.alive[i] = False
        cache.refs[i] = None
        cache.texts[i] = None
        cache.dead += 1
        while cache.n and not cache.alive[cache.n - 1]:    # trailing tombstones
            cache.n -= 1
            cache.refs.pop()
            cache.texts.pop()
            cache.dead -= 1
        if cache.dead > 32 and cache.dead * 4 > cache.n:
            self._cache_compact(cache)

    @staticmethod
    def _cache_compact(cache: _ScanCache) -> None:
        """Squeeze tombstones out, keeping the live rows in their order."""
        live = _np.flatnonzero(cache.alive[:cache.n])
        step = 16384
        for start in range(0, live.size, step):
            idx = live[start:start + step]
            end = start + idx.size
            # Destinations never overtake the sources still to be read.
            cache.matrix[start:end] = cache.matrix[idx]
            cache.norms[start:end] = cache.norms[idx]
            cache.codes[start:end] = cache.codes[idx]
        kept = live.size
        cache.alive[:kept] = True
        cache.alive[kept:cache.n] = False
        order = live.tolist()
        cache.refs = [cache.refs[i] for i in order]
        cache.texts = [cache.texts[i] for i in order]
        cache.pos = {(r[0], str(r[1])): j for j, r in enumerate(cache.refs)}
        cache.n = kept
        cache.dead = 0

    # -- scoring ----------------------------------------------------------

    @staticmethod
    def _np_scores(cache: _ScanCache, query_vector: Sequence[float]):
        """Cosine score of every used row (tombstones score 0), as float32."""
        n = cache.n
        query = _np.asarray(query_vector, dtype=_np.float32)
        query_norm = float(_np.linalg.norm(query))
        scores = _np.zeros(n, dtype=_np.float32)
        if query_norm != 0.0:
            denom = cache.norms[:n] * query_norm
            dots = cache.matrix[:n] @ query
            with _np.errstate(divide='ignore', invalid='ignore'):
                _np.divide(dots, denom, out=scores, where=denom > 0)
        return scores

    def _np_candidates(self, cache, query_vector, node_type, threshold):
        """Scores with every excluded row set to -inf, or None if nothing can match.

        Excluded: tombstones, rows of another ``node_type`` and rows below
        ``threshold`` (compared in double precision, as the scalar loop did).
        """
        if cache.n - cache.dead == 0:
            return None
        scores = self._np_scores(cache, query_vector)
        valid = scores.astype(_np.float64) >= threshold
        if cache.dead:
            valid &= cache.alive[:cache.n]
        if node_type is not None:
            code = cache.code_of.get(node_type)
            if code is None:
                return None
            valid &= cache.codes[:cache.n] == code
        _np.putmask(scores, ~valid, -_np.inf)
        return scores

    @staticmethod
    def _rank(scores, k: Optional[int]):
        """Row positions of the best ``k`` finite scores, best first.

        Ties keep row (insertion) order, exactly what a stable sort of the
        rows would give. ``k=None`` ranks every finite score.
        """
        n = scores.shape[0]
        if k is not None and k < n:
            kth = _np.partition(scores, n - k)[n - k]
            sel = _np.flatnonzero(scores >= kth if kth > -_np.inf else scores > -_np.inf)
        else:
            sel = _np.flatnonzero(scores > -_np.inf)
        if sel.size == 0:
            return sel
        order = _np.argsort(-scores[sel], kind='stable')
        ranked = sel[order]
        return ranked if k is None else ranked[:k]

    def _np_search(self, cache, query_vector, top_k, node_type, threshold):
        if top_k is not None and top_k <= 0:
            return []
        scores = self._np_candidates(cache, query_vector, node_type, threshold)
        if scores is None:
            return []
        best = self._rank(scores, top_k)
        refs, texts = cache.refs, cache.texts
        return [SimilarityResult(node_ref=refs[i], score=s, text=texts[i])
                for i, s in zip(best.tolist(), scores[best].tolist())]

    def _score_all(
        self,
        query_vector: Sequence[float],
        node_type: Optional[str] = None
    ) -> List[Tuple[NodeRef, float, str]]:
        """Cosine-score every stored embedding against a query vector.

        The general (and fallback) form. The numpy search paths do not go
        through it: they never build a tuple per stored row.
        """
        cache = self._ensure_cache()
        if cache.n - cache.dead == 0:
            return []

        if cache.matrix is not None:
            scores = self._np_scores(cache, query_vector).tolist()
            code = None
            if node_type is not None:
                code = cache.code_of.get(node_type)
                if code is None:
                    return []
            codes = cache.codes
            return [(ref, scores[i], cache.texts[i])
                    for i, ref in enumerate(cache.refs)
                    if ref is not None and (code is None or codes[i] == code)]

        scores = [self._cosine_similarity(query_vector, v) for v in cache.vectors]
        return [(ref, float(scores[i]), cache.texts[i])
                for i, ref in enumerate(cache.refs)
                if node_type is None or cache.types[i] == node_type]

    def _ranked(
        self,
        query: Union[str, Sequence[float]],
        node_type: Optional[str] = None,
        threshold: float = 0.0,
        hint: Optional[int] = None
    ):
        """Every match, best first, as an iterator that ranks lazily.

        For callers that apply a predicate before taking k (``node_filter``):
        the work done is proportional to how deep they read, not to the size
        of the store. Everything is snapshotted under the lock, so the caller
        may run arbitrary code between ``next()`` calls with the lock free.
        """
        with self._lock:
            query_vector = self._resolve_query(query)
            cache = self._ensure_cache()
            if cache.matrix is None:
                results = [SimilarityResult(node_ref=ref, score=score, text=text)
                           for ref, score, text in self._score_all(query_vector, node_type)
                           if score >= threshold]
                results.sort(key=lambda x: x.score, reverse=True)
                return iter(results)
            scores = self._np_candidates(cache, query_vector, node_type, threshold)
            if scores is None:
                return iter(())
            refs = cache.refs[:]
            texts = cache.texts[:]

        def chunks():
            k = max(64, 4 * hint) if hint else 64
            done = 0
            while True:
                best = self._rank(scores, k)
                for i in best[done:].tolist():
                    yield SimilarityResult(
                        node_ref=refs[i], score=float(scores[i]), text=texts[i])
                if best.size < k:
                    return
                done = best.size
                k *= 8

        return chunks()

    def _sqlite_vec_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        node_type: Optional[str],
        threshold: float
    ) -> List[SimilarityResult]:
        """Top-k from the vec0 index. Caller holds the lock.

        vec0 is configured with distance_metric=cosine, so its distance is
        1 - cosine similarity and the scores line up with the exact scan.
        """
        conn = self._get_conn()
        self._ensure_vec_table(len(query_vector))
        blob = self._vector_to_bytes(query_vector)

        select = (
            f"SELECT e.node_type AS node_type, e.node_id AS node_id, "
            f"       e.node_id_type AS node_id_type, e.text AS text, "
            f"       v.distance AS distance "
            f"FROM {VEC_TABLE} v JOIN embeddings e ON e.rowid = v.rowid "
        )
        if node_type is None:
            rows = conn.execute(
                select + "WHERE v.embedding MATCH ? AND k = ?", (blob, top_k)
            ).fetchall()
        else:
            rows = conn.execute(
                select + "WHERE v.node_type = ? AND v.embedding MATCH ? AND k = ?",
                (node_type, blob, top_k)
            ).fetchall()

        # Rows arrive ordered by distance ascending, i.e. score descending.
        results = []
        for row in rows:
            score = 1.0 - float(row['distance'])
            if score >= threshold:
                results.append(SimilarityResult(
                    node_ref=(row['node_type'],
                              self._restore_node_id(row['node_id'], row['node_id_type'])),
                    score=score,
                    text=row['text']
                ))
        return results

    def similarity_search(
        self,
        query: Union[str, Sequence[float]],
        top_k: Optional[int] = 10,
        node_type: Optional[str] = None,
        threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """
        Find most similar nodes by embedding.

        With sqlite_vec=True and a bounded top_k this is answered by the vec0
        virtual table; otherwise every stored vector is scored here. Asking for
        every match (top_k=None) always uses the scan - there is nothing for an
        index to narrow down.

        With numpy the scan is one matrix-vector product over a cached float32
        matrix, a partition to find the k best, and a result object for each
        of those k only. Ties between equal scores come back in insertion
        order.

        Args:
            query: Query text or vector
            top_k: Number of results to return (None returns every match)
            node_type: Filter by node type (optional)
            threshold: Minimum similarity score

        Returns:
            List of SimilarityResult sorted by score descending
        """
        with self._lock:
            query_vector = self._resolve_query(query)

            if self._sqlite_vec and top_k is not None and top_k > 0:
                # A zero vector has no direction for cosine to measure; vec0
                # returns NaN there, while the exact scan defines it as 0.0.
                if any(query_vector):
                    return self._sqlite_vec_search(query_vector, top_k, node_type, threshold)

            cache = self._ensure_cache()
            if cache.matrix is not None:
                return self._np_search(cache, query_vector, top_k, node_type, threshold)
            scored = self._score_all(query_vector, node_type)

        results = [
            SimilarityResult(node_ref=ref, score=score, text=text)
            for ref, score, text in scored
            if score >= threshold
        ]
        if top_k is None:
            results.sort(key=lambda x: x.score, reverse=True)
            return results

        # Selecting the top k rather than sorting all of them: nlargest is
        # O(n log k) and returns exactly what sorted(...)[:k] would, ties
        # included.
        return heapq.nlargest(top_k, results, key=lambda x: x.score)

    def score_map(
        self,
        query: Union[str, Sequence[float]],
        node_type: Optional[str] = None
    ) -> Dict[NodeRef, float]:
        """
        Cosine score for *every* embedded node, keyed by node ref.

        One scan serving both the seed selection and the per-node relevance
        that :meth:`SemanticGraph.semantic_multi_hop` blends in, instead of
        two.

        Always scans, even with sqlite_vec=True: the index answers top-k
        queries, and blending needs a score for every node. This is why
        semantic_multi_hop only takes the index path when blend is 0.
        """
        with self._lock:
            query_vector = self._resolve_query(query)
            cache = self._ensure_cache()
            if cache.matrix is None:
                return {ref: score for ref, score, _ in
                        self._score_all(query_vector, node_type)}
            if cache.n - cache.dead == 0:
                return {}
            scores = self._np_scores(cache, query_vector)
            if not cache.dead and node_type is None:
                return dict(zip(cache.refs, scores.tolist()))
            keep = cache.alive[:cache.n].copy()
            if node_type is not None:
                code = cache.code_of.get(node_type)
                if code is None:
                    return {}
                keep &= cache.codes[:cache.n] == code
            idx = _np.flatnonzero(keep)
            refs = cache.refs
            return {refs[i]: s for i, s in zip(idx.tolist(), scores[idx].tolist())}

    # -- portability ----------------------------------------------------

    def export_embeddings(self, graph_digest: Optional[str] = None) -> Dict[str, Any]:
        """
        Serialize every embedding into a portable, cross-language dict.

        The shape is shared by all five language ports, so a store built in
        Python can be loaded by the TypeScript, JavaScript, Rust or C++ one.

        Version 1 of the payload is unchanged; the optional metadata below is
        appended *after* ``embeddings`` as flat top-level fields, which readers
        that do not know them ignore: ``encoder_model`` (the encoder's
        ``model_name``), ``encoder_version`` (the store/encoder ``version``),
        and, when ``graph_digest`` is given, ``graph_digest`` plus
        ``graph_digest_basis``. Absent values are omitted, not written as null.
        """
        with self._lock:
            rows = self._get_conn().execute("""
                SELECT node_type, node_id, node_id_type, vector, text, model
                FROM embeddings ORDER BY rowid
            """).fetchall()

            embeddings = [{
                "type": row['node_type'],
                "id": self._restore_node_id(row['node_id'], row['node_id_type']),
                "id_type": row['node_id_type'],
                "text": row['text'],
                "model": row['model'],
                "vector": self._bytes_to_vector(row['vector']),
            } for row in rows]

            payload: Dict[str, Any] = {
                "format": EMBEDDING_FORMAT,
                "version": EMBEDDING_FORMAT_VERSION,
                "dimension": self._dimension,
                "count": len(embeddings),
                "embeddings": embeddings,
            }
            encoder_model = self._current_model()
            if encoder_model is not None:
                payload["encoder_model"] = encoder_model
            if self.version is not None:
                payload["encoder_version"] = str(self.version)
            if graph_digest is not None:
                payload["graph_digest"] = graph_digest
                payload["graph_digest_basis"] = GRAPH_DIGEST_BASIS
            return payload

    def import_embeddings(self, payload: Dict[str, Any], replace: bool = False) -> int:
        """
        Load embeddings produced by :meth:`export_embeddings`.

        Args:
            payload: Parsed sidecar dict
            replace: Drop existing embeddings first

        Returns:
            Number of embeddings loaded
        """
        if not isinstance(payload, dict) or payload.get("format") != EMBEDDING_FORMAT:
            raise EmbeddingError(
                f"not an {EMBEDDING_FORMAT} payload: "
                f"format={payload.get('format') if isinstance(payload, dict) else type(payload)!r}"
            )
        version = payload.get("version")
        if version != EMBEDDING_FORMAT_VERSION:
            raise EmbeddingError(
                f"unsupported {EMBEDDING_FORMAT} version {version!r} "
                f"(this build reads version {EMBEDDING_FORMAT_VERSION})"
            )

        with self._lock:
            self.last_import_meta = {
                key: payload[key]
                for key in ("encoder_model", "encoder_version",
                            "graph_digest", "graph_digest_basis")
                if key in payload
            }
            if replace:
                self.clear()
                self._dimension = payload.get("dimension")

            written = 0
            conn = self._get_conn()
            try:
                for item in payload.get("embeddings", []):
                    node_id = item["id"]
                    if item.get("id_type") == 'int' and not isinstance(node_id, int):
                        node_id = int(node_id)
                    vector = item["vector"]
                    self._check_dimension(vector, f"embedding for :{item['type']}:{node_id}")
                    ref = (item["type"], node_id)
                    self._write_row(
                        conn, ref, vector,
                        item.get("text", ""), item.get("model", "unknown"),
                        node_id_type=item.get("id_type") or self._node_id_type(node_id)
                    )
                    written += 1
                conn.commit()
            except BaseException:
                self._invalidate()
                raise
            return written

    def save_embeddings(
        self,
        path: Union[str, FilePath],
        graph_digest: Optional[str] = None
    ) -> int:
        """
        Write every embedding to a JSON sidecar, atomically. Returns the count.

        The file is written to a temp file and renamed into place, so a crash
        never leaves a half-written sidecar. ``graph_digest`` ties the sidecar
        to one particular graph (see :func:`graph_digest`).
        """
        payload = self.export_embeddings(graph_digest=graph_digest)
        _atomic_write_text(path, json.dumps(payload))
        return payload["count"]

    def load_embeddings(self, path: Union[str, FilePath], replace: bool = False) -> int:
        """Read a JSON sidecar written by :meth:`save_embeddings`."""
        payload = json.loads(FilePath(path).read_text(encoding='utf-8'))
        return self.import_embeddings(payload, replace=replace)

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
            self._invalidate()

    def __enter__(self) -> 'EmbeddingStore':
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# =============================================================================
# Semantic Graph
# =============================================================================

@dataclass
class SemanticSearchResult:
    """Result of semantic multi-hop search."""
    node_ref: NodeRef
    node: Optional[Node]
    score: float
    hop_count: int
    path: List[NodeRef]

    def __repr__(self) -> str:
        path_str = " -> ".join([f":{n[0]}:{n[1]}" for n in self.path])
        return f"SemanticSearchResult(score={self.score:.4f}, hops={self.hop_count}, path=[{path_str}])"


class SemanticGraph(ISONGraph):
    """
    ISONGraph extended with embedding-based semantic search.

    Combines vector similarity with graph traversal for semantic multi-hop
    queries. Every ISONGraph method still works unchanged.

    Usage:
        graph = SemanticGraph(embedding_db="memory.db")

        # Add nodes with embedding text
        graph.add_node('person', 1, name='Alice', _embed_text='Alice is a software engineer')

        # Or embed manually
        graph.embed_node(('person', 1), 'Alice is a software engineer')

        # Semantic search
        results = graph.similarity_search("engineering", top_k=5)

        # Semantic multi-hop
        results = graph.semantic_multi_hop(
            query="Who are the engineers?",
            rel_type='KNOWS',
            max_hops=2
        )

        # Save graph state and embeddings together
        graph.save("memory.isong")
    """

    def __init__(
        self,
        name: str = "semantic_graph",
        directed: bool = True,
        embedding_db: Union[str, FilePath] = ":memory:",
        encoder: Optional[EmbeddingEncoder] = None,
        auto_embed: bool = True,
        embed_fields: Optional[List[str]] = None,
        sqlite_vec: bool = False,
        version: Optional[str] = None,
        on_model_mismatch: str = "warn"
    ):
        """
        Initialize semantic graph.

        Args:
            name: Graph name
            directed: Whether edges are directed
            embedding_db: Path to embedding database
            encoder: Embedding encoder (default: SentenceTransformerEncoder,
                     constructed lazily on first use)
            auto_embed: Auto-embed nodes on add (using embed_fields)
            embed_fields: Fields to concatenate for auto-embedding
            sqlite_vec: Search through a sqlite-vec index (see :class:`EmbeddingStore`)
            version: Encoder/model version string recorded in the sidecar
                     (default: the encoder's own ``version``, if any)
            on_model_mismatch: ``"warn"`` (default), ``"error"`` or
                     ``"ignore"``: what a text search does when the
                     encoder's model differs from the stored vectors' model
        """
        super().__init__(name=name, directed=directed)

        # Lazy-load encoder to avoid import errors if not needed
        self._encoder = encoder
        self._encoder_loaded = encoder is not None

        self.embedding_store = EmbeddingStore(
            db_path=embedding_db,
            encoder=encoder,
            sqlite_vec=sqlite_vec,
            version=version,
            on_model_mismatch=on_model_mismatch
        )

        self.auto_embed = auto_embed
        self.embed_fields = embed_fields or ['name', 'description', 'content', 'text']

    # -- encoder --------------------------------------------------------

    @property
    def encoder(self) -> Optional[EmbeddingEncoder]:
        """
        The encoder in use, or None if one has not been supplied or built yet.

        Reading this never constructs a model - handy for passing the encoder
        of one graph to :meth:`load` without triggering a download.
        """
        return self._encoder

    def _get_encoder(self) -> EmbeddingEncoder:
        """Lazy-load encoder."""
        if not self._encoder_loaded:
            self._encoder = SentenceTransformerEncoder()
            self._encoder_loaded = True
            self.embedding_store.encoder = self._encoder
        return self._encoder

    def set_encoder(self, encoder: EmbeddingEncoder) -> None:
        """Install an encoder, replacing any current one."""
        self._encoder = encoder
        self._encoder_loaded = True
        self.embedding_store.encoder = encoder

    # -- nodes and embeddings -------------------------------------------

    def add_node(
        self,
        node_type: str,
        node_id: Union[int, str],
        _embed_text: Optional[str] = None,
        **properties: Any
    ) -> Node:
        """
        Add a node, optionally with embedding.

        Args:
            node_type: Node type
            node_id: Node ID
            _embed_text: Text to embed (overrides auto-embed)
            **properties: Node properties

        Returns:
            Created node
        """
        node = super().add_node(node_type, node_id, **properties)

        text = _embed_text
        if text is None and self.auto_embed:
            text = self.embed_text_for(properties)
        if text:
            self.embed_node(node.ref, text)

        return node

    @staticmethod
    def property_text(value: Any) -> str:
        """
        Render one property value as embedding text.

        ISONGraph property values are typed, so a property can be a number, a
        bool or None. Booleans render as ISON spells them - "true"/"false",
        not Python's "True"/"False" - and None contributes nothing, so the
        same graph produces the same text (and so the same vector) in every
        language port.
        """
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float) and value.is_integer() and abs(value) < 1e16:
            # Python is alone in writing an integral float as "1.0"; every
            # other port renders "1". Matching them keeps one property bag
            # mapped to one embedding text, and so to one vector.
            return str(int(value))
        return str(value)

    def embed_text_for(self, properties: Dict[str, Any]) -> str:
        """Concatenate the configured embed_fields present in properties."""
        parts = []
        for field_name in self.embed_fields:
            if field_name not in properties:
                continue
            text = self.property_text(properties[field_name])
            if text:
                parts.append(text)
        return " ".join(parts)

    def embed_node(self, node_ref: NodeRef, text: str) -> str:
        """
        Add or update embedding for a node.

        Args:
            node_ref: Node reference
            text: Text to embed

        Returns:
            Embedding ID
        """
        encoder = self._get_encoder()
        self.embedding_store.encoder = encoder
        return self.embedding_store.add(node_ref, text)

    def embed_nodes(self, items: Iterable[Tuple[NodeRef, str]]) -> int:
        """
        Embed many nodes in a single batched encoder call.

        Args:
            items: (node_ref, text) pairs

        Returns:
            Number of embeddings written
        """
        pairs = [(ref, text) for ref, text in items if text]
        if not pairs:
            return 0
        encoder = self._get_encoder()
        self.embedding_store.encoder = encoder
        return self.embedding_store.add_batch(pairs)

    def get_embedding(self, node_ref: NodeRef) -> Optional[EmbeddingRecord]:
        """Get embedding for a node."""
        return self.embedding_store.get(node_ref)

    def update_node(
        self,
        node_type: str,
        node_id: Union[int, str],
        **properties: Any
    ) -> Node:
        """
        Update node properties, keeping the embedding in step with them.

        With ``auto_embed`` on, a change to any of ``embed_fields`` re-embeds
        the node from its new properties, so searches never rank it by text
        it no longer carries. If the new properties leave nothing to embed,
        the stale vector is dropped. Updates that touch no embed field, and
        graphs with ``auto_embed`` off (whose vectors the caller manages),
        leave the embedding alone.
        """
        node = super().update_node(node_type, node_id, **properties)

        if self.auto_embed and any(f in properties for f in self.embed_fields):
            text = self.embed_text_for(node.properties)
            if text:
                self.embed_node(node.ref, text)
            else:
                self.embedding_store.remove(node.ref)

        return node

    def remove_node(self, node_type: str, node_id: Union[int, str]) -> None:
        """Remove node and its embedding."""
        ref = (node_type, node_id)
        self.embedding_store.remove(ref)
        super().remove_node(node_type, node_id)

    def embed_all_nodes(
        self,
        text_fn: Optional[Callable[[Node], str]] = None,
        skip_existing: bool = False,
        batch_size: int = 256
    ) -> int:
        """
        Embed all nodes in the graph, in batches.

        Args:
            text_fn: Function to generate embed text from node
                     (default: concatenate embed_fields)
            skip_existing: Leave nodes that already have an embedding alone
            batch_size: Texts per encoder call

        Returns:
            Number of nodes embedded
        """
        if text_fn is None:
            def text_fn(node: Node) -> str:  # noqa: F811 - documented default
                return self.embed_text_for(node.properties) or str(node.ref)

        count = 0
        batch: List[Tuple[NodeRef, str]] = []
        for node in self.nodes():
            if skip_existing and self.embedding_store.get(node.ref) is not None:
                continue
            text = text_fn(node)
            if not text:
                continue
            batch.append((node.ref, text))
            if len(batch) >= batch_size:
                count += self.embed_nodes(batch)
                batch = []
        if batch:
            count += self.embed_nodes(batch)
        return count

    # -- search ---------------------------------------------------------

    @staticmethod
    def _edge_weight(edge: Edge, weight_field: str) -> float:
        """
        An edge's numeric ``weight_field`` property clamped to [0, 1].

        A missing, non-numeric (including bool) or NaN value counts as 1.0, so
        an unweighted edge never penalises a hop.
        """
        value = edge.properties.get(weight_field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 1.0
        value = float(value)
        if value != value:  # NaN
            return 1.0
        return min(1.0, max(0.0, value))

    def _adjacent(
        self,
        node_ref: NodeRef,
        rel_type: Optional[str],
        direction: Direction
    ) -> List[Tuple[NodeRef, Edge]]:
        """(neighbor, connecting edge) pairs, in the order ``neighbors()`` yields."""
        pairs: List[Tuple[NodeRef, Edge]] = []
        if direction in (Direction.OUT, Direction.BOTH):
            for edge in self.edges(rel_type, source=node_ref):
                pairs.append((edge.target, edge))
        if direction in (Direction.IN, Direction.BOTH):
            for edge in self.edges(rel_type, target=node_ref):
                pairs.append((edge.source, edge))
        return pairs

    def _node_gate(
        self,
        node_filter: Optional[Callable[[Node], bool]],
        node_types: Optional[Union[str, Sequence[str]]]
    ) -> Optional[Callable[[NodeRef], bool]]:
        """
        Build the ``ref -> bool`` predicate behind ``node_filter``/``node_types``.

        Returns None when neither is set, so unfiltered calls take exactly the
        pre-existing code path. A node passes when its type is in
        ``node_types`` (a bare string means one type; an empty sequence passes
        nothing) *and* ``node_filter(node)`` is truthy. A ref with no node in
        the graph cannot be shown to ``node_filter``, so it fails whenever a
        ``node_filter`` is set. Answers are memoized for the one call.
        """
        if node_filter is None and node_types is None:
            return None
        types = None
        if node_types is not None:
            types = {node_types} if isinstance(node_types, str) else set(node_types)
        memo: Dict[NodeRef, bool] = {}

        def gate(ref: NodeRef) -> bool:
            cached = memo.get(ref)
            if cached is not None:
                return cached
            ok = types is None or ref[0] in types
            if ok and node_filter is not None:
                node = self._resolve_node(ref)
                ok = node is not None and bool(node_filter(node))
            memo[ref] = ok
            return ok

        return gate

    def _seed_search(
        self,
        query: Union[str, Sequence[float]],
        top_k: Optional[int],
        node_type: Optional[str],
        threshold: float,
        gate: Optional[Callable[[NodeRef], bool]]
    ) -> List[SimilarityResult]:
        """
        Top-k similarity search among the nodes that pass ``gate``.

        With a gate the store cannot truncate first: the best k *unfiltered*
        matches may all be filtered out, so every match is scored, filtered,
        and only then cut to k.
        """
        if gate is None:
            return self.embedding_store.similarity_search(query, top_k, node_type, threshold)
        ranked = self.embedding_store._ranked(query, node_type, threshold, hint=top_k)
        limit = None if top_k is None else max(top_k, 0)
        results: List[SimilarityResult] = []
        if limit == 0:
            return results
        # Best first, and stop as soon as k have passed: the gate (a user
        # callback that may touch the graph) runs on the matches that are
        # needed, not on every stored vector.
        for result in ranked:
            if gate(result.node_ref):
                results.append(result)
                if limit is not None and len(results) >= limit:
                    break
        return results

    def similarity_search(
        self,
        query: Union[str, Sequence[float]],
        top_k: Optional[int] = 10,
        node_type: Optional[str] = None,
        threshold: float = 0.0,
        node_filter: Optional[Callable[[Node], bool]] = None,
        node_types: Optional[Sequence[str]] = None
    ) -> List[SimilarityResult]:
        """
        Find similar nodes by embedding.

        Args:
            query: Query text (or a pre-computed vector)
            top_k: Number of results (None returns every match)
            node_type: Filter by type
            threshold: Minimum similarity
            node_filter: Keep only nodes for which ``node_filter(node)`` is
                truthy. Applied *before* ``top_k`` is taken, so you still get
                ``top_k`` matching nodes when that many exist. An embedding
                whose node is not in the graph fails the filter.
            node_types: Keep only nodes whose type is in this sequence
                (combines with ``node_type`` and ``node_filter`` by AND).

        Returns:
            List of SimilarityResult
        """
        if isinstance(query, str):
            encoder = self._get_encoder()
            self.embedding_store.encoder = encoder
        gate = self._node_gate(node_filter, node_types)
        return self._seed_search(query, top_k, node_type, threshold, gate)

    def similar_to_node(
        self,
        node_ref: NodeRef,
        top_k: int = 10,
        node_type: Optional[str] = None,
        threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """
        Find nodes semantically closest to one already in the graph.

        The "more like this" query - no text needed, and no encoder call:
        it searches with the node's own stored vector and drops the node
        itself from the results.

        Returns an empty list if the node has no embedding.
        """
        record = self.embedding_store.get(node_ref)
        if record is None:
            return []
        results = self.embedding_store.similarity_search(
            record.vector, None if top_k is None else top_k + 1, node_type, threshold
        )
        filtered = [r for r in results if r.node_ref != node_ref]
        return filtered if top_k is None else filtered[:top_k]

    def _resolve_node(self, node_ref: NodeRef) -> Optional[Node]:
        try:
            return self.get_node_by_ref(node_ref)
        except GraphError:
            return None

    def semantic_multi_hop(
        self,
        query: Union[str, Sequence[float]],
        rel_type: Optional[str] = None,
        max_hops: int = 2,
        top_k_seeds: int = 5,
        top_k_results: int = 10,
        direction: Direction = Direction.OUT,
        decay: float = 0.8,
        threshold: float = 0.3,
        blend: float = 0.0,
        weight_field: Optional[str] = None,
        node_filter: Optional[Callable[[Node], bool]] = None,
        node_types: Optional[Sequence[str]] = None,
        strict_filter: bool = False
    ) -> List[SemanticSearchResult]:
        """
        Semantic multi-hop search.

        Combines embedding similarity with graph traversal:
        1. Find top-k similar nodes as seeds (hop 0)
        2. Traverse graph following relationships
        3. Score paths by: max(seed_similarity, 0) * prod(decay * weight) over
           the hops, where ``weight`` is 1.0 unless ``weight_field`` is set

        Args:
            query: Query text (or a pre-computed vector)
            rel_type: Relationship type to follow (None = any)
            max_hops: Maximum hops from seeds
            top_k_seeds: Number of seed nodes from similarity search
            top_k_results: Number of results to return
            direction: Traversal direction
            decay: Score decay per hop (0-1)
            threshold: Minimum seed similarity
            blend: How much of a node's *own* similarity to the query to mix
                   into its traversal score, in [0, 1]. 0 (the default) keeps
                   pure seed-and-decay scoring. Raise it so that a node found
                   two hops out, but highly relevant in its own right, is not
                   buried under a closer but unrelated neighbour. Hop-0 seeds
                   score the same either way.
            weight_field: Name of an edge property whose numeric value
                   multiplies the decay of every hop that crosses that edge
                   (clamped to [0, 1]; a missing or non-numeric value counts
                   as 1.0, so partly-weighted graphs work). A node is scored
                   by its *best* route, which with weights can be a longer one
                   than the fewest-hops route. None (the default) ignores
                   edge properties entirely.
            node_filter: Predicate on :class:`Node`. See the filtering rules
                   below.
            node_types: Node types allowed (a type name, or a sequence).
            strict_filter: Make the filter a hard wall for traversal too.

        Filtering rules (``node_filter`` and ``node_types`` combine with AND):
          * Seeds must pass. The top ``top_k_seeds`` are chosen *among*
            passing nodes, so a filtered-out best match never occupies a seed
            slot.
          * Only passing nodes are returned.
          * By default a node that fails the filter can still be walked
            *through*: it is not returned, but it still costs a hop and its
            decay, and a returned node's ``path`` may list it. With
            ``strict_filter=True`` a failing node is never entered, so every
            node on every returned path passes.

        Returns:
            List of SemanticSearchResult sorted by score
        """
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")

        if isinstance(query, str):
            encoder = self._get_encoder()
            self.embedding_store.encoder = encoder

        gate = self._node_gate(node_filter, node_types)

        if blend > 0.0:
            # Blending needs a score for every node, so one exact scan serves
            # both the seed selection and the per-node relevance.
            scores = self.embedding_store.score_map(query)
            if not scores:
                return []
            seeds = sorted(
                ((ref, score) for ref, score in scores.items()
                 if score >= threshold and (gate is None or gate(ref))),
                key=lambda item: item[1], reverse=True
            )[:top_k_seeds]
        else:
            # Only the seeds matter, so this can be answered by the
            # sqlite-vec index when the store has one (unless a filter forces
            # a full scan - see _seed_search).
            scores = {}
            seeds = [
                (r.node_ref, r.score)
                for r in self._seed_search(query, top_k_seeds, None, threshold, gate)
            ]

        if not seeds:
            return []

        def combined(base: float, ref: NodeRef) -> float:
            if blend <= 0.0:
                return base
            return (1.0 - blend) * base + blend * scores.get(ref, 0.0)

        results: Dict[NodeRef, SemanticSearchResult] = {}

        for seed_ref, seed_score in seeds:
            # Cosine similarity is signed, and `base * decay ** hops` only
            # decays a *non-negative* base: multiplying a negative seed score
            # by 0.8 moves it toward zero, so every node it reached outranked
            # the seed itself and relevance grew with distance. A seed that
            # matches the query negatively passes no relevance along instead.
            traversal_base = max(seed_score, 0.0)

            # A seed is a hop-0 match on its own merit. An earlier seed's
            # traversal may already have recorded it with a decayed score;
            # that must not win over the node's own, higher, similarity.
            # (Only the neighbour branch below used to compare scores, so
            # a direct match reachable from a better seed was reported at
            # seed.score * decay, at hop 1, down a path it never needed.)
            existing = results.get(seed_ref)
            if existing is None or existing.score < seed_score:
                results[seed_ref] = SemanticSearchResult(
                    node_ref=seed_ref,
                    node=self._resolve_node(seed_ref),
                    score=seed_score,
                    hop_count=0,
                    path=[seed_ref]
                )

            # Breadth-first traversal. Each node keeps its best decay factor
            # and its fewest hops *separately*: with edge weights the route
            # with the best factor is not always the shortest one, and a
            # shorter route must still be expanded because it has hops left.
            # Without weights the first (shortest) arrival is both, so this
            # reduces to the plain visited-set BFS.
            best_factor: Dict[NodeRef, float] = {seed_ref: 1.0}
            best_hops: Dict[NodeRef, int] = {seed_ref: 0}
            current = deque([(seed_ref, [seed_ref], 1.0, 0)])

            while current:
                node_ref, path, factor, hop = current.popleft()

                if hop >= max_hops:
                    continue

                for neighbor, edge in self._adjacent(node_ref, rel_type, direction):
                    if neighbor in path:
                        continue
                    if strict_filter and gate is not None and not gate(neighbor):
                        continue

                    weight = (1.0 if weight_field is None
                              else self._edge_weight(edge, weight_field))
                    new_hop = hop + 1
                    # Unweighted: decay ** hops exactly, not a running product,
                    # so scores are bit-identical to the pre-weight releases.
                    new_factor = (decay ** new_hop if weight_field is None
                                  else factor * decay * weight)

                    if (new_factor <= best_factor.get(neighbor, -1.0)
                            and new_hop >= best_hops.get(neighbor, new_hop + 1)):
                        continue
                    best_factor[neighbor] = max(new_factor, best_factor.get(neighbor, -1.0))
                    best_hops[neighbor] = min(new_hop, best_hops.get(neighbor, new_hop))

                    new_path = path + [neighbor]
                    if gate is None or gate(neighbor):
                        new_score = combined(traversal_base * new_factor, neighbor)
                        # Update if better score
                        existing = results.get(neighbor)
                        if existing is None or existing.score < new_score:
                            results[neighbor] = SemanticSearchResult(
                                node_ref=neighbor,
                                node=self._resolve_node(neighbor),
                                score=new_score,
                                hop_count=new_hop,
                                path=new_path
                            )

                    current.append((neighbor, new_path, new_factor, new_hop))

        # Sort by score
        sorted_results = sorted(results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k_results]

    def _gated_shortest_path(
        self,
        start: NodeRef,
        end: NodeRef,
        rel_type: Optional[str],
        max_hops: int,
        direction: Direction,
        gate: Callable[[NodeRef], bool]
    ) -> Optional[List[NodeRef]]:
        """Fewest-hops path from start to end whose every node passes ``gate``."""
        if not gate(start) or not gate(end):
            return None
        if start == end:
            return [start]
        visited = {start}
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for neighbor, _edge in self._adjacent(path[-1], rel_type, direction):
                if neighbor in visited or not gate(neighbor):
                    continue
                if neighbor == end:
                    return path + [neighbor]
                visited.add(neighbor)
                queue.append(path + [neighbor])
        return None

    def semantic_path(
        self,
        query: Union[str, Sequence[float]],
        target_ref: NodeRef,
        rel_type: Optional[str] = None,
        max_hops: int = 5,
        top_k_seeds: int = 3,
        direction: Direction = Direction.OUT,
        decay: float = 0.9,
        threshold: float = 0.0,
        node_filter: Optional[Callable[[Node], bool]] = None,
        node_types: Optional[Sequence[str]] = None,
        strict_filter: bool = False
    ) -> Optional[SemanticSearchResult]:
        """
        Find semantically-rooted path to a target node.

        Starts from semantically similar nodes and finds path to target.

        Args:
            query: Query text (or a pre-computed vector) to find a start point
            target_ref: Target node reference
            rel_type: Relationship type
            max_hops: Maximum path length
            top_k_seeds: Number of seed nodes
            direction: Traversal direction
            decay: Score decay per hop along the path
            threshold: Minimum seed similarity. Cosine similarity is signed,
                       so a seed can legitimately score below zero - lower
                       this to search from any starting point regardless of
                       how well it matches.
            node_filter: Predicate on :class:`Node`; see semantic_multi_hop.
            node_types: Node types allowed (a type name, or a sequence).
            strict_filter: Require every node on the path to pass too.

        Filtering: seeds are chosen among passing nodes, and the *target must
        pass* or the result is None. By default the path may run through
        failing nodes (it is the fewest-hops path, as without a filter); with
        ``strict_filter=True`` the path is the fewest-hops path that uses
        passing nodes only, which can be longer or not exist.

        Returns:
            SemanticSearchResult if path found, None otherwise
        """
        if isinstance(query, str):
            encoder = self._get_encoder()
            self.embedding_store.encoder = encoder

        gate = self._node_gate(node_filter, node_types)
        if gate is not None and not gate(target_ref):
            return None

        seeds = self._seed_search(query, top_k_seeds, None, threshold, gate)

        # Every seed that reaches the target is a candidate, and the best
        # score wins: a slightly less similar seed one hop from the target
        # beats a better seed five hops away. Seeds arrive best-first, so
        # a tie goes to the more similar one. As in semantic_multi_hop, a
        # negative seed passes no relevance along rather than "decaying"
        # toward zero and outscoring itself.
        best: Optional[SemanticSearchResult] = None
        for seed in seeds:
            if strict_filter and gate is not None:
                nodes = self._gated_shortest_path(
                    seed.node_ref, target_ref, rel_type, max_hops, direction, gate)
                if nodes is None:
                    continue
            else:
                path = self.shortest_path(
                    seed.node_ref, target_ref,
                    rel_type=rel_type,
                    max_hops=max_hops,
                    direction=direction
                )
                if not path:
                    continue
                nodes = path.nodes
            length = len(nodes) - 1
            score = max(seed.score, 0.0) * (decay ** length)
            if best is None or score > best.score:
                best = SemanticSearchResult(
                    node_ref=target_ref,
                    node=self._resolve_node(target_ref),
                    score=score,
                    hop_count=length,
                    path=nodes
                )

        return best

    def _connect_components(
        self,
        keep: set,
        rel_type: Optional[str],
        direction: Direction,
        max_hops: int,
        gate: Optional[Callable[[NodeRef], bool]]
    ) -> List[NodeRef]:
        """
        Grow ``keep`` with connector nodes until it is as connected as the
        hop budget allows. Returns the connectors added, in order.

        Greedy: each round finds, over all pairs of components, the shortest
        route (at most ``max_hops`` edges, honouring ``rel_type``/
        ``direction``) whose interior nodes lie outside ``keep``, adds its
        interior, and recomputes components. Each round merges at least two,
        so it terminates; it stops when one component remains or no route fits
        the budget. ``gate`` (set for ``strict_filter``) bars failing nodes
        from serving as connectors.
        """
        added: List[NodeRef] = []

        def order(ref: NodeRef) -> Tuple[str, str, str]:
            return (ref[0], type(ref[1]).__name__, str(ref[1]))

        while len(keep) > 1:
            # Weakly connected components of the induced subgraph.
            parent = {ref: ref for ref in keep}

            def find(ref: NodeRef) -> NodeRef:
                while parent[ref] != ref:
                    parent[ref] = parent[parent[ref]]
                    ref = parent[ref]
                return ref

            for edge in self.edges(rel_type):
                if edge.source in keep and edge.target in keep:
                    parent[find(edge.source)] = find(edge.target)
            groups: Dict[NodeRef, List[NodeRef]] = {}
            for ref in keep:
                groups.setdefault(find(ref), []).append(ref)
            if len(groups) <= 1:
                break

            best_route: Optional[List[NodeRef]] = None
            for members in sorted((sorted(g, key=order) for g in groups.values()),
                                  key=lambda g: order(g[0])):
                route = self._shortest_link(
                    members, keep, rel_type, direction, max_hops, gate)
                if route is not None and (best_route is None
                                          or len(route) < len(best_route)):
                    best_route = route
            if best_route is None:
                break
            for ref in best_route[1:-1]:
                keep.add(ref)
                added.append(ref)
        return added

    def _shortest_link(
        self,
        members: List[NodeRef],
        keep: set,
        rel_type: Optional[str],
        direction: Direction,
        max_hops: int,
        gate: Optional[Callable[[NodeRef], bool]]
    ) -> Optional[List[NodeRef]]:
        """Shortest route from any of ``members`` to a kept node outside them."""
        visited = set(members)
        queue = deque([[m] for m in members])
        while queue:
            path = queue.popleft()
            if len(path) - 1 >= max_hops:
                continue
            for neighbor, _edge in self._adjacent(path[-1], rel_type, direction):
                if neighbor in visited:
                    continue
                if neighbor in keep:
                    return path + [neighbor]
                if gate is not None and not gate(neighbor):
                    continue
                if self._resolve_node(neighbor) is None:
                    continue
                visited.add(neighbor)
                queue.append(path + [neighbor])
        return None

    def semantic_subgraph(
        self,
        query: Union[str, Sequence[float]],
        rel_type: Optional[str] = None,
        max_hops: int = 2,
        top_k_seeds: int = 5,
        top_k_results: int = 10,
        direction: Direction = Direction.OUT,
        decay: float = 0.8,
        threshold: float = 0.3,
        blend: float = 0.0,
        name: Optional[str] = None,
        weight_field: Optional[str] = None,
        node_filter: Optional[Callable[[Node], bool]] = None,
        node_types: Optional[Sequence[str]] = None,
        strict_filter: bool = False,
        connect: bool = False,
        connect_max_hops: int = 2
    ) -> 'SemanticGraph':
        """
        Extract the slice of the graph a query is actually about.

        Runs :meth:`semantic_multi_hop`, then returns a new SemanticGraph
        containing exactly those nodes, every edge induced between them, and
        their embeddings - so the result can be traversed, searched or
        serialized on its own.

        This is the piece that makes an ISON graph useful as LLM context:
        `graph.semantic_subgraph(question).save(path)` yields a small, on-topic
        graph to inject rather than the whole store.

        Args:
            Same as semantic_multi_hop (including ``weight_field``,
            ``node_filter``, ``node_types`` and ``strict_filter``), plus:
            name: Name for the returned graph (default: "<name>_subgraph")
            connect: Also add *connector* nodes so the selected nodes fall
                into as few connected components as possible. Components are
                weak (edge direction ignored when judging the result), but
                connector routes honour ``rel_type`` and ``direction``. Each
                round adds the interior of the shortest route between two
                components (greedy; not a minimum Steiner tree), until one
                component remains or no route fits ``connect_max_hops``.
                Nothing but those interior nodes is added, and they are not
                marked in any way.
            connect_max_hops: Longest route, counted in edges, that may join
                two components. The default 2 allows one connector node per
                join; 3 allows two. A direct edge needs no connector (it is
                already an induced edge).

        Note:
            With ``connect=True`` the result can hold **more than**
            ``top_k_results`` nodes: ``top_k_results`` caps the *selected*
            nodes, and connectors are extra. Without ``strict_filter`` a
            connector may be a node that fails ``node_filter``/``node_types``
            (it is only a bridge); with it, connectors must pass too.

        Returns:
            A new SemanticGraph sharing this graph's encoder
        """
        if connect_max_hops < 1:
            raise ValueError(f"connect_max_hops must be >= 1, got {connect_max_hops}")

        results = self.semantic_multi_hop(
            query, rel_type=rel_type, max_hops=max_hops,
            top_k_seeds=top_k_seeds, top_k_results=top_k_results,
            direction=direction, decay=decay, threshold=threshold, blend=blend,
            weight_field=weight_field, node_filter=node_filter,
            node_types=node_types, strict_filter=strict_filter
        )

        keep = {r.node_ref for r in results}
        if connect and keep:
            gate = (self._node_gate(node_filter, node_types)
                    if strict_filter else None)
            self._connect_components(keep, rel_type, direction, connect_max_hops, gate)

        sub = SemanticGraph(
            name=name or f"{self.name}_subgraph",
            directed=self.directed,
            embedding_db=":memory:",
            encoder=self._encoder,
            auto_embed=False,
            embed_fields=list(self.embed_fields)
        )
        sub._encoder_loaded = self._encoder_loaded

        for ref in keep:
            node = self._resolve_node(ref)
            if node is None:
                continue
            sub.add_node(node.type, node.id, **node.properties)
            record = self.embedding_store.get(ref)
            if record is not None:
                sub.embedding_store.add(ref, record.text, vector=record.vector)

        for edge in self.edges(rel_type):
            if edge.source in keep and edge.target in keep:
                sub.add_edge(edge.rel_type, edge.source, edge.target, **edge.properties)

        return sub

    # -- model lifecycle ------------------------------------------------

    def stale_embeddings(self, include_unknown: bool = False) -> List[NodeRef]:
        """
        Nodes whose stored vector was not made by the current encoder.

        Stale means the vector's recorded model differs from the encoder's
        ``model_name``, or its length differs from the encoder's
        ``dimension``. Vectors recorded as ``"unknown"``/``"provided"`` (from
        ports that store no model, or supplied without an encoder) cannot be
        judged by model and are left out unless ``include_unknown`` is True;
        a dimension mismatch always counts.
        """
        encoder = self._get_encoder()
        self.embedding_store.encoder = encoder
        return self.embedding_store.stale(include_unknown=include_unknown)

    def reembed_stale(self, include_unknown: bool = False, batch_size: int = 256) -> int:
        """
        Re-encode every stale embedding with the current encoder.

        Each vector is rebuilt from the text it was originally embedded from
        (so a custom ``_embed_text`` survives); a vector stored with no text
        falls back to the node's embed fields. If the encoder's dimension
        differs from the store's, every stored vector is stale, so the store
        is emptied, its pinned dimension reset, and everything re-embedded.

        Args:
            include_unknown: Also re-embed vectors of unknown provenance
            batch_size: Texts per encoder call

        Returns:
            Number of embeddings rewritten
        """
        stale = self.stale_embeddings(include_unknown=include_unknown)
        if not stale:
            return 0

        store = self.embedding_store
        pairs: List[Tuple[NodeRef, str]] = []
        for ref in stale:
            record = store.get(ref)
            if record is None:
                continue
            text = record.text
            if not text:
                node = self._resolve_node(ref)
                text = (self.embed_text_for(node.properties) if node else "") or str(ref)
            pairs.append((ref, text))

        encoder = self._get_encoder()
        if store.dimension is not None and store.dimension != encoder.dimension:
            store.clear(reset_dimension=True)

        count = 0
        for start in range(0, len(pairs), batch_size):
            count += self.embed_nodes(pairs[start:start + batch_size])
        return count

    # -- persistence ----------------------------------------------------

    @staticmethod
    def embeddings_path_for(path: Union[str, FilePath]) -> FilePath:
        """Sidecar path holding the embeddings for a saved graph."""
        return FilePath(str(path) + EMBEDDING_SIDECAR_SUFFIX)

    def save(
        self,
        path: Union[str, FilePath],
        format: str = "auto",
        embeddings: bool = True
    ) -> None:
        """
        Save graph to file, and its embeddings alongside it.

        The ISON graph format carries nodes and edges only, so embeddings go
        to a JSON sidecar at ``<path>.embeddings.json`` that :meth:`load`
        picks up automatically. Without it a save/load round-trip silently
        returned a graph whose searches all came back empty.

        Both files are written atomically (temp file, flush, ``os.replace``),
        so a crash never leaves a truncated file. The pair cannot be replaced
        as one unit, so the sidecar also records a :func:`graph_digest` of the
        graph it belongs to; :meth:`load` checks it and catches a crash
        between the two writes, a sidecar copied from another save, and a
        graph edited afterwards. The digest is taken of the graph *as it reads
        back from the file just written*, so a round trip that normalises a
        value does not cause a false mismatch.

        With ``embeddings=False`` an existing sidecar is left in place (and
        will fail the digest check on load, as it no longer matches).

        Args:
            path: Graph file path
            format: File format
            embeddings: Write the sidecar (set False for graph-only saves)
        """
        path = FilePath(path)
        if format == "auto":
            format = "isonl" if path.suffix == ".isonl" else "ison"

        text = self.to_isonl() if format == "isonl" else self.to_ison()
        _atomic_write_text(path, text)

        if embeddings:
            reread = (ISONGraph.from_isonl(text, path.stem) if format == "isonl"
                      else ISONGraph.from_ison(text, path.stem))
            self.embedding_store.save_embeddings(
                self.embeddings_path_for(path), graph_digest=graph_digest(reread))

    @classmethod
    def load(
        cls,
        path: Union[str, FilePath],
        embedding_db: Union[str, FilePath] = ":memory:",
        format: str = "auto",
        encoder: Optional[EmbeddingEncoder] = None,
        embeddings: bool = True,
        auto_embed: bool = True,
        embed_fields: Optional[List[str]] = None,
        sqlite_vec: bool = False,
        on_mismatch: str = "error",
        version: Optional[str] = None,
        on_model_mismatch: str = "warn"
    ) -> 'SemanticGraph':
        """
        Load graph from file, restoring embeddings from its sidecar.

        Args:
            path: Graph file path
            embedding_db: Embedding database path
            format: File format
            encoder: Encoder for the loaded graph. Without one, the first
                     text search builds a SentenceTransformerEncoder, which
                     downloads a model - pass the encoder you saved with to
                     keep a MockEncoder (or a pinned model) in place.
            embeddings: Restore the sidecar written by :meth:`save`
            auto_embed: Auto-embed nodes added to the loaded graph
            embed_fields: Fields to concatenate for auto-embedding
            sqlite_vec: Index the restored vectors with sqlite-vec
            on_mismatch: What to do when the sidecar carries a graph digest
                     that does not match the graph file (a sidecar from a
                     different save, or a graph edited since):
                     ``"error"`` (default) raises
                     :class:`GraphEmbeddingMismatchError`; ``"warn"`` warns
                     and loads the sidecar's vectors anyway; ``"reembed"``
                     discards the sidecar's vectors and re-embeds every node
                     with the encoder (:meth:`embed_all_nodes`). A sidecar
                     without a digest (older releases, other ports) is loaded
                     as before and never checked.
            version: Encoder/model version string for the loaded graph's store
                     (see :class:`EmbeddingStore`). If both it (or the
                     encoder's own ``version``) and the sidecar's
                     ``encoder_version`` are set and differ, a warning is
                     issued.
            on_model_mismatch: ``"warn"``/``"error"``/``"ignore"`` for text
                     searches whose encoder model differs from the stored
                     vectors' model (see :class:`EmbeddingStore`).

        Returns:
            SemanticGraph instance

        Raises:
            GraphEmbeddingMismatchError: digest mismatch with
                ``on_mismatch="error"``
            ValueError: ``on_mismatch`` is not one of the three modes
        """
        if on_mismatch not in ("error", "warn", "reembed"):
            raise ValueError(
                f"on_mismatch must be 'error', 'warn' or 'reembed', got {on_mismatch!r}")

        # Load base graph
        base = ISONGraph.load(path, format)

        # Create semantic graph and copy data
        graph = cls(
            name=base.name,
            directed=base.directed,
            embedding_db=embedding_db,
            encoder=encoder,
            auto_embed=auto_embed,
            embed_fields=embed_fields,
            sqlite_vec=sqlite_vec,
            version=version,
            on_model_mismatch=on_model_mismatch
        )

        # Copy the loaded state across through the public API.
        #
        # This used to reach into ISONGraph's private attributes (_nodes,
        # _edges, _out_edges, _in_edges, _edge_set) to skip re-validation.
        # That worked, but it silently depended on the base library's internal
        # representation - the kind of coupling that breaks without a
        # compilation error to warn you. add_node and add_edge rebuild the
        # same indexes, and a graph small enough to load is small enough to
        # rebuild.
        #
        # super().add_node is used rather than graph.add_node so that loading
        # never triggers auto-embedding: the embeddings come from the sidecar
        # below, already computed, and re-deriving them here would need an
        # encoder and would silently overwrite what was saved.
        for node in base.nodes():
            ISONGraph.add_node(graph, node.type, node.id, **node.properties)

        for rel_type in base.edge_types():
            for edge in base.edges(rel_type):
                graph.add_edge(edge.rel_type, edge.source, edge.target,
                               **edge.properties)

        if embeddings:
            sidecar = cls.embeddings_path_for(path)
            if sidecar.is_file():
                try:
                    payload = json.loads(sidecar.read_text(encoding='utf-8'))
                except ValueError as exc:
                    graph.close()
                    raise EmbeddingError(f"unreadable embeddings sidecar {sidecar}: {exc}") from exc

                problem = None
                recorded = payload.get("graph_digest") if isinstance(payload, dict) else None
                if (recorded is not None
                        and payload.get("graph_digest_basis", GRAPH_DIGEST_BASIS)
                        == GRAPH_DIGEST_BASIS):
                    actual = graph_digest(base)
                    if recorded != actual:
                        problem = (
                            f"embeddings sidecar {sidecar} does not belong to graph "
                            f"{FilePath(path)}: it was written for graph digest "
                            f"{recorded}, but the graph reads as {actual} (the "
                            f"sidecar is from a different save, or the graph was "
                            f"edited since). Pass on_mismatch='reembed' to rebuild "
                            f"the vectors or on_mismatch='warn' to use them anyway."
                        )

                if problem is not None and on_mismatch == "error":
                    graph.close()
                    raise GraphEmbeddingMismatchError(problem)
                if problem is not None and on_mismatch == "reembed":
                    graph.embed_all_nodes()
                else:
                    if problem is not None:
                        warnings.warn(problem, UserWarning, stacklevel=2)
                    graph.embedding_store.import_embeddings(payload)
                    stored_version = graph.embedding_store.last_import_meta.get("encoder_version")
                    current_version = graph.embedding_store.version
                    if (stored_version is not None and current_version is not None
                            and str(stored_version) != str(current_version)):
                        warnings.warn(
                            f"embeddings were made by encoder version {stored_version!r}, "
                            f"but the encoder in use is version {current_version!r}",
                            UserWarning, stacklevel=2)

        return graph

    def close(self) -> None:
        """Release the embedding database connection."""
        self.embedding_store.close()

    def __enter__(self) -> 'SemanticGraph':
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        emb_count = self.embedding_store.count()
        return f"SemanticGraph(name={self.name}, nodes={self.node_count()}, edges={self.edge_count()}, embeddings={emb_count})"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Version
    '__version__',
    'EMBEDDING_FORMAT',
    'EMBEDDING_FORMAT_VERSION',
    'EMBEDDING_SIDECAR_SUFFIX',
    'GRAPH_DIGEST_BASIS',
    'UNKNOWN_MODELS',
    'graph_digest',

    # Errors
    'EmbeddingError',
    'DimensionMismatchError',
    'NoEncoderError',
    'ModelMismatchError',
    'GraphEmbeddingMismatchError',

    # Encoders
    'EmbeddingEncoder',
    'SentenceTransformerEncoder',
    'MockEncoder',

    # Store
    'EmbeddingStore',
    'EmbeddingRecord',
    'SimilarityResult',

    # Graph
    'SemanticGraph',
    'SemanticSearchResult',
]
