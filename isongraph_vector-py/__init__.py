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

import sqlite3
import struct
import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any, Dict, List, Optional, Sequence,
    Union, Callable, Iterable, Tuple
)
from pathlib import Path as FilePath

# Import from ison-graph
from ison_graph import (
    ISONGraph, Node, NodeRef,
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


# =============================================================================
# Embedding Encoder Interface
# =============================================================================

class EmbeddingEncoder(ABC):
    """Abstract base class for embedding encoders."""

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

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize encoder with a sentence-transformers model.

        Args:
            model_name: HuggingFace model name
        """
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

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self._dimension = dimension
        self.model_name = f"mock-{dimension}"

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
    Decoded snapshot of the whole table, rebuilt whenever it is written to.

    Every search reads every row (brute force), so decoding blobs and, with
    numpy, stacking them into one matrix pays for itself immediately on the
    second query: the per-query cost drops from a Python loop over N x d
    floats to a single matrix-vector product.
    """

    __slots__ = ('refs', 'types', 'texts', 'vectors', 'matrix', 'norms')

    def __init__(self, refs, types, texts, vectors, matrix, norms):
        self.refs = refs
        self.types = types
        self.texts = texts
        self.vectors = vectors
        self.matrix = matrix
        self.norms = norms


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
        sqlite_vec: bool = False
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
                 within 2e-07 of the scan, and about 4x faster over 20,000
                 vectors.

        Raises:
            EmbeddingError: sqlite_vec=True but the package is not installed
                            or the extension cannot be loaded
        """
        self.db_path = str(db_path)
        self.encoder = encoder
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
            emb_id = self._write_row(conn, node_ref, vector, text, self._model_name())
            conn.commit()
            self._cache = None
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
            for ref, text, vector in items:
                self._write_row(conn, ref, vector, text, model)
            conn.commit()
            self._cache = None
            return len(items)

    @staticmethod
    def _embedding_id(node_ref: NodeRef) -> str:
        return hashlib.md5(f"{node_ref[0]}:{node_ref[1]}".encode()).hexdigest()[:16]

    def remove(self, node_ref: NodeRef) -> bool:
        """Remove embedding for a node."""
        with self._lock:
            conn = self._get_conn()
            if self._sqlite_vec and self._vec_ready:
                row = conn.execute(
                    "SELECT rowid FROM embeddings WHERE node_type = ? AND node_id = ?",
                    (node_ref[0], str(node_ref[1]))
                ).fetchone()
                if row is not None:
                    conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (row['rowid'],))
            cursor = conn.execute("""
                DELETE FROM embeddings
                WHERE node_type = ? AND node_id = ?
            """, (node_ref[0], str(node_ref[1])))
            conn.commit()
            self._cache = None
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all embeddings."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM embeddings")
            if self._sqlite_vec and self._vec_ready:
                conn.execute(f"DELETE FROM {VEC_TABLE}")
            conn.commit()
            self._cache = None

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

    def _ensure_cache(self) -> _ScanCache:
        """Build (or reuse) the decoded snapshot every search reads. Locked."""
        if self._cache is not None:
            return self._cache

        rows = self._get_conn().execute("""
            SELECT node_type, node_id, node_id_type, vector, text
            FROM embeddings ORDER BY rowid
        """).fetchall()

        refs: List[NodeRef] = []
        types: List[str] = []
        texts: List[str] = []
        blobs: List[bytes] = []
        for row in rows:
            refs.append((row['node_type'],
                         self._restore_node_id(row['node_id'], row['node_id_type'])))
            types.append(row['node_type'])
            texts.append(row['text'])
            blobs.append(row['vector'])

        matrix = None
        norms = None
        vectors: Optional[List[List[float]]] = None
        uniform = bool(blobs) and len({len(b) for b in blobs}) == 1

        if _np is not None and uniform:
            matrix = _np.frombuffer(b"".join(blobs), dtype=_np.float32).reshape(
                len(blobs), len(blobs[0]) // 4
            )
            norms = _np.linalg.norm(matrix, axis=1)
        else:
            vectors = [self._bytes_to_vector(b) for b in blobs]

        self._cache = _ScanCache(refs, types, texts, vectors, matrix, norms)
        return self._cache

    def _score_all(
        self,
        query_vector: Sequence[float],
        node_type: Optional[str] = None
    ) -> List[Tuple[NodeRef, float, str]]:
        """Cosine-score every stored embedding against a query vector."""
        cache = self._ensure_cache()
        if not cache.refs:
            return []

        if cache.matrix is not None:
            query = _np.asarray(query_vector, dtype=_np.float32)
            query_norm = float(_np.linalg.norm(query))
            if query_norm == 0.0:
                scores = _np.zeros(len(cache.refs), dtype=_np.float32)
            else:
                denom = cache.norms * query_norm
                with _np.errstate(divide='ignore', invalid='ignore'):
                    scores = _np.where(denom > 0, (cache.matrix @ query) / denom, 0.0)
            score_list = scores.tolist()
        else:
            score_list = [self._cosine_similarity(query_vector, v)
                          for v in (cache.vectors or [])]

        out: List[Tuple[NodeRef, float, str]] = []
        for i, ref in enumerate(cache.refs):
            if node_type is not None and cache.types[i] != node_type:
                continue
            out.append((ref, float(score_list[i]), cache.texts[i]))
        return out

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

            scored = self._score_all(query_vector, node_type)

        results = [
            SimilarityResult(node_ref=ref, score=score, text=text)
            for ref, score, text in scored
            if score >= threshold
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results if top_k is None else results[:top_k]

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
            return {ref: score for ref, score, _ in self._score_all(query_vector, node_type)}

    # -- portability ----------------------------------------------------

    def export_embeddings(self) -> Dict[str, Any]:
        """
        Serialize every embedding into a portable, cross-language dict.

        The shape is shared by all five language ports, so a store built in
        Python can be loaded by the TypeScript, JavaScript, Rust or C++ one.
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

            return {
                "format": EMBEDDING_FORMAT,
                "version": EMBEDDING_FORMAT_VERSION,
                "dimension": self._dimension,
                "count": len(embeddings),
                "embeddings": embeddings,
            }

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
            if replace:
                self.clear()
                self._dimension = payload.get("dimension")

            written = 0
            conn = self._get_conn()
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
            self._cache = None
            return written

    def save_embeddings(self, path: Union[str, FilePath]) -> int:
        """Write every embedding to a JSON sidecar. Returns the count."""
        payload = self.export_embeddings()
        FilePath(path).write_text(json.dumps(payload), encoding='utf-8')
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
            self._cache = None

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
        sqlite_vec: bool = False
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
        """
        super().__init__(name=name, directed=directed)

        # Lazy-load encoder to avoid import errors if not needed
        self._encoder = encoder
        self._encoder_loaded = encoder is not None

        self.embedding_store = EmbeddingStore(
            db_path=embedding_db,
            encoder=encoder,
            sqlite_vec=sqlite_vec
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

    def similarity_search(
        self,
        query: Union[str, Sequence[float]],
        top_k: Optional[int] = 10,
        node_type: Optional[str] = None,
        threshold: float = 0.0
    ) -> List[SimilarityResult]:
        """
        Find similar nodes by embedding.

        Args:
            query: Query text (or a pre-computed vector)
            top_k: Number of results (None returns every match)
            node_type: Filter by type
            threshold: Minimum similarity

        Returns:
            List of SimilarityResult
        """
        if isinstance(query, str):
            encoder = self._get_encoder()
            self.embedding_store.encoder = encoder
        return self.embedding_store.similarity_search(
            query, top_k, node_type, threshold
        )

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
        blend: float = 0.0
    ) -> List[SemanticSearchResult]:
        """
        Semantic multi-hop search.

        Combines embedding similarity with graph traversal:
        1. Find top-k similar nodes as seeds (hop 0)
        2. Traverse graph following relationships
        3. Score paths by: max(seed_similarity, 0) * decay^hop_count

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

        Returns:
            List of SemanticSearchResult sorted by score
        """
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {blend}")

        if isinstance(query, str):
            encoder = self._get_encoder()
            self.embedding_store.encoder = encoder

        if blend > 0.0:
            # Blending needs a score for every node, so one exact scan serves
            # both the seed selection and the per-node relevance.
            scores = self.embedding_store.score_map(query)
            if not scores:
                return []
            seeds = sorted(
                ((ref, score) for ref, score in scores.items() if score >= threshold),
                key=lambda item: item[1], reverse=True
            )[:top_k_seeds]
        else:
            # Only the seeds matter, so this can be answered by the
            # sqlite-vec index when the store has one.
            scores = {}
            seeds = [
                (r.node_ref, r.score)
                for r in self.embedding_store.similarity_search(
                    query, top_k_seeds, threshold=threshold)
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

            # BFS traversal
            current = [(seed_ref, [seed_ref], 0)]
            visited = {seed_ref}

            while current:
                node_ref, path, hop = current.pop(0)

                if hop >= max_hops:
                    continue

                neighbors = self.neighbors(node_ref, rel_type, direction)

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue

                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_hop = hop + 1
                    new_score = combined(traversal_base * (decay ** new_hop), neighbor)

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

                    current.append((neighbor, new_path, new_hop))

        # Sort by score
        sorted_results = sorted(results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k_results]

    def semantic_path(
        self,
        query: Union[str, Sequence[float]],
        target_ref: NodeRef,
        rel_type: Optional[str] = None,
        max_hops: int = 5,
        top_k_seeds: int = 3,
        direction: Direction = Direction.OUT,
        decay: float = 0.9,
        threshold: float = 0.0
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

        Returns:
            SemanticSearchResult if path found, None otherwise
        """
        seeds = self.similarity_search(query, top_k_seeds, threshold=threshold)

        for seed in seeds:
            path = self.shortest_path(
                seed.node_ref, target_ref,
                rel_type=rel_type,
                max_hops=max_hops,
                direction=direction
            )
            if path:
                return SemanticSearchResult(
                    node_ref=target_ref,
                    node=self._resolve_node(target_ref),
                    score=seed.score * (decay ** path.length),
                    hop_count=path.length,
                    path=path.nodes
                )

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
        name: Optional[str] = None
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
            Same as semantic_multi_hop, plus:
            name: Name for the returned graph (default: "<name>_subgraph")

        Returns:
            A new SemanticGraph sharing this graph's encoder
        """
        results = self.semantic_multi_hop(
            query, rel_type=rel_type, max_hops=max_hops,
            top_k_seeds=top_k_seeds, top_k_results=top_k_results,
            direction=direction, decay=decay, threshold=threshold, blend=blend
        )

        keep = {r.node_ref for r in results}
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

        Args:
            path: Graph file path
            format: File format
            embeddings: Write the sidecar (set False for graph-only saves)
        """
        super().save(path, format)
        if embeddings:
            self.embedding_store.save_embeddings(self.embeddings_path_for(path))

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
        sqlite_vec: bool = False
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

        Returns:
            SemanticGraph instance
        """
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
            sqlite_vec=sqlite_vec
        )

        # Copy nodes
        for node in base.nodes():
            graph._nodes[node.type][node.id] = node

        # Copy edges
        for rel_type in base.edge_types():
            for edge in base.edges(rel_type):
                graph._edges[rel_type].append(edge)
                graph._out_edges[edge.source].append(edge)
                graph._in_edges[edge.target].append(edge)
                graph._edge_set.add(edge.key)

        if embeddings:
            sidecar = cls.embeddings_path_for(path)
            if sidecar.is_file():
                graph.embedding_store.load_embeddings(sidecar)

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

    # Errors
    'EmbeddingError',
    'DimensionMismatchError',
    'NoEncoderError',

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
