"""Durable, verifiable snapshots of one graph.

On-disk layout of a graph (``<data>/tenants/<tenant>/graphs/g-<name>/``)::

    meta.json                  immutable graph configuration (written once, atomically)
    CURRENT                    text file: the generation number to serve
    gen-000007/
        graph.isong            the graph (ISON text)
        graph.isong.embeddings.json   the vectors, tagged with the graph's digest
        MANIFEST.json          generation, graph_digest, counts, sha256 + size of each file
    gen-000006/ ...            older generations kept for fallback and for replicas

Why generations instead of overwriting ``graph.isong`` in place: the library
writes the graph file and the embeddings sidecar as two separate atomic
renames, so a crash between them leaves a *pair* that does not belong
together (the library detects this on load, by digest, but cannot repair it).
Here the pair is built completely in a temporary directory, the directory is
renamed into place, and only then is ``CURRENT`` atomically flipped. A crash
at any point leaves ``CURRENT`` naming a complete, verified generation.

Verification on load, in order: every file's size and sha256 against the
manifest; the library's own sidecar-vs-graph digest check
(``on_mismatch="error"``); the sidecar's recorded digest against the
manifest's; and the digest of the loaded graph against the manifest's. If the
generation ``CURRENT`` names fails any of them, the newest older generation
that passes is served instead and the graph is flagged ``degraded``; if none
passes the graph is quarantined (not served, not deleted).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hashlib as _hashlib

from isongraph_vector import (
    GRAPH_DIGEST_BASIS, EmbeddingEncoder, EmbeddingError, NoEncoderError, SemanticGraph,
    graph_digest,
)
from ison_graph import ISONGraph

from . import names

META_FORMAT = "igv-graph"
MANIFEST_FORMAT = "igv-snapshot"
FORMAT_VERSION = 1
GRAPH_FILE = "graph.isong"
SIDECAR_FILE = "graph.isong.embeddings.json"
MANIFEST_FILE = "MANIFEST.json"

#: Test seam for crash-safety tests: called with a point name as a save
#: progresses. Production code never sets it. (A child process in the test
#: suite installs a hook that announces the point and then sleeps, and the
#: parent SIGKILLs it there.)
fault_hook: Optional[Callable[[str], None]] = None

_LOAD_WARN_LOCK = threading.Lock()


class SnapshotError(Exception):
    """A snapshot could not be written, or failed verification."""


class ClientModeEncoder(EmbeddingEncoder):
    """Placeholder encoder for graphs whose vectors are supplied by the caller.

    Without it the library would lazily build a SentenceTransformer (and
    download a model) the first time anything asked for an encoder. This one
    refuses to encode, and stamps stored vectors with the model name
    ``"provided"``, which the library treats as "unknown provenance".
    """

    model_name = "provided"

    @property
    def dimension(self) -> int:
        raise AttributeError("client-mode graphs have no encoder dimension")

    def encode(self, text: str):
        raise NoEncoderError("this graph takes caller-supplied vectors; it has no encoder")

    def encode_batch(self, texts):
        raise NoEncoderError("this graph takes caller-supplied vectors; it has no encoder")


def _fault(point: str) -> None:
    hook = fault_hook
    if hook is not None:
        hook(point)


def sha256_file(path: Path) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def fsync_dir(path: Path) -> None:
    """Make a rename in ``path`` durable. A no-op where directories cannot be opened (Windows)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_bytes_durably(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        write_bytes_durably(tmp, data)
        if path.name == "CURRENT":
            _fault("current_tmp_written")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    fsync_dir(path.parent)


@dataclass
class GraphMeta:
    name: str
    mode: str                     # "server" | "client"
    dimension: Optional[int]      # fixed for client mode; server mode follows the encoder
    embed_fields: List[str]
    ann: Optional[str] = None
    ann_nprobe: Optional[int] = None
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> Dict[str, Any]:
        return {
            "format": META_FORMAT, "version": FORMAT_VERSION, "name": self.name,
            "directed": True, "mode": self.mode, "dimension": self.dimension,
            "embed_fields": self.embed_fields, "ann": self.ann,
            "ann_nprobe": self.ann_nprobe, "created_at": self.created_at,
        }

    @staticmethod
    def from_json(data: Any, expect_name: str) -> "GraphMeta":
        if (not isinstance(data, dict) or data.get("format") != META_FORMAT
                or data.get("version") != FORMAT_VERSION):
            raise SnapshotError("meta.json is not a graph meta document")
        if data.get("name") != expect_name or not names.is_graph_name(expect_name):
            raise SnapshotError("meta.json name does not match its directory")
        if data.get("mode") not in ("server", "client"):
            raise SnapshotError("meta.json has a bad mode")
        fields = data.get("embed_fields")
        if (not isinstance(fields, list) or not fields
                or not all(isinstance(f, str) and names.PROP_RE.match(f) for f in fields)):
            raise SnapshotError("meta.json has bad embed_fields")
        return GraphMeta(
            name=expect_name, mode=data["mode"], dimension=data.get("dimension"),
            embed_fields=list(fields), ann=data.get("ann"),
            ann_nprobe=data.get("ann_nprobe"), created_at=float(data.get("created_at", 0)),
        )


def new_semantic_graph(meta: GraphMeta, encoder: Optional[EmbeddingEncoder]) -> SemanticGraph:
    """An empty in-memory graph configured the way the server always runs them.

    ``auto_embed`` is off: the server computes embeddings itself, outside the
    graph lock, and hands the library finished vectors.
    """
    enc = encoder if meta.mode == "server" else ClientModeEncoder()
    return SemanticGraph(
        name=meta.name, directed=True, embedding_db=":memory:", encoder=enc,
        auto_embed=False, embed_fields=list(meta.embed_fields),
        on_model_mismatch="ignore", ann=meta.ann, ann_nprobe=meta.ann_nprobe,
    )


@dataclass
class Manifest:
    generation: int
    graph_digest: str
    node_count: int
    edge_count: int
    embedding_count: int
    dimension: Optional[int]
    files: Dict[str, Dict[str, Any]]
    created_at: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT, "version": FORMAT_VERSION,
            "generation": self.generation, "graph_digest": self.graph_digest,
            "node_count": self.node_count, "edge_count": self.edge_count,
            "embedding_count": self.embedding_count, "dimension": self.dimension,
            "files": self.files, "created_at": self.created_at,
        }

    @staticmethod
    def from_json(data: Any) -> "Manifest":
        try:
            if (data["format"] != MANIFEST_FORMAT or data["version"] != FORMAT_VERSION):
                raise SnapshotError("unsupported manifest")
            files = data["files"]
            for fname in (GRAPH_FILE, SIDECAR_FILE):
                entry = files[fname]
                if not (isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
                        and isinstance(entry["size"], int)):
                    raise SnapshotError("bad manifest file entry")
            digest = data["graph_digest"]
            if not (isinstance(digest, str) and digest.startswith("sha256:")):
                raise SnapshotError("bad manifest digest")
            return Manifest(
                generation=int(data["generation"]), graph_digest=digest,
                node_count=int(data["node_count"]), edge_count=int(data["edge_count"]),
                embedding_count=int(data["embedding_count"]), dimension=data.get("dimension"),
                files={k: files[k] for k in (GRAPH_FILE, SIDECAR_FILE)},
                created_at=float(data.get("created_at", 0)),
            )
        except SnapshotError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotError(f"malformed manifest ({type(exc).__name__})") from None


def semantic_digest(graph: ISONGraph) -> str:
    """Like ``graph_digest`` but ignoring null-valued properties.

    The ISON text format is columnar: a property one node has and another
    lacks comes back from a save/load as an explicit ``null`` on the other.
    Absent and null are therefore the same thing to this service (it never
    stores or returns null properties). This digest is what "the file means
    the same as memory" is checked with; the manifest still records the
    library's own ``graph_digest`` of the file, which is what the sidecar
    carries and what replicas verify.
    """
    def ref_key(ref):
        return [ref[0], 'i' if isinstance(ref[1], int) and not isinstance(ref[1], bool)
                else 's', str(ref[1])]

    def dump(obj):
        return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, default=str)

    def live(props):
        return {k: v for k, v in props.items() if v is not None}

    nodes = sorted(dump([ref_key(n.ref), live(n.properties)]) for n in graph.nodes())
    edges = sorted(dump([e.rel_type, ref_key(e.source), ref_key(e.target), live(e.properties)])
                   for e in graph.edges())
    body = "|".join([str(graph.directed), chr(10).join(nodes), chr(10).join(edges)])
    return "sem:" + _hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class Serialized:
    """A consistent in-memory image of a graph, ready to be written without any lock."""
    graph_text: str
    payload: Dict[str, Any]          # the embeddings sidecar, minus the graph digest
    semantic: str
    node_count: int
    edge_count: int
    embedding_count: int
    dimension: Optional[int]


def serialize(graph: SemanticGraph) -> Serialized:
    """Capture graph + embeddings. The caller holds the graph's read lock.

    Only memory-to-memory work happens here (text rendering, a digest, the
    vector export); the expensive parts - re-parsing, hashing the files,
    JSON-encoding the vectors, fsync - run afterwards, outside the lock, in
    ``GraphStorage.write_generation``.
    """
    payload = graph.embedding_store.export_embeddings()
    return Serialized(
        graph_text=graph.to_ison(), payload=payload, semantic=semantic_digest(graph),
        node_count=graph.node_count(), edge_count=graph.edge_count(),
        embedding_count=payload["count"], dimension=payload.get("dimension"),
    )


class GraphStorage:
    """All filesystem access for one graph directory."""

    def __init__(self, directory: Path, keep_generations: int = 3):
        self.dir = Path(directory)
        self.keep = max(2, keep_generations)

    # -- small helpers ----------------------------------------------------------

    def gen_path(self, generation: int) -> Path:
        return self.dir / f"gen-{generation:06d}"

    def read_current(self) -> Optional[int]:
        try:
            text = (self.dir / "CURRENT").read_text(encoding="ascii").strip()
        except (OSError, UnicodeDecodeError):
            return None
        return int(text) if names.GENERATION_RE.match(text) else None

    def generations(self) -> List[int]:
        out = []
        try:
            for entry in os.scandir(self.dir):
                if entry.is_dir(follow_symlinks=False) and entry.name.startswith("gen-"):
                    num = entry.name[4:]
                    if names.GENERATION_RE.match(num) and (Path(entry.path) / MANIFEST_FILE).is_file():
                        out.append(int(num))
        except FileNotFoundError:
            pass
        return sorted(out)

    def read_meta(self) -> GraphMeta:
        try:
            data = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"unreadable meta.json ({type(exc).__name__})") from None
        return GraphMeta.from_json(data, self.dir.name[2:])

    @staticmethod
    def create(parent: Path, meta: GraphMeta, keep_generations: int = 3) -> "GraphStorage":
        """Create the graph directory atomically (temp dir, then rename)."""
        final = parent / names.graph_dirname(meta.name)
        parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(meta.name)
        tmp = parent / f".tmp-create-{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        try:
            write_bytes_durably(tmp / "meta.json", json.dumps(meta.to_json(), indent=2).encode("utf-8"))
            fsync_dir(tmp)
            os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        fsync_dir(parent)
        return GraphStorage(final, keep_generations)

    def remove(self) -> None:
        """Delete the graph: rename out of the namespace first, then remove the files."""
        trash = self.dir.with_name(f".trash-{uuid.uuid4().hex[:12]}")
        os.rename(self.dir, trash)
        fsync_dir(trash.parent)
        shutil.rmtree(trash, ignore_errors=True)

    # -- writing ------------------------------------------------------------------

    def write_generation(self, image: Serialized) -> Manifest:
        """Write ``image`` as the next generation and make it current. No locks needed."""
        existing = self.generations()
        current = self.read_current() or 0
        generation = max([current] + existing) + 1
        tmp = self.dir / f".tmp-gen-{generation:06d}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            # Re-read what we are about to publish and compare it with memory.
            reread = ISONGraph.from_ison(image.graph_text, "graph")
            if semantic_digest(reread) != image.semantic:
                raise SnapshotError(
                    "the graph does not survive a save/load round trip unchanged; "
                    "refusing to publish a snapshot that differs from memory")
            digest = graph_digest(reread)
            payload = dict(image.payload)
            payload["graph_digest"] = digest
            payload["graph_digest_basis"] = GRAPH_DIGEST_BASIS
            write_bytes_durably(tmp / GRAPH_FILE, image.graph_text.encode("utf-8"))
            _fault("graph_written")
            write_bytes_durably(tmp / SIDECAR_FILE, json.dumps(payload).encode("utf-8"))
            files = {}
            for fname in (GRAPH_FILE, SIDECAR_FILE):
                file_sha, size = sha256_file(tmp / fname)
                files[fname] = {"sha256": file_sha, "size": size}
            manifest = Manifest(
                generation=generation, graph_digest=digest, node_count=image.node_count,
                edge_count=image.edge_count, embedding_count=image.embedding_count,
                dimension=image.dimension, files=files, created_at=time.time())
            write_bytes_durably(tmp / MANIFEST_FILE,
                                json.dumps(manifest.to_json(), indent=2).encode("utf-8"))
            fsync_dir(tmp)
            _fault("files_written")
            os.rename(tmp, self.gen_path(generation))
            fsync_dir(self.dir)
            _fault("generation_renamed")
            atomic_write(self.dir / "CURRENT", f"{generation:06d}\n".encode("ascii"))
            _fault("current_flipped")
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self.prune()
        return manifest

    def install_generation(self, staged: Path, generation: int) -> None:
        """Adopt an already verified generation directory (replica sync)."""
        target = self.gen_path(generation)
        if target.exists():
            shutil.rmtree(target)
        os.rename(staged, target)
        fsync_dir(self.dir)
        atomic_write(self.dir / "CURRENT", f"{generation:06d}\n".encode("ascii"))
        self.prune(keep_extra={generation})

    def prune(self, keep_extra=()) -> None:
        gens = self.generations()
        current = self.read_current()
        keep = set(gens[-self.keep:]) | set(keep_extra)
        if current is not None:
            keep.add(current)
        for g in gens:
            if g not in keep:
                shutil.rmtree(self.gen_path(g), ignore_errors=True)

    def clean_tmp(self) -> int:
        """Remove leftovers of an interrupted save/create. Only the process that
        owns the data directory may call this (it does at startup)."""
        n = 0
        try:
            for entry in os.scandir(self.dir):
                if entry.name.startswith((".tmp-", ".trash-")) or (
                        entry.name.startswith(".") and entry.name.endswith(".tmp")):
                    p = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        try:
                            p.unlink()
                        except OSError:
                            continue
                    n += 1
        except FileNotFoundError:
            pass
        return n

    # -- reading ------------------------------------------------------------------

    def verify_generation(self, gen_dir: Path) -> Manifest:
        """Check manifest well-formedness and each file's size and sha256."""
        try:
            manifest = Manifest.from_json(json.loads((gen_dir / MANIFEST_FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"unreadable manifest ({type(exc).__name__})") from None
        for fname, entry in manifest.files.items():
            path = gen_dir / fname
            try:
                digest, size = sha256_file(path)
            except OSError:
                raise SnapshotError(f"{fname} is missing or unreadable") from None
            if size != entry["size"] or digest != entry["sha256"]:
                raise SnapshotError(f"{fname} does not match its manifest checksum")
        return manifest

    def load_generation(self, gen_dir: Path, meta: GraphMeta,
                        encoder: Optional[EmbeddingEncoder]) -> Tuple[SemanticGraph, Manifest, List[str]]:
        manifest = self.verify_generation(gen_dir)
        enc = encoder if meta.mode == "server" else ClientModeEncoder()
        caught: List[str] = []
        with _LOAD_WARN_LOCK, warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            try:
                graph = SemanticGraph.load(
                    gen_dir / GRAPH_FILE, encoder=enc, auto_embed=False,
                    embed_fields=list(meta.embed_fields), on_mismatch="error",
                    on_model_mismatch="ignore", ann=meta.ann, ann_nprobe=meta.ann_nprobe)
            except (EmbeddingError, ValueError, OSError, KeyError, TypeError) as exc:
                raise SnapshotError(f"{type(exc).__name__}: {str(exc)[:200]}") from None
            caught = [str(w.message)[:300] for w in record]
        graph.name = meta.name
        try:
            if graph_digest(graph) != manifest.graph_digest:
                raise SnapshotError("loaded graph digest does not match the manifest")
            recorded = graph.embedding_store.last_import_meta.get("graph_digest")
            if recorded != manifest.graph_digest:
                raise SnapshotError("embeddings sidecar is not tagged with the manifest's graph digest")
        except BaseException:
            graph.close()
            raise
        return graph, manifest, caught

    def load(self, meta: GraphMeta, encoder: Optional[EmbeddingEncoder]):
        """Return ``(graph, manifest_or_None, degraded, warnings)``.

        See the module docstring for the fallback rules. A brand-new graph
        (no generations, no CURRENT) loads as empty with ``manifest=None``.
        """
        gens = self.generations()
        current = self.read_current()
        if not gens and current is None:
            return new_semantic_graph(meta, encoder), None, False, []
        if current is not None and current in gens:
            order = [current] + [g for g in reversed(gens) if g < current]
        else:
            order = list(reversed(gens))
        errors = []
        for gen in order:
            try:
                graph, manifest, caught = self.load_generation(self.gen_path(gen), meta, encoder)
                return graph, manifest, gen != current, caught
            except SnapshotError as exc:
                errors.append(f"gen {gen}: {exc}")
        raise SnapshotError("no generation passed verification: " + "; ".join(errors))


def acquire_dir_lock(directory: Path):
    """Exclusive, non-blocking lock so two server processes cannot share a data dir.

    Returns an open file object to keep alive for the life of the process.
    """
    directory.mkdir(parents=True, exist_ok=True)
    handle = open(directory / ".lock", "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SnapshotError(f"{directory} is in use by another igv-server process") from None
    return handle
