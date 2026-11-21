"""One adapter per Python-callable system. Each exposes the same small surface so the runner is identical.

    A = ADAPTERS[name](d=..., n_cap=..., workdir=...)
    A.add(ids, X, grp=None)    ingest a chunk (ids int64, X float32 (m,d), grp uint8 or None)
    A.finalize()               build / train / index (the "build" cost beyond ingest)
    A.set_knob(v)              accuracy knob (nprobe / ef_search); knobs() lists the sweep
    A.set_threads(t)           library thread count (1 = strictly single thread)
    A.search1(q, k)            one query -> ids (k,)
    A.searchN(Q, k)            many queries at once (uses the thread count) -> ids (Q,k) or None
    A.upsert(ids, X) / A.delete(ids)            streaming mutation, or raise NotImplementedError
    A.search_filtered1(q, k, g) restrict to grp==g (needs add(..., grp)), or NotImplementedError
    A.save(path) / A.load(path)    native persistence (returns None) ; A.disk_bytes(path)
    A.close()
Every library is used with its own documented defaults except where the knob is the point of the sweep;
common HNSW parameters are identical across libraries (M=16, efConstruction=200); the IVF cell count is the
same for every IVF system (nlist = round(4*sqrt(N)), what isongraph-vector picks automatically).
"""
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def nlist_for(n):
    return int(min(max(round(4 * math.sqrt(n)), 16), 65536))


def dir_bytes(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot


class Base:
    name = "base"
    exact = False
    knob_name = None
    version = ""
    supports_batch = False

    def __init__(self, d, n_cap, workdir):
        self.d, self.n_cap, self.workdir = d, n_cap, workdir
        os.makedirs(workdir, exist_ok=True)
        self.threads = 1

    def knobs(self, n):
        return [None]

    def finalize(self):
        pass

    def set_knob(self, v):
        pass

    def set_threads(self, t):
        self.threads = t

    def searchN(self, Q, k):
        return None

    def upsert(self, ids, X):
        raise NotImplementedError

    def delete(self, ids):
        raise NotImplementedError

    def search_filtered1(self, q, k, g):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def disk_bytes(self, path):
        return dir_bytes(path)

    def close(self):
        pass


# ----------------------------------------------------------------------------- numpy brute force
class NumpyBF(Base):
    name = "numpy_bf"
    exact = True
    supports_batch = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import numpy
        self.version = "numpy " + numpy.__version__
        self.M = np.empty((n_cap, d), dtype=np.float32)
        self.ids = np.full(n_cap, -1, dtype=np.int64)
        self.alive = np.zeros(n_cap, dtype=bool)
        self.cnt = 0
        self.slot = {}
        self.grp = None

    def add(self, ids, X, grp=None):
        m = len(ids)
        self.M[self.cnt:self.cnt + m] = X
        self.ids[self.cnt:self.cnt + m] = ids
        self.alive[self.cnt:self.cnt + m] = True
        if grp is not None:
            if self.grp is None:
                self.grp = np.full(self.n_cap, 255, dtype=np.uint8)
            self.grp[self.cnt:self.cnt + m] = grp
        self.slot.update(zip(ids.tolist(), range(self.cnt, self.cnt + m)))
        self.cnt += m

    def set_threads(self, t):
        from threadpoolctl import threadpool_limits
        self._lim = threadpool_limits(limits=t)
        self.threads = t

    def _top(self, s, k):
        s = np.where(self.alive[:self.cnt], s, -np.inf)
        p = np.argpartition(-s, k - 1)[:k]
        p = p[np.argsort(-s[p])]
        return self.ids[p]

    def search1(self, q, k):
        return self._top(self.M[:self.cnt] @ q, k)

    def searchN(self, Q, k):
        out = np.empty((len(Q), k), dtype=np.int64)
        step = max(1, 4_000_000_000 // (4 * max(self.cnt, 1)))  # bound the score matrix at ~4 GB
        for s in range(0, len(Q), step):
            S = Q[s:s + step] @ self.M[:self.cnt].T
            S = np.where(self.alive[:self.cnt][None, :], S, -np.inf)
            P = np.argpartition(-S, k - 1, axis=1)[:, :k]
            ps = np.take_along_axis(S, P, axis=1)
            o = np.argsort(-ps, axis=1)
            out[s:s + step] = self.ids[np.take_along_axis(P, o, axis=1)]
        return out

    def upsert(self, ids, X):
        new_ids, new_X = [], []
        for i, row in zip(ids.tolist(), X):
            s = self.slot.get(i)
            if s is not None and self.alive[s]:
                self.M[s] = row
            else:
                new_ids.append(i)
                new_X.append(row)
        if new_ids:
            self.add(np.array(new_ids, dtype=np.int64), np.stack(new_X))

    def delete(self, ids):
        for i in ids.tolist():
            self.alive[self.slot[i]] = False

    def search_filtered1(self, q, k, g):
        s = self.M[:self.cnt] @ q
        s = np.where(self.grp[:self.cnt] == g, s, -np.inf)
        p = np.argpartition(-s, k - 1)[:k]
        return self.ids[p[np.argsort(-s[p])]]

    def save(self, path):
        np.savez(path, M=self.M[:self.cnt], ids=self.ids[:self.cnt]) if False else None
        np.save(path + ".M.npy", self.M[:self.cnt])
        np.save(path + ".ids.npy", self.ids[:self.cnt])

    def load(self, path):
        M = np.load(path + ".M.npy")
        self.M[:len(M)] = M
        self.ids[:len(M)] = np.load(path + ".ids.npy")
        self.alive[:len(M)] = True
        self.cnt = len(M)

    def disk_bytes(self, path):
        return sum(os.path.getsize(path + s) for s in (".M.npy", ".ids.npy"))


# ----------------------------------------------------------------------------- isongraph-vector (Python)
class IsonPy(Base):
    name = "isong_py"
    exact = True
    ann = None

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        sys.path.insert(0, os.path.join(HERE, ".."))
        # isongraph-vector depends on ison-graph, which is installed in the user's global site-packages. It is
        # appended LAST so that the venv's numpy etc. still win; only ison_graph/ison_parser are taken from there.
        gs = os.path.join(os.environ.get("APPDATA", ""), "Python", "Python312", "site-packages")
        if os.path.isdir(gs) and gs not in sys.path:
            sys.path.append(gs)
        from _common import import_package
        self.mod = import_package()
        self.version = "isongraph-vector (python, repo HEAD)"
        self._new()

    def _new(self, db_path=":memory:"):
        kw = {}
        if self.ann:
            kw["ann"] = self.ann
        self.s = self.mod.EmbeddingStore(db_path, dimension=self.d, **kw)
        self.typ = None

    def set_threads(self, t):
        from threadpoolctl import threadpool_limits
        self._lim = threadpool_limits(limits=t)
        self.threads = t

    def add(self, ids, X, grp=None):
        rows = X.tolist()
        il = ids.tolist()
        if grp is None:
            ent = [(("doc", i), "", v) for i, v in zip(il, rows)]
        else:
            tn = ("t10", "t01", "rest")
            ent = [((tn[g], i), "", v) for i, v, g in zip(il, rows, grp.tolist())]
        self.s.add_batch(ent)

    def search1(self, q, k):
        return np.fromiter((r.node_ref[1] for r in self.s.similarity_search(q, top_k=k)), dtype=np.int64, count=-1)

    def upsert(self, ids, X):
        rows = X.tolist()
        self.s.add_batch([(("doc", i), "", v) for i, v in zip(ids.tolist(), rows)])

    def delete(self, ids):
        for i in ids.tolist():
            self.s.remove(("doc", i))

    def search_filtered1(self, q, k, g):
        t = ("t10", "t01")[g]
        return np.fromiter((r.node_ref[1] for r in self.s.similarity_search(q, top_k=k, node_type=t)), dtype=np.int64)

    def save(self, path):
        self.s.save_embeddings(path)

    def load(self, path):
        self.s.close()
        self._new()
        self.s.load_embeddings(path)

    def save_alt(self, path):
        """binary snapshot (the Stage D opt-in persistence)"""
        self.s.save_snapshot(path)

    def load_alt(self, path):
        self.s.close()
        self._new()
        self.s.load_snapshot(path, int_ids=lambda ty, ident: True)

    def searchN(self, Q, k):
        # numpy batch search (one matrix-matrix product per group of queries); only the exact scan has a batch path
        if self.ann:
            return None
        res = self.s.similarity_search_batch([q for q in Q], top_k=k)
        return np.array([[r.node_ref[1] for r in row] for row in res], dtype=np.int64)

    def close(self):
        self.s.close()


class IsonPyIVF(IsonPy):
    name = "isong_py_ivf"
    exact = False
    ann = "ivf"
    knob_name = "nprobe"

    def knobs(self, n):
        nl = nlist_for(n)
        base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        return [b for b in base if b < nl] + [nl]

    def finalize(self):
        self.s.build_ann_index()

    def set_knob(self, v):
        self.s.ann_nprobe = int(v)

    def stats(self):
        return self.s.ann_stats()


# ----------------------------------------------------------------------------- faiss
class FaissBase(Base):
    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import faiss
        self.faiss = faiss
        self.version = "faiss-cpu " + faiss.__version__
        self.supports_batch = True

    def set_threads(self, t):
        self.faiss.omp_set_num_threads(t)
        self.threads = t

    def search1(self, q, k):
        return self.idx.search(q[None, :], k, params=self._params())[1][0]

    def searchN(self, Q, k):
        return self.idx.search(Q, k, params=self._params())[1]

    def _params(self):
        return None

    def save(self, path):
        self.faiss.write_index(self.idx, path)

    def load(self, path):
        self.idx = self.faiss.read_index(path)

    def _sel_params(self, g):
        return self._params_with_sel(self._sels[g])

    def prepare_filter(self, grp_all):
        # IDSelectorBitmap: a packed bit per id, the cheapest selector faiss offers (IDSelectorBatch hashes every id)
        self._bits, self._sels = {}, {}
        for g in (0, 1):
            self._bits[g] = np.packbits(grp_all == g, bitorder="little")
            self._sels[g] = self.faiss.IDSelectorBitmap(len(grp_all), self.faiss.swig_ptr(self._bits[g]))

    def search_filtered1(self, q, k, g):
        return self.idx.search(q[None, :], k, params=self._sel_params(g))[1][0]


class FaissFlat(FaissBase):
    name = "faiss_flat"
    exact = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        self.idx = self.faiss.IndexIDMap2(self.faiss.IndexFlatIP(d))

    def add(self, ids, X, grp=None):
        self.idx.add_with_ids(X, ids)

    def upsert(self, ids, X):
        self.idx.remove_ids(self.faiss.IDSelectorBatch(ids))
        self.idx.add_with_ids(X, ids)

    def delete(self, ids):
        self.idx.remove_ids(self.faiss.IDSelectorBatch(ids))

    def _params_with_sel(self, sel):
        return self.faiss.SearchParameters(sel=sel)


class FaissIVF(FaissBase):
    name = "faiss_ivfflat"
    knob_name = "nprobe"

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        self.nlist = nlist_for(n_cap if n_cap < 1_000_000 else 1_000_000)
        self.nprobe = 1
        self._train_buf = []
        self._pending = []
        self.idx = None

    def configure(self, n):
        self.nlist = nlist_for(n)

    def knobs(self, n):
        nl = nlist_for(n)
        return [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256] if b < nl] + [nl]

    def add(self, ids, X, grp=None):
        # buffer until finalize(): IVF must be trained on data first (40 points per cell, the same sample
        # size isongraph-vector's k-means uses). Training is part of the build cost.
        if self.idx is not None:          # already built: streaming insert
            self.idx.add_with_ids(X, ids)
        else:
            self._pending.append((ids, X))

    def finalize(self):
        f = self.faiss
        n = sum(len(i) for i, _ in self._pending)
        self.nlist = nlist_for(n)
        q = f.IndexFlatIP(self.d)
        self.idx = f.IndexIVFFlat(q, self.d, self.nlist, f.METRIC_INNER_PRODUCT)
        want = min(n, 40 * self.nlist)
        rng = np.random.default_rng(0)
        allX = np.concatenate([x for _, x in self._pending]) if len(self._pending) > 1 else self._pending[0][1]
        sample = allX[np.sort(rng.choice(n, want, replace=False))] if want < n else allX
        self.idx.train(sample)
        del allX
        for ids, X in self._pending:
            self.idx.add_with_ids(X, ids)
        self._pending = []

    def set_knob(self, v):
        self.nprobe = int(v)

    def _params(self):
        return self.faiss.SearchParametersIVF(nprobe=self.nprobe)

    def _params_with_sel(self, sel):
        return self.faiss.SearchParametersIVF(sel=sel, nprobe=self.nprobe)

    def upsert(self, ids, X):
        self.idx.remove_ids(self.faiss.IDSelectorBatch(ids))
        self.idx.add_with_ids(X, ids)

    def delete(self, ids):
        self.idx.remove_ids(self.faiss.IDSelectorBatch(ids))

    def load(self, path):
        self.idx = self.faiss.read_index(path)


class FaissHNSW(FaissBase):
    name = "faiss_hnsw"
    knob_name = "efSearch"

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        f = self.faiss
        self.idx = f.IndexHNSWFlat(d, 16, f.METRIC_INNER_PRODUCT)
        self.idx.hnsw.efConstruction = 200
        self.ef = 16
        self._next = 0

    def knobs(self, n):
        return [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400, 640]

    def add(self, ids, X, grp=None):
        assert ids[0] == self._next, "faiss HNSW assigns sequential ids"
        self.idx.add(X)
        self._next += len(ids)

    def set_knob(self, v):
        self.ef = int(v)

    def _params(self):
        return self.faiss.SearchParametersHNSW(efSearch=self.ef)

    def _params_with_sel(self, sel):
        return self.faiss.SearchParametersHNSW(sel=sel, efSearch=self.ef)

    def load(self, path):
        self.idx = self.faiss.read_index(path)


# ----------------------------------------------------------------------------- hnswlib
class Hnswlib(Base):
    name = "hnswlib"
    knob_name = "ef"
    supports_batch = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import hnswlib
        self.h = hnswlib
        self.version = "hnswlib " + getattr(hnswlib, "__version__", "0.8.0")
        self.idx = hnswlib.Index(space="ip", dim=d)
        self.idx.init_index(max_elements=n_cap, ef_construction=200, M=16, allow_replace_deleted=True)
        self.idx.set_num_threads(32)
        self.threads = 32
        self.grp = None

    def knobs(self, n):
        return [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400, 640]

    def set_threads(self, t):
        self.idx.set_num_threads(t)
        self.threads = t

    def add(self, ids, X, grp=None):
        self.idx.add_items(X, ids, num_threads=self.threads)

    def set_knob(self, v):
        self.idx.set_ef(int(v))

    def search1(self, q, k):
        return self.idx.knn_query(q[None, :], k=k, num_threads=1)[0][0]

    def searchN(self, Q, k):
        return self.idx.knn_query(Q, k=k, num_threads=self.threads)[0]

    def upsert(self, ids, X):
        self.idx.add_items(X, ids, num_threads=self.threads, replace_deleted=False)

    def delete(self, ids):
        for i in ids.tolist():
            self.idx.mark_deleted(i)

    def prepare_filter(self, grp_all):
        self.grp = grp_all

    def search_filtered1(self, q, k, g):
        grp = self.grp
        return self.idx.knn_query(q[None, :], k=k, num_threads=1, filter=lambda i: grp[i] == g)[0][0]

    def save(self, path):
        self.idx.save_index(path)

    def load(self, path):
        self.idx = self.h.Index(space="ip", dim=self.d)
        self.idx.load_index(path, max_elements=self.n_cap)
        self.idx.set_num_threads(self.threads)


# ----------------------------------------------------------------------------- usearch
class Usearch(Base):
    name = "usearch"
    knob_name = "expansion_search"
    supports_batch = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import usearch
        from usearch.index import Index
        self.Index = Index
        self.version = "usearch " + usearch.__version__
        self.idx = Index(ndim=d, metric="ip", dtype="f32", connectivity=16, expansion_add=200, expansion_search=16)
        self.threads = 32

    def knobs(self, n):
        return [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400, 640]

    def add(self, ids, X, grp=None):
        self.idx.add(ids.astype(np.uint64), X, threads=self.threads, copy=True)

    def set_knob(self, v):
        self.idx.expansion_search = int(v)

    def search1(self, q, k):
        return self.idx.search(q, k, threads=1).keys.astype(np.int64)

    def searchN(self, Q, k):
        return self.idx.search(Q, k, threads=self.threads).keys.astype(np.int64)

    def upsert(self, ids, X):
        self.idx.remove(ids.astype(np.uint64))
        self.idx.add(ids.astype(np.uint64), X, threads=self.threads, copy=True)

    def delete(self, ids):
        self.idx.remove(ids.astype(np.uint64))

    def save(self, path):
        self.idx.save(path)

    def load(self, path):
        self.idx = self.Index(ndim=self.d, metric="ip", dtype="f32", connectivity=16, expansion_add=200, expansion_search=16)
        self.idx.load(path)


# ----------------------------------------------------------------------------- chromadb
class Chroma(Base):
    name = "chromadb"
    knob_name = "ef_search"
    supports_batch = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import chromadb
        self.chromadb = chromadb
        self.version = "chromadb " + chromadb.__version__
        self.path = os.path.join(workdir, "chroma")
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
        shutil.rmtree(self.path, ignore_errors=True)
        self.client = chromadb.PersistentClient(path=self.path)
        self.col = self.client.create_collection(
            "bench", configuration={"hnsw": {"space": "ip", "ef_construction": 200, "max_neighbors": 16, "ef_search": 16}})
        self.has_meta = False

    def knobs(self, n):
        return [10, 16, 32, 64, 128, 256, 512]

    def add(self, ids, X, grp=None):
        step = 5000
        for s in range(0, len(ids), step):
            kw = {}
            if grp is not None:
                kw["metadatas"] = [{"grp": int(g)} for g in grp[s:s + step]]
            self.col.add(ids=[str(i) for i in ids[s:s + step].tolist()], embeddings=X[s:s + step], **kw)

    def _reopen(self):
        # chromadb 1.5.9 only applies a changed ef_search when the collection's HNSW index is reloaded
        # (verified: same-handle recall is unchanged after modify(), correct after reopening the client).
        from chromadb.api.shared_system_client import SharedSystemClient
        self.col = None
        self.client = None
        SharedSystemClient.clear_system_cache()
        self.client = self.chromadb.PersistentClient(path=self.path)
        self.col = self.client.get_collection("bench")

    def set_knob(self, v):
        self.col.modify(configuration={"hnsw": {"ef_search": int(v)}})
        self._reopen()

    def _ids(self, r):
        return np.array([[int(x) for x in row] for row in r["ids"]], dtype=np.int64)

    def search1(self, q, k):
        r = self.col.query(query_embeddings=[q], n_results=k, include=[])
        return self._ids(r)[0]

    def searchN(self, Q, k):
        out = []
        for s in range(0, len(Q), 500):
            out.append(self._ids(self.col.query(query_embeddings=Q[s:s + 500], n_results=k, include=[])))
        return np.concatenate(out)

    def upsert(self, ids, X):
        step = 5000
        for s in range(0, len(ids), step):
            self.col.upsert(ids=[str(i) for i in ids[s:s + step].tolist()], embeddings=X[s:s + step])

    def delete(self, ids):
        step = 5000
        for s in range(0, len(ids), step):
            self.col.delete(ids=[str(i) for i in ids[s:s + step].tolist()])

    def search_filtered1(self, q, k, g):
        r = self.col.query(query_embeddings=[q], n_results=k, where={"grp": int(g)}, include=[])
        return self._ids(r)[0]

    def save(self, path):
        pass  # PersistentClient writes as it goes; "save" cost is whatever add() already paid

    def load(self, path):
        self._reopen()

    def disk_bytes(self, path):
        return dir_bytes(self.path)

    def close(self):
        self.client = None


# ----------------------------------------------------------------------------- qdrant (local mode)
class QdrantLocal(Base):
    name = "qdrant_local"
    exact = True  # local mode is a brute-force numpy scan, not HNSW

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import qdrant_client
        from qdrant_client import QdrantClient, models
        self.models = models
        self.version = "qdrant-client " + __import__("importlib.metadata", fromlist=["version"]).version("qdrant-client") + " (local mode)"
        self.path = os.path.join(workdir, "qdrant")
        shutil.rmtree(self.path, ignore_errors=True)
        self.QC = QdrantClient
        import warnings
        warnings.filterwarnings("ignore")
        self.c = QdrantClient(path=self.path)
        self.c.create_collection("bench", vectors_config=models.VectorParams(size=d, distance=models.Distance.COSINE))

    def add(self, ids, X, grp=None):
        step = 2000
        for s in range(0, len(ids), step):
            payload = None if grp is None else [{"grp": int(g)} for g in grp[s:s + step]]
            self.c.upload_collection("bench", vectors=X[s:s + step], ids=ids[s:s + step].tolist(), payload=payload, batch_size=step)

    def search1(self, q, k):
        r = self.c.query_points("bench", query=q, limit=k)
        return np.array([p.id for p in r.points], dtype=np.int64)

    def upsert(self, ids, X):
        step = 2000
        for s in range(0, len(ids), step):
            self.c.upload_collection("bench", vectors=X[s:s + step], ids=ids[s:s + step].tolist(), batch_size=step)

    def delete(self, ids):
        self.c.delete("bench", points_selector=self.models.PointIdsList(points=ids.tolist()))

    def search_filtered1(self, q, k, g):
        m = self.models
        f = m.Filter(must=[m.FieldCondition(key="grp", match=m.MatchValue(value=int(g)))])
        r = self.c.query_points("bench", query=q, limit=k, query_filter=f)
        return np.array([p.id for p in r.points], dtype=np.int64)

    def save(self, path):
        self.c.close()

    def load(self, path):
        self.c = self.QC(path=self.path)

    def disk_bytes(self, path):
        return dir_bytes(self.path)

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- lancedb
class LanceBase(Base):
    knob_name = "nprobes"
    index = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import lancedb
        import pyarrow as pa
        self.lancedb, self.pa = lancedb, pa
        self.version = "lancedb " + lancedb.__version__
        self.path = os.path.join(workdir, "lance")
        shutil.rmtree(self.path, ignore_errors=True)
        self.db = lancedb.connect(self.path)
        self.tbl = None
        self.nprobes = 1
        self.has_grp = False

    def _tab(self, ids, X, grp):
        pa = self.pa
        cols = {"id": pa.array(ids, pa.int64()),
                "vector": pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1)), self.d)}
        if grp is not None:
            cols["grp"] = pa.array(grp.astype(np.int8), pa.int8())
        return pa.table(cols)

    def add(self, ids, X, grp=None):
        t = self._tab(ids, X, grp)
        if self.tbl is None:
            self.tbl = self.db.create_table("bench", data=t)
        else:
            self.tbl.add(t)

    def set_knob(self, v):
        self.nprobes = int(v)

    def _q(self, q):
        s = self.tbl.search(q, vector_column_name="vector")
        if self.index:
            s = s.nprobes(self.nprobes)
        return s

    def search1(self, q, k):
        return np.asarray(self._q(q).limit(k).select(["id"]).to_arrow()["id"].to_numpy())

    def upsert(self, ids, X):
        self.tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(self._tab(ids, X, None))

    def delete(self, ids):
        self.tbl.delete("id IN (" + ",".join(map(str, ids.tolist())) + ")")

    def search_filtered1(self, q, k, g):
        return np.asarray(self._q(q).where(f"grp = {int(g)}", prefilter=True).limit(k).select(["id"]).to_arrow()["id"].to_numpy())

    def save(self, path):
        pass  # lance writes datasets as it goes

    def load(self, path):
        self.db = self.lancedb.connect(self.path)
        self.tbl = self.db.open_table("bench")

    def disk_bytes(self, path):
        return dir_bytes(self.path)


class LanceIVF(LanceBase):
    name = "lancedb_ivfflat"

    def knobs(self, n):
        nl = nlist_for(n)
        return [b for b in [1, 2, 4, 8, 16, 32, 64, 128, 256] if b < nl] + [nl]

    def finalize(self):
        n = self.tbl.count_rows()
        self.tbl.create_index(metric="dot", index_type="IVF_FLAT", num_partitions=nlist_for(n), vector_column_name="vector")


class LanceFlat(LanceBase):
    name = "lancedb_flat"
    exact = True
    index = False

    def _q(self, q):
        return self.tbl.search(q, vector_column_name="vector").metric("dot")


# ----------------------------------------------------------------------------- sqlite-vec
class SqliteVec(Base):
    name = "sqlite_vec"
    exact = True

    def __init__(self, d, n_cap, workdir):
        super().__init__(d, n_cap, workdir)
        import sqlite3
        import sqlite_vec
        self.sv = sqlite_vec
        self.version = "sqlite-vec " + sqlite_vec.__version__ + " / sqlite " + sqlite3.sqlite_version
        self.path = os.path.join(workdir, "sv.db")
        for s in ("", "-wal", "-shm"):
            if os.path.exists(self.path + s):
                os.remove(self.path + s)
        self.sqlite3 = sqlite3
        self._open()
        self.has_grp = False

    def _open(self):
        self.db = self.sqlite3.connect(self.path)
        self.db.enable_load_extension(True)
        self.sv.load(self.db)
        self.db.enable_load_extension(False)

    def _create(self, grp):
        self.has_grp = grp is not None
        meta = ", grp integer" if self.has_grp else ""
        self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS v USING vec0(embedding float[{self.d}] distance_metric=cosine{meta})")

    def add(self, ids, X, grp=None):
        self._create(grp)
        cur = self.db.cursor()
        if grp is None:
            cur.executemany("INSERT INTO v(rowid, embedding) VALUES (?, ?)", ((int(i), x.tobytes()) for i, x in zip(ids, X)))
        else:
            cur.executemany("INSERT INTO v(rowid, embedding, grp) VALUES (?, ?, ?)", ((int(i), x.tobytes(), int(g)) for i, x, g in zip(ids, X, grp)))
        self.db.commit()

    def search1(self, q, k):
        rows = self.db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k = ?", (q.tobytes(), k)).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def upsert(self, ids, X):
        cur = self.db.cursor()
        for i, x in zip(ids.tolist(), X):
            cur.execute("DELETE FROM v WHERE rowid = ?", (i,))
            cur.execute("INSERT INTO v(rowid, embedding) VALUES (?, ?)", (i, x.tobytes()))
        self.db.commit()

    def delete(self, ids):
        self.db.executemany("DELETE FROM v WHERE rowid = ?", ((i,) for i in ids.tolist()))
        self.db.commit()

    def search_filtered1(self, q, k, g):
        rows = self.db.execute("SELECT rowid FROM v WHERE embedding MATCH ? AND k = ? AND grp = ?", (q.tobytes(), k, int(g))).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def save(self, path):
        self.db.commit()
        self.db.close()

    def load(self, path):
        self._open()

    def disk_bytes(self, path):
        return sum(os.path.getsize(self.path + s) for s in ("", "-wal") if os.path.exists(self.path + s))

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass


ADAPTERS = {c.name: c for c in (NumpyBF, IsonPy, IsonPyIVF, FaissFlat, FaissIVF, FaissHNSW, Hnswlib, Usearch, Chroma,
                                QdrantLocal, LanceIVF, LanceFlat, SqliteVec)}
