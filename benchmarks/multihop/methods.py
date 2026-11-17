"""Retrieval methods and metrics.  Every function returns rankings as a list of
int arrays (passage ids, best first, length <= MAXK) so all methods are scored by
the same code."""
from __future__ import annotations

import re
import time

import numpy as np
import scipy.sparse as sp

from common import KS, import_lib

MAXK = max(KS)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def metrics_at(rankings, questions, k):
    """Per-question arrays for a returned list truncated to k.
    recall = |gold ∩ top-k| / |gold|; all = 1 if every gold passage is within top-k;
    mrr = 1/rank of the first gold passage within top-k (0 if none: MRR@k);
    n_ret = number of passages actually returned (<= k)."""
    n = len(questions)
    rec, alls, mrr, nret = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i, (r, q) in enumerate(zip(rankings, questions)):
        gold = set(q["gold"])
        r = [int(x) for x in r[:k]]
        nret[i] = len(r)
        hit = len(gold.intersection(r))
        rec[i] = hit / len(gold)
        alls[i] = float(hit == len(gold))
        for pos, p in enumerate(r):
            if p in gold:
                mrr[i] = 1.0 / (pos + 1)
                break
    return {"recall": rec, "all": alls, "mrr": mrr, "n_ret": nret}


def set_metrics(sets, questions):
    """Metrics for unordered returned sets (no MRR, no k): recall, all, size."""
    n = len(questions)
    rec, alls, size = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, (s, q) in enumerate(zip(sets, questions)):
        gold = set(q["gold"])
        hit = len(gold.intersection(s))
        rec[i] = hit / len(gold)
        alls[i] = float(hit == len(gold))
        size[i] = len(s)
    return {"recall": rec, "all": alls, "size": size}


# ----------------------------------------------------------------------------
# dense vector search (what any vector store does)
# ----------------------------------------------------------------------------
def dense_scores(Q, P):
    return Q @ P.T  # cosine: both L2-normalised


def topn_from_scores(S, n):
    idx = np.argpartition(-S, n, axis=1)[:, :n]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, idx], axis=1, kind="stable")
    return idx[rows, order]


def vector_rankings(Q, P, n=MAXK):
    S = dense_scores(Q, P)
    return list(topn_from_scores(S, n)), S


# ----------------------------------------------------------------------------
# BM25 (Okapi, k1=1.2 b=0.75), own implementation
# ----------------------------------------------------------------------------
_STOP = set("""a an and are as at be but by for from has have he her his i in is it its of on or she that the their there
they this to was were which who whom will with you your we our us not no do does did so than then these those what when
where while about after also been being both can could during each had how into more most other over such up very would""".split())
_TOK = re.compile(r"[a-z0-9]+")


def _tok(s):
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        vocab, rows, cols, vals = {}, [], [], []
        lens = []
        for i, d in enumerate(docs):
            toks = _tok(d)
            lens.append(len(toks))
            tf = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                tf[j] = tf.get(j, 0) + 1
            for j, c in tf.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n = len(docs)
        self.vocab = vocab
        lens = np.array(lens, dtype=np.float64)
        avg = lens.mean()
        tfm = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
        df = np.asarray((tfm > 0).sum(0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        tfm = tfm.tocoo()
        denom = tfm.data + k1 * (1 - b + b * lens[tfm.row] / avg)
        w = tfm.data * (k1 + 1) / denom * idf[tfm.col]
        self.W = sp.csr_matrix((w, (tfm.row, tfm.col)), shape=tfm.shape).T.tocsr()  # vocab x docs

    def query_matrix(self, queries):
        rows, cols = [], []
        for i, q in enumerate(queries):
            for t in set(_tok(q)):
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(queries), len(self.vocab)))

    def scores(self, queries):
        return np.asarray((self.query_matrix(queries) @ self.W).todense(), dtype=np.float32)


def rrf(rank_lists_full, n=MAXK, k0=60):
    """Reciprocal rank fusion over full-score matrices' rank positions.
    rank_lists_full: list of score matrices (Q x N)."""
    fused = None
    for S in rank_lists_full:
        order = np.argsort(-S, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(S.shape[0])[:, None]
        ranks[rows, order] = np.arange(S.shape[1])[None, :]
        contrib = 1.0 / (k0 + ranks + 1)
        fused = contrib if fused is None else fused + contrib
    return list(topn_from_scores(fused.astype(np.float32), n))


# ----------------------------------------------------------------------------
# cross-encoder rerank of vector top-N
# ----------------------------------------------------------------------------
def rerank_scores(queries, cand, corpus, model_id, batch=128):
    import torch
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_id, device="cuda" if torch.cuda.is_available() else "cpu", max_length=256)
    pairs = []
    for q, c in zip(queries, cand):
        for p in c:
            pairs.append((q, f"{corpus[int(p)]['title']}. {corpus[int(p)]['text']}"))
    s = ce.predict(pairs, batch_size=batch, show_progress_bar=False)
    return np.asarray(s, dtype=np.float32).reshape(len(queries), -1)


def rerank_rankings(cand, scores, n=MAXK):
    out = []
    for c, s in zip(cand, scores):
        o = np.argsort(-s, kind="stable")[:n]
        out.append(np.asarray(c)[o])
    return out


# ----------------------------------------------------------------------------
# Personalised PageRank re-ranking (independent of the library; HippoRAG-style)
# ----------------------------------------------------------------------------
def build_adjacency(n, edges, fams, directed_mention, weighted):
    """Sparse adjacency A (row = source).  mention edges are directed unless
    directed_mention is False (then symmetrised); entity/knn are symmetric."""
    rows, cols, vals = [], [], []
    for f in fams:
        if f.startswith("mention"):
            s, d = edges[f]
            rows += [s]; cols += [d]; vals += [np.ones(len(s))]
            if not directed_mention:
                rows += [d]; cols += [s]; vals += [np.ones(len(s))]
        else:
            a, b, w = edges[f]
            w = w.astype(np.float64) if weighted else np.ones(len(a))
            rows += [a, b]; cols += [b, a]; vals += [w, w]
    if not rows:
        return sp.csr_matrix((n, n))
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def ppr_rankings(S, A, seeds_k, alpha, beta, n=MAXK, iters=25, block=256):
    """S: Q x N dense cosine scores.  Seeds = top `seeds_k` by cosine, weight ∝ max(sim,0).
    p <- alpha * P^T p + (1-alpha) * s   (dangling mass restarts at s), 25 power iterations.
    final = (1-beta) * p/max(p) + beta * max(sim,0)/max(sim)   -> top n."""
    nq, N = S.shape
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    PT = (sp.diags(inv) @ A).T.tocsr().astype(np.float32)   # column-stochastic for non-dangling nodes
    dangling = (deg == 0).astype(np.float32)
    out = []
    for s0 in range(0, nq, block):
        Sb = S[s0:s0 + block]
        nb = Sb.shape[0]
        top = np.argpartition(-Sb, seeds_k, axis=1)[:, :seeds_k]
        rows = np.arange(nb)[:, None]
        R = np.zeros((N, nb), dtype=np.float32)
        R[top, rows] = np.maximum(Sb[rows, top], 0).astype(np.float32)
        R += 0.0
        sums = R.sum(0, keepdims=True)
        sums[sums == 0] = 1.0
        R /= sums
        p = R.copy()
        for _ in range(iters):
            dm = (dangling[:, None] * p).sum(0, keepdims=True)
            p = alpha * (PT @ p) + (alpha * dm + (1 - alpha)) * R
        p = p.T
        pn = p / np.maximum(p.max(1, keepdims=True), 1e-12)
        sim = np.maximum(Sb, 0)
        sn = sim / np.maximum(sim.max(1, keepdims=True), 1e-12)
        fin = (1 - beta) * pn + beta * sn
        out += list(topn_from_scores(fin.astype(np.float32), n))
    return out


# ----------------------------------------------------------------------------
# ISONGraph Vector (library under test, unmodified)
# ----------------------------------------------------------------------------
def build_lib_graph(corpus, P, edges, fams):
    lib = import_lib()
    from ison_graph import Direction  # noqa: F401
    g = lib.SemanticGraph(name="mh", directed=True, embedding_db=":memory:", auto_embed=False)
    for c in corpus:
        g.add_node("passage", c["id"], title=c["title"])
    g.embedding_store.add_batch([(("passage", c["id"]), c["title"], P[c["id"]].tolist()) for c in corpus])
    ref = lambda i: ("passage", int(i))  # noqa: E731
    for f in fams:
        if f.startswith("mention"):
            s, d = edges[f]
            for a, b in zip(s.tolist(), d.tolist()):
                g.add_edge("MENTIONS", ref(a), ref(b), w=1.0)
        else:
            a_, b_, w_ = edges[f]
            rel = {"entity": "SHARES_ENTITY", "entity_rand": "SHARES_ENTITY", "knn": "NEAR"}[f]
            for a, b, w in zip(a_.tolist(), b_.tolist(), w_.tolist()):
                g.add_edge(rel, ref(a), ref(b), w=float(w))
                g.add_edge(rel, ref(b), ref(a), w=float(w))
    return g


def lib_multihop_rankings(g, Q, params, n=MAXK, idx=None, timing=None):
    """Run semantic_multi_hop with a precomputed query vector for each query."""
    from ison_graph import Direction
    direction = Direction.BOTH if params["direction"] == "both" else Direction.OUT
    out = []
    qs = range(len(Q)) if idx is None else idx
    for i in qs:
        t0 = time.perf_counter()
        res = g.semantic_multi_hop(
            Q[i].tolist(), max_hops=params["hops"], top_k_seeds=params["seeds"], top_k_results=n,
            direction=direction, decay=params["decay"], threshold=-1.0, blend=params["blend"],
            weight_field=params["weight_field"])
        if timing is not None:
            timing.append(time.perf_counter() - t0)
        out.append(np.array([r.node_ref[1] for r in res], dtype=np.int64))
    return out


def lib_subgraph_sets(g, Q, params, k, connect, connect_hops, idx=None, timing=None):
    from ison_graph import Direction
    direction = Direction.BOTH if params["direction"] == "both" else Direction.OUT
    out = []
    qs = range(len(Q)) if idx is None else idx
    for i in qs:
        t0 = time.perf_counter()
        sub = g.semantic_subgraph(
            Q[i].tolist(), max_hops=params["hops"], top_k_seeds=params["seeds"], top_k_results=k,
            direction=direction, decay=params["decay"], threshold=-1.0, blend=params["blend"],
            weight_field=params["weight_field"], connect=connect, connect_max_hops=connect_hops)
        if timing is not None:
            timing.append(time.perf_counter() - t0)
        out.append(set(int(n.id) for n in sub.nodes()))
    return out
