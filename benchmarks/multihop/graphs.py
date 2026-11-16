"""Graph construction that uses ONLY passage text/titles (and, for kNN, the passage
embeddings).  No question, no answer, no supporting-fact annotation is touched.

Edge families
  mention : directed  A -> B  when B's title (parenthetical stripped, >=4 chars,
            case-sensitive exact token match) occurs in A's text.  Titles that
            occur in more than MENTION_CAP passages are dropped as hubs.
            (A "hyperlink" proxy: the HF copies carry no Wikipedia link markup.)
  entity  : undirected A -- B when they share >=1 'entity', an entity being a
            maximal run of capitalised tokens (with of/the/de/... connectors),
            sentence-initial single tokens and a small stop list removed, plus
            the passage's own title.  Entities with document frequency outside
            [2, ENTITY_CAP] are dropped (singletons carry no edge, hubs no
            information).  Edge weight w = n/(n+1) for n shared entities.
  knn     : undirected A -- B when B is among A's KNN_K nearest passages (cosine,
            same encoder as retrieval) or vice-versa.  Weight w = cosine.
Union graphs combine families under separate relationship types.
Constants are fixed a priori (not tuned on any question).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

MENTION_CAP = 30
ENTITY_CAP = 15
KNN_K = 5

_TOK = re.compile(r"\w[\w'’\-]*|[.!?]")
_WORD = re.compile(r"\w+")
_CONNECT = {"of", "the", "de", "del", "la", "le", "von", "van", "der", "den", "du", "da", "di", "bin", "al", "and", "for", "in"}
_CAP_STOP = {
    "the", "a", "an", "in", "on", "at", "of", "he", "she", "it", "his", "her", "its", "they", "their", "this", "that",
    "these", "those", "after", "before", "during", "as", "by", "for", "from", "with", "when", "while", "since", "although",
    "however", "also", "but", "and", "or", "if", "then", "there", "who", "which", "what", "where", "one", "two", "three",
    "first", "second", "following", "following", "both", "some", "many", "most", "other", "such", "since", "until", "upon",
    "into", "over", "under", "between", "among", "through", "about", "against", "within", "without", "he", "we", "i", "you",
    "mr", "mrs", "dr", "st", "jr", "sr", "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "born", "died", "known", "called", "named", "currently", "originally", "today", "now",
}


def strip_title(t: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()


def _words(text):
    return _WORD.findall(text)


def mention_edges(corpus):
    """-> (src, dst) int arrays, directed mentioner -> mentioned."""
    title_map = defaultdict(list)
    maxn = 1
    for c in corpus:
        base = strip_title(c["title"])
        if len(base) < 4:
            continue
        toks = tuple(_words(base))
        if not toks:
            continue
        title_map[toks].append(c["id"])
        maxn = max(maxn, len(toks))
    maxn = min(maxn, 12)
    pairs = defaultdict(set)  # title_key -> set of mentioning passages
    for c in corpus:
        toks = _words(c["text"])
        n = len(toks)
        for i in range(n):
            for L in range(1, min(maxn, n - i) + 1):
                key = tuple(toks[i:i + L])
                if key in title_map:
                    pairs[key].add(c["id"])
    src, dst = [], []
    for key, mentioners in pairs.items():
        if len(mentioners) > MENTION_CAP:
            continue
        for a in mentioners:
            for b in title_map[key]:
                if a != b:
                    src.append(a)
                    dst.append(b)
    pr = sorted(set(zip(src, dst)))
    return (np.array([p[0] for p in pr], dtype=np.int64), np.array([p[1] for p in pr], dtype=np.int64))


def extract_entities(text: str):
    toks = _TOK.findall(text)
    ents = set()
    i, n = 0, len(toks)
    sent_start = True
    while i < n:
        t = toks[i]
        if t in ".!?":
            sent_start = True
            i += 1
            continue
        if t[0].isupper():
            j = i + 1
            last_cap = i
            while j < n:
                u = toks[j]
                if u not in ".!?" and u[0].isupper():
                    last_cap = j
                    j += 1
                elif (u.lower() in _CONNECT and j + 1 < n
                      and toks[j + 1] not in ".!?" and toks[j + 1][0].isupper()):
                    j += 1
                else:
                    break
            span = toks[i:last_cap + 1]
            initial = sent_start
            while span and span[0].lower() in _CAP_STOP:
                span = span[1:]
                initial = False
            # a lone sentence-initial capitalised word is just a capitalised word
            if span and not (len(span) == 1 and initial):
                ents.add(" ".join(span).lower())
            i = last_cap + 1
            sent_start = False
            continue
        sent_start = False
        i += 1
    return ents


def entity_edges(corpus):
    """-> (a, b, w) arrays, a<b, undirected."""
    inv = defaultdict(list)
    for c in corpus:
        ents = extract_entities(c["text"])
        ents.add(strip_title(c["title"]).lower())
        for e in ents:
            inv[e].append(c["id"])
    shared = defaultdict(int)
    for e, docs in inv.items():
        if not (2 <= len(docs) <= ENTITY_CAP):
            continue
        docs = sorted(set(docs))
        for x in range(len(docs)):
            for y in range(x + 1, len(docs)):
                shared[(docs[x], docs[y])] += 1
    items = sorted(shared.items())
    a = np.array([k[0] for k, _ in items], dtype=np.int64)
    b = np.array([k[1] for k, _ in items], dtype=np.int64)
    w = np.array([n / (n + 1.0) for _, n in items], dtype=np.float32)
    return a, b, w


def knn_edges(P: np.ndarray, k: int = KNN_K, block: int = 2048):
    """Symmetrised kNN graph from L2-normalised passage embeddings. -> (a, b, w), a<b."""
    n = P.shape[0]
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Pt = torch.from_numpy(P).to(dev)
        nbrs = []
        for s in range(0, n, block):
            sim = Pt[s:s + block] @ Pt.T
            idx = torch.arange(s, min(s + block, n), device=dev)
            sim[torch.arange(len(idx), device=dev), idx] = -2.0
            top = torch.topk(sim, k, dim=1).indices.cpu().numpy()
            nbrs.append(top)
        nbrs = np.concatenate(nbrs, 0)
    except Exception:
        nbrs = np.zeros((n, k), dtype=np.int64)
        for s in range(0, n, block):
            sim = P[s:s + block] @ P.T
            for r in range(sim.shape[0]):
                sim[r, s + r] = -2.0
            nbrs[s:s + block] = np.argpartition(-sim, k, axis=1)[:, :k]
    pairs = set()
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            pairs.add((i, j) if i < j else (j, i))
    pr = sorted(pairs)
    a = np.array([p[0] for p in pr], dtype=np.int64)
    b = np.array([p[1] for p in pr], dtype=np.int64)
    w = np.clip((P[a] * P[b]).sum(1), 0.0, 1.0).astype(np.float32)
    return a, b, w


VARIANTS = {
    # name -> families
    "mention": ("mention",),
    "entity": ("entity",),
    "knn": ("knn",),
    "mention+entity": ("mention", "entity"),
    "all": ("mention", "entity", "knn"),
}


CONTROLS = {
    # negative controls: the SAME edge lists with node labels randomly permuted (seeded), i.e. an isomorphic
    # graph whose edges no longer line up with the passages' text.  Any gain here is an artefact of
    # expansion/re-ranking mechanics, not of graph content.
    "ctrl_mention": ("mention_rand",),
    "ctrl_entity": ("entity_rand",),
}
ALL_VARIANTS = {**VARIANTS, **CONTROLS}


def permute_edges(edges, n, seed=7):
    rng = np.random.default_rng(seed)
    pi = rng.permutation(n)
    out = {}
    s, d = edges["mention"]
    out["mention_rand"] = (pi[s], pi[d])
    a, b, w = edges["entity"]
    out["entity_rand"] = (pi[a], pi[b], w)
    return out


def graph_diagnostics(fams, edges, questions, n_nodes):
    """Descriptive statistics.  Gold annotations are used HERE ONLY (reporting how
    useful an already-built graph is); they never influence construction."""
    und = defaultdict(set)    # undirected adjacency union over families in `fams`
    n_edges = 0
    for f in fams:
        if f.startswith("mention"):
            s, d = edges[f]
            n_edges += len(s)
            for a, b in zip(s.tolist(), d.tolist()):
                und[a].add(b)
                und[b].add(a)
        else:
            a_, b_, _ = edges[f]
            n_edges += len(a_)
            for a, b in zip(a_.tolist(), b_.tolist()):
                und[a].add(b)
                und[b].add(a)
    deg = np.array([len(und[i]) for i in range(n_nodes)])
    linked1 = linked2 = tot = 0
    frac_ctx = []
    for q in questions:
        g = q["gold"]
        ctx = set(q["context"])
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                tot += 1
                if g[y] in und[g[x]]:
                    linked1 += 1
                elif und[g[x]] & und[g[y]]:
                    linked2 += 1
        nb = set()
        for x in g:
            nb |= und[x]
        nb -= set(g)
        if nb:
            frac_ctx.append(len(nb & ctx) / len(nb))
    return {
        "n_edges_undirected_union": int(sum(len(v) for v in und.values()) // 2),
        "n_edges_by_family_directed_or_undirected": int(n_edges),
        "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)), "max_degree": int(deg.max()),
        "isolated_frac": float((deg == 0).mean()),
        "gold_pairs_directly_linked": linked1 / max(tot, 1),
        "gold_pairs_linked_within_2": (linked1 + linked2) / max(tot, 1),
        "neighbours_of_gold_that_are_in_same_question_context": float(np.mean(frac_ctx)) if frac_ctx else None,
    }
