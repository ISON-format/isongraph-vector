"""Embed corpus passages and questions once, cache as .npy outside the repo.

Files (MH_DATA_DIR):  emb_<ds>__<model>__passages.npy   (N x d, float32, L2-normalised)
                      emb_<ds>__<model>__queries.npy    (Q x d, float32, L2-normalised)
Passage text embedded = "<title>. <text>" ; max_seq_length 256 for both models.
bge-small-en-v1.5 gets its documented query instruction prefix on questions only.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import DATA_DIR, MODELS, DATASETS, load_json


def emb_paths(ds, model):
    return (DATA_DIR / f"emb_{ds}__{model}__passages.npy", DATA_DIR / f"emb_{ds}__{model}__queries.npy")


def load_embeddings(ds, model):
    p, q = emb_paths(ds, model)
    return np.load(p), np.load(q)


def embed(ds, model, force=False):
    pp, qp = emb_paths(ds, model)
    if pp.exists() and qp.exists() and not force:
        print(f"[embed] cached {ds}/{model}")
        return
    import torch
    from sentence_transformers import SentenceTransformer
    hf_id, qprefix, pprefix = MODELS[model]
    corpus = load_json(DATA_DIR / f"corpus_{ds}.json")
    questions = load_json(DATA_DIR / f"questions_{ds}.json")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(hf_id, device=dev)
    m.max_seq_length = 256
    t0 = time.time()
    ptexts = [pprefix + f"{c['title']}. {c['text']}" for c in corpus]
    P = m.encode(ptexts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False).astype(np.float32)
    Q = m.encode([qprefix + q["question"] for q in questions], batch_size=128, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    np.save(pp, P)
    np.save(qp, Q)
    print(f"[embed] {ds}/{model}: P{P.shape} Q{Q.shape} on {dev} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    for ds in DATASETS:
        for model in MODELS:
            embed(ds, model)
