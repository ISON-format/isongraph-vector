"""Embed a real passage corpus with all-MiniLM-L6-v2 (384-d) for the competitive benchmark.

Corpus: BeIR/hotpotqa (Wikipedia abstracts, 5.2M passages). We take a seeded random
sample of N passages (default 1,060,000: 1M for the scale tests + headroom for the
streaming-mutation inserts) and 2,000 real HotpotQA questions as queries.
Output (raw little-endian float32, row-major, L2-normalised): base.f32, queries.f32, meta.json

Run with any Python that has sentence-transformers + torch (a CUDA GPU makes it ~10 minutes):
    python prep_text.py --out D:/Goldmine/ISON/bench_data/prep/text384
"""
import argparse, json, time, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1_060_000)
ap.add_argument("--queries", type=int, default=2000)
ap.add_argument("--max-seq", type=int, default=128)
ap.add_argument("--seed", type=int, default=20261115)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import pandas as pd, torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

cp = hf_hub_download("BeIR/hotpotqa", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
qp = hf_hub_download("BeIR/hotpotqa", "queries/queries-00000-of-00001.parquet", repo_type="dataset")
corpus = pd.read_parquet(cp)
queries = pd.read_parquet(qp)
print("corpus", len(corpus), "queries", len(queries), corpus.columns.tolist(), queries.columns.tolist())
rng = np.random.default_rng(a.seed)
sel = np.sort(rng.choice(len(corpus), size=a.n, replace=False))
rng.shuffle(sel)  # random order: prefixes of the base file are random samples
texts = [(corpus.iloc[i]["title"] + ". " + corpus.iloc[i]["text"]).strip() for i in sel]
qsel = rng.choice(len(queries), size=a.queries, replace=False)
qtexts = [queries.iloc[i]["text"] for i in qsel]

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
model.max_seq_length = a.max_seq
if dev == "cuda":
    model.half()
def enc(ts, bs=512):
    return model.encode(ts, batch_size=bs, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False).astype(np.float32)

t0 = time.time()
qv = enc(qtexts)
qv /= np.linalg.norm(qv, axis=1, keepdims=True)
qv.tofile(os.path.join(a.out, "queries.f32"))
# stream in slices so a crash does not lose everything
out = open(os.path.join(a.out, "base.f32"), "wb")
step = 20000
for s in range(0, a.n, step):
    v = enc(texts[s:s + step])
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    (v / nrm).astype(np.float32).tofile(out)
    if (s // step) % 5 == 0:
        print(s + step, "done", round(time.time() - t0), "s", flush=True)
out.close()
json.dump({"name": "text384", "n": a.n, "d": 384, "q": a.queries, "source": "BeIR/hotpotqa corpus (seeded random sample) + queries",
           "model": "sentence-transformers/all-MiniLM-L6-v2", "device": dev, "max_seq_length": a.max_seq,
           "fp16": dev == "cuda", "seed": a.seed, "metric": "cosine (vectors L2-normalised)"},
          open(os.path.join(a.out, "meta.json"), "w"), indent=1)
print("done", round(time.time() - t0), "s")
