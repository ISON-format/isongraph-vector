"""Sample questions and build one pooled retrieval corpus per dataset.

Sources (dev sets only, pinned HuggingFace revisions):
  hotpotqa : hotpotqa/hotpot_qa            distractor/validation  rev 1908d6af...
  2wiki    : framolfese/2WikiMultihopQA    data/validation        rev fe713bfb...
  musique  : dgslibisey/MuSiQue            musique_ans_v1.0_dev   rev c8f4f8c9...
             (the answerable dev split of the official MuSiQue v1.0 release)

For each dataset: N_QUESTIONS questions are sampled with a fixed seed; their
context paragraphs (gold + the dataset's own distractors) are pooled and
de-duplicated by (title, text) into ONE corpus that every question is searched
against.  Questions are split into a `dev` split (hyper-parameter selection)
and a `test` split (reported numbers).

Outputs (outside the repo, in MH_DATA_DIR):  corpus_<ds>.json, questions_<ds>.json
Outputs (in the repo):                        manifests/<ds>_sample.json (ids+split)
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

from common import (DATA_DIR, MANIFEST_DIR, N_DEV, N_QUESTIONS, RAW_DIR, SEED,
                    dump_json)

SOURCES = {
    "hotpotqa": {"repo": "hotpotqa/hotpot_qa", "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
                 "file": "distractor/validation-00000-of-00001.parquet"},
    "2wiki": {"repo": "framolfese/2WikiMultihopQA", "revision": "fe713bfbd1afbca1a65246741a75890405d56a3a",
              "file": "data/validation-00000-of-00001.parquet"},
    "musique": {"repo": "dgslibisey/MuSiQue", "revision": "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321",
                "file": "musique_ans_v1.0_dev.jsonl"},
}


def download(ds):
    from huggingface_hub import hf_hub_download
    s = SOURCES[ds]
    return hf_hub_download(s["repo"], s["file"], repo_type="dataset", revision=s["revision"],
                           local_dir=str(RAW_DIR / s["repo"].replace("/", "__")))


def load_raw(ds):
    """-> list of dict(id, question, type, hops, paragraphs=[(title,text)], gold=indices)"""
    path = download(ds)
    out = []
    if ds in ("hotpotqa", "2wiki"):
        import pandas as pd
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            titles = list(r["context"]["title"])
            sents = r["context"]["sentences"]
            paras = [(t, " ".join(x.strip() for x in s).strip()) for t, s in zip(titles, sents)]
            gold_titles = set(r["supporting_facts"]["title"])
            gold = [i for i, (t, _) in enumerate(paras) if t in gold_titles]
            missing = gold_titles - {t for t, _ in paras}
            if missing:
                continue  # supporting title absent from the context: drop
            out.append({"id": r["id"], "question": r["question"], "type": r["type"],
                        "hops": len(gold_titles), "paragraphs": paras, "gold": gold})
    else:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                paras = [(p["title"], p["paragraph_text"].strip()) for p in r["paragraphs"]]
                gold = [i for i, p in enumerate(r["paragraphs"]) if p["is_supporting"]]
                prefix = r["id"].split("_")[0]
                out.append({"id": r["id"], "question": r["question"], "type": prefix,
                            "hops": int(prefix[0]), "paragraphs": paras, "gold": gold})
    return out


def build(ds):
    rows = load_raw(ds)
    rows.sort(key=lambda r: r["id"])
    rng = random.Random(SEED)
    sample = rng.sample(rows, min(N_QUESTIONS, len(rows)))
    order = list(range(len(sample)))
    rng.shuffle(order)
    split = {}
    for rank, i in enumerate(order):
        split[sample[i]["id"]] = "dev" if rank < N_DEV else "test"

    key2pid, corpus = {}, []
    questions = []
    for r in sample:
        pids = []
        for title, text in r["paragraphs"]:
            k = (title, text)
            if k not in key2pid:
                key2pid[k] = len(corpus)
                corpus.append({"id": len(corpus), "title": title, "text": text})
            pids.append(key2pid[k])
        gold = sorted({pids[i] for i in r["gold"]})
        questions.append({"id": r["id"], "question": r["question"], "type": r["type"], "hops": r["hops"],
                          "gold": gold, "n_gold": len(gold), "split": split[r["id"]],
                          "context": pids})
    dump_json(corpus, DATA_DIR / f"corpus_{ds}.json")
    dump_json(questions, DATA_DIR / f"questions_{ds}.json")
    dump_json({"dataset": ds, "source": SOURCES[ds], "seed": SEED, "n_questions": len(questions),
               "n_dev": sum(q["split"] == "dev" for q in questions),
               "n_test": sum(q["split"] == "test" for q in questions),
               "corpus_size": len(corpus), "n_available": len(rows),
               "ids": [[q["id"], q["split"]] for q in questions],
               "corpus_sha1": hashlib.sha1(json.dumps([(c["title"], c["text"]) for c in corpus]).encode()).hexdigest()},
              MANIFEST_DIR / f"{ds}_sample.json")
    c = Counter(q["type"] for q in questions)
    h = Counter(q["n_gold"] for q in questions)
    print(f"{ds}: available={len(rows)} sampled={len(questions)} corpus={len(corpus)} "
          f"passages (raw pooled {sum(len(r['paragraphs']) for r in sample)}) types={dict(c)} gold-count={dict(h)}")


if __name__ == "__main__":
    import sys
    for ds in (sys.argv[1:] or SOURCES):
        build(ds)
