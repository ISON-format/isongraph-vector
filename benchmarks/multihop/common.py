"""Shared constants and helpers for the multi-hop retrieval benchmark."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DATA_DIR = Path(os.environ.get("MH_DATA_DIR", r"D:\Goldmine\ISON\bench_data\multihop"))
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = HERE / "results"
MANIFEST_DIR = HERE / "manifests"
for _d in (DATA_DIR, RAW_DIR, RESULTS_DIR, MANIFEST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = ["hotpotqa", "2wiki", "musique"]
MODELS = {
    # short name -> (HF id, query prefix, passage prefix)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge": ("BAAI/bge-small-en-v1.5",
            "Represent this sentence for searching relevant passages: ", ""),
}
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEED = 13
N_QUESTIONS = 1500      # sampled per dataset
N_DEV = 500             # hyper-parameter selection split; the rest (1000) is the test split
KS = (5, 10, 20)


def import_lib():
    """Import isongraph_vector from the repo checkout, unmodified."""
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    from _common import import_package
    return import_package()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj, path, indent=None):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
