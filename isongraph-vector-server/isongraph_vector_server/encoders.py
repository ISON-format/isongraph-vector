"""Build the one shared encoder from config and bound how it is used.

The encoder is shared by every tenant, so it is the one place where tenants
compete for a resource. :class:`BoundedEncoder` caps simultaneous calls and
makes a caller wait only a bounded time for a slot (raising ``Overloaded``),
so a slow remote embedding endpoint cannot absorb every worker thread.
Texts are encoded *outside* any graph lock: a slow encoder stalls the one
request, never a whole graph.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

from isongraph_vector import EmbeddingEncoder, EmbeddingError

from .config import EncoderConfig
from .errors import ApiError, Overloaded, Unavailable

ENCODE_CHUNK = 64


class BoundedEncoder(EmbeddingEncoder):
    def __init__(self, inner: EmbeddingEncoder, max_concurrency: int, wait: float = 5.0,
                 serialize: bool = False):
        self._inner = inner
        self._sem = threading.BoundedSemaphore(max(1, max_concurrency))
        self._wait = wait
        self._serial = threading.Lock() if serialize else None
        self.model_name = getattr(inner, "model_name", "unknown")
        self.version = getattr(inner, "version", None)

    @property
    def dimension(self) -> int:
        return int(self._inner.dimension)

    def _call(self, fn, *args):
        if not self._sem.acquire(timeout=self._wait):
            raise Overloaded("embedding capacity exhausted; retry shortly",
                             headers={"Retry-After": "1"})
        try:
            if self._serial is not None:
                with self._serial:
                    return fn(*args)
            return fn(*args)
        except ApiError:
            raise
        except EmbeddingError as exc:
            # Do not echo the encoder's message: it can quote the upstream URL or body.
            raise Unavailable("the embedding service failed", code="encoder_failed") from exc
        finally:
            self._sem.release()

    def encode(self, text: str) -> List[float]:
        return self._call(self._inner.encode, text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        texts = list(texts)
        out: List[List[float]] = []
        for i in range(0, len(texts), ENCODE_CHUNK):
            out.extend(self._call(self._inner.encode_batch, texts[i:i + ENCODE_CHUNK]))
        return out


def build_encoder(cfg: EncoderConfig, env=None) -> Optional[EmbeddingEncoder]:
    env = os.environ if env is None else env
    if cfg.kind == "none":
        return None
    if cfg.kind == "mock":
        from isongraph_vector import MockEncoder
        inner: EmbeddingEncoder = MockEncoder(cfg.dimension, version=cfg.version)
        serialize = False
    elif cfg.kind == "http":
        from isongraph_vector import HttpEncoder
        key = env.get(cfg.api_key_env) if cfg.api_key_env else None
        inner = HttpEncoder(cfg.url, cfg.model, api_key=key, dimension=cfg.dimension,
                            timeout=cfg.timeout, api=cfg.api, version=cfg.version)
        serialize = False
    elif cfg.kind == "sentence-transformers":
        from isongraph_vector import SentenceTransformerEncoder
        inner = SentenceTransformerEncoder(cfg.model or "all-MiniLM-L6-v2", version=cfg.version)
        serialize = True   # one model instance; do not assume concurrent encode() is safe
    else:   # pragma: no cover - validated in config
        raise ValueError(cfg.kind)
    return BoundedEncoder(inner, cfg.max_concurrency, serialize=serialize)
