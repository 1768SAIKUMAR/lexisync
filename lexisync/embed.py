"""Stage 3a - Embeddings and reranking models (+ offline stand-ins).

Bi-encoder (BAAI/bge-small-en-v1.5, 384-d): encodes query and passage
SEPARATELY. Every chunk is embedded once at index time; a query costs one
encoding + one matrix product. Fast, but the query never "sees" the passage.

Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2): reads "[query] [SEP]
[passage]" TOGETHER, so attention can compare words directly. Much more
accurate, but one forward pass per pair - so we only rerank the top 30.
Its raw output is a logit; sigmoid maps it to a 0..1 relevance, which the
abstention gate in Stage 4 thresholds.

Offline stand-ins (no download, deterministic) let tests and CI run anywhere:
  HashingEncoder  - words + word bigrams hashed into 384 dims, L2-normalised
  KeywordReranker - share of query content words found in the passage
They are NOT used for reported results.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_WORD = re.compile(r"[a-z0-9]+(?:[.()][a-z0-9]+)*\)?")
_STOP = set("""a an the of to in on for and or by with is are be been was were it its this
that these those as at from which what when how who whom whose where whether can
could shall should may might will would do does did under any all such than other
into their there them they he she his her not no""".split())


def offline_mode() -> bool:
    """LEXISYNC_OFFLINE=1 forces the stand-ins (tests, CI, no internet)."""
    return os.environ.get("LEXISYNC_OFFLINE", "") == "1"


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class Encoder(Protocol):
    name: str
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class Reranker(Protocol):
    name: str
    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


# ------------------------------------------------------------- real models
class STEncoder:
    def __init__(self, model_name: str, query_prefix: str = BGE_QUERY_PREFIX,
                 batch_size: int = 32, seed: int = 42):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.manual_seed(seed)
        self.name = model_name
        self.prefix = query_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # bge was trained with an instruction prefix on queries only;
        # passages are encoded as-is. Forgetting the prefix costs recall.
        if is_query:
            texts = [self.prefix + t for t in texts]
        return self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512):
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        logits = np.asarray(self.model.predict([(query, p) for p in passages],
                                               convert_to_numpy=True), dtype=np.float32)
        if logits.min() >= 0 and logits.max() <= 1:   # already probabilities
            return logits
        return 1.0 / (1.0 + np.exp(-logits))          # sigmoid -> 0..1


# ------------------------------------------------------ offline stand-ins
class HashingEncoder:
    name = "offline-hashing-384"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _bucket(self, tok: str) -> tuple[int, float]:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)   # stable, unlike hash()
        return h % self.dim, (1.0 if (h >> 64) & 1 else -1.0)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            ws = [w.rstrip("s") if len(w) > 3 else w for w in content_words(t)]
            for tok in ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]:
                j, sign = self._bucket(tok)
                out[i, j] += sign
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class KeywordReranker:
    name = "offline-keyword"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        q = {w.rstrip("s") for w in content_words(query)}
        if not q:
            return np.zeros(len(passages), dtype=np.float32)
        out = []
        for p in passages:
            ws = {w.rstrip("s") for w in content_words(p)}
            frac = len(q & ws) / len(q)
            out.append(1 / (1 + math.exp(-10 * (frac - 0.5))))   # squash to 0..1
        return np.asarray(out, dtype=np.float32)


# ------------------------------------------------------------------ factory
def get_encoder(cfg: dict) -> Encoder:
    r = cfg["retrieval"]
    if offline_mode() or r.get("offline"):
        return HashingEncoder()
    return STEncoder(r["embed_model"], r["query_prefix"], seed=cfg["seed"])


def get_reranker(cfg: dict) -> Reranker:
    r = cfg["retrieval"]
    if offline_mode() or r.get("offline"):
        return KeywordReranker()
    return CrossEncoderReranker(r["reranker_model"])
