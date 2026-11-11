"""Stage 3b - Hybrid retrieval: BM25 + dense, RRF fusion, cross-encoder rerank,
and a query planner for two-part questions.

Modes (compared in the evaluation):
  bm25             lexical only
  dense            bge embeddings, cosine similarity
  hybrid           RRF(bm25, dense) -> top `candidates`
  hybrid_rerank    hybrid, then cross-encoder rescoring of the candidates
  hybrid_rerank_mq hybrid_rerank per sub-query (planner), best of each part kept

RRF: score(d) = sum over rankers of 1 / (k + rank_r(d)), k = 60.
Fusing RANKS avoids comparing a BM25 score (unbounded, ~0-40) with a cosine
(-1..1). k = 60 dampens the head: rank 1 vs rank 2 differ by ~1.6 %, so one
ranker alone can't dominate, and a passage ranked well by BOTH wins.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import load_config
from .chunking import chunk_document, index_text
from .embed import content_words, get_encoder, get_reranker
from .ingest import ingest_pdf
from .models import Chunk, Document

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_mq")
INDEX_VERSION = 4        # bump when ingestion/chunking output changes

# ------------------------------------------------------------- tokenizer
# Legal references first, so "83(5)", "7.2", "2016/679" survive as one token;
# then ordinary words. Punctuation-splitting tokenizers would turn "7.2" into
# "7" and "2", and every clause numbered 7.x or x.2 would match.
_REF = re.compile(r"\d+(?:\.\d+)+|\d+\(\d+\)(?:\([a-z]{1,3}\))?|\d+/\d+|[a-z]+\d*|\d+")
_STOP = set("""a an the of to in on for and or by with is are be been was were it its this
that these those as at from which what when how who whom where whether can could shall
should may might will would do does did any all such than other into their there them
they not no""".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    for t in _REF.findall(text):
        if t in _STOP:
            continue
        if t.isalpha() and len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]                                   # fines -> fine
        out.append(t)
        # every prefix of a reference: 83(5)(a) -> 83(5), 83 ; 7.2.1 -> 7.2, 7
        if re.match(r"^\d+[.(]", t):
            for m in re.finditer(r"(?=[.(])", t):
                if m.start():
                    out.append(t[:m.start()])
    return out


# ------------------------------------------------------------------ planner
_SPLIT = re.compile(r",?\s+and\s+(?=(?:what|when|which|how|who|where|whether|why|"
                    r"does|do|is|are|can|must)\b)", re.I)


def plan_query(query: str) -> list[str]:
    """Split "A, and what/which/how ... B" into sub-queries. A vague second
    part ("and which ones apply?") inherits the topic words of the first part,
    otherwise it would retrieve almost anything."""
    parts = [p.strip(" ?,.") for p in _SPLIT.split(query) if p and p.strip(" ?,.")]
    parts = [p for p in parts if not re.fullmatch(r"(?i)what|when|which|how|who|where|"
                                                    r"whether|why|does|do|is|are|can|must", p)]
    if len(parts) < 2:
        return [query]
    topic = [w for w in content_words(parts[0])
             if w not in {"maximum", "minimum", "much", "many", "company"}][:5]
    out = [parts[0]]
    for p in parts[1:]:
        cw = content_words(p)
        if len(cw) < 6 and not set(cw) & set(topic):
            p = f"{p} ({' '.join(topic)})"
        out.append(p)
    return out


# -------------------------------------------------------------------- index
@dataclass
class Hit:
    chunk: Chunk
    score: float                       # mode-specific final score
    bm25_rank: int | None = None
    dense_rank: int | None = None
    relevance: float | None = None     # cross-encoder sigmoid, 0..1
    subquery: str = ""


@dataclass
class SearchResult:
    query: str
    mode: str
    hits: list[Hit]
    subqueries: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)       # ms

    @property
    def best_relevance(self) -> float | None:
        rel = [h.relevance for h in self.hits if h.relevance is not None]
        return max(rel) if rel else None


class Index:
    def __init__(self, chunks: list[Chunk], docs: list[Document], cfg: dict,
                 encoder=None, reranker=None):
        self.cfg = cfg
        self.chunks = chunks
        self.docs = {d.doc_id: d for d in docs}
        self._encoder = encoder
        self._reranker = reranker
        r = cfg["retrieval"]
        self.texts = [index_text(c) for c in chunks]
        self.tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens, k1=r["bm25_k1"], b=r["bm25_b"])
        self.emb = self.encoder.encode(self.texts) if chunks else np.zeros((0, 384))
        self.encoder_name = self.encoder.name

    # lazily created so a loaded index doesn't load models until needed
    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = get_encoder(self.cfg)
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(self.cfg)
        return self._reranker

    # -------------------------------------------------------------- persist
    def save(self, path: Path, fingerprint: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"fingerprint": fingerprint, "chunks": self.chunks, "docs": self.docs,
                 "emb": self.emb, "encoder_name": self.encoder_name}
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path, cfg: dict, fingerprint: str) -> "Index | None":
        try:
            with open(path, "rb") as f:
                st = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
            return None
        if st.get("fingerprint") != fingerprint:
            return None
        self = cls.__new__(cls)
        self.cfg, self.chunks, self.docs = cfg, st["chunks"], st["docs"]
        self._encoder = self._reranker = None
        r = cfg["retrieval"]
        self.texts = [index_text(c) for c in self.chunks]
        self.tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens, k1=r["bm25_k1"], b=r["bm25_b"])
        self.emb, self.encoder_name = st["emb"], st["encoder_name"]
        return self

    # --------------------------------------------------------------- search
    def _mask(self, doc_ids) -> np.ndarray:
        if not doc_ids:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([c.doc_id in doc_ids for c in self.chunks])

    def rank_bm25(self, query: str, mask: np.ndarray) -> list[int]:
        s = self.bm25.get_scores(tokenize(query))
        s = np.where(mask, s, -np.inf)
        order = np.argsort(-s, kind="stable")
        return [int(i) for i in order if np.isfinite(s[i]) and s[i] > 0]

    def rank_dense(self, query: str, mask: np.ndarray) -> list[int]:
        q = self.encoder.encode([query], is_query=True)[0]
        s = np.where(mask, self.emb @ q, -np.inf)       # cosine (unit vectors)
        order = np.argsort(-s, kind="stable")
        return [int(i) for i in order if np.isfinite(s[i])]

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
        return fused

    def _single(self, query: str, mode: str, k: int, mask, timings) -> list[Hit]:
        r = self.cfg["retrieval"]
        t0 = time.perf_counter()
        b = self.rank_bm25(query, mask) if mode != "dense" else []
        d = self.rank_dense(query, mask) if mode != "bm25" else []
        b_rank = {i: n for n, i in enumerate(b, 1)}
        d_rank = {i: n for n, i in enumerate(d, 1)}
        if mode == "bm25":
            ids, scores = b, [1.0 / n for n in range(1, len(b) + 1)]
        elif mode == "dense":
            ids, scores = d, [1.0 / n for n in range(1, len(d) + 1)]
        else:
            fused = self.rrf([b[:100], d[:100]], r["rrf_k"])
            ids = sorted(fused, key=lambda i: -fused[i])
            scores = [fused[i] for i in ids]
        timings["retrieval"] = timings.get("retrieval", 0) + (time.perf_counter() - t0) * 1000

        if mode in ("hybrid_rerank", "hybrid_rerank_mq"):
            ids = ids[: r["candidates"]]
            t1 = time.perf_counter()
            rel = self.reranker.score(query, [self.texts[i] for i in ids])
            timings["rerank"] = timings.get("rerank", 0) + (time.perf_counter() - t1) * 1000
            order = np.argsort(-rel, kind="stable")
            return [Hit(self.chunks[ids[j]], float(rel[j]), b_rank.get(ids[j]),
                        d_rank.get(ids[j]), float(rel[j]), query) for j in order[:max(k, 10)]]
        return [Hit(self.chunks[i], s, b_rank.get(i), d_rank.get(i), None, query)
                for i, s in zip(ids[:max(k, 10)], scores)]

    def search(self, query: str, mode: str = "hybrid_rerank_mq", k: int | None = None,
               doc_ids: list[str] | None = None) -> SearchResult:
        """Return the top hits (at least 10 kept for recall@10; callers use
        the first k). doc_ids restricts the search to the selected documents."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = k or self.cfg["retrieval"]["top_k"]
        mask = self._mask(doc_ids)
        timings: dict = {}
        if mode != "hybrid_rerank_mq":
            return SearchResult(query, mode, self._single(query, mode, k, mask, timings),
                                [query], timings)
        subs = plan_query(query)
        if len(subs) == 1:
            return SearchResult(query, mode, self._single(query, mode, k, mask, timings),
                                subs, timings)
        per_part = [self._single(s, mode, k, mask, timings) for s in subs]
        # 1) the best passage of every part is guaranteed a slot,
        # 2) remaining slots go to the highest relevance across all parts.
        chosen: dict[str, Hit] = {}
        for hits in per_part:
            if hits and hits[0].chunk.chunk_id not in chosen:
                chosen[hits[0].chunk.chunk_id] = hits[0]
        pool = sorted((h for hits in per_part for h in hits), key=lambda h: -h.score)
        for h in pool:
            if h.chunk.chunk_id not in chosen:
                chosen[h.chunk.chunk_id] = h
        guaranteed = list(chosen.values())[:len(per_part)]
        rest = sorted(list(chosen.values())[len(per_part):], key=lambda h: -h.score)
        hits = sorted(guaranteed, key=lambda h: -h.score) + rest
        return SearchResult(query, mode, hits[:max(k, 10)], subs, timings)


# --------------------------------------------------------------- building
def fingerprint(paths: list[Path], cfg: dict, offline: bool) -> str:
    """Index cache key: file contents + model names + pipeline version. Any
    change (new PDF, other model, new chunker) forces a rebuild."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    r = cfg["retrieval"]
    h.update(json.dumps([INDEX_VERSION, offline, r["embed_model"],
                         cfg.get("doc_names", {})]).encode())
    return h.hexdigest()[:16]


def build_index(paths: list[Path], cfg: dict | None = None, cache: Path | None = None,
                names: dict[str, str] | None = None, progress=None) -> Index:
    from .embed import offline_mode
    cfg = cfg or load_config()
    paths = [Path(p) for p in paths]
    offline = offline_mode() or bool(cfg["retrieval"].get("offline"))
    fp = fingerprint(paths, cfg, offline)
    if cache and cache.exists():
        idx = Index.load(cache, cfg, fp)
        if idx is not None:
            return idx
    docs, chunks = [], []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), p.name)
        name = (names or {}).get(p.stem) or cfg.get("doc_names", {}).get(p.stem)
        doc = ingest_pdf(p, name=name, cfg=cfg)
        docs.append(doc)
        chunks.extend(chunk_document(doc))
    idx = Index(chunks, docs, cfg)
    if cache:
        idx.save(cache, fp)
    return idx
