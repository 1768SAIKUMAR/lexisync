"""Stage 4b - Grounded answers: gate -> prompt -> JSON -> verify -> retry -> confidence.

  1. Retrieval gate   best rerank relevance < 0.10 -> abstain WITHOUT calling the
                      LLM (no chance to hallucinate) and list the documents searched.
  2. Prompt           numbered passages [C1]..[Ck] (label + title + text) + rules.
  3. JSON answer      claims (text, citations, exact quote) -> answer -> missing ->
                      abstain -> explanation -> conflicts. Evidence first, decision last.
  4. Verifier         each claim's quote must be in a cited passage (rapidfuzz
                      partial_ratio >= 85 or word overlap >= 0.6) AND every number
                      in the claim must occur in that passage.
  5. Retry loop       failed claims -> reformulated query -> retrieve -> regenerate
                      (max 2 attempts); claims still failing are dropped; no
                      verified claim left -> abstain.
  6. Confidence       HIGH / MEDIUM / LOW / INSUFFICIENT EVIDENCE.
  7. Citations        [C#] markers replaced by real labels ("GDPR p. 83, Article 83(5)").
Also: closed-book baseline (no retrieval) for the mandatory comparison.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz

from .chunking import index_text
from .llm import ANSWER_SCHEMA, LLM
from .models import Chunk
from .retrieve import Hit, Index

SYSTEM_PROMPT = """You are a careful legal analyst. Answer ONLY from the numbered passages.

Rules:
1. Make ONE claim per fact. Every claim must cite at least one passage id, e.g.
   ["C2"], and include an exact quote of 5-25 words copied character-for-character
   from that passage.
2. Use only facts stated in the passages. No outside knowledge, no guessing.
3. Copy numbers, amounts, percentages and dates exactly as written. Never swap
   numbers between passages or between a rule and its exception.
4. Keep the default rule and its exceptions apart; say which is which.
5. Answer EVERY part of the question. If the passages give several amounts, tiers,
   deadlines or conditions (e.g. a lower and a higher fine), give each one as its
   own claim with its own citation.
6. "conflicts": only when two passages state DIFFERENT values for the same thing;
   say which source you used. Otherwise an empty list.
7. "missing": only parts of the question that NO passage addresses. Otherwise an
   empty list.
8. If the passages do not answer the question at all, return no claims and set
   "abstain": true.
9. "answer": plain-language answer, 1-5 sentences. EVERY sentence must end with the
   [C#] marker(s) of the passage(s) it comes from.
10. "explanation": one or two sentences on why the cited evidence supports the answer.

Write the JSON fields in this order: claims, answer, missing, abstain, explanation,
conflicts."""

CLOSED_BOOK_PROMPT = """You are a legal analyst. Answer the question from your own
knowledge. You have no documents. Return JSON with the same fields; leave
"citations" empty and "quote" as an empty string. If you do not know, abstain."""

INSUFFICIENT = "INSUFFICIENT EVIDENCE"
ABSTAIN_TEXT = "The documents do not contain enough evidence to answer this question."


# ------------------------------------------------------------------ results
@dataclass
class Claim:
    text: str
    citations: list[str]                 # passage ids as given: ["C1"]
    quote: str
    labels: list[str] = field(default_factory=list)      # resolved citation labels
    chunk_ids: list[str] = field(default_factory=list)
    verified: bool = False
    quote_score: float = 0.0             # best fuzzy score over cited passages
    numbers_ok: bool = True
    reason: str = ""                     # why verification failed
    attempt: int = 1


@dataclass
class AnswerResult:
    question: str
    answer: str
    confidence: str
    abstained: bool
    abstain_reason: str = ""
    claims: list[Claim] = field(default_factory=list)
    dropped: list[Claim] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    explanation: str = ""
    conflicts: list[str] = field(default_factory=list)
    passages: list[Hit] = field(default_factory=list)    # what the LLM saw (last attempt)
    best_relevance: float | None = None
    subqueries: list[str] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)    # "GDPR (88 pages)"
    timings: dict = field(default_factory=dict)          # ms: retrieval, rerank, llm, total
    tokens: dict = field(default_factory=dict)           # prompt, completion
    attempts: list[dict] = field(default_factory=list)   # retry log
    mode: str = ""
    llm: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passages"] = [{"label": h.chunk.label, "chunk_id": h.chunk.chunk_id,
                          "relevance": h.relevance, "score": h.score} for h in self.passages]
        return d


# --------------------------------------------------------------- helpers
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[-‐‑–—]\s*\n\s*", "", s)                 # hyphenated line breaks
    return re.sub(r"\s+", " ", s).strip()


_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:[  ,.]\d{3})+|\d+(?:[.,]\d+)?)"
                  r"(\s*(?:million|m\b|mn\b|billion|bn\b))?", re.I)


def numbers_in(text: str) -> set[float]:
    """Numbers as values, so '20 000 000', '20,000,000', '20000000' and
    'EUR 20 million' all become 20000000.0. Passage markers [C3] are ignored."""
    text = re.sub(r"\[C\d+\]", " ", unicodedata.normalize("NFKC", text))
    out = set()
    for m in _NUM.finditer(text):
        raw, mult = m.group(1), (m.group(2) or "").strip().lower()
        if re.fullmatch(r"\d{1,3}(?:[  ,.]\d{3})+", raw):
            val = float(re.sub(r"[  ,.]", "", raw))       # thousands separators
        else:
            val = float(raw.replace(",", "."))
        if mult in ("million", "m", "mn"):
            val *= 1e6
        elif mult in ("billion", "bn"):
            val *= 1e9
        out.add(round(val, 6))
    return out


def quote_found(quote: str, passage: str, fuzzy_min: float, overlap_min: float) -> tuple[bool, float]:
    """Exact-ish match: rapidfuzz partial_ratio (best-matching substring,
    tolerant to PDF line breaks and small OCR/whitespace differences) or the
    share of quote words that occur in the passage."""
    q, p = _norm(quote), _norm(passage)
    if not q:
        return False, 0.0
    score = fuzz.partial_ratio(q, p)
    qw = re.findall(r"\w+", q)
    pw = set(re.findall(r"\w+", p))
    overlap = sum(w in pw for w in qw) / len(qw) if qw else 0.0
    return (score >= fuzzy_min or overlap >= overlap_min), max(score / 100, overlap)


def parse_json(text: str) -> dict:
    """Tolerant JSON parsing: strips ```json fences and surrounding prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"claims": [], "answer": "", "missing": [], "abstain": True,
            "explanation": "model output was not valid JSON", "conflicts": []}


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = [f"Question: {question}", "", "Passages:"]
    for i, h in enumerate(hits, 1):
        c = h.chunk
        head = c.label + (f" — {c.title}" if c.title else "")
        parts.append(f"[C{i}] {head}\n{c.text}\n")
    parts.append("Return the JSON object now.")
    return "\n".join(parts)


def verify_claims(raw_claims: list[dict], hits: list[Hit], cfg: dict, attempt: int) -> list[Claim]:
    a = cfg["answer"]
    by_id = {f"C{i}": h.chunk for i, h in enumerate(hits, 1)}
    out = []
    for rc in raw_claims or []:
        cits = [c.strip("[] ") for c in (rc.get("citations") or []) if isinstance(c, str)]
        cl = Claim(text=str(rc.get("text", "")).strip(), citations=cits,
                   quote=str(rc.get("quote", "")).strip(), attempt=attempt)
        chunks = [by_id[c] for c in cits if c in by_id]
        cl.labels = [c.label for c in chunks]
        cl.chunk_ids = [c.chunk_id for c in chunks]
        if not cl.text:
            continue
        if not chunks:
            cl.reason = "cites no retrieved passage"
            out.append(cl)
            continue
        nq = len(cl.quote.split())
        found = [(quote_found(cl.quote, c.text, a["fuzzy_min"], a["overlap_min"]), c)
                 for c in chunks]
        cl.quote_score = max(s for (_, s), _ in found)
        ok_chunks = [c for (ok, _), c in found if ok]
        claim_nums = numbers_in(cl.text)
        nums_ok_chunks = [c for c in ok_chunks
                          if claim_nums <= numbers_in(index_text(c))]
        cl.numbers_ok = bool(nums_ok_chunks) or not claim_nums
        if nq < 3:
            cl.reason = "quote too short to verify"
        elif not ok_chunks:
            cl.reason = f"quote not found in cited passage (score {cl.quote_score:.2f})"
        elif claim_nums and not nums_ok_chunks:
            missing = claim_nums - set().union(*(numbers_in(index_text(c)) for c in ok_chunks))
            cl.reason = "number(s) not in cited passage: " + ", ".join(
                f"{int(n):,}" if float(n).is_integer() else f"{n:g}" for n in sorted(missing))
        else:
            cl.verified = True
        out.append(cl)
    return out


def _with_labels(text: str, hits: list[Hit]) -> str:
    """Replace [C2] / [C1, C3] markers with the real citation labels."""
    labels = {f"C{i}": h.chunk.label for i, h in enumerate(hits, 1)}

    def sub(m):
        ids = re.findall(r"C\d+", m.group(0))
        names = [labels[i] for i in ids if i in labels]
        return f"({'; '.join(names)})" if names else ""
    return re.sub(r"\[(?:C\d+(?:\s*,\s*)?)+\]", sub, text).replace(" .", ".").strip()


def grounded_answer(text: str, hits: list[Hit], claims: list[Claim]) -> str:
    """Keep only answer sentences that (a) carry a [C#] marker pointing at a passage
    that backs a verified claim and (b) contain no number missing from those
    passages. The model's free text is never shown unchecked."""
    by_id = {f"C{i}": h.chunk for i, h in enumerate(hits, 1)}
    backing = {cid for c in claims for cid in c.chunk_ids}
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip()):
        ids = re.findall(r"C\d+", " ".join(re.findall(r"\[[^\]]*\]", sent)))
        chunks = [by_id[i] for i in ids if i in by_id and by_id[i].chunk_id in backing]
        if not chunks:
            continue
        nums = numbers_in(sent)
        if nums and not nums <= set().union(*(numbers_in(index_text(c)) for c in chunks)):
            continue
        kept.append(sent)
    return _with_labels(" ".join(kept), hits) if kept else ""


def _claims_answer(claims: list[Claim]) -> str:
    return " ".join(f"{c.text.rstrip('.')} ({'; '.join(c.labels)})." for c in claims)


def _confidence(claims: list[Claim], dropped: list[Claim], rel: float | None,
                missing: list[str], cfg: dict) -> str:
    a = cfg["answer"]
    if not claims:
        return INSUFFICIENT
    if rel is not None and rel < a["low_relevance"] or len(dropped) >= len(claims):
        return "LOW"
    if (not dropped and not missing and rel is not None and rel >= a["high_relevance"]
            and all(c.verified for c in claims)):
        return "HIGH"
    return "MEDIUM"


def _searched(index: Index, doc_ids: list[str] | None) -> list[str]:
    docs = [d for d in index.docs.values() if not doc_ids or d.doc_id in doc_ids]
    return [f"{d.name} ({d.report.n_pages} pages)" for d in docs]


# ------------------------------------------------------------------- main
def answer_question(question: str, index: Index, llm: LLM, cfg: dict, *,
                    mode: str = "hybrid_rerank_mq", k: int | None = None,
                    verify: bool = True, closed_book: bool = False,
                    doc_ids: list[str] | None = None) -> AnswerResult:
    t_start = time.perf_counter()
    a = cfg["answer"]
    k = k or cfg["retrieval"]["top_k"]
    res = AnswerResult(question=question, answer="", confidence=INSUFFICIENT,
                       abstained=True, mode="closed_book" if closed_book else mode,
                       llm=llm.name, searched=_searched(index, doc_ids))
    tokens = {"prompt": 0, "completion": 0}
    timings = {"retrieval": 0.0, "rerank": 0.0, "llm": 0.0}

    # ----- closed-book baseline: no retrieval, nothing can be verified
    if closed_book:
        r = llm.generate(CLOSED_BOOK_PROMPT, f"Question: {question}", ANSWER_SCHEMA)
        timings["llm"] = r.ms
        j = parse_json(r.text)
        res.answer = j.get("answer", "")
        res.abstained = bool(j.get("abstain")) or not res.answer
        res.confidence = "UNGROUNDED (no retrieval)" if not res.abstained else INSUFFICIENT
        res.claims = [Claim(str(c.get("text", "")), [], "", reason="no evidence (closed book)")
                      for c in j.get("claims", []) if isinstance(c, dict)]
        res.tokens = {"prompt": r.prompt_tokens, "completion": r.completion_tokens}
        res.timings = {**timings, "total": (time.perf_counter() - t_start) * 1000}
        return res

    query = question
    verified: dict[str, Claim] = {}          # claim text -> verified claim
    kept_hits: list[Hit] = []                 # passages backing verified claims
    last: dict = {}
    failed: list[Claim] = []
    for attempt in range(1, (a["max_attempts"] if verify else 1) + 1):
        sr = index.search(query, mode=mode, k=k, doc_ids=doc_ids)
        timings["retrieval"] += sr.timings.get("retrieval", 0)
        timings["rerank"] += sr.timings.get("rerank", 0)
        if attempt == 1:
            res.best_relevance, res.subqueries = sr.best_relevance, sr.subqueries
            # ----- 1. retrieval gate: no LLM call when nothing relevant was found
            if sr.best_relevance is not None and sr.best_relevance < a["abstain_threshold"]:
                res.abstain_reason = (f"best passage relevance {sr.best_relevance:.2f} is below "
                                      f"the threshold {a['abstain_threshold']:.2f}")
                res.answer = f"{ABSTAIN_TEXT} Searched: {', '.join(res.searched)}."
                res.passages = sr.hits[:k]
                res.timings = {**timings, "total": (time.perf_counter() - t_start) * 1000}
                res.attempts.append({"attempt": 1, "query": query, "gate": "abstain",
                                     "best_relevance": sr.best_relevance})
                return res
        # passages for this attempt: keep evidence of verified claims, add new hits
        hits, seen = list(kept_hits), {h.chunk.chunk_id for h in kept_hits}
        for h in sr.hits:
            if len(hits) >= k + len(kept_hits):
                break
            if h.chunk.chunk_id not in seen:
                hits.append(h)
                seen.add(h.chunk.chunk_id)

        # ----- 2-3. generate
        r = llm.generate(SYSTEM_PROMPT, build_prompt(question, hits), ANSWER_SCHEMA)
        timings["llm"] += r.ms
        tokens["prompt"] += r.prompt_tokens
        tokens["completion"] += r.completion_tokens
        last = parse_json(r.text)
        res.passages = hits

        # ----- 4. verify
        claims = verify_claims(last.get("claims", []), hits, cfg, attempt)
        if not verify:                       # ablation: accept claims unchecked
            for c in claims:
                c.verified = bool(c.labels)
        failed = [c for c in claims if not c.verified]
        for c in claims:
            if c.verified and c.text not in verified:
                verified[c.text] = c
                for cid in c.chunk_ids:
                    h = next(h for h in hits if h.chunk.chunk_id == cid)
                    if h not in kept_hits:
                        kept_hits.append(h)
        res.attempts.append({
            "attempt": attempt, "query": query, "passages": [h.chunk.label for h in hits],
            "claims": len(claims), "verified": len(claims) - len(failed),
            "rejected": [{"claim": c.text, "reason": c.reason} for c in failed],
            "model_abstained": bool(last.get("abstain")),
        })
        if not failed or not verify:
            break
        # ----- 5. reformulate: question + the content words of the failed claims
        extra = " ".join(dict.fromkeys(w for c in failed for w in re.findall(r"[A-Za-z]{4,}|\d[\d .,%]*", c.text)))
        query = f"{question} {extra}"[:400]

    res.dropped = failed if verify else []
    res.claims = list(verified.values())
    res.missing = [str(m) for m in last.get("missing", []) if m]
    res.explanation = str(last.get("explanation", ""))
    res.conflicts = [str(c) for c in last.get("conflicts", []) if c]
    res.tokens = tokens

    if not res.claims:
        res.abstained = True
        res.abstain_reason = ("the model found no answer in the passages"
                              if last.get("abstain") or not last.get("claims")
                              else "no claim could be verified against its cited passage")
        res.answer = f"{ABSTAIN_TEXT} Searched: {', '.join(res.searched)}."
        res.confidence = INSUFFICIENT
    else:
        res.abstained = False
        # every displayed sentence must be backed; otherwise use the verified claims
        res.answer = ("" if res.dropped else
                      grounded_answer(str(last.get("answer", "")), res.passages, res.claims))
        if not res.answer:
            res.answer = _claims_answer(res.claims)
        res.confidence = _confidence(res.claims, res.dropped, res.best_relevance,
                                     res.missing, cfg)
    res.timings = {**timings, "total": (time.perf_counter() - t_start) * 1000}
    return res
