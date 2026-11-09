"""Stage 3 tests: tokenizer, planner, RRF, modes, rerank, cache. Offline models."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ["LEXISYNC_OFFLINE"] = "1"          # stand-in models, no downloads

from lexisync import load_config                               # noqa: E402
from lexisync.embed import HashingEncoder, KeywordReranker     # noqa: E402
from lexisync.retrieve import (MODES, Index, build_index, plan_query,  # noqa: E402
                               tokenize)

GDPR = Path(__file__).resolve().parent.parent / "data" / "docs" / "gdpr.pdf"


@pytest.fixture(scope="module")
def idx(pdfs, tmp_path_factory):
    paths = [pdfs["eu_regulation.pdf"], pdfs["contract.pdf"]]
    return build_index(paths, cache=tmp_path_factory.mktemp("ix") / "index.pkl",
                       names={"eu_regulation": "REG", "contract": "MSA"})


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_keeps_legal_references():
    t = tokenize("Clause 7.2 and Article 83(5)(a) of Regulation 2016/679 impose fines")
    assert {"7.2", "83(5)(a)", "83(5)", "83", "2016/679"} <= set(t)
    assert "fine" in t and "the" not in t and "and" not in t


# ------------------------------------------------------------------ planner
def test_planner_splits_two_part_question():
    q = ("Under the GDPR, what is the maximum fine a company can receive, "
         "and which kinds of violations trigger the higher fine?")
    parts = plan_query(q)
    assert len(parts) == 2
    assert parts[1].startswith("which kinds of violations")


def test_planner_vague_part_inherits_topic():
    parts = plan_query("What is the liability cap in Section 7, and when does it apply?")
    assert parts[1].startswith("when does it apply (")
    assert "liability" in parts[1] and "cap" in parts[1]


def test_planner_leaves_single_question_alone():
    q = "What are the rights and freedoms of data subjects?"
    assert plan_query(q) == [q]


# ---------------------------------------------------------------------- RRF
def test_rrf_rewards_agreement_between_rankers():
    fused = Index.rrf([[1, 2, 3], [2, 3, 1]], k=60)
    assert max(fused, key=fused.get) == 2               # ranked high by both
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)


# --------------------------------------------------------------- the modes
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_clause_by_number(idx, mode):
    r = idx.search("What does clause 7.2 say about the total liability?", mode, k=4)
    assert "clause 7.2" in [h.chunk.label.split(", ")[1] for h in r.hits[:4]]
    assert r.timings["retrieval"] >= 0


def test_bm25_hits_exact_reference_first(idx):
    r = idx.search("Article 83(5)", "bm25", k=3)
    assert r.hits[0].chunk.clause == "83(5)"


def test_rerank_gives_relevance_between_0_and_1(idx):
    r = idx.search("maximum administrative fines 20 000 000 EUR", "hybrid_rerank", k=4)
    assert all(0.0 <= h.relevance <= 1.0 for h in r.hits)
    assert r.hits[0].chunk.clause == "83(5)"
    assert r.best_relevance == r.hits[0].relevance
    assert "rerank" in r.timings


def test_non_rerank_modes_have_no_relevance(idx):
    assert idx.search("fines", "hybrid", k=4).best_relevance is None


def test_multi_query_keeps_best_passage_of_each_part(idx):
    q = "What is the maximum administrative fine, and what does clause 7.2 cap?"
    r = idx.search(q, "hybrid_rerank_mq", k=4)
    assert len(r.subqueries) == 2
    top = [h.chunk.label.split(", ")[1] for h in r.hits[:4]]
    assert "Article 83(5)" in top and "clause 7.2" in top


def test_doc_filter(idx):
    r = idx.search("liability", "hybrid", k=10, doc_ids=["contract"])
    assert r.hits and all(h.chunk.doc_id == "contract" for h in r.hits)


def test_unknown_mode_rejected(idx):
    with pytest.raises(ValueError):
        idx.search("x", "magic")


# -------------------------------------------------------------- cache
def test_index_cache_roundtrip_and_invalidation(pdfs, tmp_path):
    cache = tmp_path / "i.pkl"
    a = build_index([pdfs["contract.pdf"]], cache=cache)
    b = build_index([pdfs["contract.pdf"]], cache=cache)          # loaded
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
    assert np.allclose(a.emb, b.emb)
    c = build_index([pdfs["contract.pdf"], pdfs["table.pdf"]], cache=cache)
    assert len(c.chunks) > len(a.chunks)                          # rebuilt


# ------------------------------------------------------- offline models
def test_offline_models_are_deterministic_and_normalised():
    e = HashingEncoder()
    v1, v2 = e.encode(["administrative fines"]), e.encode(["administrative fines"])
    assert np.allclose(v1, v2) and np.isclose(np.linalg.norm(v1), 1.0)
    s = KeywordReranker().score("administrative fines", ["fines are administrative", "cats"])
    assert s[0] > 0.9 > 0.1 > s[1]


# ------------------------------------------- real GDPR, offline models
@pytest.mark.skipif(not GDPR.exists(), reason="data/docs/gdpr.pdf not downloaded")
def test_gdpr_fines_question_finds_83_4_and_83_5():
    idx = build_index([GDPR], cfg=load_config())
    q = ("Under the GDPR, what is the maximum fine a company can receive, "
         "and which kinds of violations trigger the higher fine?")
    top = {h.chunk.clause for h in idx.search(q, "hybrid_rerank", k=4).hits[:4]}
    assert {"83(4)", "83(5)"} <= top


# ------------------------------------- real models (only on request)
@pytest.mark.skipif(os.environ.get("LEXISYNC_REAL") != "1" or not GDPR.exists(),
                    reason="set LEXISYNC_REAL=1 to test the downloaded models")
def test_real_models_gdpr(monkeypatch):
    monkeypatch.setenv("LEXISYNC_OFFLINE", "0")
    idx = build_index([GDPR], cfg=load_config())
    r = idx.search("What is the maximum fine under the GDPR?", "hybrid_rerank_mq", k=4)
    assert {"83(4)", "83(5)"} & {h.chunk.clause for h in r.hits[:4]}
    assert r.best_relevance > 0.5
    r = idx.search("What is the governing law of the subcontractor agreement?",
                   "hybrid_rerank", k=4)
    assert r.best_relevance < 0.1          # should trigger the abstention gate
