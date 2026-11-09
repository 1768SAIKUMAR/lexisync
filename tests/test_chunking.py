"""Stage 2 tests: clause-aware chunking."""
import re
from pathlib import Path

import pytest

from lexisync.chunking import (MAX_CHARS, MIN_CHARS, _Draft, _merge_tiny, chunk_document,
                               detect_scheme, index_text)
from lexisync.ingest import ingest_pdf

GDPR = Path(__file__).resolve().parent.parent / "data" / "docs" / "gdpr.pdf"


@pytest.fixture(scope="module")
def eu(pdfs):
    return chunk_document(ingest_pdf(pdfs["eu_regulation.pdf"], name="REG"))


@pytest.fixture(scope="module")
def contract(pdfs):
    return chunk_document(ingest_pdf(pdfs["contract.pdf"], name="MSA"))


def _by_label(chunks):
    return {c.label: c for c in chunks}


# --------------------------------------------------------------- scheme
def test_scheme_detection(pdfs):
    assert detect_scheme(ingest_pdf(pdfs["eu_regulation.pdf"]).blocks) == "eu"
    assert detect_scheme(ingest_pdf(pdfs["contract.pdf"]).blocks) == "contract"
    assert detect_scheme(ingest_pdf(pdfs["headers.pdf"]).blocks) == "page"


# ------------------------------------------------------------------- EU
def test_recitals_and_preamble(eu):
    labels = [c.label for c in eu]
    assert labels[0] == "REG p. 1, Preamble"
    assert "REG p. 1, Recital (1)" in labels and "REG p. 1, Recital (2)" in labels
    assert _by_label(eu)["REG p. 1, Recital (2)"].kind == "recital"


def test_article_heading_title_and_chapter_context(eu):
    c = _by_label(eu)["REG p. 1, Article 1(1)"]
    assert c.text.startswith("Article 1\n1. This Regulation")   # heading merged forward
    assert c.title == "Subject-matter"
    assert c.context == "CHAPTER I General provisions"
    assert not any(c.text.strip() in ("Article 1", "Article 83") for c in eu)


def test_definitions_are_numbered_paragraphs_of_article_4(eu):
    d = [c for c in eu if c.kind == "definition"]
    assert [c.clause for c in d] == ["4(1)", "4(2)"]
    assert "'consent' means" in d[1].text
    assert d[1].title == "Definitions"


def test_long_paragraph_split_at_points_with_lead_in(eu):
    parts = [c for c in eu if c.clause == "83(2)"]
    assert len(parts) >= 2
    assert all(len(c.text) <= MAX_CHARS for c in parts)
    assert all(len(c.text) >= MIN_CHARS for c in parts)          # no tiny tail
    assert re.match(r"^\(\w\)", parts[1].text)                   # starts at a point
    assert "due regard shall be given" in parts[1].context       # lead-in carried
    assert parts[0].page_start == 0 and parts[-1].page_end == 1


def test_label_matches_assignment_format(eu):
    c = [c for c in eu if c.clause == "83(5)"][0]
    assert c.label == "REG p. 2, Article 83(5)"
    assert "20 000 000 EUR" in c.text
    assert c.bboxes and all(p == 1 for p, _ in c.bboxes)


def test_index_text_contains_reference_and_title(eu):
    c = [c for c in eu if c.clause == "83(5)"][0]
    it = index_text(c)
    assert "Article 83(5)" in it and "administrative fines" in it
    assert c.text in it                                         # text kept verbatim


# --------------------------------------------------------------- contract
def test_contract_clauses_headings_and_inline_headings(contract):
    by = _by_label(contract)
    assert by["MSA p. 1, clause 7.2"].title == "Liability"
    assert "EUR 2,000,000" in by["MSA p. 1, clause 7.2"].text
    assert by["MSA p. 1, clause 7.1"].text.startswith("7. Liability\n7.1")
    assert by["MSA p. 1, clause 1"].title == "Definitions"       # inline heading
    assert by["MSA p. 2, clause 9.1"].title == "Indemnities"


# ---------------------------------------------------------- tables, fallback
def test_table_is_own_chunk_with_caption(pdfs):
    chunks = chunk_document(ingest_pdf(pdfs["table.pdf"], name="X"))
    t = [c for c in chunks if c.kind == "table"]
    assert len(t) == 1
    assert t[0].label == "X p. 1, Table 9.1"
    assert "| Total | 3,500,000 | - |" in t[0].text


def test_page_fallback_one_section_per_page(pdfs):
    chunks = chunk_document(ingest_pdf(pdfs["headers.pdf"], name="H"))
    assert [c.section for c in chunks] == [f"Page {i}" for i in range(1, 6)]
    assert [c.label for c in chunks][:2] == ["H p. 1", "H p. 2"]


# ------------------------------------------------------------- merge rule
def test_tiny_chunk_merges_only_into_same_clause():
    seg = lambda t: (t, 0, (0, 0, 1, 1))                         # noqa: E731
    a = _Draft("Article 5", "5(1)", "", "text", "", [seg("x" * 200)])
    b = _Draft("Article 5", "5(1)", "", "text", "", [seg("short tail")])
    c = _Draft("Article 5", "5(2)", "", "text", "", [seg("tiny other clause")])
    out = _merge_tiny([a, b, c])
    assert len(out) == 2
    assert out[0].text.endswith("short tail")
    assert out[1].clause == "5(2)"                               # not merged across clauses


# --------------------------------------------- real GDPR (only if present)
@pytest.mark.skipif(not GDPR.exists(), reason="data/docs/gdpr.pdf not downloaded")
def test_real_gdpr_structure():
    chunks = chunk_document(ingest_pdf(GDPR))
    recitals = {c.section for c in chunks if c.kind == "recital"}
    articles = {c.section for c in chunks if c.section.startswith("Article")}
    assert len(recitals) == 173 and len(articles) == 99
    c = [c for c in chunks if c.clause == "83(5)"][0]
    assert c.label == "GDPR p. 83, Article 83(5)"
    assert "20 000 000 EUR" in c.text and "4 %" in c.text
    assert [c for c in chunks if c.clause == "4(11)"][0].kind == "definition"
