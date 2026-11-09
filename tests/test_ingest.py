"""Stage 1 tests: each synthetic PDF checks one ingestion behaviour."""
import pytest

from lexisync.ingest import (CLAUSE_START, header_key, ingest_pdf, normalize_text,
                             order_blocks, tesseract_available)
from lexisync.models import Block


def _texts(doc):
    return [b.text for b in doc.blocks]


def _first_tokens(doc):
    return [b.text.split()[0] for b in doc.blocks]


# ---------------------------------------------------------------- normalise
def test_nfkc_fixes_ligatures_and_hyphenation():
    assert normalize_text("ﬁne and eﬀect") == "fine and effect"
    assert normalize_text("process-\ning") == "processing"
    assert normalize_text("EUR 20") == "EUR 20"


def test_header_key_normalises_digits():
    assert header_key("Page 3 of 10") == header_key("Page 17 of 10")


@pytest.mark.parametrize("line,is_clause", [
    ("Article 83", True), ("7.2 The total liability", True), ("7. Liability", True),
    ("(3) Member States", True), ("83(5) Infringements", True),
    ("(a) the nature", False), ("The controller shall", False), ("2016 was", False),
])
def test_clause_start_pattern(line, is_clause):
    assert bool(CLAUSE_START.match(line)) is is_clause


# ------------------------------------------------------------ reading order
def test_two_column_reading_order(pdfs):
    doc = ingest_pdf(pdfs["two_column.pdf"])
    assert doc.report.two_column_pages == [1]
    assert _first_tokens(doc) == [
        "FULL-TITLE", "LEFT-1", "LEFT-2", "LEFT-3", "RIGHT-1", "RIGHT-2", "RIGHT-3",
        "FULL-SEPARATOR", "LEFT-4", "LEFT-5", "RIGHT-4", "RIGHT-5"]


def test_single_indented_block_is_not_two_column(pdfs):
    doc = ingest_pdf(pdfs["single_indented.pdf"])
    assert doc.report.two_column_pages == []
    assert _first_tokens(doc) == ["PARA-1", "PARA-2", "PARA-3"]


def test_one_side_by_side_pair_is_not_enough():
    mk = lambda x0, y0, x1, y1, t: Block(0, (x0, y0, x1, y1), t)  # noqa: E731
    blocks = [mk(50, 100, 280, 150, "A"), mk(320, 100, 545, 150, "B"),
              mk(50, 200, 545, 260, "C")]
    ordered, two_col = order_blocks(blocks, 595, 0.55, 2)
    assert not two_col and [b.text for b in ordered] == ["A", "B", "C"]


# ----------------------------------------------------------- header/footer
def test_running_headers_and_footers_removed(pdfs):
    doc = ingest_pdf(pdfs["headers.pdf"])
    joined = " ".join(_texts(doc))
    assert "ACME Services Agreement" not in joined
    assert "Confidential" not in joined
    assert sum(t.startswith("BODY-") for t in _texts(doc)) == 5
    assert doc.report.n_removed_lines == 10
    assert len(doc.report.removed_header_lines) == 2


# ----------------------------------------------------------------- footnotes
def test_footnote_vs_recital(pdfs):
    doc = ingest_pdf(pdfs["footnotes.pdf"])
    kinds = {b.text[:20]: b.kind for b in doc.blocks}
    assert kinds["(1) OJ L 119, 4.5.20"] == "footnote"
    assert kinds["(1) Natural persons "] == "text"     # same "(1)" but body font, high
    assert doc.blocks[-1].kind == "footnote"           # read after main text


# --------------------------------------------------------------------- tables
def test_table_markdown_and_merged_cells(pdfs):
    doc = ingest_pdf(pdfs["table.pdf"])
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    md = tables[0].text
    assert md.splitlines()[0] == "| Item | Cap (EUR) | Notes |"
    assert "| Data breach | 1,500,000 aggregate | 1,500,000 aggregate |" in md
    assert "| Total | 3,500,000 | - |" in md
    prose = " ".join(b.text for b in doc.blocks if b.kind == "text")
    assert "IP indemnity" not in prose                 # not duplicated as prose
    assert "BODY after table" in prose


# --------------------------------------------------------------------- scanned
def test_scanned_page_detected(pdfs):
    doc = ingest_pdf(pdfs["scanned.pdf"])
    assert doc.report.scanned_pages == [2]
    assert not doc.pages[0].is_scanned


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")
def test_scanned_page_ocr_keeps_clauses(pdfs):
    doc = ingest_pdf(pdfs["scanned.pdf"])
    ocr = [b for b in doc.blocks if b.source == "ocr"]
    assert doc.report.ocr_pages == [2]
    assert ocr[0].text.startswith("Article 5")
    assert any(b.text.startswith("1. Personal data") for b in ocr)
    assert any(b.text.startswith("2. The controller") for b in ocr)
    # boxes are in PDF points, inside the page
    assert all(0 <= b.bbox[0] < b.bbox[2] <= 595 for b in ocr)


def test_scanned_page_flagged_when_tesseract_missing(pdfs, monkeypatch):
    monkeypatch.setattr("lexisync.ingest.tesseract_available", lambda: False)
    doc = ingest_pdf(pdfs["scanned.pdf"])
    assert doc.report.ocr_skipped_pages == [2]
    assert all(b.page == 0 for b in doc.blocks)


# ------------------------------------------------------------------ contracts
def test_contract_split_at_numbered_clauses(pdfs):
    doc = ingest_pdf(pdfs["contract.pdf"])
    starts = [b.text.split()[0] for b in doc.blocks]
    assert starts == ["MASTER", "1.", "7.", "7.1", "7.2", "9.", "9.1"]
    assert doc.blocks[5].page == 1
    assert [b.order for b in doc.blocks] == list(range(len(doc.blocks)))


def test_split_inside_one_block_keeps_points_together():
    from lexisync.ingest import split_at_clauses
    lines = [("4. Infringements of the following", (0, 0, 100, 10)),
             ("(a) the obligations of the controller;", (0, 10, 100, 20)),
             ("5. Infringements of the following", (0, 30, 100, 40)),
             ("(a) the basic principles", (0, 40, 100, 50))]
    out = split_at_clauses(lines)
    assert [t.split("\n")[0][:2] for t, _ in out] == ["4.", "5."]
    assert out[0][1] == (0, 0, 100, 20) and out[1][1] == (0, 30, 100, 50)


def test_report_shows_counts_and_page_numbers(pdfs):
    from lexisync.ingest import format_report
    text = format_report(ingest_pdf(pdfs["two_column.pdf"]).report)
    assert "two-column pages: 1  (p. 1)" in text
    assert "scanned pages: 0" in text
