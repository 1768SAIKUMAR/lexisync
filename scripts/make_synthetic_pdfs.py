"""Synthetic PDFs that each isolate one layout problem, used by the tests.

    python scripts/make_synthetic_pdfs.py data/synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

W, H = 595, 842  # A4 in points
LOREM = ("The parties agree that the obligations set out in this clause apply "
         "for the full term of the agreement and survive termination.")


def _box(page, rect, text, size=10.0):
    rect = (rect[0], rect[1], rect[2], max(rect[3], rect[1] + size * 1.6))
    rc = page.insert_textbox(pymupdf.Rect(rect), text, fontsize=size, fontname="helv")
    assert rc >= 0, f"text overflow in {rect}: {text[:30]}"


def make_two_column(path: Path) -> None:
    """Title, two-column band, full-width separator, second two-column band."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _box(p, (50, 60, 545, 90), "FULL-TITLE Regulation on two-column layout", 14)
    for i in range(3):
        y = 110 + i * 90
        _box(p, (50, y, 285, y + 80), f"LEFT-{i + 1} {LOREM}")
        _box(p, (310, y, 545, y + 80), f"RIGHT-{i + 1} {LOREM}")
    _box(p, (50, 390, 545, 430), f"FULL-SEPARATOR {LOREM}")
    for i in range(2):
        y = 450 + i * 90
        _box(p, (50, y, 285, y + 80), f"LEFT-{i + 4} {LOREM}")
        _box(p, (310, y, 545, y + 80), f"RIGHT-{i + 4} {LOREM}")
    doc.save(path)


def make_single_column_indented(path: Path) -> None:
    """One column with a single indented paragraph: must NOT be two-column."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _box(p, (50, 60, 545, 140), f"PARA-1 {LOREM} {LOREM}")
    _box(p, (320, 160, 545, 230), f"PARA-2 indented note {LOREM}")
    _box(p, (50, 250, 545, 330), f"PARA-3 {LOREM} {LOREM}")
    doc.save(path)


def make_headers(path: Path, n: int = 5) -> None:
    """Running header with a page number and a running footer on every page."""
    doc = pymupdf.open()
    for i in range(n):
        p = doc.new_page(width=W, height=H)
        _box(p, (50, 20, 545, 40), f"ACME Services Agreement - Page {i + 1} of {n}", 8)
        _box(p, (50, 100, 545, 400), f"BODY-{i + 1} {LOREM} {LOREM}", 11)
        _box(p, (50, 805, 545, 825), f"Confidential - Draft {2024 + i}", 8)
    doc.save(path)


def make_footnotes(path: Path) -> None:
    """A recital '(1)' in body text (not a footnote) and a real small footnote."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _box(p, (50, 80, 545, 160), f"(1) Natural persons should have control of their own data. {LOREM}", 11)
    _box(p, (50, 180, 545, 300), f"(2) The protection of natural persons is a right. {LOREM}", 11)
    _box(p, (50, 740, 545, 770), "(1) OJ L 119, 4.5.2016, p. 1.", 7)
    doc.save(path)


def make_table(path: Path) -> Path:
    """A ruled 4x3 table whose row 3 has a cell merged across two columns."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _box(p, (50, 60, 545, 90), "Table 9.1 Indemnity obligations", 12)
    xs, ys = [60, 220, 380, 540], [110, 140, 170, 200, 230]
    cells = [["Item", "Cap (EUR)", "Notes"],
             ["IP indemnity", "2,000,000", "Uncapped for fraud"],
             ["Data breach", "1,500,000", None],      # None = merged with left
             ["Total", "3,500,000", "-"]]
    shape = p.new_shape()
    for y in ys:
        shape.draw_line((xs[0], y), (xs[-1], y))
    for r in range(4):
        for c, x in enumerate(xs):
            if r == 2 and c == 2:   # skip the inner line -> merged cell
                continue
            shape.draw_line((x, ys[r]), (x, ys[r + 1]))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            if val is None:
                continue
            text = "1,500,000 aggregate" if (r == 2 and c == 1) else val
            p.insert_text((xs[c] + 4, ys[r] + 19), text, fontsize=9, fontname="helv")
    _box(p, (50, 260, 545, 320), f"BODY after table. {LOREM}", 10)
    doc.save(path)
    return path


def make_scanned(path: Path) -> None:
    """Page 1 digital, page 2 image-only (rendered text, no text layer)."""
    src = pymupdf.open()
    p = src.new_page(width=W, height=H)
    _box(p, (60, 80, 535, 140), "Article 5\nPrinciples relating to processing", 14)
    _box(p, (60, 160, 535, 260), "1. Personal data shall be processed lawfully, "
         "fairly and in a transparent manner in relation to the data subject.", 12)
    _box(p, (60, 280, 535, 380), "2. The controller shall be responsible for, and be "
         "able to demonstrate compliance with, paragraph 1.", 12)
    pix = p.get_pixmap(dpi=200)

    doc = pymupdf.open()
    d = doc.new_page(width=W, height=H)
    _box(d, (50, 80, 545, 200), f"Article 4\nDefinitions\nDIGITAL page. {LOREM}", 11)
    s = doc.new_page(width=W, height=H)
    s.insert_image(s.rect, pixmap=pix)
    doc.save(path)


def make_contract(path: Path) -> None:
    """A two-page contract with headings, inline headings and sub-clauses."""
    doc = pymupdf.open()
    p = doc.new_page(width=W, height=H)
    _box(p, (50, 60, 545, 90), "MASTER SERVICES AGREEMENT", 14)
    _box(p, (50, 100, 545, 170), "1. Definitions. In this Agreement \"Services\" "
         "means the services described in Schedule 1.", 10)
    _box(p, (50, 178, 545, 200), "7. Liability", 12)
    _box(p, (50, 205, 545, 260), "7.1 Neither party excludes liability for death "
         "or personal injury caused by negligence.", 10)
    _box(p, (50, 265, 545, 330), "7.2 The total liability of the Supplier shall "
         "not exceed EUR 2,000,000 in any contract year.", 10)
    p2 = doc.new_page(width=W, height=H)
    _box(p2, (50, 58, 545, 80), "9. Indemnities", 12)
    _box(p2, (50, 85, 545, 150), "9.1 The Supplier shall indemnify the Customer "
         "against third-party IP claims.", 10)
    doc.save(path)


ALL = {
    "two_column.pdf": make_two_column,
    "single_indented.pdf": make_single_column_indented,
    "headers.pdf": make_headers,
    "footnotes.pdf": make_footnotes,
    "table.pdf": make_table,
    "scanned.pdf": make_scanned,
    "contract.pdf": make_contract,
}


def build_all(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in ALL.items():
        fn(out_dir / name)
        paths[name] = out_dir / name
    return paths


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    for name, path in build_all(out).items():
        print("wrote", path)
