"""OCR accuracy on a sample (bonus feature 1).

Digital pages have exact ground truth (their text layer). We render each page
to an image at the OCR resolution, run the same OCR path the ingester uses on
scanned pages, and compare OCR text with the true text:

  CER  character error rate = edit distance / reference length (lower is better)
  WER  word error rate      = word-level edit distance / reference words
  clause-number recall      = share of clause starts ("1.", "(a)", "Article 83")
                              that survive OCR, since chunking depends on them

    python scripts/ocr_benchmark.py data/docs/gdpr.pdf --pages 5 33 82 83
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexisync.ingest import CLAUSE_START, normalize_text, ocr_page, tesseract_available  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(s)).strip()


def scanned_copy(page: pymupdf.Page, dpi: int) -> pymupdf.Page:
    """An image-only copy of `page` (what a scanner would produce)."""
    pix = page.get_pixmap(dpi=dpi)
    doc = pymupdf.open()
    p = doc.new_page(width=page.rect.width, height=page.rect.height)
    p.insert_image(p.rect, pixmap=pix)
    return p


def clause_starts(lines: list[str]) -> list[str]:
    return [m.group(0).strip() for l in lines if (m := CLAUSE_START.match(l))]


def benchmark(pdf: Path, pages: list[int], dpi: int = 200) -> dict:
    doc = pymupdf.open(pdf)
    rows = []
    for pno in pages:
        page = doc[pno - 1]
        ref_lines = [l for l in page.get_text("text").splitlines() if l.strip()]
        ref = _norm(" ".join(ref_lines))
        lines, psm = ocr_page(scanned_copy(page, dpi), dpi)
        hyp_lines = [t for t, _, _ in lines]
        hyp = _norm(" ".join(hyp_lines))
        cer = Levenshtein.distance(ref, hyp) / max(1, len(ref))
        rw, hw = ref.split(), hyp.split()
        wer = Levenshtein.distance(rw, hw) / max(1, len(rw))
        ref_cl, hyp_cl = clause_starts(ref_lines), clause_starts(hyp_lines)
        found = sum(1 for c in ref_cl if c in hyp_cl)
        rows.append({"page": pno, "chars": len(ref), "cer": round(cer, 4), "wer": round(wer, 4),
                     "clause_starts": len(ref_cl),
                     "clause_recall": round(found / len(ref_cl), 3) if ref_cl else None,
                     "psm": psm})
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 4)  # noqa: E731
    return {"pdf": pdf.name, "dpi": dpi, "pages": rows,
            "mean_cer": mean("cer"), "mean_wer": mean("wer")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default=str(ROOT / "data" / "docs" / "gdpr.pdf"))
    ap.add_argument("--pages", nargs="*", type=int, default=[2, 33, 82, 83])
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    if not tesseract_available():
        sys.exit("Tesseract is not installed (brew install tesseract).")
    res = benchmark(Path(a.pdf), a.pages, a.dpi)
    out = ROOT / "reports" / "ocr_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"{'page':>5} {'chars':>6} {'CER':>7} {'WER':>7} {'clause recall':>14} psm")
    for r in res["pages"]:
        cr = "-" if r["clause_recall"] is None else f"{r['clause_recall']:.0%}"
        print(f"{r['page']:>5} {r['chars']:>6} {r['cer']:>7.2%} {r['wer']:>7.2%} {cr:>14} {r['psm']}")
    print(f"mean CER {res['mean_cer']:.2%} · mean WER {res['mean_wer']:.2%} · saved {out}")


if __name__ == "__main__":
    main()
