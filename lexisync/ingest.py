"""Stage 1 - Ingestion: PDF -> ordered, cleaned Blocks + a structure report.

Pipeline per document
  1. Pass over all pages to find running headers/footers (repeated lines in the
     top/bottom band; digits normalised so "Page 3" == "Page 7").
  2. Per page:
       a. scanned?  (< 40 chars of text AND images cover > 50 %)  -> OCR
       b. tables    (find_tables, cell text via clipping, merged cells filled,
                     stored as one Markdown block; table text removed from prose)
       c. text      (get_text("dict"), NFKC, header/footer lines dropped)
       d. footnotes (small font AND low on page AND starts with "(n)")
       e. reading order (two-column only if blocks truly sit side by side)
       f. split blocks where a new numbered clause starts
  3. Structure report.
"""
from __future__ import annotations

import logging
import re
import shutil
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from . import load_config
from .models import BBox, Block, Document, PageInfo, StructureReport

log = logging.getLogger(__name__)

# A line that starts a new numbered clause: "7.2 ...", "7. Liability", "(3) ...",
# "Article 83", "83(5)". Lettered points "(a)" are NOT clause starts - they stay
# inside their paragraph so a chunk keeps its list of conditions together.
CLAUSE_START = re.compile(
    r"^\s*(?:"
    r"Article\s+\d+[a-z]?\b"             # Article 83
    r"|\d+(?:\.\d+)+\.?\s"               # 7.2 / 7.2.1 / 7.2.
    r"|\d+\.\s+\S"                       # 7. Liability / 1. Definitions.
    r"|\(\d+\)\s"                        # (1) recital / numbered paragraph
    r"|\d+\(\d+\)\s"                     # 83(5)
    r")",
    re.IGNORECASE,
)
FOOTNOTE_START = re.compile(r"^\s*\(\d+\)")


# --------------------------------------------------------------------- helpers
def normalize_text(s: str) -> str:
    """NFKC folds compatibility characters: the ligature 'ﬁ' -> 'fi',
    non-breaking space -> space, full-width digits -> ASCII. Then collapse
    whitespace and re-join words hyphenated across a line break."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("­", "")                      # soft hyphen
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)          # hyphenation at line end
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def header_key(s: str) -> str:
    """Key used to compare candidate header/footer lines across pages."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\d+", "#", s)                      # page numbers don't hide repetition
    return re.sub(r"\s+", " ", s).strip()


def union_bbox(boxes: list[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _center_in(b: BBox, r: BBox) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


def _v_overlap(a: BBox, b: BBox) -> bool:
    return min(a[3], b[3]) - max(a[1], b[1]) > 0


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


# ------------------------------------------------------- headers and footers
def _band_lines(page: pymupdf.Page, band: float) -> list[tuple[str, BBox]]:
    """Text lines lying in the top or bottom band of the page."""
    h = page.rect.height
    out = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            y0, y1 = ln["bbox"][1], ln["bbox"][3]
            if y1 <= band * h or y0 >= (1 - band) * h:
                text = "".join(sp["text"] for sp in ln["spans"]).strip()
                if text:
                    out.append((text, tuple(ln["bbox"])))
    return out


def find_running_lines(doc: pymupdf.Document, band: float, ratio: float) -> set[str]:
    """Header/footer keys that repeat on >= `ratio` of pages (and >= 2 pages)."""
    if len(doc) < 2:
        return set()
    counts: Counter[str] = Counter()
    for page in doc:
        counts.update({header_key(t) for t, _ in _band_lines(page, band)})
    need = max(2, ratio * len(doc))
    return {k for k, c in counts.items() if c >= need}


# --------------------------------------------------------------------- tables
def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    clean = [[c.replace("|", "/").replace("\n", " ").strip() for c in r] for r in rows]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


def may_have_table(page: pymupdf.Page, band: float = 0.08) -> bool:
    """Cheap pre-check before the slow find_tables(): a ruled table needs at
    least 2 real horizontal and 2 real vertical lines outside the header band.
    A footnote separator (a 52 x 1 pt rectangle) or a header rule doesn't
    qualify. On the GDPR this cuts ingestion from ~22 s to ~1 s."""
    h = page.rect.height
    horiz = vert = 0
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "l":
                r = pymupdf.Rect(it[1], it[2])
            elif it[0] == "re":
                r = pymupdf.Rect(it[1])
                if r.width >= 20 and r.height >= 10:       # a cell drawn as a box
                    horiz, vert = horiz + 2, vert + 2
                    continue
            else:
                continue
            if r.y1 < band * h or r.y0 > (1 - band) * h:
                continue
            if r.width >= 20 and r.height < 2:
                horiz += 1
            elif r.height >= 10 and r.width < 2:
                vert += 1
            if horiz >= 2 and vert >= 2:
                return True
    return False


def extract_tables(page: pymupdf.Page) -> list[tuple[BBox, list[list[str]]]]:
    """Find tables; read each cell by clipping; fill merged (None) cells.

    PyMuPDF reports a cell covered by a merged neighbour as None. We fill it
    with the value to its left (horizontal span) or, in the first column, the
    value above (vertical span), so every row is self-contained when chunked.
    """
    out = []
    try:
        tables = page.find_tables()
    except Exception as e:  # noqa: BLE001
        log.warning("find_tables failed on page %d: %s", page.number, e)
        return out
    for tab in tables.tables:
        grid: list[list[str | None]] = []
        for row in tab.rows:
            vals = []
            for cell in row.cells:
                if cell is None:
                    vals.append(None)
                else:
                    vals.append(normalize_text(page.get_text("text", clip=pymupdf.Rect(cell))))
            grid.append(vals)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v is None:
                    if j > 0 and row[j - 1] is not None:
                        row[j] = row[j - 1]
                    elif i > 0 and j < len(grid[i - 1]) and grid[i - 1][j] is not None:
                        row[j] = grid[i - 1][j]
                    else:
                        row[j] = ""
        if len(grid) >= 2 or (grid and len(grid[0]) >= 2):
            out.append((tuple(tab.bbox), grid))  # type: ignore[arg-type]
    return out


# ------------------------------------------------------------------------ OCR
def _ocr_lines(page: pymupdf.Page, dpi: int, psm: int) -> list[tuple[str, BBox, float]]:
    """OCR one page; group words into lines; return (text, bbox_pt, height_pt)."""
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(img, config=f"--psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    scale = 72.0 / dpi                                   # pixels -> PDF points
    lines: dict[tuple, list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) > 0:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(i)
    out = []
    for key in sorted(lines, key=lambda k: min(data["top"][i] for i in lines[k])):
        idx = sorted(lines[key], key=lambda i: data["left"][i])
        text = " ".join(data["text"][i] for i in idx)
        x0 = min(data["left"][i] for i in idx) * scale
        y0 = min(data["top"][i] for i in idx) * scale
        x1 = max(data["left"][i] + data["width"][i] for i in idx) * scale
        y1 = max(data["top"][i] + data["height"][i] for i in idx) * scale
        out.append((normalize_text(text), (x0, y0, x1, y1), y1 - y0))
    return out


def ocr_page(page: pymupdf.Page, dpi: int) -> tuple[list[tuple[str, BBox, float]], int]:
    """OCR twice and keep the run that preserves more clause numbers.

    psm 3 (automatic page segmentation) treats numbers hanging in the left
    margin - recital "(4)", paragraph "1." - as a separate column and drops or
    detaches them; psm 6 ("one uniform block of text") keeps "(4) The ..." on
    one line. Chunking depends on those numbers, so we run both and keep the
    one with more clause starts (tie -> psm 3, the better general layout).
    Measured on GDPR p. 2: clause-number recall 12 % (psm 3) vs ~100 % (psm 6).
    Doubles OCR time, but only for scanned pages."""
    best = _ocr_lines(page, dpi, psm=3)
    alt = _ocr_lines(page, dpi, psm=6)
    n3 = sum(bool(CLAUSE_START.match(t)) for t, _, _ in best)
    n6 = sum(bool(CLAUSE_START.match(t)) for t, _, _ in alt)
    return (alt, 6) if n6 > n3 else (best, 3)


# ---------------------------------------------------------- reading order
def order_blocks(blocks: list[Block], page_width: float, max_w_ratio: float,
                 min_pairs: int) -> tuple[list[Block], bool]:
    """Return blocks in reading order and whether the page is two-column.

    A page is two-column only if at least `min_pairs` left-column blocks sit
    side by side (vertically overlapping) with right-column blocks. One
    indented paragraph or a signature line is not enough. Full-width blocks
    (titles, tables spanning the page) separate horizontal bands; inside each
    band we read the left column top-down, then the right column.
    """
    mid = page_width / 2
    max_w = max_w_ratio * page_width

    def side(b: Block) -> str:
        w = b.bbox[2] - b.bbox[0]
        if w <= max_w and b.bbox[2] <= mid + 0.05 * page_width:
            return "L"
        if w <= max_w and b.bbox[0] >= mid - 0.05 * page_width:
            return "R"
        return "F"   # full width / crosses the gutter

    by_y = sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    left = [b for b in by_y if side(b) == "L"]
    right = [b for b in by_y if side(b) == "R"]
    pairs = sum(1 for lb in left for rb in right if _v_overlap(lb.bbox, rb.bbox))
    if pairs < min_pairs:
        return by_y, False

    ordered: list[Block] = []
    band: list[Block] = []

    def flush():
        ordered.extend(b for b in band if side(b) == "L")
        ordered.extend(b for b in band if side(b) == "R")
        band.clear()

    for b in by_y:
        if side(b) == "F":
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered, True


# -------------------------------------------------------- clause splitting
def split_at_clauses(lines: list[tuple[str, BBox]]) -> list[tuple[str, BBox]]:
    """Group a block's lines into paragraphs; start a new one whenever a line
    begins a new numbered clause. Each paragraph gets the union bbox of its
    own lines - tighter boxes mean more precise highlights later."""
    groups: list[list[tuple[str, BBox]]] = []
    for text, bbox in lines:
        if not groups or CLAUSE_START.match(text):
            groups.append([])
        groups[-1].append((text, bbox))
    return [(normalize_text("\n".join(t for t, _ in g)), union_bbox([b for _, b in g]))
            for g in groups if any(t.strip() for t, _ in g)]


# ------------------------------------------------------------------- main
def _image_cover(page: pymupdf.Page) -> float:
    area = page.rect.width * page.rect.height
    covered = 0.0
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            covered += r.width * r.height
    return min(1.0, covered / area) if area else 0.0


def _doc_name(path: Path) -> str:
    stem = re.sub(r"[_\-]+", " ", path.stem).strip()
    return stem.upper() if len(stem) <= 6 else stem.title()


def ingest_pdf(path: str | Path, doc_id: str | None = None, name: str | None = None,
               cfg: dict | None = None) -> Document:
    cfg = (cfg or load_config())["ingest"]
    path = Path(path)
    doc_id = doc_id or re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    name = name or _doc_name(path)
    pdf = pymupdf.open(path)
    band = cfg["header_footer_band"]
    report = StructureReport(doc_id=doc_id, n_pages=len(pdf))

    running = find_running_lines(pdf, band, cfg["header_repeat_ratio"])
    removed: Counter[str] = Counter()
    have_tess = None  # checked lazily, only if a scanned page appears

    pages: list[PageInfo] = []
    raw: list[tuple[PageInfo, list[Block]]] = []
    sizes: list[float] = []   # (size per char) for the body-font median

    for page in pdf:
        w, h = page.rect.width, page.rect.height
        info = PageInfo(number=page.number, width=w, height=h)
        text_all = page.get_text("text").strip()
        info.n_chars = len(text_all)
        info.image_cover = _image_cover(page)
        info.is_scanned = (info.n_chars < cfg["scanned_max_chars"]
                           and info.image_cover > cfg["scanned_min_image_cover"])
        blocks: list[Block] = []

        if info.is_scanned:
            if have_tess is None:
                have_tess = tesseract_available()
            if not have_tess:
                info.ocr_skipped = True
                log.warning("%s p.%d is scanned but Tesseract is missing - skipped",
                            doc_id, page.number + 1)
            else:
                lines, psm = ocr_page(page, cfg["ocr_dpi"])
                info.ocr_used, info.ocr_psm = True, psm
                kept = []
                for text, bb, hgt in lines:
                    in_band = bb[3] <= band * h or bb[1] >= (1 - band) * h
                    if in_band and header_key(text) in running:
                        removed[header_key(text)] += 1
                        continue
                    kept.append((text, bb))
                    sizes.extend([hgt] * len(text))
                for text, bb in split_at_clauses(kept):
                    blocks.append(Block(page.number, bb, text, source="ocr",
                                        font_size=0.0))
            pages.append(info)
            raw.append((info, blocks))
            continue

        # -- tables first, so their text is not duplicated as prose
        tables = extract_tables(page) if may_have_table(page) else []
        info.n_tables = len(tables)
        for bb, grid in tables:
            blocks.append(Block(page.number, bb, table_to_markdown(grid), kind="table"))

        # -- text blocks
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            lines: list[tuple[str, BBox]] = []
            line_sizes: list[float] = []
            for ln in blk["lines"]:
                lb = tuple(ln["bbox"])
                text = "".join(sp["text"] for sp in ln["spans"])
                if not text.strip():
                    continue
                if any(_center_in(lb, tb) for tb, _ in tables):
                    continue
                in_band = lb[3] <= band * h or lb[1] >= (1 - band) * h
                if in_band and header_key(text) in running:
                    removed[header_key(text)] += 1
                    continue
                lines.append((text, lb))
                for sp in ln["spans"]:
                    line_sizes.extend([sp["size"]] * len(sp["text"].strip()))
            if not lines:
                continue
            size = statistics.median(line_sizes) if line_sizes else 0.0
            sizes.extend(line_sizes)
            for text, bb in split_at_clauses(lines):
                blocks.append(Block(page.number, bb, text, font_size=size))
        pages.append(info)
        raw.append((info, blocks))

    # -- footnotes need the document-wide body font size, so classify after pass
    body = statistics.median(sizes) if sizes else 0.0
    all_blocks: list[Block] = []
    for info, blocks in raw:
        for b in blocks:
            if (b.kind == "text" and b.source == "digital" and body
                    and b.font_size < cfg["footnote_font_ratio"] * body
                    and b.bbox[1] >= cfg["footnote_min_y_ratio"] * info.height
                    and FOOTNOTE_START.match(b.text)):
                b.kind = "footnote"
        info.n_footnotes = sum(b.kind == "footnote" for b in blocks)
        # footnotes are read after the main text of their page
        main = [b for b in blocks if b.kind != "footnote"]
        notes = [b for b in blocks if b.kind == "footnote"]
        ordered, info.two_column = order_blocks(
            main, info.width, cfg["column_max_width_ratio"], cfg["min_column_pairs"])
        for b in ordered + sorted(notes, key=lambda b: b.bbox[1]):
            b.order = len(all_blocks)
            all_blocks.append(b)

    report.scanned_pages = [p.number + 1 for p in pages if p.is_scanned]
    report.ocr_pages = [p.number + 1 for p in pages if p.ocr_used]
    report.ocr_skipped_pages = [p.number + 1 for p in pages if p.ocr_skipped]
    report.two_column_pages = [p.number + 1 for p in pages if p.two_column]
    report.n_tables = sum(p.n_tables for p in pages)
    report.n_footnotes = sum(p.n_footnotes for p in pages)
    report.n_blocks = len(all_blocks)
    report.removed_header_lines = sorted(removed)
    report.n_removed_lines = sum(removed.values())
    pdf.close()
    return Document(doc_id, name, str(path), pages, all_blocks, report)


def format_report(r: StructureReport) -> str:
    """Counts first, then the page numbers, so "1 (p. 88)" can't be misread."""
    def pg(xs):
        return f"{len(xs)}" + (f"  (p. {', '.join(map(str, xs))})" if xs else "")
    return "\n".join([
        f"Document: {r.doc_id}",
        f"  pages: {r.n_pages}   blocks: {r.n_blocks}",
        f"  scanned pages: {pg(r.scanned_pages)}",
        f"    OCR'd: {pg(r.ocr_pages)}   skipped (no Tesseract): {pg(r.ocr_skipped_pages)}",
        f"  two-column pages: {pg(r.two_column_pages)}",
        f"  tables: {r.n_tables}   footnotes: {r.n_footnotes}",
        f"  removed header/footer lines: {r.n_removed_lines} "
        f"({len(r.removed_header_lines)} distinct patterns)",
    ])


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    for p in sys.argv[1:]:
        print(format_report(ingest_pdf(p).report))
