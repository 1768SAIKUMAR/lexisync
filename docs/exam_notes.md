# Exam notes

## Stage 1 — Ingestion (`lexisync/ingest.py`)

**Goal.** Turn a PDF into `Block`s (text, page, bounding box, kind) in the order a
human reads them, with noise removed. Everything downstream (chunking, retrieval,
citations, highlighting) inherits whatever ingestion gets wrong, so this stage is
"garbage in, garbage out" insurance.

**Why `get_text("dict")` and not `get_text("text")`?** The dict output gives
blocks → lines → spans, each with a bbox and font size. We need bboxes for
highlighting and two-column detection, and font size for footnote detection.
Plain text throws both away.

**NFKC normalisation.** Unicode "compatibility" characters look the same but are
different code points: the ligature `ﬁ` (U+FB01) vs `f`+`i`, non-breaking space vs
space. Without NFKC, BM25 would not match "fine" against "ﬁne" — a real risk for
GDPR, where "fines" is a key word. We also remove soft hyphens and re-join words
split across a line break.

**Reading order / two columns.** A naive top-to-bottom sort on a two-column
page interleaves the columns (left line 1, right line 1, …), which mixes
clauses. We call a page two-column only if ≥ 2 left-column blocks vertically
overlap right-column blocks. One pair could be a label next to a value or an
indented note, so it does not count. Full-width blocks (title, a separator
paragraph, a wide table) split the page into horizontal bands; within each band
we read the left column top-down, then the right. *Tested:* `two_column.pdf` gives
FULL-TITLE, L1-3, R1-3, FULL-SEPARATOR, L4-5, R4-5; `single_indented.pdf` is not
called two-column.

**Running headers/footers.** A line is removed if it lies in the top/bottom 8 % of
the page and its normalised text repeats on ≥ 40 % of pages (and ≥ 2 pages).
Digits are replaced by `#` first, so "Page 3 of 5" and "Page 4 of 5" match.
Why remove them? They would appear in many chunks, add retrieval noise, and could
be cited as evidence. Why the 40 % ratio rather than "every page"? The first page
often has a different header, and some pages are landscape tables.

**Footnotes.** Three conditions must all hold: font < 0.9 × the document's median
body size, block starts in the lower 40 % of the page, and text starts with
"(n)". Each condition alone fails: EU recitals also start with "(1)" (so the
pattern alone is wrong), small print appears elsewhere, and so do low blocks.
Footnotes are kept (kind = "footnote") but moved after the page's main text, so
they don't split a clause in two. The median is document-wide, computed per
character, so a page full of footnotes cannot move it.

**Tables.** `find_tables()` detects ruled tables; each cell's text is read by
clipping to the cell bbox (more reliable than splitting a text line). Merged
cells come back as `None`, and we fill them from the left (horizontal span) or
from above (vertical span) so every row is self-contained when it ends up in a
chunk. The table is stored as one Markdown block; its text is excluded from
prose so it is not indexed twice. *Tested:* merged "1,500,000 aggregate" cell.

**Scanned pages.** A page is scanned if it has < 40 text characters and images
cover > 50 % of it: both conditions, because a page with a logo and no text, or a
short text page, is not a scan. We OCR at 200 dpi (a speed/accuracy trade-off:
300 dpi is slower, below 150 hurts small print), group Tesseract's words into
lines by (block, paragraph, line) ids, and convert pixel coordinates to PDF points
(× 72 / dpi) so highlights land in the right place. If no clause number survives
in a page that looks legal, we re-run with `--psm 6` ("one uniform block"), which
tends to keep line-start numbers like "1." attached, and keep the run with more
clause starts. If Tesseract is missing, the page is flagged `ocr_skipped` and the
system keeps working.

**Clause splitting.** A PDF text block can hold several clauses. We start a new
paragraph whenever a line begins a numbered clause (`Article 83`, `7.2`, `7.`,
`(3)`, `83(5)`), but not at lettered points `(a)`. Points belong to their
paragraph: "(a) … (b) …" are the conditions of 83(4). Each paragraph gets the
union bbox of its own lines, so highlights are tight.

**Structure report.** Pages, scanned/OCR/skipped pages, two-column pages,
tables, footnotes, removed header lines. It is shown in the UI after indexing
(assignment step 2) and is the first sanity check on a new PDF in the exam.

**Known limitations (say these honestly).** Unruled tables (whitespace only)
are often missed by `find_tables`; three-column layouts are read as one column;
footnotes without "(n)" markers (e.g. "¹") are not detected; OCR quality is not
measured yet (Stage 7 adds an OCR benchmark).
