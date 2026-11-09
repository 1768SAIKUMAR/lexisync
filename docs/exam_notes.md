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

**Verified on the real GDPR (OJ L 119, 88 pages).** 1,639 blocks, 0 scanned
pages, 0 tables (correct: the GDPR has none), 21 footnotes (the "(1) OJ C 229…"
references on p. 1-2), 352 running-header lines removed (4 patterns: date,
"L 119/#", "EN", "Official Journal of the European Union"). Only p. 88 is
two-column: the signature block "For the European Parliament | For the Council",
which really is side by side. Article 83 comes out as heading → title →
"1." → "2." → points (a), (b)… in order.

**Speed fix.** `find_tables()` took 21 of 22 s on the GDPR because every page
has *some* drawn line (header rule, footnote separator). `may_have_table()` first
counts real ruling lines: ≥ 2 horizontal (≥ 20 pt wide) and ≥ 2 vertical
(≥ 10 pt tall) outside the header band. Result: 22 s → 1 s, identical output.
Matters in the exam: the unseen PDF is indexed live.

## Stage 2 — Clause-aware chunking (`lexisync/chunking.py`)

**Goal.** One chunk = one citable legal unit, so every citation can name
exactly one clause, and the verifier can check a quote against exactly that text.

**Why not fixed-size chunks?** A 500-token window can cut Article 83(4) in half,
or put the end of 83(4) and the start of 83(5) in one chunk. Then "up to
10 000 000 EUR" and "up to 20 000 000 EUR" sit in the same passage, the LLM can
swap them, and a citation can't say which paragraph it came from.

**Three schemes, chosen per document** by counting markers:
- `eu` (≥ 2 blocks that are exactly "Article N"): Recitals "(n)" before the
  first article; "Article N" + title line; paragraphs "1." → clause "N(1)";
  "(n)" inside an article → "N(n)" (definitions, e.g. 4(11) 'consent');
  CHAPTER / Section lines become *context*, not chunks.
- `contract` (≥ 3 clause markers): "7. Liability" heading, inline heading
  "1. Definitions. In this…", clause "7.2 …", numbered section without heading.
- `page` fallback: one section per page (e.g. a judgment without numbering).

*Why decide per document?* "1." means "paragraph 1 of this Article" in a
regulation but "Section 1" in a contract. The scheme resolves the ambiguity.

*Why must the whole block be "Article N"?* Lines inside paragraphs also start
with "Article 6(1)…" (30 such blocks in the GDPR). Matching only a block that is
*just* the heading avoids false article starts.

**Merge rule (< 120 chars).** A tiny chunk joins the previous chunk *of the same
clause* only. "Article 83" (10 chars) is merged *forward* into 83(1). A genuinely
short paragraph like 68(2) "The Board shall be represented by its Chair." stays
alone, because merging it into 68(1) would make the citation wrong.

**Split rule (> 1500 chars).** Split preferably before points (a), (b)… (each point
is its own block), otherwise at sentence ends. Later parts get the paragraph's
lead-in sentence as *context* ("…due regard shall be given to the following:"),
so "(i) where measures…" still makes sense alone. A leftover piece under 120
chars is folded back (no "(j)" orphan).

**What each chunk keeps.** doc id, section, clause, title, verbatim text, page
range, the bbox of every block it came from, kind (text / recital / definition
/ table / page), context, and a label like **"GDPR p. 83, Article 83(5)"**.

**`index_text()`**: what retrieval sees = label + title + context + text. So a
query "administrative fines" matches the title, and "83(5)" matches the label,
while `text` stays verbatim for the quote verifier and the highlight.

**Tables.** Their own chunk; a preceding "Table 9.1 …" caption becomes the title,
so the label reads "… p. 18, Table 9.1", as in the assignment example.

**Verified on the GDPR.** 638 chunks (405 article paragraphs, 207 recital
chunks covering all 173 recitals, 26 definitions), all 99 articles found,
median 408 chars, max 1535. 83(4) holds €10M / 2 %, 83(5) holds €20M / 4 %,
each in its own chunk.

**Limitations.** Annexes and sub-paragraph numbering like "1a" are not special-
cased; a contract with unusual numbering ("Clause Seven") falls back to pages.

## Stage 3 — Retrieval (`lexisync/retrieve.py`, `lexisync/embed.py`)

**Goal.** For a question, return the k (default 4) chunks most likely to contain
the answer, with a 0..1 relevance score that Stage 4 uses to decide whether to
abstain.

**BM25 (lexical).** Scores a chunk by the query words it contains, weighting rare
words more (IDF) and normalising by chunk length (b = 0.75), with saturating
term frequency (k1 = 1.5). *Why we need it:* clause numbers. "7.2" or "83(5)" are
exact strings; an embedding model sees "7.2" as a vague number. The tokenizer
keeps references as one token (`7.2`, `83(5)(a)`, `2016/679`) and also emits their
prefixes (`83(5)(a)` → `83(5)`, `83`), so "Article 83" still matches 83(5). A naive
tokenizer would split "7.2" into "7" and "2" and match every clause 7.x and x.2.

**Dense (semantic).** bge-small-en-v1.5 maps text to a 384-d unit vector; relevance
= cosine similarity = dot product. *Why:* paraphrases. "penalty" ↔ "administrative
fine", "can I be forgotten" ↔ "right to erasure". bge was trained with an
instruction prefix on *queries only* ("Represent this sentence for searching
relevant passages: "); passages are encoded plain. This is a **bi-encoder**: query
and passage never see each other, so all 638 chunks are embedded once at index
time and a query is one matrix-vector product (milliseconds).

**Reciprocal Rank Fusion (k = 60).** score(d) = Σ 1 / (60 + rank_r(d)). *Why ranks,
not scores?* BM25 is unbounded (0–40), cosine is −1..1; adding them is
meaningless and normalising them is fragile. RRF only needs ranks. k = 60 (the
value from the original RRF paper) flattens the head so that one ranker alone
can't dominate: a passage ranked 3rd by both beats one ranked 1st by one and
absent from the other. Top 30 fused candidates go on.

**Cross-encoder rerank.** ms-marco-MiniLM-L-6-v2 reads "[query] [SEP] [passage]"
*together*, so attention compares query and passage tokens directly, which is much more
accurate than cosine between two separately made vectors, but one transformer
pass per pair. So: cheap retrieval for recall (top 30), expensive model for
precision (re-order 30). Output logit → sigmoid → relevance 0..1. The
**best relevance** feeds the abstention gate (Stage 4).

**Query planner (multi-query, `_mq`).** "What is the maximum fine, and which
violations trigger the higher fine?" needs two passages: 83(4) and 83(5). One
embedding of the whole question lands between topics. The planner splits on
", and what/which/when/how…", runs each part separately, and guarantees the best
passage of *each* part a slot in the final top-k; remaining slots go to the
highest relevance. A vague part ("and when does it apply?") gets the first part's
topic words appended, or it would match anything.

**Five modes** (for the mandatory comparison and recall@k): `bm25`, `dense`,
`hybrid`, `hybrid_rerank`, `hybrid_rerank_mq`. Assignment requires dense-only vs
hybrid **or** no-retrieval vs retrieval; we report both.

**Index cache.** Pickle of chunks + embeddings, keyed by a fingerprint (SHA-256 of
the PDF bytes + embedding model + pipeline version). Same PDFs → loads in ~0.05 s;
any change → rebuilt. BM25 is rebuilt on load (fast) rather than pickled.

**Offline stand-ins.** `LEXISYNC_OFFLINE=1` swaps in a hashing encoder (words and
bigrams hashed into 384 dims) and a keyword-overlap reranker, so tests and CI
run without downloading models. Never used for reported numbers.

**Document filter.** `doc_ids` restricts search to the documents of the
current case (UI case selection).
