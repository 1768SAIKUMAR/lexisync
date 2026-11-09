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

**Measured on the real models (MacBook, GDPR).** Best cross-encoder relevance:
answerable fines question 0.36 (single query) / 0.57 (planner); unanswerable
"governing law of the subcontractor agreement" **0.00**. The planner raised the
best relevance from 0.36 to 0.57 and put 83(5) first. First rerank call 2.4 s
(model load), afterwards ~0.15–0.3 s.

## Stage 4 — Grounded answers (`answer.py`, `llm.py`, `pipeline.py`)

**Goal.** Every sentence the user sees is backed by a verified quote from a cited
passage; otherwise the system abstains. The LLM is never trusted on its own.

**1. Retrieval gate (before the LLM).** If the best rerank relevance < 0.10,
abstain immediately and list the documents searched. *Why before the LLM?* An
LLM given irrelevant passages tends to answer anyway from its own knowledge; if it
never sees the question, it can't hallucinate. *Why 0.10?* Measured: answerable
0.36–0.57, unanswerable 0.00. The evaluation logs best relevance for all
answerable vs unanswerable questions to justify the value.

**2. Prompt.** Passages numbered [C1]..[Ck], each with label + title + text, so
the model can refer to them unambiguously and we can map [C2] back to "GDPR p. 83,
Article 83(5)". Rules: only the passages; exact 5–25-word quote per claim; copy
numbers exactly, never swap them; keep default rules and exceptions apart; report
conflicts; partial answers name what's missing; abstain if nothing fits.

**3. JSON schema, in this order:** claims (text, citations, quote) → answer →
missing → abstain → explanation → conflicts. An LLM generates left to right, so
field order is reasoning order: it must first commit to evidence, and only at the
end decide whether to abstain. If "abstain" came first, the model would decide
before looking at evidence and then rationalise. Ollama's `format` = the schema
constrains decoding to valid JSON; temperature 0 + seed 42 make it deterministic.

**4. Verifier (code, not the LLM).** For each claim:
- the quote must appear in a cited passage: rapidfuzz `partial_ratio` ≥ 85 (best
  matching substring; tolerant to PDF line breaks and spacing) **or** ≥ 60 % of the
  quote's words occur in the passage;
- every number in the claim must occur in that passage. Numbers are compared
  as values: "20 000 000", "20,000,000" and "EUR 20 million" are all 2e7. This
  catches the classic error of putting 83(4)'s €10M into 83(5)'s claim.
- a claim citing a passage id that wasn't given is rejected.

**5. Retry loop (planner + verifier agent, bonus feature 5).** If any claim fails:
reformulate the query (question + content words of the failed claims), retrieve
again, regenerate, keeping the passages that back already-verified claims. Max 2
attempts (step cap). Claims still failing are dropped; if none survive → abstain
("no claim could be verified"). Each attempt is logged: query, passages, verified
count, rejected claims with reasons. That log is the retry log the assignment asks for.

**Never show unverified text.** If any claim was dropped, the answer is rebuilt
from the verified claims only, instead of using the model's free-text answer.

**6. Confidence.**
- INSUFFICIENT EVIDENCE: abstained.
- HIGH: every claim verified first time, nothing missing, relevance ≥ 0.7.
- LOW: relevance < 0.3, or at least as many claims dropped as kept.
- MEDIUM: everything else (e.g. partial answer, or a claim dropped).

**7. Closed-book baseline.** Same LLM, no retrieval, no passages: confidence
"UNGROUNDED". Gives the mandatory *no-retrieval vs retrieval* comparison.

**Backends.** Ollama (local qwen2.5:7b, default), any OpenAI-compatible API
(cloud fallback, allowed by the assignment), MockLLM (deterministic; scripted
responses make every branch of the verifier testable).

**Why verify in code rather than ask a second LLM?** It is deterministic, fast
(< 1 ms), explainable in one sentence, and can't itself hallucinate. Limitation:
it checks that the quote exists and the numbers match, not that the claim's
*interpretation* of the quote is right (e.g. a negation the model ignored).

**Real-model fix after the first run.** qwen2.5:7b answered only the 83(5) tier
and wrote a second sentence ("applies to basic principles, data subjects'
rights…") with no [C#] marker, so it was shown unverified. Fixes: the prompt asks
for one claim per fact and every tier as its own claim; `grounded_answer()` shows
only answer sentences that carry a [C#] marker pointing at a passage backing a
verified claim, with all numbers present in that passage; otherwise the answer
is rebuilt from the verified claims. A good example for "what did you learn from
error analysis".

## Stage 5 — UI (`app.py`, `lexisync/render.py`)

**Layout (assignment §2 flow).** Sidebar: case (create/select), documents to
search, retrieval mode, passages slider (default 4), verifier toggle,
no-retrieval baseline toggle, LLM (Ollama / mock fallback if Ollama is down).
Tabs: **Ask** · **Documents & structure** · **Evaluation**.

**Cases.** A case = a folder `data/cases/<name>/docs/*.pdf` + its own index
cache. First launch seeds a "gdpr-demo" case from `data/docs`. Upload or URL
adds PDFs; the index rebuilds automatically when files change (cache key =
file paths + modification times; plus the content fingerprint in Stage 3).

**Ask tab shows** the confidence badge, the answer with citations, citation chips,
the "why this evidence supports it" explanation, missing parts, conflicts,
timings (retrieval / rerank / LLM / total) and tokens, the claim verification
table (✅/❌, cited label, quote, match score, attempt, reason), the verifier retry
log, the planner's sub-queries, every retrieved passage with relevance and
BM25/dense ranks, and **the cited PDF page with the cited region highlighted**.

**Highlighting (`render.py`).** The bbox of each block survives from ingestion
→ chunk.bboxes → here. Render the page at zoom z (points → pixels ×z), draw a
translucent yellow rectangle per box, optional crop to the region. Because the
chunk is one clause, the highlight covers exactly 83(5) with its points
(a)–(e) and stops before paragraph 6 (bonus feature 4: bounding-box citations).

**Model loading.** Encoder, reranker and LLM client are created once per process
(`embed._CACHE`, `st.cache_resource`), so the first question takes seconds and
later ones only retrieval + LLM time.

**Abstention is visible** (graded under "Functional UI"): red INSUFFICIENT
EVIDENCE badge, the reason (e.g. "best passage relevance 0.00 is below the
threshold 0.10") and the list of documents searched.

**Tests.** Streamlit AppTest runs the app headless with offline models + mock LLM:
it starts, answers, shows claims/passages/citation, the docs table and the empty
evaluation tab; a pixel test checks the highlight lands inside the box.

**Second document check (ePrivacy Directive 2002/58/EC, 11 pages).** The 2002
Official Journal is genuinely two-column: detected on all 11 pages, read
column-by-column; 49/49 recitals and 21/21 articles found. That's the evidence
that the two-column rule works on real input, not just synthetic tests.

## Stage 6 — Evaluation (`eval/run_eval.py`)

**Question set (`eval/questions.jsonl`, 20 questions).** 11 factual (8 GDPR,
3 ePrivacy), 5 multi-hop (two paragraphs of one document, or two documents:
m03 needs ePrivacy Art. 2(f) → GDPR Art. 94(2) → GDPR Art. 4(11)), 4 unanswerable.
Gold = document + article + paragraph; key facts with "|" alternatives
("20 000 000|20 million"). Every gold label and key fact was checked against the
actual chunk text by script before running.

**Retrieval metrics.** recall@k = share of a question's gold passages in the
top k (a multi-hop question with 2 gold passages and 1 found scores 0.5);
MRR = mean of 1/rank of the first gold passage. Reported at *paragraph* level
(83(5) exactly) and *article* level (any part of Article 83).

**Abstention threshold.** For every question the best cross-encoder relevance
is logged; the report sweeps thresholds and counts unanswerable questions
abstained vs answerable questions wrongly abstained. The chosen 0.10 must
sit between the two groups.

**End-to-end configurations** (ablations):
`no_retrieval` (closed book, the mandatory baseline) · `dense_only` · `full`
(hybrid + rerank + planner + verifier, k = 6) · `full_no_planner` ·
`full_no_verifier` · `full_k4`.

**Metrics.** Automatic correctness (all key facts present, not abstained;
for unanswerable: abstained), a `manual_correct` column in answers.csv that I
grade myself (0/1) and that survives re-runs, citation precision (share of cited
passages in a gold article), abstention accuracy (unanswerable → abstained),
false-abstention rate (answerable → abstained), latency (mean, p95), tokens.

**Engineering.** Progress is appended to `runs/<config>.jsonl` after every
question; `--resume` skips finished questions; `--summary-only` recomputes
metrics after manual grading; `--tag` keeps before/after runs apart for the
error analysis.

**Held-out test (`eval/heldout.jsonl`).** 8 questions on a document never used
during development (Basic Law of Germany), written before running; no
changes to code or thresholds afterwards. This tests generalisation, which is
the same situation as the exam's unseen PDF.

**Robustness fix from the first full evaluation run.** The first `dense_only`
question hung for 300 s: with JSON-constrained decoding, a 7B model can get
stuck emitting whitespace and never close the object. Fixes: `num_predict` =
1024 tokens per answer (normal answers use ~350), a 180 s timeout with a clear
error, and the evaluation retries a failed question once, then records it as
an ERROR (counted as wrong, not as an abstention) and continues.

## Stage 7 — Engineering (Makefile, Docker, CI, diagram)

**What it is.** Everything that makes the project reproducible by someone else.

- **Pinned `requirements.txt`** — exact versions, so the same code gives the same result next
  month. **`config.yaml`** holds every number we tuned plus `seed: 42` (LLM seed, numpy, torch).
- **Makefile** — one short command per task (`make test`, `make app`, `make eval`, `make help`).
  It is documentation that runs.
- **Dockerfile** — `python:3.12-slim` + Tesseract + CPU-only torch; the embedding and reranker
  models are downloaded at build time so the container works offline. **docker-compose** runs
  two services: `ollama` (the LLM server) and `app`; the app finds Ollama via
  `OLLAMA_HOST_URL=http://ollama:11434` (inside Docker, `localhost` would be the app itself).
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — on every push: Ubuntu, Python 3.12,
  Tesseract, then `pytest` with `LEXISYNC_OFFLINE=1` and `LEXISYNC_LLM=mock`. Why offline/mock?
  CI has no GPU and no 4.7 GB model; the tests check *our logic* (chunking, fusion, verifier,
  abstention) with small stand-ins and a scripted LLM, so they are fast and deterministic.
- **OCR benchmark** (`scripts/ocr_benchmark.py`) — synthetic scanned pages whose true text we
  know. CER = character edit distance / length; WER = the same for words. Results at 200 dpi:
  **CER 3.2 %, WER 3.5 %** (150 dpi 4.2 %, 300 dpi 3.5 % — 200 dpi is the speed/accuracy
  sweet spot). Clause-number recall on page 2 was 12 %: Tesseract's automatic layout mode
  (psm 3) treated the margin numbers "(1)", "(2)" as a separate column and dropped them. Fix:
  run psm 3 and psm 6 (one uniform block) and keep the result with more clause starts → 88 %.
- **Architecture diagram** — `scripts/make_architecture.py` draws it from code (SVG → PDF/PNG),
  so the numbers on it can be updated after the final evaluation.

## Exam demo script (5 minutes)

1. **(0:00) Start** — `make app` already running, Ollama running, "Mock" NOT selected.
   Create a new case "exam", upload the examiner's PDF (or paste its URL).
2. **(0:30) Structure report** — Documents tab: "it detected N pages, two-column on pages …,
   tables on …, headers removed, OCR on …". This shows the ingestion is not blind.
3. **(1:00) Prose question** — ask it. Point at: the answer, the citation chips
   ("p. 12, Article 5(2)"), the confidence badge, the claim table (every claim ✔ verified,
   with its quote), and the highlighted region on the rendered page.
4. **(2:00) Two-page multi-hop question** — show the planner split ("searched as 2 parts"),
   two citations on two different pages, open both highlighted pages.
5. **(3:30) Unanswerable question** — INSUFFICIENT EVIDENCE in ~0.3 s, best relevance ≈ 0,
   "no LLM call was made"; list of documents searched. Say: *"it refuses rather than guess."*
6. **(4:30) Close** — point at the diagram: gate → LLM → verifier → citations.

If something goes wrong: a slow answer is the 7B model on a laptop (say so, keep talking);
if a claim shows ✘ it was *dropped* — that is the verifier working, show it.

## Likely theory questions (short answers)

- **Why hybrid retrieval?** BM25 is exact on legal references and rare words ("83(5)", "Article
  17"); dense finds paraphrases ("right to be forgotten" ↔ "erasure"). They fail on different
  questions, so fusing them recovers more (R@10 0.95 vs 0.89 / 0.93).
- **Why RRF instead of adding scores?** BM25 scores (0–30) and cosine (0–1) are on different
  scales; RRF uses only ranks: score = Σ 1/(60 + rank). No normalisation, robust.
- **Bi-encoder vs cross-encoder?** Bi-encoder embeds query and passage separately → fast,
  pre-computable, used for search over all chunks. Cross-encoder reads both together with full
  attention → much more accurate but one forward pass per pair, so only on the top 30.
- **Why does hybrid alone have lower R@1 than BM25?** Fusion promotes passages that are
  "OK in both" over one that is perfect in one; the reranker fixes the order (R@1 0.44 → 0.62).
- **How do you prevent hallucination?** (1) gate before the LLM, (2) prompt: passages only,
  each claim needs a passage id + verbatim quote, (3) code verifier: fuzzy quote match ≥ 85
  and exact number match, (4) only verified sentences shown, (5) abstain if nothing survives.
  The LLM is never trusted on its own word.
- **Why temperature 0 and a seed?** Reproducible answers — the same question gives the same
  answer, which evaluation and an exam demo need.
- **Why JSON with "abstain" last?** The model writes claims first, then decides; deciding
  "abstain" before looking at the evidence makes it refuse too easily or too rarely.
- **Why clause-aware chunks instead of fixed 500 tokens?** A fixed window cuts "Article 83(5)"
  in half and the citation becomes wrong; legal text is already divided into units that
  people cite, so we use them.
- **What is recall@k / MRR?** recall@k: fraction of questions whose gold passage is in the
  top k. MRR: mean of 1/rank of the first correct passage (1.0 = always first).
- **What does the ablation show?** Removing one component at a time (planner, verifier,
  reranker) and measuring the drop proves each component earns its place.
- **What is the held-out test for?** All thresholds were tuned on GDPR/ePrivacy; the Basic Law
  test, run once without retuning, estimates how it does on an unseen document — like the
  exam.
- **Limitations?** 7B model omits parts of long answers; slow on a laptop; complex tables and
  bad scans; threshold calibrated on one corpus.

## Error analysis and the "second look" fix

- **Symptom.** In the full system f02 (lower fine tier, Article 83(4)) and f04 (definition of
  consent, Article 4(11)) ended as INSUFFICIENT EVIDENCE although retrieval had found the
  right passages.
- **Diagnosis from the run log.** `claims 0, dropped 0, tries 1`, reason "the model found no
  answer in the passages". So the *model* refused; the verifier dropped nothing, and the
  retry loop (which only reacts to rejected claims) never ran. The same questions were
  answered in other configurations, so it is model variability, not missing evidence.
  (Ollama on a Mac GPU is not bit-for-bit deterministic even with temperature 0 + seed.)
- **Fix.** If the model returns no claims while the best relevance is ≥ 0.30 (clearly
  relevant passages), ask once more with the same passages plus a note that a relevance
  model judged them relevant. Safety is unchanged: (1) truly unanswerable questions are
  stopped by the gate at 0.10 and never reach this, (2) whatever comes back still passes the
  verifier (quote + numbers), so a pushed model cannot sneak in an invented claim (tested).
- **Exam phrasing.** "I looked at the failures, found the cause in the logs, fixed that
  specific failure mode, measured before/after, and checked on a held-out document that I
  did not overfit."
- **Measured.** Before → after (full system, 20 questions): correct 75 % → 80 %, false
  abstention 12 % → 6 %, citation precision 82 % → 83 %, unanswerable still 4/4, latency
  +4 s. f02 fixed; f04 still refused (honest remaining weakness).

## Held-out result (German Basic Law, never used in development)

- Retrieval recall@3 1.00, MRR 0.90; 7/8 correct; 3/3 unanswerable refused (2 by the gate,
  "how many members currently" by the model); 0/5 answerable wrongly refused; ~10 s/answer.
- A scoring bug first showed recall 0.00: the file `basic_law.pdf` becomes the id `basic-law`,
  the questions said `basic_law`. Lesson: when a metric says 0 but the answers look right,
  suspect the measurement first. Fixed by normalising ids (with a test).
- The one error, h02 (voting age): the right passage, Article 38(2), was retrieved, but the
  model cited Article 54(1) (Federal President, age 40), which shares the words "entitled to
  vote in Bundestag elections". The verifier accepted it because the quote is genuinely
  there. **Exam answer:** "My verifier guarantees faithfulness to the cited passage, not
  relevance to the question. A fix would be a second check that the claim answers the
  question (e.g. cross-encoder score of question vs. claim), but I did not tune after the
  held-out run, to keep the test honest."

## Real root cause found in the UI: cut-off output (token limit)

- **Symptom (UI, fines question):** INSUFFICIENT EVIDENCE although the passages were
  perfect (relevance 0.99, Article 83(4)–(6)). Tokens shown: 1984 + **2048** = 2 attempts ×
  the 1024-token limit → both outputs were **cut off mid-JSON**.
- **Why it looked like a refusal:** invalid JSON was parsed as "no claims, abstain", and the
  message said "the model found no answer". The earlier diagnosis of f02/f04 ("model
  variability") was most likely the same bug — they also took 80–145 s.
- **Fix (general, not tuned to a question):**
  1. Detect truncation (`done_reason == "length"` from Ollama).
  2. Salvage: "claims" is the first JSON field, so every *complete* claim before the cut is
     recovered and verified as usual.
  3. If nothing complete survived, the second look adds "your answer was cut off, give at
     most 4 claims with short quotes".
  4. Prompt rule 11: at most 6 claims, short quotes.
  5. The reason shown is now honest: "output was cut off at the token limit".
- **Exam lesson:** "The UI's token counter showed exactly twice the limit — the metric told
  me the cause. Observability (timings, tokens, retry log) is part of the design."
- **Measured after the cut-off fix (full system, 20 questions):** 85 % correct, 16/16
  answerable answered (false abstention 0 %), 4/4 unanswerable refused, 28 s average
  (faster, because no more 1024-token runaways). Citation precision 79 % (was 83 %): more
  answers, some also citing a neighbouring article. Story: 75 % → 80 % (second look) →
  85 % (real root cause fixed).
