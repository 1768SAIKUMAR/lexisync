# LexiSync — evidence-grounded legal document analyst

LexiSync answers questions about a set of legal PDFs (regulations, contracts, constitutions).
Every claim in an answer cites the **document, page and clause** it comes from
(e.g. `GDPR p. 83, Article 83(5)`), the cited page is shown with the **cited region
highlighted**, and the system **abstains** ("INSUFFICIENT EVIDENCE") when the documents do not
contain the answer. It runs fully locally: open-weight models, no paid API.

![Architecture](docs/architecture.png)

A Mermaid version of the diagram is in [docs/architecture.md](docs/architecture.md);
a printable one-page PDF is in [docs/architecture.pdf](docs/architecture.pdf).

## How it works (one paragraph per stage)

1. **Ingestion** (`lexisync/ingest.py`) — PyMuPDF text blocks with font sizes and boxes,
   NFKC normalisation, two-column reading order, running header/footer and footnote removal,
   tables (`find_tables`, merged cells filled) converted to Markdown, scanned pages sent to
   Tesseract OCR. A structure report lists what was detected on which page.
2. **Clause-aware chunking** (`lexisync/chunking.py`) — chunks follow the legal structure
   (Article 83 → paragraphs 83(5), recitals, definitions 4(11); contract clauses 7 → 7.2).
   Tiny pieces are merged within the same clause, long ones split at points with the
   lead-in kept. Each chunk keeps its page(s), bounding boxes and a citation label.
3. **Retrieval** (`lexisync/retrieve.py`) — BM25 with a legal-reference tokenizer plus
   `BAAI/bge-small-en-v1.5` embeddings, fused with Reciprocal Rank Fusion (k = 60), then the
   top 30 are reranked by the cross-encoder `ms-marco-MiniLM-L-6-v2`. A query planner splits
   multi-part questions so each part gets its own evidence.
4. **Grounded answers** (`lexisync/answer.py`) — a retrieval gate abstains without calling the
   LLM when the best relevance is below 0.10. Otherwise `qwen2.5:7b` (Ollama, temperature 0,
   seed 42, JSON output) writes claims, each with a passage id and a verbatim quote. Code then
   checks every quote against the cited passage (rapidfuzz) and every number; failing claims
   trigger one reformulated retry and are dropped if still unsupported. Only verified,
   cited sentences are shown, with a confidence level HIGH / MEDIUM / LOW.
5. **UI** (`app.py`, `lexisync/render.py`) — Streamlit app with cases (separate document
   sets), upload or URL, answer + citations + claim table, and the cited page rendered with
   the cited region highlighted.
6. **Evaluation** (`eval/run_eval.py`) — retrieval recall@k/MRR for five retrieval modes and
   end-to-end comparison of six system configurations (ablations), plus a held-out test.
7. **Engineering** — pinned requirements, `config.yaml` with seed 42, Makefile, Dockerfile +
   docker-compose (app + Ollama), GitHub Actions CI, 80+ tests that run offline.

## Quickstart (macOS / Linux)

Requirements: Python 3.12+, [Tesseract](https://github.com/tesseract-ocr/tesseract)
(`brew install tesseract` / `apt install tesseract-ocr`) and [Ollama](https://ollama.com).

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup          # pinned dependencies
make model          # embedding + reranker models, and `ollama pull qwen2.5:7b`
make docs           # GDPR + ePrivacy into data/docs (EUR-Lex may require a browser download)
make test           # unit tests
make app            # http://localhost:8501
```

Command line:

```bash
make structure                    # what ingestion found in each PDF
make index                        # build the cached search index
make ask Q="What are the maximum fines under the GDPR?"      # compare retrieval modes
make answer Q="What are the maximum fines under the GDPR?"   # full pipeline
make eval                          # full evaluation -> reports/main/report.md
make heldout                       # held-out document in data/heldout
make ocr                           # OCR accuracy benchmark
make help                          # every target
```

### Docker

```bash
make docker      # = docker compose up --build ; app on http://localhost:8501
docker compose exec ollama ollama pull qwen2.5:7b    # first time only
```

The app container talks to the Ollama container through `OLLAMA_HOST_URL`.

## Configuration

Everything tunable is in [`config.yaml`](config.yaml): seed, ingestion thresholds, models,
BM25 and RRF parameters, `top_k`, the abstention threshold, verifier thresholds and the
LLM backend (`ollama`, an OpenAI-compatible server, or `mock`). Environment overrides:
`LEXISYNC_OFFLINE=1` (small stand-in models for tests), `LEXISYNC_LLM=mock`,
`OLLAMA_HOST_URL`, `LEXISYNC_CASES`, `LEXISYNC_REPORTS`.

## Results

Corpus: GDPR (88 pages) and the ePrivacy Directive (11 pages, two-column) → 747 chunks;
structure check found 99/99 GDPR articles, 173/173 recitals, 21/21 ePrivacy articles and
49/49 recitals. Test set `eval/questions.jsonl`: 20 questions written and verified by hand —
11 factual, 5 multi-hop (3 across both documents), 4 unanswerable.

**Retrieval** (paragraph-level gold passages, 16 answerable questions):

| mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.52 | 0.61 | 0.77 | 0.89 | 0.76 |
| dense (bge-small) | 0.53 | 0.77 | 0.83 | 0.93 | 0.76 |
| hybrid (RRF) | 0.44 | 0.71 | 0.93 | 0.95 | 0.68 |
| hybrid + rerank | 0.62 | 0.83 | 0.86 | 0.92 | 0.83 |
| hybrid + rerank + planner | **0.71** | **0.93** | **0.93** | 0.93 | **0.93** |

Abstention gate: at every threshold from 0.01 to 0.30, 4/4 unanswerable questions abstained
and 0/16 answerable questions were wrongly abstained (chosen threshold 0.10).

**End-to-end** (20 questions, qwen2.5:7b, automatic key-fact scoring):

| configuration | correct | citation precision | abstention accuracy (4 unanswerable) | false abstention (16 answerable) | latency |
|---|---|---|---|---|---|
| closed-book LLM (no retrieval) | 35 % | – | 75 % | 44 % | 16 s |
| dense retrieval only | 60 % | 76 % | 100 % | 19 % | 33 s |
| full system (hybrid + rerank + planner + verifier, k = 6) | **75 %** | **82 %** | **100 %** | 12 % | 28 s |
| full without planner | 70 % | 76 % | 100 % | 12 % | 31 s |
| full without verifier | 75 % | 75 % | 100 % | 6 % | 24 s |
| full with k = 4 | 70 % | 81 % | 100 % | 25 % | 29 s |
| **full + second look (final)** | **80 %** | **83 %** | **100 %** | **6 %** | 32 s |

What this shows:
- **Retrieval is essential**: without documents the model is right 35 % of the time and
  invented an answer for one unanswerable question; with the full pipeline 75 % and no
  invented answers. The four unanswerable questions are stopped by the gate in ~0.3 s,
  without an LLM call.
- **Hybrid + rerank beats dense alone** (75 % vs 60 %), and the **planner** adds 5 points of
  correctness and 6 points of citation precision.
- **The verifier raises citation precision** (75 % → 82 %) at the same correctness: it
  removes claims whose quote or numbers are not in the cited passage.
- **Error analysis of the two wrong refusals** (f02 lower fine tier, f04 definition of
  consent): retrieval found the right passages, but the model itself answered "abstain"
  (0 claims, 0 dropped, 1 try), so the verifier's retry loop never started. The same
  questions were answered in other configurations, so this is model variability, not
  missing evidence. Fix: a **second look** — when the model refuses although the best
  passage relevance is ≥ 0.30, it is asked once more with the same passages and a note that
  they were judged relevant; whatever it returns is still verified. **Result:** correct
  75 % → 80 %, false abstention 12 % → 6 %, citation precision 82 % → 83 %, still 4/4
  unanswerable refused; cost +4 s average latency. f02 is now answered; f04 (definition of
  consent) is still refused after the second look — a remaining 7B-model weakness.
  Because this fix was motivated by test-set errors, the held-out document is the honest
  check of it (`reports/fix/` holds the after-run, `reports/main/` the before-run).
- **k = 6 beats k = 4**: with only 4 passages two multi-hop questions lost evidence
  (false abstention 25 %).
- Full report: `reports/main/report.md`; manual grades go in the `manual_correct` column of
  `reports/main/answers.csv` (then `python eval/run_eval.py --summary-only`).

**Held-out test** (run once after all tuning, nothing changed afterwards): the German Basic
Law (English translation, a document type never used during development, 631 chunks),
8 questions in `eval/heldout.jsonl` — 4 factual, 1 multi-hop, 3 unanswerable.

| | result |
|---|---|
| retrieval recall@3 (hybrid + rerank) | 1.00 (MRR 0.90; BM25 alone 0.70) |
| correct answers | 88 % (7/8) |
| citation precision | 80 % |
| unanswerable refused | 3/3 (2 by the gate, 1 by the model) |
| answerable wrongly refused | 0/5 |
| average latency | 10 s |

The one error (h02, "from what age can people vote?") is instructive: retrieval had the
right passage (Article 38(2), age 18) in the top 3, but the model cited Article 54(1)
(election of the Federal President, age 40), which contains the exact phrase "entitled to
vote in Bundestag elections". The verifier passed it because the quote *is* in that
passage: it checks that a claim is supported by its citation, not that the citation answers
the question. As agreed for a held-out test, this was not tuned away.

**OCR** (synthetic scanned pages with known text, 200 dpi): mean character error rate 3.2 %,
word error rate 3.5 %; clause-number recall on the hardest page improved from 12 % to 88 %
by running two Tesseract page-segmentation modes and keeping the one that finds more
clause numbers.

## Limitations (honest)

- A 7B local model sometimes leaves out one part of a long multi-part answer (e.g. the second
  fine tier of Article 83); the verifier guarantees what *is* shown is supported, not that
  everything relevant is shown. Missing parts are listed under "missing".
- Answers are slow on a laptop CPU/GPU (15–40 s per question; ~80 s with a retry).
- Tables and scanned pages are handled, but complex nested tables and poor scans lose
  structure; OCR text keeps page-level, not word-perfect, highlighting.
- The abstention threshold was chosen on our own question set; on the held-out document it
  still refused 3/3 unanswerable and 0/5 answerable questions.
- The verifier proves a claim is *in* the cited passage, not that the passage answers the
  question (held-out h02: a quote about the President's minimum age answered a voting-age
  question).
- This is a reading aid, not legal advice.

## Project layout

```
lexisync/      models, ingest, chunking, embed, retrieve, llm, answer, pipeline, render
app.py         Streamlit UI
eval/          questions.jsonl, heldout.jsonl, run_eval.py
scripts/       download_docs, make_synthetic_pdfs, build_index, ocr_benchmark, make_architecture
tests/         pytest suite (offline models + mock LLM)
docs/          architecture diagram (PNG/PDF/SVG + Mermaid), exam notes
reports/       evaluation outputs
```
