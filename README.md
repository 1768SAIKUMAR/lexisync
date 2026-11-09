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

**End-to-end** (six configurations: closed-book LLM, dense-only, full system, full without
planner, full without verifier, full with k = 4): see `reports/main/report.md` after
`make eval`. Correctness is scored automatically by key facts and checked by hand in the
`manual_correct` column of `reports/main/answers.csv`.

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
- The abstention threshold was chosen on our own question set; a very different document
  type may need recalibration (that is what the held-out test checks).
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
