# LexiSync — evidence-grounded legal document analyst

[![CI](https://github.com/1768SAIKUMAR/lexisync/actions/workflows/ci.yml/badge.svg)](https://github.com/1768SAIKUMAR/lexisync/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![LLM](https://img.shields.io/badge/LLM-qwen2.5%3A7b%20(local)-green)
![Tests](https://img.shields.io/badge/tests-95%20passing-brightgreen)

LexiSync answers questions about public legal PDFs — regulations, statutes, contracts,
judgments — and follows three rules:

| | Rule | How |
|---|---|---|
| 📌 | **Cite** every claim | document, page and clause, e.g. `GDPR p. 83, Article 83(5)`, with the cited region highlighted on the page |
| ✅ | **Verify** every claim | code checks that each quote is really in the cited passage and that every number matches |
| 🛑 | **Refuse** when unsure | "INSUFFICIENT EVIDENCE" plus the documents searched, instead of a guess |

It runs **fully locally** (open-weight models, no paid API).

**Author:** Sai Kumar Chintala · DA3 Deep Learning & GenAI final project (Prof. Dr. Gayan de Silva)
· **Repository:** https://github.com/1768SAIKUMAR/lexisync

```text
Q: Under the GDPR, what is the maximum fine a company can receive, and which
   kinds of violations trigger the higher fine?

A: [MEDIUM]  Up to 20 000 000 EUR, or in the case of an undertaking, up to 4 % of the
   total worldwide annual turnover of the preceding financial year, whichever is higher
   (GDPR p. 83, Article 83(5); GDPR p. 83, Article 83(6)). The basic principles for
   processing, including conditions for consent, pursuant to Articles 5, 6, 7 and 9; ...
   2 claims verified · cited page shown with Article 83(5) highlighted · 39 s
```

---

## Contents

1. [Using the app, step by step](#1-using-the-app-step-by-step)
2. [Architecture](#2-architecture)
3. [The pipeline, stage by stage](#3-the-pipeline-stage-by-stage)
4. [Getting started](#4-getting-started)
5. [Evaluation and results](#5-evaluation-and-results)
6. [Tested on unseen documents](#6-tested-on-unseen-documents)
7. [Optional extended features (assignment §4)](#7-optional-extended-features-assignment-4)
8. [Project structure](#8-project-structure)
9. [Testing and reproducibility](#9-testing-and-reproducibility)
10. [Limitations](#10-limitations)

---

## 1. Using the app, step by step

The Streamlit app (`make app`, then http://localhost:8501) follows the order of a real review.

| Step | Where | What you do | What you see |
|---|---|---|---|
| 1 | Sidebar → **New case** | Create a case (a separate set of 2–5 documents) | The case in the case list |
| 2 | **Documents & structure** | Upload PDFs, or paste a PDF URL and click **Fetch** | Indexing runs once (≈1–3 min for 100+ pages), then is cached |
| 3 | **Documents & structure** | Check the structure report | Pages, chunks, scanned/OCR'd pages, two-column pages, tables, footnotes, removed header lines; per-document details |
| 4 | **Ask** | Type a question (or click an example) | — |
| 5 | **Ask** | Read the answer | Confidence badge, cited answer, citation chips, "why this evidence supports the answer", timings and tokens |
| 6 | **Ask** | Check the evidence | Claim-verification table (quote, match score, attempt), retrieved passages with scores, the cited page with the cited region highlighted |
| 7 | **History** | Review earlier questions | Every question of the case with confidence, citations and latency; **Ask again**; CSV download |
| 8 | **Evaluation** | See how well it works | Final run, held-out run, all six configurations, recall@k |

A question the documents cannot answer (*"What is the governing law of the subcontractor
agreement?"*) returns **INSUFFICIENT EVIDENCE**, the reason, and the list of documents searched.

**Sidebar settings:** retrieval mode (default `hybrid_rerank_mq`), passages given to the LLM
(default 6), verifier on/off, a **no-retrieval baseline** switch (the LLM answers from memory,
for comparison), and the LLM (Ollama, or a scripted mock for offline demos).

---

## 2. Architecture

![Architecture](docs/architecture.png)

One-page printable version: [docs/architecture.pdf](docs/architecture.pdf).

```mermaid
flowchart TD
    subgraph INDEX["Indexing: once per document"]
        A["PDF upload or URL"] --> B["Ingestion<br/>text blocks, reading order, 2 columns,<br/>headers/footers, footnotes, tables"]
        B --> S{"Scanned page?"}
        S -- yes --> O["Tesseract OCR, 200 dpi"] --> C
        S -- no --> C["Clause-aware chunking<br/>Article 83(5) · clause 7.2 · section 1(2)<br/>page + bounding boxes + label"]
        C --> X[("Index<br/>BM25 + bge-small embeddings")]
    end

    subgraph ANSWER["Answering: per question"]
        Q["Question"] --> P["Query planner<br/>split multi-part questions"]
        P --> R["Hybrid retrieval<br/>BM25 + dense, RRF k = 60, top 30"]
        X --> R
        R --> K["Cross-encoder rerank<br/>relevance 0..1, keep top 6"]
        K --> G{"Best relevance<br/>&ge; 0.10?"}
        G -- no --> AB["INSUFFICIENT EVIDENCE<br/>+ documents searched<br/>(no LLM call)"]
        G -- yes --> L["LLM qwen2.5:7b, temperature 0, seed 42<br/>JSON: claims with passage id + quote"]
        L --> T{"Output cut off or<br/>model refused?"}
        T -- "yes, relevant passages" --> L
        T -- no --> V{"Verifier: quote in cited passage<br/>and all numbers match?"}
        V -- "claim fails (1st try)" --> R
        V -- "no claim survives" --> AB
        V -- pass --> CI["Citations + confidence<br/>HIGH / MEDIUM / LOW"]
        CI --> UI["Streamlit UI<br/>answer, claim table,<br/>cited page highlighted"]
        AB --> UI
    end
```

---

## 3. The pipeline, stage by stage

| # | Stage | What happens | Code |
|---|---|---|---|
| 1 | **Ingestion** | PyMuPDF reads text blocks with font size and position; NFKC normalisation; two-column pages read column by column; running headers/footers (repeated on ≥ 40 % of pages) and footnotes (small font near the bottom) kept out of the main text; tables converted to Markdown with merged cells filled; pages with almost no text but a large image sent to Tesseract OCR. A structure report records what was found on which page. | `lexisync/ingest.py` |
| 2 | **Clause-aware chunking** | Chunks follow the legal structure, not a fixed size. The layout is detected automatically: **EU acts** (recitals, Article 83 → paragraphs 83(5), definitions 4(11)), **contracts** (7. Liability → clause 7.2), **UK-style statutes** (section 1 → subsections 1(2), group headings as context), and a **per-page fallback** for anything else (e.g. judgments). Points (a), (b) stay with their paragraph; pieces < 120 characters are merged within the same clause, pieces > 1500 are split at points with the lead-in kept. Each chunk stores its pages, bounding boxes and a citation label. | `lexisync/chunking.py` |
| 3 | **Indexing** | BM25 with a tokenizer that keeps legal references (`7.2`, `83(5)(a)`) whole, plus dense embeddings (`BAAI/bge-small-en-v1.5`, 384-d). Cached on disk, rebuilt only when a file or the model changes. | `lexisync/retrieve.py`, `lexisync/embed.py` |
| 4 | **Query planning + hybrid retrieval** | Multi-part questions ("A, and which B?") are split, each part searched separately. BM25 and dense rankings are fused with Reciprocal Rank Fusion (k = 60) → top 30 candidates. | `lexisync/retrieve.py` |
| 5 | **Reranking** | A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads question and passage together and gives a relevance 0–1; the best 6 passages go to the LLM, with the best passage of each sub-question guaranteed. | `lexisync/retrieve.py` |
| 6 | **Abstention gate** | If the best relevance is below 0.10, answer INSUFFICIENT EVIDENCE immediately — no LLM call. | `lexisync/answer.py` |
| 7 | **Generation** | A local LLM (`qwen2.5:7b` via Ollama, temperature 0, seed 42) sees only the numbered passages and returns JSON: claims (each with a passage id and a verbatim quote), answer, missing parts, abstain flag, explanation, conflicts. An OpenAI-compatible API or a mock can be used instead. | `lexisync/llm.py`, `lexisync/answer.py` |
| 8 | **Verification** | Code, not the LLM, checks every claim: quote found in the cited passage (rapidfuzz `partial_ratio` ≥ 85 or word overlap ≥ 0.6) and every number matching. Failed claims → one reformulated search and retry (step cap 2); still failing → dropped. A refusal despite relevant passages gets one "second look"; output cut off at the token limit keeps its complete claims. Only cited, verified sentences are shown. | `lexisync/answer.py` |
| 9 | **Confidence + display** | HIGH (relevance ≥ 0.70, nothing dropped or missing) · MEDIUM · LOW (relevance < 0.30 or most claims dropped) · INSUFFICIENT EVIDENCE. The UI renders the cited page with the region highlighted and saves the question to the case history. | `lexisync/answer.py`, `app.py`, `lexisync/render.py` |

---

## 4. Getting started

**Prerequisites:** Python 3.12+, [Ollama](https://ollama.com), and
[Tesseract](https://github.com/tesseract-ocr/tesseract) for scanned pages
(`brew install tesseract` / `apt install tesseract-ocr`).

```bash
# 1. environment and pinned dependencies
python3 -m venv .venv && source .venv/bin/activate
make setup

# 2. models: embedding + reranker, and the local LLM (ollama pull qwen2.5:7b)
make model

# 3. sample documents: GDPR + ePrivacy into data/docs
#    (EUR-Lex may block scripts — then download the two PDFs in a browser)
make docs

# 4. run the tests, then the app
make test
make app                 # http://localhost:8501
```

**Command line** (`make help` lists every target):

```bash
make structure                                              # what ingestion found in each PDF
make index                                                  # build the cached search index
make ask    Q="What are the maximum fines under the GDPR?"  # compare retrieval modes
make answer Q="What are the maximum fines under the GDPR?"  # full pipeline in the terminal
make eval                                                   # full evaluation -> reports/main/
make heldout                                                # held-out document in data/heldout
make ocr                                                    # OCR accuracy benchmark
```

**Docker** (app + Ollama in two containers): `make docker`, then once
`docker compose exec ollama ollama pull qwen2.5:7b`.

**Configuration:** everything tunable is in [`config.yaml`](config.yaml) — seed, ingestion
thresholds, models, BM25/RRF parameters, `top_k`, abstention and verifier thresholds, LLM
backend. Environment overrides: `LEXISYNC_OFFLINE=1` (small stand-in models), `LEXISYNC_LLM=mock`,
`OLLAMA_HOST_URL`, `LEXISYNC_CASES`, `LEXISYNC_REPORTS`.

---

## 5. Evaluation and results

### 5.1 Question sets

| Set | Documents | Questions | Use |
|---|---|---|---|
| **Development set** `eval/questions.jsonl` | GDPR (88 p.) + ePrivacy Directive (11 p., two-column) → 747 chunks | 20: 11 factual, 5 multi-hop (3 across both documents), 4 unanswerable | Evaluation, ablations, error analysis |
| **Held-out set** `eval/heldout.jsonl` | German Basic Law (English), never used in development → 631 chunks | 8: 4 factual, 1 multi-hop, 3 unanswerable | Run **once** after all tuning, nothing changed afterwards |

Every question has gold articles/paragraphs and key facts. Answers are scored automatically
(key facts) **and manually** (`manual_correct` column). Structure check on the corpus: 99/99 GDPR
articles, 173/173 recitals, 21/21 ePrivacy articles, 49/49 recitals detected.

### 5.2 Retrieval (16 answerable questions, paragraph-level gold)

| Mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.52 | 0.61 | 0.77 | 0.89 | 0.76 |
| Dense (bge-small) | 0.53 | 0.77 | 0.83 | 0.93 | 0.76 |
| Hybrid (RRF) | 0.44 | 0.71 | 0.93 | 0.95 | 0.68 |
| Hybrid + rerank | 0.62 | 0.83 | 0.86 | 0.92 | 0.83 |
| **Hybrid + rerank + planner** | **0.71** | **0.93** | **0.93** | 0.93 | **0.93** |

Abstention gate: at every threshold from 0.01 to 0.30, 4/4 unanswerable questions were stopped
and 0/16 answerable ones; the threshold is 0.10.

### 5.3 End-to-end (20 questions, qwen2.5:7b)

| Configuration | Correct | Citation precision | Unanswerable refused | Answerable wrongly refused | Latency |
|---|---|---|---|---|---|
| No retrieval (LLM from memory) | 35 % | – | 75 % | 44 % | 16 s |
| Dense retrieval only | 60 % | 76 % | 100 % | 19 % | 33 s |
| Full system (k = 6) | 75 % | 82 % | 100 % | 12 % | 28 s |
| Full without planner | 70 % | 76 % | 100 % | 12 % | 31 s |
| Full without verifier | 75 % | 75 % | 100 % | 6 % | 24 s |
| Full with k = 4 | 70 % | 81 % | 100 % | 25 % | 29 s |
| **Final system** | **85 %** (manual: **85 %**) | 79 % | **100 %** | **0 %** | **28 s** |

**What this shows**

- **Retrieval is essential** (the mandatory comparison): from memory the model is right 35 %
  of the time and invented an answer to an unanswerable question; with retrieval, 85 % and no
  invented answers.
- **Hybrid + rerank beats dense only** (75 % vs 60 %); the **planner** adds 5 points of
  correctness and 6 of citation precision; **6 passages beat 4** (with 4, multi-hop questions
  lost one source).
- **The verifier raises citation precision** (75 % → 82 %) at equal correctness.
- **Manual grading** of the final run: 17/20 = 85 %, identical to the automatic check on every
  question. Wrong: f08 (answer stops before the DPO conditions), m02 (second part cites the
  wrong recitals instead of Article 34's "high risk"), m03 (does not say that references now
  point to the GDPR).

### 5.4 Error analysis: 75 % → 85 %

| Step | Finding | Change | Correct | Wrongly refused |
|---|---|---|---|---|
| Baseline | f02, f04 refused although the right passages were retrieved | — | 75 % | 12 % |
| Second look | Logs: the model returned no claims | Re-ask once when relevance ≥ 0.30 | 80 % | 6 % |
| Root cause | UI token counter read exactly 2 × 1024: JSON **cut off at the output limit** was read as a refusal | Detect truncation, keep complete claims, ask for a shorter answer, brevity rule, honest reason | **85 %** | **0 %** |

Runs: `reports/main/` (before), `reports/fix/` (second look), `reports/final/` (final).

### 5.5 Held-out document (German Basic Law, run once)

| Metric | Result |
|---|---|
| Retrieval recall@3 (hybrid + rerank) | 1.00 (MRR 0.90; BM25 alone 0.70) |
| Correct answers | 7/8 (88 %) |
| Citation precision | 80 % |
| Unanswerable refused | 3/3 (2 by the gate, 1 by the model) |
| Answerable wrongly refused | 0/5 |
| Average latency | 10 s |

The one error is instructive: for *"from what age can people vote?"* the right passage
(Article 38(2), age 18) was retrieved, but the model cited Article 54(1) (Federal President, age
40), which contains the words "entitled to vote in Bundestag elections". The verifier accepted
it because the quote *is* in that passage. As agreed for a held-out test, it was not tuned away.

### 5.6 OCR

Synthetic scanned pages with known text, 200 dpi: character error rate **3.2 %**, word error
rate **3.5 %**. Running two Tesseract page-segmentation modes and keeping the one that finds
more clause numbers raised clause-number recall on the hardest page from 12 % to 88 %.

---

## 6. Tested on unseen documents

Beyond the evaluation sets, the app was tried live, as in the exam, on documents it had never
seen. Three were fetched by URL.

| Document | Added by | Structure | Plain question | Two-part question | Unanswerable |
|---|---|---|---|---|---|
| EU AI Act (Regulation 2024/1689), 144 p. | upload | 977 chunks, 575 header lines removed | ✅ HIGH — Article 99(3), €35M / 7 % | ✅ HIGH — definition (Article 3(1), p. 46) + application date; date cited from Recital (179) rather than Article 113 | ✅ refused by the gate (relevance 0.02, 0.3 s) |
| Assaults on Emergency Workers (Offences) Act 2018, 9 p. | URL, legislation.gov.uk | 20 chunks, `section 1(2)` labels | ✅ HIGH — 12 months, section 1(2) | ⚠️ MEDIUM — pages 6 + 7 combined; definition incomplete (constables missing) | ✅ refused by the model (relevance 0.78: right topic, missing fact) |
| US Supreme Court slip opinion (Moore v. Harper, 2023), 65 p. | URL, supremecourt.gov | 120 page-level chunks (judgment) | — | — | ✅ question about a different case refused (relevance 0.00) |

Found and fixed through these trials (each with a regression test): the **statute layout** (UK
acts were first cited by page only), the **page fallback** when a document only looks like a
statute (a judgment's page tops), and an answer from one case leaking into another after
switching cases.

---

## 7. Optional extended features (assignment §4)

| # | Feature | Status | Evidence |
|---|---|---|---|
| 1 | OCR for scanned pages | ✅ | Scanned pages detected and flagged; OCR with two page-segmentation modes; CER 3.2 %, WER 3.5 % on a sample (`scripts/ocr_benchmark.py`); flagged when Tesseract is missing |
| 2 | Table extraction | ✅ partly | Merged cells filled, Markdown output, captions become labels ("Table 9.1"); tested on a synthetic merged-cell table; no accuracy figure vs. manual ground truth |
| 4 | Bounding-box citations | ✅ partly | Cited region highlighted on the rendered page, with zoom; precision measured at passage, not region, level |
| 5 | Planner + verifier agent loop | ✅ | Query planner; verifier rejects unsupported claims, reformulates and retries (step cap 2); retry log shown in the UI |
| 7 | Ablation table | ✅ | Six configurations (§5.3): with/without retrieval, dense only, without planner, without verifier, k = 4 |
| 9 | Multi-document cross-referencing | ✅ partly | Three cross-document questions (GDPR + ePrivacy); conflicting values reported in a `conflicts` field |
| 11 | Latency and cost report | ✅ | Seconds per stage and tokens per question in the UI, the history and every report; optimisations measured: table pre-check (GDPR ingestion 22 s → 1 s), truncation handling (32 s → 28 s, false abstention 6 % → 0 %) |
| 3, 6, 8, 10 | VLM figures, fine-tuned reranker, autoencoder, public deployment | – | Not implemented |

---

## 8. Project structure

```text
lexisync/
├── lexisync/                 core library
│   ├── models.py             data classes: Block, Document, Chunk (+ citation labels)
│   ├── ingest.py             PDF → text blocks: reading order, headers/footers, footnotes, tables, OCR
│   ├── chunking.py           layout detection (EU / contract / statute / page) + clause-aware chunks
│   ├── embed.py              sentence-transformer embeddings and cross-encoder
│   ├── retrieve.py           BM25 + dense + RRF, reranking, query planner, cached index
│   ├── llm.py                Ollama / OpenAI-compatible / mock backends
│   ├── answer.py             gate, prompt, JSON parsing, verifier, retries, confidence
│   ├── pipeline.py           one-call facade + command line
│   └── render.py             page images with highlighted citations
├── app.py                    Streamlit UI: Ask · Documents & structure · Evaluation · History
├── eval/                     question sets + run_eval.py (retrieval, end-to-end, ablations)
├── reports/                  evaluation outputs (main, fix, final, heldout)
├── scripts/                  document download, synthetic PDFs, OCR benchmark, diagram
├── tests/                    95 offline tests (small stand-in models + mock LLM)
├── docs/                     architecture diagram (PNG/PDF/SVG + Mermaid), exam notes
├── config.yaml               every tunable setting, seed 42
├── Makefile                  one command per task (make help)
├── Dockerfile, docker-compose.yml
└── requirements.txt          pinned dependencies
```

---

## 9. Testing and reproducibility

- **95 tests** (`make test`) run offline with small stand-in models and a scripted LLM: ingestion
  on synthetic PDFs (two-column, headers, footnotes, tables, scanned pages, contracts, EU acts),
  chunking for every layout, retrieval, answering and verification, the UI (headless), and the
  evaluation scripts. Bugs found in use were turned into regression tests.
- **CI:** GitHub Actions runs the suite on every push (badge at the top).
- **Reproducibility:** pinned `requirements.txt`, `seed: 42` in `config.yaml`, LLM at temperature 0
  with a fixed seed, cached indexes keyed by a fingerprint of files, models and settings.
- **History:** built stage by stage in small commits (ingestion → chunking → retrieval →
  answering → UI → evaluation → engineering, then fixes from evaluation and live trials).

---

## 10. Limitations

- **Completeness:** a 7B local model sometimes leaves items out of a long answer (one GDPR fine
  tier; constables in the UK act's definition). The verifier guarantees that what is shown is
  supported, not that everything relevant is shown.
- **Support ≠ relevance:** the verifier proves a quote is in the cited passage, not that the
  passage answers the question (held-out voting-age example).
- **Recitals vs articles:** an explanatory recital can outrank the binding article that says the
  same thing (AI Act application date).
- **Speed:** 15–40 s per answer on a laptop; indexing a 100+ page PDF takes 1–3 minutes.
- **Layouts:** complex nested tables, poor scans, and slightly varying running headers (US slip
  opinions) are handled only partly; OCR highlights are page-level.
- **Names:** document names come from file names (`ukpga_20180023_en.pdf` → "Ukpga 20180023 En").

LexiSync is a reading aid for public documents, not legal advice.
