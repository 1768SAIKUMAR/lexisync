# One-command runs.  make help   lists everything.
PY ?= python

.PHONY: help setup model docs synthetic structure chunks index ask answer app eval \
        eval-retrieval heldout ocr test docker

help:             ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ -/'

setup:            ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

model:            ## download embedding + reranker models and the local LLM
	$(PY) -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('models ready')"
	ollama pull qwen2.5:7b

docs:             ## download the public PDFs into data/docs (EUR-Lex may need a browser)
	$(PY) scripts/download_docs.py data/docs

synthetic:        ## build the synthetic layout test PDFs
	$(PY) scripts/make_synthetic_pdfs.py data/synthetic

structure:        ## structure report for every PDF in data/docs
	$(PY) -m lexisync.ingest data/docs/*.pdf

chunks:           ## chunking summary + first chunks for every PDF in data/docs
	for f in data/docs/*.pdf; do $(PY) -m lexisync.chunking $$f 8; done

index:            ## build the search index for data/docs (cached)
	$(PY) scripts/build_index.py

ask:              ## compare all retrieval modes on one question: make ask Q="..."
	$(PY) scripts/build_index.py --query "$(Q)"

answer:           ## full pipeline on one question: make answer Q="..."
	$(PY) -m lexisync.pipeline "$(Q)"

app:              ## start the web UI on http://localhost:8501
	$(PY) -m streamlit run app.py

eval:             ## full evaluation (retrieval + 6 configs), resumable
	caffeinate -i $(PY) eval/run_eval.py --resume 2>/dev/null || $(PY) eval/run_eval.py --resume

eval-retrieval:   ## retrieval-only evaluation (recall@k, MRR, threshold)
	$(PY) eval/run_eval.py --retrieval-only

heldout:          ## held-out test on an unseen document in data/heldout
	$(PY) eval/run_eval.py --questions eval/heldout.jsonl --docs data/heldout --tag heldout --configs full

ocr:              ## OCR accuracy benchmark (CER/WER, clause-number recall)
	$(PY) scripts/ocr_benchmark.py data/docs/gdpr.pdf

test:             ## unit tests (offline models, mock LLM)
	LEXISYNC_OFFLINE=1 $(PY) -m pytest -q

docker:           ## build and start app + Ollama in Docker
	docker compose up --build
