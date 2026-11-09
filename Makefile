# One-command runs. Targets are added stage by stage.
PY ?= python3

.PHONY: setup model docs synthetic structure chunks index ask test answer app

setup:            ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

model:            ## download the embedding + reranker models (and the LLM, if Ollama is installed)
	$(PY) -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('models ready')"
	-ollama pull qwen2.5:7b

docs:             ## download the public legal PDFs into data/docs
	$(PY) scripts/download_docs.py data/docs

synthetic:        ## build the synthetic layout test PDFs
	$(PY) scripts/make_synthetic_pdfs.py data/synthetic

structure:        ## print the structure report for every PDF in data/docs
	$(PY) -m lexisync.ingest data/docs/*.pdf

test:             ## run the unit tests
	$(PY) -m pytest -q

chunks:           ## show chunking summary + first chunks for every PDF in data/docs
	for f in data/docs/*.pdf; do $(PY) -m lexisync.chunking $$f 8; done

index:            ## build the search index for data/docs (cached in data/index)
	$(PY) scripts/build_index.py

ask:              ## compare all retrieval modes on one question: make ask Q="..."
	$(PY) scripts/build_index.py --query "$(Q)"

answer:           ## full pipeline (retrieval + LLM + verifier): make answer Q="..."
	$(PY) -m lexisync.pipeline "$(Q)"

app:              ## start the web UI on http://localhost:8501
	$(PY) -m streamlit run app.py
