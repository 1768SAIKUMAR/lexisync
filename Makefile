# One-command runs. Targets are added stage by stage.
PY ?= python3

.PHONY: setup docs synthetic structure chunks test

setup:            ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

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
