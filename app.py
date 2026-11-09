"""LexiSync - Streamlit UI.   Run:  streamlit run app.py   (or: make app)

Sidebar : cases (create/select), documents in the case, retrieval mode,
          passages slider, verifier toggle, no-retrieval baseline toggle, LLM.
Tabs    : Ask | Documents & structure | Evaluation
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from lexisync import ROOT, load_config
from lexisync.answer import AnswerResult, answer_question
from lexisync.chunking import chunk_summary
from lexisync.llm import MockLLM, get_llm
from lexisync.render import chunk_pages, render_page
from lexisync.retrieve import MODES, build_index

CASES = Path(os.environ.get("LEXISYNC_CASES", ROOT / "data" / "cases"))
SEED = Path(os.environ.get("LEXISYNC_SEED", ROOT / "data" / "docs"))
REPORTS = Path(os.environ.get("LEXISYNC_REPORTS", ROOT / "reports"))
BADGE = {"HIGH": ("#0F6E56", "#E1F5EE"), "MEDIUM": ("#854F0B", "#FAEEDA"),
         "LOW": ("#993C1D", "#FAECE7"), "INSUFFICIENT EVIDENCE": ("#A32D2D", "#FCEBEB")}
EXAMPLES = [
    "Under the GDPR, what is the maximum fine a company can receive, and which kinds "
    "of violations trigger the higher fine?",
    "What does the GDPR define as consent?",
    "What is the governing law of the subcontractor agreement?",
]

st.set_page_config(page_title="LexiSync", page_icon="⚖️", layout="wide")
cfg = load_config()


# ------------------------------------------------------------------ cases
def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "case"


def list_cases() -> list[str]:
    CASES.mkdir(parents=True, exist_ok=True)
    if not any(CASES.iterdir()):                 # first launch: seed from data/docs
        demo = CASES / "gdpr-demo" / "docs"
        demo.mkdir(parents=True)
        for p in SEED.glob("*.pdf"):
            shutil.copy(p, demo / p.name)
    return sorted(p.name for p in CASES.iterdir() if p.is_dir())


def case_pdfs(case: str) -> list[Path]:
    d = CASES / case / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return sorted(d.glob("*.pdf"))


@st.cache_resource(show_spinner=False)
def load_index(case: str, files_key: tuple):
    """Cached per case + file set; the pickle cache in the case folder makes a
    restart fast too."""
    paths = [Path(p) for p, _ in files_key]
    return build_index(paths, cfg, cache=CASES / case / "index.pkl")


def files_key(paths: list[Path]) -> tuple:
    return tuple((str(p), p.stat().st_mtime) for p in paths)


def ollama_ok() -> bool:
    try:
        return requests.get(cfg["llm"]["ollama_host"] + "/api/tags", timeout=1).ok
    except requests.RequestException:
        return False


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("⚖️ LexiSync")
    st.caption("Evidence-grounded legal document analyst")
    cases = list_cases()
    case = st.selectbox("Case", cases, index=0)
    with st.expander("New case"):
        new = st.text_input("Case name", placeholder="e.g. Vendor contract review")
        if st.button("Create case") and new:
            (CASES / slug(new) / "docs").mkdir(parents=True, exist_ok=True)
            st.rerun()

    pdfs = case_pdfs(case)
    st.subheader("Documents")
    selected = st.multiselect("Search in", [p.stem for p in pdfs],
                              default=[p.stem for p in pdfs],
                              help="Restrict the search to these documents.")
    st.subheader("Settings")
    mode = st.selectbox("Retrieval mode", MODES, index=MODES.index("hybrid_rerank_mq"))
    k = st.slider("Passages given to the LLM", 1, 8, cfg["retrieval"]["top_k"])
    verify = st.toggle("Verifier (check quotes & numbers)", value=True)
    closed_book = st.toggle("No-retrieval baseline", value=False,
                            help="Ask the LLM without any documents (for comparison).")
    has_ollama = ollama_ok()
    backend = st.radio("LLM", ["Ollama (local)", "Mock (offline demo)"],
                       index=0 if has_ollama else 1, horizontal=True)
    if backend.startswith("Ollama") and not has_ollama:
        st.error("Ollama is not running. Start it with `brew services start ollama`.")
    st.caption(f"Model: {cfg['llm']['ollama_model'] if backend.startswith('Ollama') else 'mock'}"
               f" · temperature 0 · seed {cfg['seed']}")

index = None
if pdfs:
    with st.spinner("Indexing documents (first time per case only)..."):
        index = load_index(case, files_key(pdfs))
doc_ids = [d.doc_id for d in index.docs.values()
           if Path(d.path).stem in selected] if index else []

tab_ask, tab_docs, tab_eval = st.tabs(["Ask", "Documents & structure", "Evaluation"])


# ---------------------------------------------------------------- helpers
def badge(conf: str) -> str:
    fg, bg = BADGE.get(conf, ("#444441", "#F1EFE8"))
    return (f"<span style='background:{bg};color:{fg};padding:4px 12px;border-radius:12px;"
            f"font-weight:600;font-size:0.9rem'>{conf}</span>")


def chip(text: str) -> str:
    return (f"<span style='border:1px solid #888;border-radius:10px;padding:2px 9px;"
            f"margin:2px;display:inline-block;font-size:0.85rem'>📄 {text}</span>")


def show_page_for(chunk, key: str):
    doc = index.docs[chunk.doc_id]
    pages = chunk_pages(chunk)
    for p, boxes in pages.items():
        crop = st.toggle("Zoom to cited region", value=False, key=f"{key}-{p}")
        st.image(render_page(doc.path, p, boxes, crop=crop),
                 caption=f"{doc.name}, page {p + 1}: highlighted = {chunk.label}",
                 width="stretch")


# --------------------------------------------------------------------- Ask
with tab_ask:
    if not index:
        st.info("This case has no documents yet. Add PDFs in **Documents & structure**.")
    else:
        st.write("**Example questions**")
        cols = st.columns(len(EXAMPLES))
        for c, ex in zip(cols, EXAMPLES):
            if c.button(ex[:60] + ("…" if len(ex) > 60 else ""), width="stretch"):
                st.session_state["q"] = ex
        q = st.text_area("Your question", key="q", height=80,
                         placeholder="e.g. What is the maximum fine under the GDPR?")
        if st.button("Ask", type="primary", disabled=not q.strip()):
            llm = get_llm(cfg) if backend.startswith("Ollama") else MockLLM()
            with st.spinner("Retrieving, answering and verifying..."):
                try:
                    st.session_state["res"] = answer_question(
                        q.strip(), index, llm, cfg, mode=mode, k=k, verify=verify,
                        closed_book=closed_book, doc_ids=doc_ids)
                except RuntimeError as e:
                    st.error(str(e))

        r: AnswerResult | None = st.session_state.get("res")
        if r:
            st.divider()
            st.markdown(badge(r.confidence), unsafe_allow_html=True)
            st.markdown(f"#### {r.answer}")
            labels = list(dict.fromkeys(l for c in r.claims for l in c.labels))
            if labels:
                st.markdown(" ".join(chip(l) for l in labels), unsafe_allow_html=True)
            if r.abstained and r.abstain_reason:
                st.caption(f"Abstained because {r.abstain_reason}.")
            if r.explanation and not r.abstained:
                st.markdown(f"**Why this evidence supports the answer:** {r.explanation}")
            if r.missing:
                st.warning("Not answered by the documents: " + "; ".join(r.missing))
            if r.conflicts:
                st.warning("Conflicting sources: " + "; ".join(r.conflicts))

            t = r.timings
            m = st.columns(5)
            m[0].metric("Retrieval", f"{t.get('retrieval', 0):.0f} ms")
            m[1].metric("Rerank", f"{t.get('rerank', 0):.0f} ms")
            m[2].metric("LLM", f"{t.get('llm', 0) / 1000:.1f} s")
            m[3].metric("Total", f"{t.get('total', 0) / 1000:.1f} s")
            m[4].metric("Tokens", f"{r.tokens.get('prompt', 0)} + {r.tokens.get('completion', 0)}")

            st.subheader("Claim verification")
            rows = [{"": "✅" if c.verified else "❌", "claim": c.text,
                     "cited": "; ".join(c.labels), "quote": c.quote,
                     "match": round(c.quote_score, 2), "attempt": c.attempt,
                     "reason": c.reason or "verified"}
                    for c in r.claims + r.dropped]
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                             column_config={
                                 "": st.column_config.TextColumn(width=40),
                                 "claim": st.column_config.TextColumn(width="large"),
                                 "cited": st.column_config.TextColumn(width="medium"),
                                 "quote": st.column_config.TextColumn(width="large"),
                                 "match": st.column_config.NumberColumn(width=60),
                                 "attempt": st.column_config.NumberColumn(width=60)})
            else:
                st.caption("No claims (abstained before or after generation).")

            left, right = st.columns([1, 1])
            with left:
                if len(r.attempts) > 1 or any(a.get("rejected") for a in r.attempts):
                    with st.expander(f"Verifier retry log ({len(r.attempts)} attempts)"):
                        st.json(r.attempts)
                if r.subqueries and len(r.subqueries) > 1:
                    st.caption("Query planner split the question into: " +
                               " | ".join(r.subqueries))

                st.subheader("Retrieved passages")
                if r.best_relevance is not None:
                    st.caption(f"Best relevance {r.best_relevance:.2f} "
                               f"(abstain below {cfg['answer']['abstain_threshold']:.2f})")
                for i, h in enumerate(r.passages, 1):
                    rel = f"relevance {h.relevance:.2f}" if h.relevance is not None \
                        else f"score {h.score:.3f}"
                    ranks = f"BM25 #{h.bm25_rank or '-'} · dense #{h.dense_rank or '-'}"
                    with st.expander(f"[C{i}] {h.chunk.label} · {rel} · {ranks}"):
                        if h.chunk.title:
                            st.caption(h.chunk.title)
                        st.text(h.chunk.text)

            with right:
                st.subheader("Cited page")
                cited = [h.chunk for h in r.passages
                         if h.chunk.chunk_id in {cid for c in r.claims for cid in c.chunk_ids}]
                show = cited or [h.chunk for h in r.passages[:1]]
                if show:
                    pick = st.radio("Citation", [c.label for c in show], horizontal=True,
                                    label_visibility="collapsed")
                    chunk = next(c for c in show if c.label == pick)
                    if not cited:
                        st.caption("Nothing was cited, so this is the top retrieved passage.")
                    show_page_for(chunk, "ask")


# ------------------------------------------------------- Documents & structure
with tab_docs:
    st.subheader(f"Documents in case “{case}”")
    up = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if up and st.button(f"Add {len(up)} file(s)"):
        for f in up:
            (CASES / case / "docs" / f.name).write_bytes(f.getbuffer())
        st.rerun()
    c1, c2 = st.columns([3, 1])
    url = c1.text_input("…or add a PDF from a URL", placeholder="https://.../document.pdf")
    if c2.button("Fetch", width="stretch") and url:
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "LexiSync/0.1"})
            if not resp.content.startswith(b"%PDF"):
                st.error("That URL returned a web page, not a PDF (EUR-Lex does this for "
                         "scripts). Download it in your browser and upload it instead.")
            else:
                name = slug(Path(url.split("?")[0]).stem or "document") + ".pdf"
                (CASES / case / "docs" / name).write_bytes(resp.content)
                st.rerun()
        except requests.RequestException as e:
            st.error(f"Download failed: {e}")

    if st.button("Re-index documents", help="Normally automatic when files change."):
        load_index.clear()
        (CASES / case / "index.pkl").unlink(missing_ok=True)
        st.rerun()

    if index:
        rows = []
        for d in index.docs.values():
            rp = d.report
            ch = [c for c in index.chunks if c.doc_id == d.doc_id]
            rows.append({"document": d.name, "pages": rp.n_pages, "chunks": len(ch),
                         "scanned pages": len(rp.scanned_pages), "OCR'd": len(rp.ocr_pages),
                         "two-column pages": len(rp.two_column_pages), "tables": rp.n_tables,
                         "footnotes": rp.n_footnotes, "header lines removed": rp.n_removed_lines})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for d in index.docs.values():
            rp = d.report
            ch = [c for c in index.chunks if c.doc_id == d.doc_id]
            with st.expander(f"{d.name}: structure details"):
                st.write(chunk_summary(ch))
                if rp.scanned_pages:
                    st.warning(f"Scanned pages: {rp.scanned_pages} · OCR'd: {rp.ocr_pages}"
                               + (f" · skipped (no Tesseract): {rp.ocr_skipped_pages}"
                                  if rp.ocr_skipped_pages else ""))
                st.write(f"Two-column pages: {rp.two_column_pages or 'none'}")
                st.write(f"Removed header/footer patterns: {rp.removed_header_lines}")
                st.dataframe(pd.DataFrame([{"label": c.label, "title": c.title,
                                            "kind": c.kind, "chars": len(c.text)}
                                           for c in ch]), hide_index=True, height=260)
                if st.button(f"Remove {d.name} from case", key=f"rm-{d.doc_id}"):
                    Path(d.path).unlink(missing_ok=True)
                    st.rerun()


# ---------------------------------------------------------------- Evaluation
with tab_eval:
    summary = REPORTS / "eval_summary.json"
    if not summary.exists():
        st.info("No evaluation yet. Run `make eval` (Stage 6); results appear here.")
    else:
        s = json.loads(summary.read_text())
        st.caption(f"Run: {s.get('timestamp', '')} · {s.get('n_questions', '?')} questions · "
                   f"LLM {s.get('llm', '')}")
        cfgs = pd.DataFrame(s["configs"]).T
        head = s.get("headline", "full")
        if head in cfgs.index:
            h = cfgs.loc[head]
            m = st.columns(5)
            for col, (name, key) in zip(m, [("Correct (auto)", "correctness"),
                                             ("Citation precision", "citation_precision"),
                                             ("Abstention accuracy", "abstention_accuracy"),
                                             ("False abstentions", "false_abstention_rate"),
                                             ("Latency", "latency_s")]):
                v = h.get(key)
                col.metric(name, "-" if v is None or pd.isna(v) else
                           (f"{v:.1f} s" if key == "latency_s" else f"{v:.0%}"))
        st.subheader("All setups")
        metric_cols = [c for c in ["correctness", "manual_correctness", "citation_precision",
                                   "abstention_accuracy", "false_abstention_rate"]
                       if c in cfgs.columns]
        st.bar_chart(cfgs[metric_cols], stack=False)
        if "latency_s" in cfgs.columns:
            st.subheader("Latency per question (s)")
            st.bar_chart(cfgs[["latency_s"]])
        if "retrieval" in s:
            st.subheader("Retrieval recall@k (paragraph level)")
            rec = pd.DataFrame({m: {int(mt.group(1)): v for k_, v in d.items()
                                    if (mt := re.fullmatch(r"recall@(\d+)", k_))}
                                for m, d in s["retrieval"].items()})
            st.line_chart(rec.sort_index())
            st.dataframe(pd.DataFrame(s["retrieval"]).T, width="stretch")
        answers = REPORTS / "eval_answers.csv"
        if answers.exists():
            st.subheader("All answers")
            st.dataframe(pd.read_csv(answers), width="stretch", height=400)
