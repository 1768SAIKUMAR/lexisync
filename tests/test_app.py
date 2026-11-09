"""Stage 5 tests: the Streamlit app runs, answers, abstains and shows evidence
(headless AppTest, offline models, mock LLM, temporary case folder)."""
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(pdfs, tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    shutil.copy(pdfs["eu_regulation.pdf"], seed / "reg.pdf")
    monkeypatch.setenv("LEXISYNC_OFFLINE", "1")
    monkeypatch.setenv("LEXISYNC_CASES", str(tmp_path / "cases"))
    monkeypatch.setenv("LEXISYNC_SEED", str(seed))
    monkeypatch.setenv("LEXISYNC_REPORTS", str(tmp_path / "reports"))   # isolated
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_starts_with_seeded_case(app):
    assert app.sidebar.selectbox[0].value == "gdpr-demo"
    assert "reg" in app.sidebar.multiselect[0].value
    # LLM choice depends on whether Ollama is running on this machine
    assert app.sidebar.radio[0].value in ("Ollama (local)", "Mock (offline demo)")
    assert [t.label for t in app.tabs] == ["Ask", "Documents & structure", "Evaluation"]


def test_ask_shows_answer_badge_claims_and_page(app):
    app.sidebar.radio[0].set_value("Mock (offline demo)").run()   # never call a real LLM
    app.text_area(key="q").set_value("What are the maximum administrative fines?").run()
    app.button[[b.label for b in app.button].index("Ask")].click().run()
    assert not app.exception, app.exception
    md = " ".join(m.value for m in app.markdown)
    assert any(c in md for c in ("HIGH", "MEDIUM", "LOW"))
    assert "Article 83" in md                                 # citation chip / answer
    assert len(app.dataframe) >= 1                           # claim verification table
    assert any(e.label.startswith("[C1]") for e in app.expander)
    picker = [r for r in app.radio if r.value and "Article 83" in str(r.value)]
    assert picker                                            # cited-page selector shown


def test_documents_tab_reports_structure(app):
    tables = [d.value for d in app.dataframe]
    assert any("pages" in t.columns and "chunks" in t.columns for t in tables)


def test_evaluation_tab_without_results(app):
    assert any("No evaluation yet" in i.value for i in app.info)


def test_evaluation_tab_renders_real_summary(pdfs, tmp_path, monkeypatch):
    """A summary with paragraph- AND article-level keys (recall@1_article) must render."""
    import json
    seed = tmp_path / "seed"
    seed.mkdir()
    shutil.copy(pdfs["eu_regulation.pdf"], seed / "reg.pdf")
    rep = tmp_path / "reports"
    rep.mkdir()
    retrieval = {m: {"recall@1": .5, "recall@1_article": .6, "recall@3": .7,
                     "recall@3_article": .8, "mrr": .6, "mrr_article": .7}
                 for m in ("bm25", "dense")}
    cfgs = {"full": {"n": 2, "correctness": .5, "manual_correctness": None,
                     "citation_precision": 1.0, "abstention_accuracy": 1.0,
                     "false_abstention_rate": 0.0, "latency_s": 20.0, "tokens": 900}}
    (rep / "eval_summary.json").write_text(json.dumps(
        {"timestamp": "t", "n_questions": 2, "llm": "mock", "headline": "full",
         "configs": cfgs, "retrieval": retrieval}))
    monkeypatch.setenv("LEXISYNC_OFFLINE", "1")
    monkeypatch.setenv("LEXISYNC_CASES", str(tmp_path / "cases"))
    monkeypatch.setenv("LEXISYNC_SEED", str(seed))
    monkeypatch.setenv("LEXISYNC_REPORTS", str(rep))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Retrieval recall@k" in s.value for s in at.subheader)
