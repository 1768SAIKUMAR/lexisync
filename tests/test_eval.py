"""Stage 6 tests: scoring functions and a full offline evaluation run."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from run_eval import key_facts_ok, matches  # noqa: E402

from lexisync.models import Chunk  # noqa: E402


def test_key_facts_alternatives_and_all_required():
    kf = ["20 000 000|20 million", "4 %|4%"]
    assert key_facts_ok("Up to 20 million euros or 4% of turnover.", kf)
    assert not key_facts_ok("Up to 20 million euros.", kf)          # 4 % missing
    assert key_facts_ok("anything", [])


def test_gold_matching_levels():
    c = Chunk("gdpr:1", "gdpr", "GDPR", "Article 83", "83(5)", "", "x", 82, 82)
    assert matches(c, {"doc": "gdpr", "section": "Article 83", "clause": "83(4)"}, "article")
    assert not matches(c, {"doc": "gdpr", "section": "Article 83", "clause": "83(4)"}, "paragraph")
    assert matches(c, {"doc": "gdpr", "section": "Article 83", "clause": ""}, "paragraph")
    assert not matches(c, {"doc": "eprivacy", "section": "Article 83", "clause": ""}, "article")


def test_offline_evaluation_end_to_end(pdfs, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    shutil.copy(pdfs["eu_regulation.pdf"], docs / "reg.pdf")
    qs = tmp_path / "q.jsonl"
    qs.write_text("\n".join(json.dumps(q) for q in [
        {"id": "a", "type": "factual", "question": "What are the administrative fines?",
         "gold": [{"doc": "reg", "section": "Article 83", "clause": "83(5)"}],
         "key_facts": ["20 000 000"]},
        {"id": "u", "type": "unanswerable", "question": "Who won the 1998 World Cup?",
         "gold": [], "key_facts": []}]))
    env = {**os.environ, "LEXISYNC_OFFLINE": "1", "LEXISYNC_LLM": "mock"}
    tag = "pytest_tmp"
    out = subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py"), "--questions",
                          str(qs), "--docs", str(docs), "--tag", tag,
                          "--configs", "full", "no_retrieval"],
                         env=env, capture_output=True, text=True, timeout=300)
    try:
        assert out.returncode == 0, out.stderr[-2000:]
        rep = ROOT / "reports" / tag
        s = json.loads((rep / "summary.json").read_text())
        assert set(s["configs"]) == {"full", "no_retrieval"}
        assert "recall@5" in s["retrieval"]["bm25"]
        assert (rep / "answers.csv").read_text().count("\n") == 5      # header + 2x2
        assert "## End-to-end" in (rep / "report.md").read_text()
        # resume: nothing re-run, files unchanged
        before = (rep / "runs" / "full.jsonl").read_text()
        subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py"), "--questions",
                        str(qs), "--docs", str(docs), "--tag", tag, "--configs", "full",
                        "--resume"], env=env, capture_output=True, timeout=300)
        assert (rep / "runs" / "full.jsonl").read_text() == before
    finally:
        shutil.rmtree(ROOT / "reports" / tag, ignore_errors=True)


def test_eval_survives_llm_failure(pdfs, tmp_path, monkeypatch):
    import run_eval
    from lexisync import load_config
    from lexisync.retrieve import build_index
    monkeypatch.setenv("LEXISYNC_OFFLINE", "1")

    class Broken:
        name = "broken"
        def generate(self, *a, **k):
            raise RuntimeError("Ollama did not answer within 180 s.")

    cfg = load_config()
    idx = build_index([pdfs["eu_regulation.pdf"]], cfg)
    qs = [{"id": "a", "type": "factual", "question": "What are the administrative fines?",
           "gold": [{"doc": "eu-regulation", "section": "Article 83", "clause": ""}],
           "key_facts": ["fine"]}]
    run_file = tmp_path / "full.jsonl"
    run_eval.run_config("full", idx, Broken(), cfg, qs, run_file, resume=False)
    row = json.loads(run_file.read_text())
    assert row["confidence"] == "ERROR" and "180 s" in row["error"]


def test_gold_doc_id_spelling_is_normalised():
    from types import SimpleNamespace
    from eval.run_eval import matches
    ch = SimpleNamespace(doc_id="basic-law", section="Article 38", clause="38(2)")
    assert matches(ch, {"doc": "basic_law", "section": "Article 38", "clause": "38(2)"}, "paragraph")
    assert not matches(ch, {"doc": "gdpr", "section": "Article 38", "clause": "38(2)"}, "article")
