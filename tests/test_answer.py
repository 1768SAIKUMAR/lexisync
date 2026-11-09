"""Stage 4 tests: gate, verifier, retry loop, numbers, confidence, baseline.
Offline models + scripted MockLLM, so every branch is exercised deterministically."""
import copy
import os
import re

import pytest

os.environ["LEXISYNC_OFFLINE"] = "1"

from lexisync import load_config                                   # noqa: E402
from lexisync.answer import (INSUFFICIENT, _with_labels, answer_question,  # noqa: E402
                             numbers_in, parse_json, quote_found)
from lexisync.llm import MockLLM                                   # noqa: E402
from lexisync.retrieve import build_index                          # noqa: E402

Q = "What is the maximum administrative fine for infringing the provisions?"
GOOD_QUOTE = "subject to administrative fines up to 20 000 000 EUR"


@pytest.fixture(scope="module")
def idx(pdfs, tmp_path_factory):
    return build_index([pdfs["eu_regulation.pdf"], pdfs["contract.pdf"]],
                       cache=tmp_path_factory.mktemp("ix") / "i.pkl",
                       names={"eu_regulation": "REG", "contract": "MSA"})


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


def cid_of(prompt: str, label_part: str) -> str:
    """Passage id ([C3]) that the prompt assigned to a label."""
    m = re.search(r"^\[(C\d+)\] [^\n]*" + re.escape(label_part), prompt, re.M)
    assert m, f"{label_part} not in prompt"
    return m.group(1)


def reply(claims, answer="", abstain=False, missing=()):
    return {"claims": claims, "answer": answer, "missing": list(missing),
            "abstain": abstain, "explanation": "test", "conflicts": []}


def good(prompt):
    c = cid_of(prompt, "Article 83(5)")
    return reply([{"text": "Fines can reach 20 000 000 EUR or 4 % of turnover.",
                   "citations": [c], "quote": GOOD_QUOTE}],
                 answer=f"Up to EUR 20 million or 4 % of turnover [{c}].")


# ------------------------------------------------------------- helpers
def test_numbers_in_normalises_formats():
    assert numbers_in("20 000 000 EUR") == numbers_in("EUR 20,000,000") == {20000000.0}
    assert numbers_in("EUR 20 million") == {20000000.0}
    assert numbers_in("4 % of turnover [C3]") == {4.0}          # [C3] ignored
    assert 2.0 in numbers_in("2.0")


def test_quote_found_tolerates_line_breaks_but_not_inventions():
    passage = "be subject to administrative\nfines up to 20 000 000 EUR, or in the case"
    assert quote_found("subject to administrative fines up to 20 000 000 EUR", passage, 85, .6)[0]
    assert not quote_found("the controller must appoint a data protection officer", passage, 85, .6)[0]


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('```json\n{"answer": "x", "claims": []}\n```')["answer"] == "x"
    assert parse_json("Sure! {\"answer\": \"y\"} hope it helps")["answer"] == "y"
    assert parse_json("not json at all")["abstain"] is True


def test_citation_markers_become_labels(idx):
    hits = idx.search(Q, "hybrid_rerank", k=4).hits[:4]
    out = _with_labels("Up to 20M [C1].", hits)
    assert out == f"Up to 20M ({hits[0].chunk.label})."


# ---------------------------------------------------------------- gate
def test_gate_abstains_without_calling_llm(idx, cfg):
    cfg["answer"]["abstain_threshold"] = 1.01             # nothing can pass
    llm = MockLLM()
    r = answer_question(Q, idx, llm, cfg, mode="hybrid_rerank")
    assert r.abstained and r.confidence == INSUFFICIENT
    assert llm.calls == []                                # LLM never called
    assert "REG (2 pages)" in r.answer and "MSA (2 pages)" in r.answer
    assert "below the threshold" in r.abstain_reason


# ------------------------------------------------------------ verifier
def test_verified_claim_and_label_replacement(idx, cfg):
    r = answer_question(Q, idx, MockLLM([good]), cfg, mode="hybrid_rerank")
    assert not r.abstained
    assert [c.verified for c in r.claims] == [True]
    assert r.claims[0].labels == ["REG p. 2, Article 83(5)"]
    assert "(REG p. 2, Article 83(5))" in r.answer and "[C" not in r.answer
    assert r.confidence in ("HIGH", "MEDIUM")


def test_swapped_number_is_rejected(idx, cfg):
    def swapped(prompt):
        c = cid_of(prompt, "Article 83(5)")
        return reply([{"text": "Fines can reach 10 000 000 EUR.",       # wrong number
                       "citations": [c], "quote": GOOD_QUOTE}])
    r = answer_question(Q, idx, MockLLM([swapped, swapped]), cfg, mode="hybrid_rerank")
    assert r.abstained
    assert "10000000" in r.dropped[0].reason.replace(",", "")
    assert r.abstain_reason == "no claim could be verified against its cited passage"


def test_retry_loop_recovers_and_logs(idx, cfg):
    def invented(prompt):
        c = cid_of(prompt, "Article 83(5)")
        return reply([{"text": "Fines can reach 20 000 000 EUR.", "citations": [c],
                       "quote": "the supervisory authority may order a ban on processing"}])
    llm = MockLLM([invented, good])
    r = answer_question(Q, idx, llm, cfg, mode="hybrid_rerank")
    assert len(llm.calls) == 2 and len(r.attempts) == 2
    assert r.attempts[0]["rejected"][0]["reason"].startswith("quote not found")
    assert r.attempts[1]["query"] != Q                    # reformulated query
    assert r.claims and r.claims[0].attempt == 2 and not r.abstained


def test_partially_verified_answer_drops_bad_claim(idx, cfg):
    def mixed(prompt):
        c = cid_of(prompt, "Article 83(5)")
        return reply([
            {"text": "Fines can reach 20 000 000 EUR.", "citations": [c], "quote": GOOD_QUOTE},
            {"text": "Fines are always paid within 30 days.", "citations": [c],
             "quote": "fines must always be paid within thirty days of notice"}],
            answer=f"Up to 20M [{c}], paid within 30 days [{c}].")
    r = answer_question(Q, idx, MockLLM([mixed, mixed]), cfg, mode="hybrid_rerank")
    assert len(r.claims) == 1 and len(r.dropped) == 1
    assert "30 days" not in r.answer                      # rebuilt from verified claims
    assert r.confidence in ("MEDIUM", "LOW")


def test_citing_a_non_existing_passage_fails(idx, cfg):
    bad = reply([{"text": "x y z", "citations": ["C99"], "quote": GOOD_QUOTE}])
    r = answer_question(Q, idx, MockLLM([bad, bad]), cfg, mode="hybrid_rerank")
    assert r.abstained and r.dropped[0].reason == "cites no retrieved passage"


def test_model_abstention_is_respected(idx, cfg):
    r = answer_question(Q, idx, MockLLM([reply([], abstain=True)]), cfg, mode="hybrid_rerank")
    assert r.abstained and r.abstain_reason == "the model found no answer in the passages"


def test_verifier_off_accepts_unchecked_claims(idx, cfg):
    def invented(prompt):
        c = cid_of(prompt, "Article 83(5)")
        return reply([{"text": "Fines can reach 99 EUR.", "citations": [c],
                       "quote": "completely invented words that are nowhere"}])
    r = answer_question(Q, idx, MockLLM([invented]), cfg, mode="hybrid_rerank", verify=False)
    assert not r.abstained and len(r.attempts) == 1      # ablation: no checking


def test_missing_part_lowers_confidence(idx, cfg):
    def partial(prompt):
        out = good(prompt)
        out["missing"] = ["which violations trigger the fine"]
        return out
    r = answer_question(Q, idx, MockLLM([partial]), cfg, mode="hybrid_rerank")
    assert r.missing and r.confidence != "HIGH"


# ------------------------------------------------------------ baseline
def test_closed_book_baseline(idx, cfg):
    llm = MockLLM([reply([{"text": "Up to 4 % of turnover.", "citations": [], "quote": ""}],
                         answer="Up to 4 % of turnover.")])
    r = answer_question(Q, idx, llm, cfg, closed_book=True)
    assert r.mode == "closed_book" and r.confidence.startswith("UNGROUNDED")
    assert "Passages" not in llm.calls[0]                 # no retrieval in the prompt
    assert r.timings["retrieval"] == 0


def test_result_serialises(idx, cfg):
    import json
    r = answer_question(Q, idx, MockLLM([good]), cfg, mode="hybrid_rerank")
    d = r.to_dict()
    json.dumps(d)
    assert d["passages"][0]["label"] and "llm" in d["timings"]


def test_unmarked_or_unbacked_sentences_are_not_shown(idx, cfg):
    def chatty(prompt):
        out = good(prompt)
        c = out["claims"][0]["citations"][0]
        out["answer"] = (f"Fines can reach 20 000 000 EUR [{c}]. "
                         "It also applies to all cats in Europe. "            # no marker
                         f"The fine is 99 EUR [{c}].")                         # bad number
        return out
    r = answer_question(Q, idx, MockLLM([chatty]), cfg, mode="hybrid_rerank")
    assert "20 000 000 EUR (REG p. 2, Article 83(5))" in r.answer
    assert "cats" not in r.answer and "99" not in r.answer


def test_answer_without_markers_falls_back_to_verified_claims(idx, cfg):
    def nomarkers(prompt):
        out = good(prompt)
        out["answer"] = "Up to 20 million euros, and it covers many violations."
        return out
    r = answer_question(Q, idx, MockLLM([nomarkers]), cfg, mode="hybrid_rerank")
    assert r.answer == "Fines can reach 20 000 000 EUR or 4 % of turnover (REG p. 2, Article 83(5))."
