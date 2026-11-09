"""Stage 6 - Evaluation.

    python eval/run_eval.py                          # retrieval + all end-to-end configs
    python eval/run_eval.py --retrieval-only         # recall@k / MRR only (fast)
    python eval/run_eval.py --configs full no_retrieval
    python eval/run_eval.py --resume                 # continue an interrupted run
    python eval/run_eval.py --summary-only           # recompute after manual grading
    python eval/run_eval.py --tag after_fix          # keep before/after results apart
    python eval/run_eval.py --questions eval/heldout.jsonl --docs data/heldout --tag heldout

Outputs (reports/<tag>/):
  retrieval.json       recall@1/3/5/10 + MRR per mode, article and paragraph level
  relevance.csv        best rerank relevance per question (abstention threshold)
  runs/<config>.jsonl  one line per answered question (written immediately)
  answers.csv          all answers + a manual_correct column for YOU to fill in
  summary.json         metrics per config (read by the UI's Evaluation tab)
  report.md            Markdown report
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexisync import load_config               # noqa: E402
from lexisync.answer import answer_question     # noqa: E402
from lexisync.llm import get_llm                # noqa: E402
from lexisync.retrieve import MODES, build_index  # noqa: E402

KS = (1, 3, 5, 10)
CONFIGS = {
    # name:              (mode,               k, verify, closed_book)
    "no_retrieval":      (None,               0, False, True),
    "dense_only":        ("dense",            6, True,  False),
    "full":              ("hybrid_rerank_mq", 6, True,  False),
    "full_no_planner":   ("hybrid_rerank",    6, True,  False),
    "full_no_verifier":  ("hybrid_rerank_mq", 6, False, False),
    "full_k4":           ("hybrid_rerank_mq", 4, True,  False),
}


# ------------------------------------------------------------------ helpers
def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace(" ", " ")
    return re.sub(r"\s+", " ", s)


def norm_doc(doc_id: str) -> str:
    """Document ids are slugs of the file name ("basic_law.pdf" -> "basic-law");
    accept either spelling in the question files."""
    return re.sub(r"[^a-z0-9]+", "-", doc_id.lower()).strip("-")


def matches(chunk, gold: dict, level: str) -> bool:
    if norm_doc(chunk.doc_id) != norm_doc(gold["doc"]) or chunk.section != gold["section"]:
        return False
    return level == "article" or not gold["clause"] or chunk.clause == gold["clause"]


def key_facts_ok(answer: str, key_facts: list[str]) -> bool:
    """Every key fact must appear; '|' separates accepted alternatives."""
    a = norm(answer)
    return all(any(norm(alt) in a for alt in kf.split("|")) for kf in key_facts)


# ---------------------------------------------------------------- retrieval
def eval_retrieval(index, questions, out_dir: Path) -> dict:
    """recall@k = share of a question's gold passages found in the top k,
    averaged over answerable questions; MRR = mean of 1/rank of the first gold
    passage. Article level: right article; paragraph level: right paragraph."""
    answerable = [q for q in questions if q["gold"]]
    res: dict = {}
    rel_rows = []
    for mode in MODES:
        stats = {lvl: {k: [] for k in KS} | {"rr": []} for lvl in ("article", "paragraph")}
        for q in answerable:
            hits = index.search(q["question"], mode=mode, k=10).hits[:10]
            for lvl in stats:
                found_at = []
                for g in q["gold"]:
                    r = next((i for i, h in enumerate(hits, 1) if matches(h.chunk, g, lvl)), None)
                    found_at.append(r)
                for k in KS:
                    stats[lvl][k].append(sum(1 for r in found_at if r and r <= k) / len(found_at))
                first = min((r for r in found_at if r), default=None)
                stats[lvl]["rr"].append(1 / first if first else 0.0)
        res[mode] = {}
        for lvl, st in stats.items():
            suffix = "" if lvl == "paragraph" else "_article"
            for k in KS:
                res[mode][f"recall@{k}{suffix}"] = round(statistics.mean(st[k]), 3)
            res[mode][f"mrr{suffix}"] = round(statistics.mean(st["rr"]), 3)
    # best relevance (cross-encoder) for answerable vs unanswerable -> threshold
    for q in questions:
        r = index.search(q["question"], mode="hybrid_rerank", k=4)
        rel_rows.append({"id": q["id"], "type": q["type"],
                         "best_relevance": round(r.best_relevance or 0, 4)})
    with open(out_dir / "relevance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "type", "best_relevance"])
        w.writeheader()
        w.writerows(rel_rows)
    (out_dir / "retrieval.json").write_text(json.dumps(res, indent=2))
    return res


def threshold_table(rel_csv: Path) -> list[dict]:
    rows = list(csv.DictReader(open(rel_csv)))
    ans = [float(r["best_relevance"]) for r in rows if r["type"] != "unanswerable"]
    una = [float(r["best_relevance"]) for r in rows if r["type"] == "unanswerable"]
    out = []
    for t in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        out.append({"threshold": t,
                    "unanswerable_abstained": f"{sum(v < t for v in una)}/{len(una)}",
                    "answerable_wrongly_abstained": f"{sum(v < t for v in ans)}/{len(ans)}"})
    return out


# ---------------------------------------------------------------- end-to-end
def run_config(name, index, llm, cfg, questions, run_file: Path, resume: bool):
    mode, k, verify, closed = CONFIGS[name]
    done = set()
    if resume and run_file.exists():
        done = {json.loads(l)["id"] for l in run_file.read_text().splitlines() if l.strip()}
    elif run_file.exists():
        run_file.unlink()
    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        t = time.perf_counter()
        r, err = None, ""
        for attempt in (1, 2):                 # one retry, then record the failure
            try:
                r = answer_question(q["question"], index, llm, cfg,
                                    mode=mode or "hybrid_rerank_mq", k=k or None,
                                    verify=verify, closed_book=closed)
                break
            except Exception as e:  # noqa: BLE001 - keep the run going
                err = f"{type(e).__name__}: {e}"
                print(f"  [{name}] {q['id']}: attempt {attempt} failed: {err}", flush=True)
        if r is None:
            row = {"id": q["id"], "type": q["type"], "config": name, "question": q["question"],
                   "answer": "", "confidence": "ERROR", "abstained": False, "error": err,
                   "abstain_reason": "", "claims": 0, "dropped": 0, "attempts": 0,
                   "cited": [], "best_relevance": None,
                   "latency_s": round(time.perf_counter() - t, 2), "timings": {},
                   "tokens": {}, "missing": []}
            with open(run_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            continue
        cited = [{"label": l, "doc": cid.split(":")[0], "chunk_id": cid}
                 for c in r.claims for l, cid in zip(c.labels, c.chunk_ids)]
        row = {"id": q["id"], "type": q["type"], "config": name, "question": q["question"],
               "answer": r.answer, "confidence": r.confidence, "abstained": r.abstained,
               "abstain_reason": r.abstain_reason, "claims": len(r.claims),
               "dropped": len(r.dropped), "attempts": len(r.attempts), "cited": cited,
               "best_relevance": r.best_relevance, "latency_s": round(time.perf_counter() - t, 2),
               "timings": r.timings, "tokens": r.tokens, "missing": r.missing}
        with open(run_file, "a") as f:                     # progress saved per question
            f.write(json.dumps(row) + "\n")
        print(f"  [{name}] {i}/{len(questions)} {q['id']}: {r.confidence} "
              f"({row['latency_s']} s)", flush=True)


def citation_precision(row, q, chunk_index) -> float | None:
    """Share of cited passages that belong to a gold article (article level)."""
    if not row["cited"] or not q["gold"]:
        return None
    ok = 0
    for c in row["cited"]:
        ch = chunk_index.get(c["chunk_id"])
        ok += bool(ch and any(matches(ch, g, "article") for g in q["gold"]))
    return ok / len(row["cited"])


def summarise(out_dir: Path, questions, chunk_index, cfg, llm_name) -> dict:
    qs = {q["id"]: q for q in questions}
    manual = {}
    ans_csv = out_dir / "answers.csv"
    if ans_csv.exists():                       # keep YOUR grading across reruns
        for r in csv.DictReader(open(ans_csv)):
            if r.get("manual_correct", "").strip() in ("0", "1"):
                manual[(r["config"], r["id"])] = int(r["manual_correct"])
    configs, all_rows = {}, []
    for run_file in sorted((out_dir / "runs").glob("*.jsonl")):
        rows = [json.loads(l) for l in run_file.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        name = run_file.stem
        correct, cprec, lat, toks = [], [], [], []
        una_abst, ans_abst, man = [], [], []
        errors = 0
        for row in rows:
            q = qs[row["id"]]
            if row.get("error"):                 # counts as wrong, not as abstention
                errors += 1
                ok = False
            elif q["type"] == "unanswerable":
                ok = row["abstained"]
                una_abst.append(row["abstained"])
            else:
                ok = (not row["abstained"]) and key_facts_ok(row["answer"], q["key_facts"])
                ans_abst.append(row["abstained"])
                p = citation_precision(row, q, chunk_index)
                if p is not None:
                    cprec.append(p)
            row["auto_correct"] = int(ok)
            row["manual_correct"] = manual.get((name, row["id"]), "")
            if row["manual_correct"] != "":
                man.append(row["manual_correct"])
            correct.append(ok)
            lat.append(row["latency_s"])
            toks.append(row["tokens"].get("prompt", 0) + row["tokens"].get("completion", 0))
            all_rows.append(row)
        m = lambda xs: round(statistics.mean(xs), 3) if xs else None  # noqa: E731
        configs[name] = {
            "n": len(rows), "correctness": m(correct),
            "manual_correctness": m(man) if man else None,
            "citation_precision": m(cprec),
            "abstention_accuracy": m(una_abst),
            "false_abstention_rate": m(ans_abst),
            "latency_s": m(lat), "latency_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2),
            "tokens": m(toks), "errors": errors,
        }
    fields = ["config", "id", "type", "question", "answer", "confidence", "abstained",
              "auto_correct", "manual_correct", "claims", "dropped", "attempts",
              "best_relevance", "latency_s", "cited"]
    with open(ans_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({**row, "cited": "; ".join(c["label"] for c in row["cited"])})
    summary = {"timestamp": time.strftime("%Y-%m-%d %H:%M"), "n_questions": len(questions),
               "llm": llm_name, "headline": "full", "configs": configs}
    if (out_dir / "retrieval.json").exists():
        summary["retrieval"] = json.loads((out_dir / "retrieval.json").read_text())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir, summary)
    return summary


def write_report(out_dir: Path, s: dict) -> None:
    L = [f"# LexiSync evaluation ({out_dir.name})", "",
         f"{s['n_questions']} questions · LLM {s['llm']} · {s['timestamp']}", ""]
    if "retrieval" in s:
        L += ["## Retrieval (paragraph level; article level in brackets)", "",
              "| mode | R@1 | R@3 | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|"]
        for mode, d in s["retrieval"].items():
            cells = [f"{d[f'recall@{k}']:.2f} ({d[f'recall@{k}_article']:.2f})" for k in KS]
            L.append(f"| {mode} | " + " | ".join(cells) +
                     f" | {d['mrr']:.2f} ({d['mrr_article']:.2f}) |")
        L.append("")
    rel = out_dir / "relevance.csv"
    if rel.exists():
        L += ["## Abstention threshold (best cross-encoder relevance)", "",
              "| threshold | unanswerable abstained | answerable wrongly abstained |",
              "|---|---|---|"]
        L += [f"| {r['threshold']} | {r['unanswerable_abstained']} | "
              f"{r['answerable_wrongly_abstained']} |" for r in threshold_table(rel)]
        L.append("")
    if s["configs"]:
        L += ["## End-to-end", "",
              "| config | correct (auto) | correct (manual) | citation precision | "
              "abstention acc. | false abstention | latency (s) | tokens |",
              "|---|---|---|---|---|---|---|---|"]
        f = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
        for name, c in s["configs"].items():
            L.append(f"| {name} | {f(c['correctness'])} | {f(c['manual_correctness'])} | "
                     f"{f(c['citation_precision'])} | {f(c['abstention_accuracy'])} | "
                     f"{f(c['false_abstention_rate'])} | {c['latency_s']} | {c['tokens']} |")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    questions = load_questions(Path(a.questions))
    out_dir = ROOT / "reports" / a.tag
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.docs).glob("*.pdf"))
    index = build_index(paths, cfg, cache=Path(a.docs) / ".index.pkl")
    chunk_index = {c.chunk_id: c for c in index.chunks}
    llm = get_llm(cfg)
    print(f"{len(questions)} questions · {len(paths)} documents · {len(index.chunks)} chunks · "
          f"LLM {llm.name}")

    if not a.summary_only:
        print("retrieval evaluation ...")
        eval_retrieval(index, questions, out_dir)
        if not a.retrieval_only:
            for name in a.configs:
                print(f"config {name} ...")
                run_config(name, index, llm, cfg, questions,
                           out_dir / "runs" / f"{name}.jsonl", a.resume)
    s = summarise(out_dir, questions, chunk_index, cfg, llm.name)
    print((out_dir / "report.md").read_text())
    if a.tag == "main":       # the UI reads reports/eval_summary.json
        (ROOT / "reports" / "eval_summary.json").write_text(json.dumps(s, indent=2))
        (ROOT / "reports" / "eval_answers.csv").write_text((out_dir / "answers.csv").read_text())


if __name__ == "__main__":
    main()
