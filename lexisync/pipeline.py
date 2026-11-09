"""Stage 4c - The pipeline object used by the UI and the evaluation.

    from lexisync.pipeline import LexiSync
    ls = LexiSync.from_folder("data/docs")
    r = ls.ask("What is the maximum fine under the GDPR?")
    print(r.answer, r.confidence)
"""
from __future__ import annotations

from pathlib import Path

from . import ROOT, load_config
from .answer import AnswerResult, answer_question
from .llm import LLM, get_llm
from .retrieve import Index, build_index


class LexiSync:
    def __init__(self, index: Index, cfg: dict | None = None, llm: LLM | None = None):
        self.cfg = cfg or load_config()
        self.index = index
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm(self.cfg)
        return self._llm

    @classmethod
    def from_paths(cls, paths: list[Path], cfg: dict | None = None, llm: LLM | None = None,
                   cache: Path | None = None, progress=None) -> "LexiSync":
        cfg = cfg or load_config()
        idx = build_index(paths, cfg, cache=cache, progress=progress)
        return cls(idx, cfg, llm)

    @classmethod
    def from_folder(cls, folder: str | Path = ROOT / "data" / "docs", **kw) -> "LexiSync":
        paths = sorted(Path(folder).glob("*.pdf"))
        kw.setdefault("cache", ROOT / "data" / "index" / "index.pkl")
        return cls.from_paths(paths, **kw)

    def ask(self, question: str, **kw) -> AnswerResult:
        return answer_question(question, self.index, self.llm, self.cfg, **kw)


def format_result(r: AnswerResult) -> str:
    lines = [f"Q: {r.question}", f"[{r.confidence}]  {r.answer}"]
    if r.abstained and r.abstain_reason:
        lines.append(f"  (abstained: {r.abstain_reason})")
    for c in r.claims:
        lines.append(f"  ✓ {c.text}  <- {'; '.join(c.labels)}  quote: \"{c.quote}\"")
    for c in r.dropped:
        lines.append(f"  ✗ dropped: {c.text}  ({c.reason})")
    if r.missing:
        lines.append(f"  missing: {'; '.join(r.missing)}")
    if r.conflicts:
        lines.append(f"  conflicts: {'; '.join(r.conflicts)}")
    if len(r.attempts) > 1:
        lines.append(f"  verifier retries: {len(r.attempts) - 1}")
    t = r.timings
    lines.append(f"  time: retrieval {t.get('retrieval', 0):.0f} ms, rerank "
                 f"{t.get('rerank', 0):.0f} ms, LLM {t.get('llm', 0):.0f} ms, total "
                 f"{t.get('total', 0):.0f} ms; tokens {r.tokens}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    ls = LexiSync.from_folder()
    for q in sys.argv[1:] or ["What is the maximum fine under the GDPR?"]:
        print(format_result(ls.ask(q)), "\n")
