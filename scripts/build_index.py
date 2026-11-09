"""Build (or load from cache) the search index for every PDF in data/docs.

    python scripts/build_index.py                 # real models
    LEXISYNC_OFFLINE=1 python scripts/build_index.py
    python scripts/build_index.py --query "What is the maximum GDPR fine?"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexisync import load_config                       # noqa: E402
from lexisync.chunking import chunk_summary            # noqa: E402
from lexisync.ingest import format_report              # noqa: E402
from lexisync.retrieve import MODES, build_index       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    ap.add_argument("--query", default=None)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config()
    paths = sorted(Path(a.docs).glob("*.pdf"))
    if not paths:
        sys.exit(f"no PDFs in {a.docs}")
    t = time.time()
    idx = build_index(paths, cfg, cache=ROOT / "data" / "index" / "index.pkl",
                      progress=lambda i, n, name: print(f"[{i + 1}/{n}] {name}"))
    print(f"index ready in {time.time() - t:.1f} s   encoder={idx.encoder_name}")
    for d in idx.docs.values():
        print(format_report(d.report))
        print("  " + chunk_summary([c for c in idx.chunks if c.doc_id == d.doc_id]))
    if a.query:
        for m in MODES:
            r = idx.search(a.query, m, a.k)
            rel = f"  best relevance={r.best_relevance:.2f}" if r.best_relevance is not None else ""
            ms = " ".join(f"{k}={v:.0f}ms" for k, v in r.timings.items())
            print(f"\n{m}{rel}  ({ms})")
            for h in r.hits[: a.k]:
                extra = f"  rel={h.relevance:.2f}" if h.relevance is not None else ""
                print(f"  - {h.chunk.label}{extra}")


if __name__ == "__main__":
    main()
