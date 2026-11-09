"""Download the public legal PDFs used for development and evaluation.

    python scripts/download_docs.py            # into data/docs/
All sources are public EU legislation from EUR-Lex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

DOCS = {
    # file name               : URL
    "gdpr.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
    "eprivacy.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32002L0058",
    "ai_act.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689",
}
HEADERS = {"User-Agent": "LexiSync-student-project/0.1"}


def download(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"skip {name} (exists)")
            continue
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            sys.exit(f"{name}: response is not a PDF ({r.headers.get('content-type')})")
        dest.write_bytes(r.content)
        print(f"wrote {dest} ({len(r.content) // 1024} KB)")


if __name__ == "__main__":
    download(Path(sys.argv[1] if len(sys.argv) > 1 else "data/docs"))
