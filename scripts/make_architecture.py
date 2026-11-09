"""Draw the one-page architecture diagram: docs/architecture.{svg,pdf,png}.

    python scripts/make_architecture.py
Edit RESULTS below after the final evaluation run.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RESULTS = [
    "20 questions (11 factual, 5 multi-hop incl. 3 cross-document, 4 unanswerable) + held-out "
    "set on an unseen document · recall@1/3/5/10 + MRR per retrieval mode",
    "recall@3: BM25 0.61 · dense 0.77 · hybrid 0.71 · +rerank 0.83 · +planner 0.93 (MRR 0.93) · OCR CER 3.2 %",
    "end-to-end correct: closed-book 35 % · dense-only 60 % · full 75 % → 80 % with second look (cit. precision 83 %) "
    "· unanswerable abstained 4/4 · held-out Basic Law: 7/8 correct, 3/3 refused",
]

W, H = 1600, 1030
IDX = ("#E1F5EE", "#0F6E56", "#085041")
ANS = ("#EEEDFE", "#534AB7", "#3C3489")
SAFE = ("#FAECE7", "#993C1D", "#712B13")
OUT = ("#F1EFE8", "#5F5E5A", "#2C2C2A")


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, t, size=16, col="#333", bold=False, anchor="start"):
    w = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}"{w} '
            f'text-anchor="{anchor}">{esc(t)}</text>')


def box(x, y, w, h, title, lines, c):
    fill, stroke, tcol = c
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="2"/>',
           text(x + w / 2, y + 34, title, 21, tcol, True, "middle")]
    out += [text(x + w / 2, y + 64 + i * 24, l, 16, "#333", anchor="middle")
            for i, l in enumerate(lines)]
    return "\n".join(out)


def arrow(points, col="#555", width=2.5):
    """Polyline with a hand-drawn triangle head (no SVG markers: not every
    renderer supports them). Segments must be horizontal or vertical."""
    d = "M" + " L".join(f"{x} {y}" for x, y in points)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    dx, dy = (x2 > x1) - (x2 < x1), (y2 > y1) - (y2 < y1)
    a, b = 12, 7
    tip = (x2, y2)
    base = (x2 - dx * a, y2 - dy * a)
    left = (base[0] - dy * b, base[1] + dx * b)
    right = (base[0] + dy * b, base[1] - dx * b)
    head = " ".join(f"{p[0]},{p[1]}" for p in (tip, left, right))
    return (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}"/>'
            f'<polygon points="{head}" fill="{col}"/>')


def build() -> str:
    bw, bh, xs = 330, 150, [60, 440, 820, 1200]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         text(60, 60, "LexiSync: evidence-grounded legal document analyst", 32, "#222", True),
         text(60, 92, "Every claim cites page + clause, is verified against its source, and the "
              "system abstains when the documents lack evidence.", 18, "#555"),
         text(60, 140, "INDEXING  (once per document)", 17, IDX[1], True)]
    top = [("Documents", ["2–5 public legal PDFs", "GDPR (88 p.), ePrivacy (11 p.,",
                          "two-column), any upload / URL", "cases = separate document sets"], OUT),
           ("1  Ingestion", ["PyMuPDF dict · NFKC · reading order",
                             "2-column bands · header/footer removal",
                             "footnotes · tables (find_tables)",
                             "scanned → Tesseract OCR 200 dpi"], IDX),
           ("2  Clause-aware chunking", ["EU: Article 83 → 83(5), recitals, 4(11)",
                                         "contract: 7. Liability → 7.2",
                                         "merge < 120 chars · split > 1500",
                                         "label 'GDPR p. 83, Article 83(5)' + bboxes"], IDX),
           ("3  Index", ["BM25 (legal-ref tokenizer: 7.2, 83(5))",
                         "bge-small-en-v1.5 embeddings (384-d)",
                         "fingerprinted pickle cache", "GDPR + ePrivacy: 747 chunks"], IDX)]
    for x, (t, l, c) in zip(xs, top):
        s.append(box(x, 155, bw, bh, t, l, c))
    for i in range(3):
        s.append(arrow([(xs[i] + bw, 230), (xs[i + 1] - 2, 230)]))

    s.append(text(60, 400, "ANSWERING  (per question)", 17, ANS[1], True))
    row_a = [("4  Query planner", ["splits 'A, and which B?'", "vague part inherits topic",
                                   "each part searched separately",
                                   "best passage of each part kept"], ANS),
             ("5  Hybrid retrieval", ["BM25 rank + dense rank", "Reciprocal Rank Fusion, k = 60",
                                      "top 30 candidates", "(ranks, not raw scores)"], ANS),
             ("6  Cross-encoder rerank", ["ms-marco-MiniLM-L-6-v2",
                                          "reads query + passage together",
                                          "sigmoid → relevance 0..1",
                                          "top k = 4–6 passages"], ANS),
             ("7  Retrieval gate", ["best relevance < 0.10 ?", "→ ABSTAIN, no LLM call",
                                    "list documents searched",
                                    "(answerable ≥ 0.3, unanswerable ≈ 0)"], SAFE)]
    for x, (t, l, c) in zip(xs, row_a):
        s.append(box(x, 415, bw, bh, t, l, c))
    for i in range(3):
        s.append(arrow([(xs[i] + bw, 490), (xs[i + 1] - 2, 490)]))

    row_b = [("10  UI (Streamlit)", ["answer + confidence badge", "citation chips · timings",
                                     "claim verification table",
                                     "cited PDF page, region highlighted"], OUT),
             ("9  Citations & confidence", ["[C2] → 'GDPR p. 83, Article 83(5)'",
                                            "only cited, checked sentences shown",
                                            "HIGH / MEDIUM / LOW /",
                                            "INSUFFICIENT EVIDENCE"], ANS),
             ("8b  Verifier (code)", ["quote in cited passage: rapidfuzz",
                                      "partial_ratio ≥ 85 or overlap ≥ 0.6",
                                      "every number must match (20M ≠ 10M)",
                                      "fail → reformulate, retry (max 2)"], SAFE),
             ("8a  LLM: qwen2.5:7b (Ollama)", ["temperature 0 · seed 42 · JSON schema",
                                               "claims (text, [C#], quote) → answer",
                                               "→ missing → abstain (decide last)",
                                               "rules: only passages, no swapped numbers"], ANS)]
    for x, (t, l, c) in zip(xs, row_b):
        s.append(box(x, 655, bw, bh, t, l, c))
    for i in (3, 2, 1):
        s.append(arrow([(xs[i], 730), (xs[i - 1] + bw + 2, 730)]))

    s.append(arrow([(1365, 565), (1365, 653)], ANS[1]))                       # gate -> LLM
    s.append(text(1378, 615, "relevance ≥ 0.10", 15, ANS[2]))
    s.append(arrow([(1365, 305), (1365, 362), (605, 362), (605, 413)], IDX[1]))  # index -> retr.
    s.append(text(700, 354, "index used by retrieval", 15, IDX[1]))
    s.append(arrow([(920, 655), (920, 612), (700, 612), (700, 567)], SAFE[1]))  # retry loop
    s.append(text(712, 603, "failed claim → new query", 15, SAFE[1]))

    s.append(f'<rect x="60" y="822" width="1470" height="34" rx="8" fill="{SAFE[0]}" '
             f'stroke="{SAFE[1]}" stroke-width="1.5"/>')
    s.append(text(75, 845, "Abstention: gate (step 7), model says no answer, or no claim "
                  "survives the verifier  →  INSUFFICIENT EVIDENCE + list of documents searched",
                  16, SAFE[2]))
    s.append(text(60, 900, "EVALUATION", 17, "#222", True))
    for i, line in enumerate(RESULTS):
        s.append(text(60, 928 + i * 26, line, 16))
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    svg = build()
    (out / "architecture.svg").write_text(svg)
    pdf = pymupdf.open("pdf", pymupdf.open("svg", svg.encode()).convert_to_pdf())
    pdf.save(out / "architecture.pdf")
    pdf[0].get_pixmap(dpi=110).save(out / "architecture.png")
    print("wrote docs/architecture.svg, .pdf, .png")


if __name__ == "__main__":
    main()
