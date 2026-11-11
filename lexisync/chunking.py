"""Stage 2 - Clause-aware chunking: ordered Blocks -> Chunks (one legal unit each).

Why not fixed-size windows? A 500-token window can cut Article 83(4) in half
or glue 83(4) to 83(5); then a citation can't name one clause and the
verifier can't check it. Here a chunk is a legal unit: a recital, a numbered
paragraph ("83(5)"), a definition ("4(11)") or a contract clause ("7.2").

Three schemes, chosen per document by counting structural markers:
  eu        "Article N" headings (+ title line), paragraphs "1." -> "N(1)",
            "(n)" inside an article -> "N(n)" (definitions in Art. 4),
            recitals "(n)" before the first article, CHAPTER/Section = context.
  contract  "7. Liability" headings, inline "1. Definitions. ...",
            "7.2 ..." clauses, numbered sections without headings.
  page      fallback: one section per page.
Then: merge tiny chunks (< 120 chars) into the previous chunk of the same
clause; split long ones (> 1500 chars) at points (a), (b)... or sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import union_bbox
from .models import BBox, Block, Chunk, Document

MIN_CHARS = 120
MAX_CHARS = 1500

ARTICLE_HEAD = re.compile(r"^Article\s+(\d+[a-z]?)\s*$", re.I)        # whole block
ARTICLE_INLINE = re.compile(r"^Article\s+(\d+[a-z]?)\n(.+)$", re.I | re.S)
CHAPTER = re.compile(r"^(CHAPTER|TITLE|PART)\s+([IVXLC]+|\d+)\s*$", re.I)
SECTION_EU = re.compile(r"^Section\s+(\d+)\s*$")
PARA_NUM = re.compile(r"^(\d+)\.(?:\s|$)")                                     # "1. ..."
PAREN_NUM = re.compile(r"^\((\d+)\)(?:\s|$)")                                  # "(1) ..."
POINT = re.compile(r"^\(([a-z]{1,3}|[ivx]+)\)\s")                        # "(a) ..."
CLAUSE_DOTTED = re.compile(r"^(\d+)\.(\d+(?:\.\d+)*)\.?\s")              # "7.2 ..."
HEADING_CONTRACT = re.compile(r"^(\d+)\.\s+([A-Z][^.\n]{0,60})$")        # "7. Liability"
INLINE_HEADING = re.compile(r"^(\d+)\.\s+([A-Z][\w ,&/'-]{1,50})\.\s+(\S.*)$", re.S)
# UK-style statutes: "1\nCommon assault and battery" (section number + title,
# one block), subsections "(1)", group headings like "Aggravation" above them.
STATUTE_HEAD = re.compile(r"^(\d{1,3}[A-Z]{0,2})(?:\n|\s{1,4})([A-Z\u201c\"][^\n]{1,100})$")
RUNNING_HEAD = re.compile(r"^[^\n]*\b(?:Act|Regulations?|Order) \d{4}\b[^\n]*\n\d{1,4}$")
SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


@dataclass
class _Draft:
    """A chunk being built: text segments with their provenance."""
    section: str
    clause: str
    title: str
    kind: str
    context: str
    segs: list[tuple[str, int, BBox]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(t for t, _, _ in self.segs)

    def add(self, b: Block, text: str | None = None) -> None:
        self.segs.append((text if text is not None else b.text, b.page, b.bbox))


def detect_scheme(blocks: list[Block]) -> str:
    n_art = sum(bool(ARTICLE_HEAD.match(b.text) or ARTICLE_INLINE.match(b.text))
                for b in blocks)
    n_contract = sum(bool(CLAUSE_DOTTED.match(b.text) or HEADING_CONTRACT.match(b.text)
                          or INLINE_HEADING.match(b.text)) for b in blocks)
    if n_art >= 2:
        return "eu"
    n_stat = sum(bool(_statute_head(b.text)) for b in blocks)
    n_paren = sum(bool(PAREN_NUM.match(b.text)) for b in blocks)
    if n_stat >= 2 and n_paren >= 2:
        return "statute"
    if n_contract >= 3:
        return "contract"
    return "page"


# ------------------------------------------------------------------- schemes
def _eu_drafts(blocks: list[Block]) -> list[_Draft]:
    drafts: list[_Draft] = []
    cur: _Draft | None = None
    article = ""          # "83"
    title = ""
    chapter = ""          # "CHAPTER VIII Remedies, liability and penalties"
    expect = None         # "chapter_title" | "article_title" | "section_title"
    in_body = False       # True after the first Article

    def start(section, clause, kind, ctx=None):
        nonlocal cur
        cur = _Draft(section, clause, title if section.startswith("Article") else "",
                     kind, chapter if ctx is None else ctx)
        drafts.append(cur)
        return cur

    for b in blocks:
        t = b.text.strip()
        first = t.split("\n", 1)[0]

        if expect == "chapter_title":
            chapter = f"{chapter} {first}"
            expect = None
            continue
        if expect == "section_title":
            chapter = f"{chapter} / {first}"
            expect = None
            continue
        if expect == "article_title":
            expect = None
            if not (PARA_NUM.match(t) or PAREN_NUM.match(t)) and len(t) < 200:
                title = t.replace("\n", " ")
                if cur is not None and cur.section == f"Article {article}":
                    cur.title = title
                continue

        if m := CHAPTER.match(t):
            chapter, expect = f"{m.group(1).upper()} {m.group(2)}", "chapter_title"
            continue
        if (m := SECTION_EU.match(t)) and in_body:
            chapter, expect = re.sub(r" / Section.*$", "", chapter) + f" / Section {m.group(1)}", "section_title"
            continue
        m = ARTICLE_HEAD.match(t)
        mi = None if m else ARTICLE_INLINE.match(t)
        if m or mi:
            article, title, in_body = (m or mi).group(1), "", True
            start(f"Article {article}", "", "heading")
            cur.add(b, f"Article {article}")
            if mi:  # "Article 4\nDefinitions\n..." in one block
                rest = mi.group(2).split("\n", 1)
                title = cur.title = rest[0].strip()
                if len(rest) > 1:
                    cur.add(b, rest[1])
            else:
                expect = "article_title"
            continue

        if not in_body:                                   # preamble + recitals
            if m := PAREN_NUM.match(t):
                start(f"Recital ({m.group(1)})", "", "recital", ctx="Recitals")
            elif cur is None:
                start("Preamble", "", "text", ctx="Preamble")
            cur.add(b)
            continue

        if b.kind == "table":
            start(f"Article {article}", cur.clause if cur else "", "table")
            cur.add(b)
            continue
        m = PARA_NUM.match(t) or PAREN_NUM.match(t)
        if m:
            kind = "definition" if "definition" in title.lower() and PAREN_NUM.match(t) else "text"
            start(f"Article {article}", f"{article}({m.group(1)})", kind)
            cur.add(b)
            continue
        if cur is None:
            start(f"Article {article}", "", "text")
        cur.add(b)                                        # points (a), continuations
    return drafts


def _contract_drafts(blocks: list[Block]) -> list[_Draft]:
    drafts: list[_Draft] = []
    cur: _Draft | None = None
    section, title = "", ""

    def start(clause, kind="text"):
        nonlocal cur
        cur = _Draft(f"Section {section}" if section else "", clause, title, kind, "")
        drafts.append(cur)

    for b in blocks:
        t = b.text.strip()
        if b.kind == "table":
            start(cur.clause if cur else "", "table")
            cur.add(b)
        elif m := CLAUSE_DOTTED.match(t):                 # "7.2 The total..."
            if m.group(1) != section:
                section, title = m.group(1), ""
            start(f"{m.group(1)}.{m.group(2)}")
            cur.add(b)
        elif m := INLINE_HEADING.match(t):                # "1. Definitions. In this..."
            section, title = m.group(1), m.group(2).strip()
            start(section)
            cur.add(b)
        elif m := HEADING_CONTRACT.match(t):              # "7. Liability"
            section, title = m.group(1), m.group(2).strip()
            start("", "heading")
            cur.add(b)
        elif m := PARA_NUM.match(t):                      # "3. The Supplier shall..."
            section, title = m.group(1), ""
            start(section)
            cur.add(b)
        else:
            if cur is None:
                start("")
            cur.add(b)
    return drafts


def _statute_head(t: str):
    """A statute section heading: number + title, no full stop at the end."""
    m = STATUTE_HEAD.match(t.strip())
    return m if m and not m.group(2).rstrip().endswith((".", ";", ",")) else None


def _is_crosshead(t: str) -> bool:
    """A group heading such as 'Aggravation' or 'General provisions'."""
    t = t.strip()
    return ("\n" not in t and 2 < len(t) < 60 and t[0].isupper()
            and not t.endswith((".", ";", ",", ":")) and not re.search(r"\d", t))


def _statute_drafts(blocks: list[Block]) -> list[_Draft]:
    """Statutes (UK acts and similar): Section 1 -> subsections 1(1), 1(2);
    points (a), (b) stay with their subsection; group headings become context."""
    drafts: list[_Draft] = []
    cur: _Draft | None = None
    section, title, group, in_body = "", "", "", False

    def start(clause, kind="text", sec=None, ctx=None):
        nonlocal cur
        cur = _Draft(sec or f"Section {section}", clause, title if sec is None else "",
                     kind, group if ctx is None else ctx)
        drafts.append(cur)

    texts = [b.text.strip() for b in blocks]

    def following(i: int) -> str:
        for t in texts[i + 1:]:
            if not RUNNING_HEAD.match(t):
                return t
        return ""

    def body_follows(i: int) -> bool:
        """A real heading is followed by its text, not by another heading
        (a contents list) or a title page ('ELIZABETH II')."""
        n = following(i)
        if PAREN_NUM.match(n):
            return True
        first = n.split("\n", 1)[0]
        return bool(n) and n[0].isupper() and not _statute_head(n) \
            and not _is_crosshead(n) and not first.isupper()

    for i, b in enumerate(blocks):
        t = texts[i]
        nxt = following(i)
        if RUNNING_HEAD.match(t):                         # "...Act 2018 (c. 23)\n2"
            continue
        if (m := _statute_head(t)) and body_follows(i):
            section, title, in_body = m.group(1), m.group(2).strip(), True
            start("", "heading")
            cur.add(b)
            continue
        if _is_crosshead(t) and _statute_head(nxt):       # "Aggravation" above section 1
            group = t
            continue
        if not in_body:                                   # cover, contents, enacting words
            if cur is None or cur.section != "Preamble":
                start("", "text", sec="Preamble", ctx="Preamble")
            cur.add(b)
            continue
        if b.kind == "table":
            start(cur.clause if cur else "", "table")
            cur.add(b)
            continue
        if m := PAREN_NUM.match(t):
            start(f"{section}({m.group(1)})")
            cur.add(b)
            continue
        cur.add(b)                                        # points (a), continuations
    return drafts


def _page_drafts(blocks: list[Block]) -> list[_Draft]:
    out: list[_Draft] = []
    for b in blocks:
        if b.kind == "table":
            out.append(_Draft(f"Page {b.page + 1}", "", "", "table", ""))
            out[-1].add(b)
            continue
        if not out or out[-1].kind != "page" or out[-1].segs[-1][1] != b.page:
            out.append(_Draft(f"Page {b.page + 1}", "", "", "page", ""))
        out[-1].add(b)
    return out


# ------------------------------------------------------------ merge / split
TABLE_CAPTION = re.compile(r"^Table\s+[\w.]+", re.I)


def _attach_captions(drafts: list[_Draft]) -> list[_Draft]:
    """A short 'Table 9.1 ...' chunk right before a table becomes the table's
    title, so the table can be cited as 'Table 9.1'."""
    out: list[_Draft] = []
    for d in drafts:
        prev = out[-1] if out else None
        if (d.kind == "table" and prev is not None and len(prev.text) < 200
                and TABLE_CAPTION.match(prev.segs[-1][0])):
            cap = prev.segs.pop()
            d.title = cap[0].replace("\n", " ")
            d.segs.insert(0, cap)
            if not prev.segs:
                out.pop()
        out.append(d)
    return out


def _merge_tiny(drafts: list[_Draft]) -> list[_Draft]:
    """Merge a chunk under MIN_CHARS into the previous chunk *of the same
    clause* (same section and clause). A heading-only chunk ("7. Liability")
    is merged forward into the first clause of its section instead."""
    out: list[_Draft] = []
    carry: _Draft | None = None
    for d in drafts:
        if carry is not None:
            if d.section == carry.section:
                d.segs = carry.segs + d.segs
                d.title = d.title or carry.title
            else:
                out.append(carry)
            carry = None
        if d.kind == "heading":
            d.kind = "text"
            if len(d.text) < MIN_CHARS:        # heading (+ short lead-in) only
                carry = d
                continue
        prev = out[-1] if out else None
        if (len(d.text) < MIN_CHARS and prev is not None and d.kind != "table"
                and prev.kind != "table"
                and (prev.section, prev.clause) == (d.section, d.clause)):
            prev.segs.extend(d.segs)
            continue
        out.append(d)
    if carry is not None:
        out.append(carry)
    return out


def _split_text(text: str) -> list[str]:
    """Split one long text at sentence ends, packing pieces up to MAX_CHARS."""
    parts, buf = [], ""
    for s in SENTENCE_END.split(text):
        if buf and len(buf) + len(s) + 1 > MAX_CHARS:
            parts.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        parts.append(buf)
    return parts


def _split_long(d: _Draft) -> list[_Draft]:
    """Split a chunk over MAX_CHARS. Prefer boundaries before points (a), (b)
    (each point is usually its own block); fall back to sentences. Later
    pieces get the lead-in sentence as context, so "(f) the degree of
    cooperation..." still knows it is a factor for deciding a fine."""
    if len(d.text) <= MAX_CHARS or d.kind == "table":
        return [d]
    segs: list[tuple[str, int, BBox]] = []
    for t, p, bb in d.segs:                    # oversize single blocks -> sentences
        segs.extend((s, p, bb) for s in (_split_text(t) if len(t) > MAX_CHARS else [t]))
    pieces: list[list[tuple[str, int, BBox]]] = [[]]
    size = 0
    for seg in segs:
        at_point = bool(POINT.match(seg[0]))
        if pieces[-1] and size + len(seg[0]) > MAX_CHARS and (at_point or size > 0):
            pieces.append([])
            size = 0
        pieces[-1].append(seg)
        size += len(seg[0]) + 1
    if len(pieces) > 1 and sum(len(t) for t, _, _ in pieces[-1]) < MIN_CHARS:
        pieces[-2].extend(pieces.pop())      # don't leave a tiny tail like "(j) ..."
    first = re.sub(r"^(Article \d+[a-z]?\s*)?(\(?\d+\)?\.?)\s*", "", d.text.replace("\n", " "))
    lead = re.split(r"(?<=[:.])\s", first, maxsplit=1)[0][:200]
    out = []
    for i, segs_i in enumerate(pieces):
        ctx = d.context if i == 0 else f"{d.context} | {lead} (cont.)".strip(" |")
        out.append(_Draft(d.section, d.clause, d.title, d.kind, ctx, segs_i))
    return out


# ---------------------------------------------------------------------- main
def chunk_document(doc: Document, include_footnotes: bool = False) -> list[Chunk]:
    blocks = [b for b in doc.blocks if include_footnotes or b.kind != "footnote"]
    scheme = detect_scheme(blocks)
    drafts = {"eu": _eu_drafts, "contract": _contract_drafts,
              "statute": _statute_drafts, "page": _page_drafts}[scheme](blocks)
    if scheme == "statute" and len({d.section for d in drafts
                                    if d.section.startswith("Section ")}) < 2:
        # looked like a statute (e.g. "2\nMOORE v. HARPER" page tops in a judgment)
        # but no real sections were confirmed: cite by page instead
        drafts = _page_drafts(blocks)
    drafts = _attach_captions([d for d in drafts if d.segs])
    drafts = [d for d in _merge_tiny(drafts) if d.text.strip()]
    final = [p for d in drafts for p in _split_long(d)]

    chunks = []
    for i, d in enumerate(final):
        pages = [p for _, p, _ in d.segs]
        by_page: dict[int, list[BBox]] = {}
        for _, p, bb in d.segs:
            by_page.setdefault(p, []).append(bb)
        boxes = [(p, bb) for p, bbs in by_page.items() for bb in bbs]
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}:{i:04d}", doc_id=doc.doc_id, doc_name=doc.name,
            section=d.section, clause=d.clause, title=d.title, text=d.text.strip(),
            page_start=min(pages), page_end=max(pages), bboxes=boxes, kind=d.kind,
            context=d.context))
    return chunks


def index_text(c: Chunk) -> str:
    """What retrieval sees: label + title + context + text. The heading words
    ("administrative fines") and the reference ("Article 83(5)") become
    searchable, while c.text stays verbatim for quoting and verification."""
    return "\n".join(x for x in (c.label, c.title, c.context, c.text) if x)


def chunk_summary(chunks: list[Chunk]) -> str:
    from collections import Counter
    kinds = Counter(c.kind for c in chunks)
    lens = sorted(len(c.text) for c in chunks)
    med = lens[len(lens) // 2] if lens else 0
    return (f"{len(chunks)} chunks  kinds={dict(kinds)}  "
            f"median={med} chars  max={max(lens, default=0)}")


if __name__ == "__main__":
    import sys

    from .ingest import ingest_pdf
    doc = ingest_pdf(sys.argv[1])
    chunks = chunk_document(doc)
    print(f"scheme={detect_scheme(doc.blocks)}  {chunk_summary(chunks)}")
    for c in chunks[: int(sys.argv[2]) if len(sys.argv) > 2 else 10]:
        print(f"- {c.label} | {c.title[:40]} | {c.text[:70]!r}")
