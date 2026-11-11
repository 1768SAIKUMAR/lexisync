"""Data model shared by every stage.

Design rule: provenance is never lost. A Block knows its page and bounding box;
a Chunk keeps the list of (page, bbox) of the blocks it came from. That is what
lets the UI highlight the exact region behind a citation at the very end.
Pages are stored 0-based internally and shown 1-based to users.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points


@dataclass
class Block:
    """One unit of text in reading order (paragraph, table or footnote)."""
    page: int                  # 0-based page index
    bbox: BBox
    text: str
    kind: str = "text"         # "text" | "table" | "footnote"
    font_size: float = 0.0
    source: str = "digital"    # "digital" | "ocr"
    order: int = 0             # position in the document's reading order


@dataclass
class PageInfo:
    number: int                # 0-based
    width: float
    height: float
    n_chars: int = 0
    image_cover: float = 0.0
    is_scanned: bool = False
    ocr_used: bool = False
    ocr_psm: int | None = None
    ocr_skipped: bool = False  # scanned but Tesseract unavailable
    two_column: bool = False
    n_tables: int = 0
    n_footnotes: int = 0


@dataclass
class StructureReport:
    doc_id: str
    n_pages: int = 0
    scanned_pages: list[int] = field(default_factory=list)   # 1-based
    ocr_pages: list[int] = field(default_factory=list)
    ocr_skipped_pages: list[int] = field(default_factory=list)
    two_column_pages: list[int] = field(default_factory=list)
    n_tables: int = 0
    n_footnotes: int = 0
    n_blocks: int = 0
    removed_header_lines: list[str] = field(default_factory=list)
    n_removed_lines: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    doc_id: str                # slug used internally, e.g. "gdpr"
    name: str                  # short display name used in labels, e.g. "GDPR"
    path: str
    pages: list[PageInfo]
    blocks: list[Block]
    report: StructureReport


@dataclass
class Chunk:
    """Retrieval unit produced by chunking.py (Stage 2)."""
    chunk_id: str
    doc_id: str
    doc_name: str
    section: str               # e.g. "Article 83" or "7"
    clause: str                # e.g. "83(5)" or "7.2"; "" if none
    title: str                 # e.g. "General conditions for imposing administrative fines"
    text: str
    page_start: int            # 0-based
    page_end: int
    bboxes: list[tuple[int, BBox]] = field(default_factory=list)
    kind: str = "text"         # "text" | "table" | "recital" | "definition" | "page"
    context: str = ""          # e.g. chapter heading, used as context only

    @property
    def label(self) -> str:
        """Human-readable citation: 'GDPR p. 83, Article 83(5)',
        'GDPR p. 2, Recital (4)', 'MSA p. 1, clause 7.2', 'X p. 3'."""
        if self.page_start == self.page_end:
            pages = f"p. {self.page_start + 1}"
        else:
            pages = f"pp. {self.page_start + 1}-{self.page_end + 1}"
        if self.section.startswith("Article"):
            ref = f"Article {self.clause}" if self.clause else self.section
        elif self.section.startswith(("Recital", "Preamble", "Page")):
            ref = "" if self.section.startswith("Page") else self.section
        elif self.clause and "(" in self.clause:      # statute: section 1(2)
            ref = f"section {self.clause}"
        elif self.clause:
            ref = f"clause {self.clause}"
        else:
            ref = self.section.lower()
        if self.kind == "table":
            import re
            m = re.match(r"Table\s+[\w.]+", self.title, re.I)
            ref = f"{ref}, {m.group(0)}" if m and ref else (m.group(0) if m else f"{ref} table".strip())
        return f"{self.doc_name} {pages}, {ref}" if ref else f"{self.doc_name} {pages}"
