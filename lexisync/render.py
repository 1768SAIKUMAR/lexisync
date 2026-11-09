"""Stage 5a - Render a PDF page with the cited regions highlighted.

The bounding boxes come all the way from ingestion (Block.bbox, in PDF points)
through chunking (Chunk.bboxes = [(page, bbox), ...]). Rendering at `zoom`
scales points to pixels, so a box (x0, y0, x1, y1) is drawn at zoom * box.
"""
from __future__ import annotations

import io

import pymupdf
from PIL import Image, ImageDraw

from .models import BBox, Chunk

HIGHLIGHT = (255, 214, 0, 90)      # translucent yellow
OUTLINE = (220, 120, 0, 255)


def render_page(pdf_path: str, page: int, boxes: list[BBox] | None = None,
                zoom: float = 1.6, crop: bool = False, pad: float = 40) -> bytes:
    """PNG bytes of `page` (0-based) with `boxes` highlighted. With crop=True the
    image is cut to the highlighted region (+ pad points), for a close-up."""
    with pymupdf.open(pdf_path) as doc:
        p = doc[page]
        clip = None
        if crop and boxes:
            x0 = min(b[0] for b in boxes) - pad
            y0 = min(b[1] for b in boxes) - pad
            x1 = max(b[2] for b in boxes) + pad
            y1 = max(b[3] for b in boxes) + pad
            clip = pymupdf.Rect(max(0, x0), max(0, y0), min(p.rect.width, x1),
                                min(p.rect.height, y1))
        pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
    if boxes:
        ox, oy = (clip.x0, clip.y0) if clip else (0, 0)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for b in boxes:
            r = [(b[0] - ox) * zoom - 2, (b[1] - oy) * zoom - 2,
                 (b[2] - ox) * zoom + 2, (b[3] - oy) * zoom + 2]
            d.rectangle(r, fill=HIGHLIGHT, outline=OUTLINE, width=2)
        img = Image.alpha_composite(img, overlay)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def chunk_pages(chunk: Chunk) -> dict[int, list[BBox]]:
    """Group a chunk's boxes by page: {page: [bbox, ...]}."""
    out: dict[int, list[BBox]] = {}
    for p, bb in chunk.bboxes:
        out.setdefault(p, []).append(bb)
    return out
