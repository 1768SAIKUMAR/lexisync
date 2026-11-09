"""Stage 5: page rendering with highlighted citation regions."""
import io

from PIL import Image

from lexisync.chunking import chunk_document
from lexisync.ingest import ingest_pdf
from lexisync.render import chunk_pages, render_page


def test_render_highlights_cited_region(pdfs):
    path = pdfs["contract.pdf"]
    c = [c for c in chunk_document(ingest_pdf(path)) if c.clause == "7.2"][0]
    (page, boxes), = chunk_pages(c).items()
    plain = Image.open(io.BytesIO(render_page(str(path), page, None, zoom=1)))
    marked = Image.open(io.BytesIO(render_page(str(path), page, boxes, zoom=1)))
    assert plain.size == marked.size == (595, 842)
    x0, y0, x1, y1 = (int(v) for v in boxes[0])
    inside = [marked.getpixel((x, y)) for x in range(x0 + 2, x1 - 2, 7)
              for y in range(y0 + 1, y1 - 1, 3)]
    tinted = [p for p in inside if p[0] > 230 and p[1] > 200 and p[2] < 200]
    assert len(tinted) > len(inside) / 2                # background turned yellow
    assert plain.getpixel((x1 - 3, y0 + 2)) == (255, 255, 255)
    assert marked.getpixel((10, 10)) == (255, 255, 255)  # untouched outside


def test_render_crop_zooms_to_region(pdfs):
    path = pdfs["contract.pdf"]
    c = [c for c in chunk_document(ingest_pdf(path)) if c.clause == "7.2"][0]
    (page, boxes), = chunk_pages(c).items()
    crop = Image.open(io.BytesIO(render_page(str(path), page, boxes, zoom=1, crop=True)))
    assert crop.size[1] < 200                           # just the clause, not the page
