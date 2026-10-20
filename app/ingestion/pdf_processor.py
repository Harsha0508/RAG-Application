"""
PDF Processor — Converts PDF pages to images for OCR processing.

Uses PyMuPDF (fitz) to render each page as a PIL Image at configurable DPI.
Also detects whether pages have embedded (selectable) text vs scanned images.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import fitz  # PyMuPDF
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


class PageResult(NamedTuple):
    """Result for a single processed PDF page."""
    page_number: int
    image: Image.Image
    embedded_text: str
    has_embedded_text: bool


# Minimum characters to consider a page as having usable embedded text
_MIN_TEXT_LENGTH = 50


def render_pdf_pages(pdf_path: str | Path) -> list[PageResult]:
    """
    Render all pages of a PDF or load an image file as PIL Images.

    For each PDF page, also extracts any embedded (selectable) text.
    For image files (.png, .jpg, .jpeg, .webp), loads the image directly for OCR.

    Args:
        pdf_path: Path to the PDF or image file.

    Returns:
        List of PageResult tuples with page images and text info.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        RuntimeError: If the file cannot be opened.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    suffix = pdf_path.suffix.lower()

    
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        try:
            img = Image.open(str(pdf_path)).convert("RGB")
            logger.info("Loaded image '%s' (%dx%d)", pdf_path.name, img.width, img.height)
            return [PageResult(
                page_number=1,
                image=img,
                embedded_text="",
                has_embedded_text=False,
            )]
        except Exception as e:
            raise RuntimeError(f"Failed to open image '{pdf_path.name}': {e}") from e

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF '{pdf_path.name}': {e}") from e

    results: list[PageResult] = []
    dpi = settings.OCR_DPI
    zoom = dpi / 72.0  # PDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    logger.info("Processing PDF '%s' (%d pages) at %d DPI", pdf_path.name, len(doc), dpi)

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        try:
            # Extract any embedded text
            embedded_text = page.get_text("text").strip()
            has_text = len(embedded_text) >= _MIN_TEXT_LENGTH

            # Render page as pixmap → PIL Image
            pixmap = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

            results.append(PageResult(
                page_number=page_num,
                image=img,
                embedded_text=embedded_text,
                has_embedded_text=has_text,
            ))

            if has_text:
                logger.debug("Page %d: embedded text found (%d chars), OCR may be skipped",
                             page_num, len(embedded_text))
            else:
                logger.debug("Page %d: no/insufficient embedded text, OCR required", page_num)

        except Exception as e:
            logger.error("Failed to process page %d of '%s': %s", page_num, pdf_path.name, e)
            # Continue with remaining pages
            continue

    doc.close()
    logger.info("Rendered %d/%d pages from '%s'", len(results), page_idx + 1, pdf_path.name)
    return results
