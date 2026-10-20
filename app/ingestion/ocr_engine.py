"""
OCR Engine - Extracts text from page images using RapidOCR.

This module keeps the public helper functions used by the ingestion pipeline,
while swapping the inference backend away from PaddleOCR.
"""

from __future__ import annotations
from rapidocr_onnxruntime import RapidOCR
import logging
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Keep OCR inference lightweight and predictable on CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

_ocr_engine: RapidOCR | None = None


def _get_ocr_engine() -> RapidOCR:
    """Get or initialize the RapidOCR engine (singleton)."""
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("Initializing RapidOCR engine (first run may download cached models)...")
        _ocr_engine = RapidOCR(print_verbose=False)
        logger.info("RapidOCR engine initialized successfully")
    return _ocr_engine


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL Image for OCR.

    The preprocessing is intentionally conservative because RapidOCR already
    performs its own OCR-specific preprocessing internally. We only normalize
    the input to a 3-channel RGB array and optionally apply a light denoise /
    threshold pass when callers explicitly request it.
    """
    rgb_image = image.convert("RGB")
    img_array = np.array(rgb_image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)


def _extract_text_candidates(item: Any) -> list[str]:
    """Recursively collect text values from a RapidOCR result object."""
    if item is None:
        return []

    if hasattr(item, "txts"):
        txts = getattr(item, "txts")
        if txts:
            return [str(text).strip() for text in txts if str(text).strip()]

    if hasattr(item, "text"):
        text = str(getattr(item, "text")).strip()
        if text:
            return [text]

    if isinstance(item, str):
        text = item.strip()
        return [text] if text else []

    if isinstance(item, tuple):

        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            return _extract_text_candidates(item[0])

        # Some wrappers return (result, elapse). Preserve the result object.
        if len(item) == 2 and isinstance(item[1], (int, float)):
            return _extract_text_candidates(item[0])

        # OCR row shapes: (box, text, score) or (box, (text, score)).
        if len(item) >= 2 and isinstance(item[1], str):
            text = item[1].strip()
            return [text] if text else []
        if len(item) >= 2 and isinstance(item[1], (tuple, list)) and item[1]:
            text = str(item[1][0]).strip()
            return [text] if text else []

        texts: list[str] = []
        for value in item:
            texts.extend(_extract_text_candidates(value))
        return texts

    if isinstance(item, list):
        if len(item) >= 2 and isinstance(item[1], str):
            text = item[1].strip()
            return [text] if text else []

        texts: list[str] = []
        for value in item:
            texts.extend(_extract_text_candidates(value))
        return texts

    if hasattr(item, "word_results"):
        word_results = getattr(item, "word_results")
        if word_results:
            texts: list[str] = []
            for value in word_results:
                texts.extend(_extract_text_candidates(value))
            if texts:
                return texts

    return []


def extract_text_from_image(
    image: Image.Image,
    preprocess: bool = True,
) -> str:
    """
    Extract text from a single page image using RapidOCR.

    Args:
        image: PIL Image of a document page.
        preprocess: Whether to apply local image preprocessing before OCR.

    Returns:
        Extracted text as a single string, preserving reading order.
        Returns empty string if OCR fails or finds no text.
    """
    ocr = _get_ocr_engine()

    try:
        img_input = preprocess_image(image) if preprocess else np.array(image.convert("RGB"))
        # result = ocr(img_input)

        result, _ = ocr(img_input)

        texts = _extract_text_candidates(result)
        if not texts and isinstance(result, tuple) and result:
            texts = _extract_text_candidates(result[0])

        extracted = "\n".join(texts).strip()
        if extracted:
            logger.debug("OCR extracted %d characters from image", len(extracted))
        else:
            logger.debug("OCR returned no text for image")
        return extracted

    except Exception as e:
        logger.error("OCR extraction failed: %s", e, exc_info=True)
        return ""


def extract_text_from_pages(
    pages: list[tuple[int, Image.Image, bool]],
) -> dict[int, str]:
    """
    Extract text from multiple page images.

    For pages with embedded text, uses the embedded text directly.
    For scanned pages, runs OCR.
    """
    results: dict[int, str] = {}

    for page_num, image, has_embedded_text in pages:
        if has_embedded_text:
            logger.debug("Page %d: using embedded text (skipping OCR)", page_num)
            continue

        logger.debug("Page %d: running OCR", page_num)
        text = extract_text_from_image(image, preprocess=True)
        results[page_num] = text

    return results

