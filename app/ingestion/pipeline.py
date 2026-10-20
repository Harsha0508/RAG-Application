"""
Ingestion Pipeline — Orchestrates the full PDF → Vector Store flow.

Steps:
1. PDF → page images (PyMuPDF)
2. For each page: extract embedded text or run OCR
3. Chunk all page text
4. Embed all chunks (batch)
5. Upsert to Qdrant vector store
6. Move source PDF to processed/ (or failed/ on error)
"""

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a single PDF document."""
    file_name: str
    success: bool
    pages_processed: int
    chunks_created: int
    chunks_added: int  # After deduplication
    processing_time_seconds: float
    error: str | None = None


def ingest_pdf(
    file_path: str | Path,
    vector_store=None,
    embedder=None,
) -> IngestionResult:
    """
    Ingest a single PDF file through the full pipeline.

    Args:
        file_path: Path to the PDF file.
        vector_store: VectorStore instance (created if None).
        embedder: Embedder instance (created if None).

    Returns:
        IngestionResult with processing stats.
    """
    file_path = Path(file_path)
    file_name = file_path.name
    start_time = time.time()

    logger.info("=== Starting ingestion: %s ===", file_name)

    try:
        # Lazy imports to avoid circular dependencies and slow startup
        from app.ingestion.pdf_processor import render_pdf_pages
        from app.ingestion.ocr_engine import extract_text_from_image
        from app.ingestion.text_chunker import chunk_document
        from app.embeddings.embedder import Embedder
        from app.vector_store.qdrant_store import VectorStore

        # Initialize components if not provided
        if embedder is None:
            embedder = Embedder.get_instance()
        if vector_store is None:
            vector_store = VectorStore()

        # Step 1: Render PDF pages
        logger.info("[1/5] Rendering PDF pages...")
        pages = render_pdf_pages(file_path)
        if not pages:
            raise RuntimeError(f"No pages could be rendered from {file_name}")

        # Step 2: Extract text (embedded or OCR)
        logger.info("[2/5] Extracting text from %d pages...", len(pages))
        page_texts: dict[int, str] = {}

        for page_result in pages:
            page_num = page_result.page_number
            if page_result.has_embedded_text:
                page_texts[page_num] = page_result.embedded_text
                logger.debug("Page %d: used embedded text", page_num)
            else:
                ocr_text = extract_text_from_image(page_result.image)
                if ocr_text:
                    page_texts[page_num] = ocr_text
                    logger.debug("Page %d: OCR extracted %d chars", page_num, len(ocr_text))
                else:
                    logger.warning("Page %d: no text extracted", page_num)

        if not page_texts:
            raise RuntimeError(f"No text could be extracted from {file_name}")

        # Step 3: Chunk all text
        logger.info("[3/5] Chunking text...")
        chunks = chunk_document(page_texts, file_name)
        if not chunks:
            raise RuntimeError(f"No chunks produced from {file_name}")

        # Step 4: Embed all chunks
        logger.info("[4/5] Embedding %d chunks...", len(chunks))
        texts_to_embed = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts_to_embed)

        # Step 5: Add to vector store
        logger.info("[5/5] Adding to vector store...")
        chunks_added = vector_store.add_documents(
            chunk_ids=[c.chunk_id for c in chunks],
            texts=texts_to_embed,
            embeddings=embeddings,
            source_files=[c.source_file for c in chunks],
            page_numbers=[c.page_number for c in chunks],
            chunk_indices=[c.chunk_index for c in chunks],
            chunks=chunks,
        )

        # Move to processed directory
        _move_to_processed(file_path)

        elapsed = time.time() - start_time
        logger.info(
            "=== Ingestion complete: %s | %d pages, %d chunks, %d added | %.1fs ===",
            file_name, len(pages), len(chunks), chunks_added, elapsed,
        )

        return IngestionResult(
            file_name=file_name,
            success=True,
            pages_processed=len(pages),
            chunks_created=len(chunks),
            chunks_added=chunks_added,
            processing_time_seconds=round(elapsed, 2),
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Ingestion failed for '%s': %s", file_name, e, exc_info=True)

        # Move to failed directory
        _move_to_failed(file_path)

        return IngestionResult(
            file_name=file_name,
            success=False,
            pages_processed=0,
            chunks_created=0,
            chunks_added=0,
            processing_time_seconds=round(elapsed, 2),
            error=str(e),
        )


def _move_to_processed(file_path: Path) -> None:
    """Move a successfully processed file to the processed directory."""
    try:
        dest = settings.PROCESSED_DIR / file_path.name
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
        logger.info("Moved '%s' to processed/", file_path.name)
    except Exception as e:
        logger.warning("Could not move '%s' to processed/: %s", file_path.name, e)
        # Try to delete instead
        try:
            file_path.unlink()
            logger.info("Deleted '%s' after processing", file_path.name)
        except Exception:
            pass


def _move_to_failed(file_path: Path) -> None:
    """Move a failed file to the failed directory."""
    try:
        if file_path.exists():
            dest = settings.FAILED_DIR / file_path.name
            settings.FAILED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest))
            logger.info("Moved '%s' to failed/", file_path.name)
    except Exception as e:
        logger.warning("Could not move '%s' to failed/: %s", file_path.name, e)

