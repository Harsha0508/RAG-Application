"""
Text Chunker - Splits extracted text into overlapping chunks for embedding.

Uses sentence-boundary-aware splitting to avoid breaking mid-sentence.
Each chunk carries metadata for traceability back to source document/page.
"""

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field

from config import settings

logger = logging.getLogger(__name__)

# Sentence boundary regex - matches common sentence-ending punctuation
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?。！？])\s+')


@dataclass
class TextChunk:
    """A chunk of text with source metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def _generate_chunk_id(source_file: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic, UUID-safe chunk ID."""
    key = f"{source_file}::page{page_number}::chunk{chunk_index}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex-based boundary detection."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source_file: str,
    page_number: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks with sentence-boundary awareness.

    The chunker first splits text into sentences, then groups sentences
    into chunks that fit within `chunk_size` characters, with `chunk_overlap`
    characters of overlap between consecutive chunks.

    Args:
        text: Raw text to chunk.
        source_file: Name of the source PDF file.
        page_number: Page number this text came from.
        chunk_size: Max characters per chunk (default from settings).
        chunk_overlap: Overlap characters between chunks (default from settings).

    Returns:
        List of TextChunk objects.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    text = text.strip()

    # If the entire text fits in one chunk, return it as-is
    if len(text) <= chunk_size:
        chunk_id = _generate_chunk_id(source_file, page_number, 0)
        return [TextChunk(
            chunk_id=chunk_id,
            text=text,
            source_file=source_file,
            page_number=page_number,
            chunk_index=0,
            char_start=0,
            char_end=len(text),
        )]

    # Split into sentences for boundary-aware chunking
    sentences = _split_into_sentences(text)

    if not sentences:
        # Fallback: if no sentence boundaries found, use raw sliding window
        return _sliding_window_chunk(text, source_file, page_number, chunk_size, chunk_overlap)

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_length = 0
    chunk_index = 0
    char_position = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk_size
        if current_length + sentence_len + 1 > chunk_size and current_sentences:
            # Emit the current chunk
            chunk_text_str = " ".join(current_sentences)
            chunk_id = _generate_chunk_id(source_file, page_number, chunk_index)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text_str,
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
                char_start=char_position,
                char_end=char_position + len(chunk_text_str),
            ))
            chunk_index += 1

            # Calculate overlap: keep trailing sentences that fit within overlap
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) + 1 <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break

            char_position += len(chunk_text_str) - overlap_len
            current_sentences = overlap_sentences
            current_length = overlap_len

        current_sentences.append(sentence)
        current_length += sentence_len + 1  # +1 for space

    # Emit the final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunk_id = _generate_chunk_id(source_file, page_number, chunk_index)
        chunks.append(TextChunk(
            chunk_id=chunk_id,
            text=chunk_text_str,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            char_start=char_position,
            char_end=char_position + len(chunk_text_str),
        ))

    logger.debug(
        "Chunked page %d of '%s': %d chunks (size=%d, overlap=%d)",
        page_number, source_file, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _sliding_window_chunk(
    text: str,
    source_file: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Fallback: raw character-based sliding window chunking."""
    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunk_id = _generate_chunk_id(source_file, page_number, chunk_index)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text_str,
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
                char_start=start,
                char_end=end,
            ))
            chunk_index += 1

        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size  # Safety: prevent infinite loop
        start += step

    return chunks


def chunk_document(
    page_texts: dict[int, str],
    source_file: str,
) -> list[TextChunk]:
    """
    Chunk all pages of a document.

    Args:
        page_texts: Dict mapping page_number -> text content.
        source_file: Name of the source PDF file.

    Returns:
        Flat list of all TextChunks across all pages.
    """
    all_chunks: list[TextChunk] = []

    for page_num in sorted(page_texts.keys()):
        text = page_texts[page_num]
        chunks = chunk_text(text, source_file, page_num)
        all_chunks.extend(chunks)

    logger.info(
        "Document '%s': %d total chunks from %d pages",
        source_file, len(all_chunks), len(page_texts),
    )
    return all_chunks
