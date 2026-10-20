"""
Hardcoded constants for the RAG OCR system.
These values are not configurable via environment variables.
"""

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

# Minimum text length to consider a page as having embedded text
MIN_EMBEDDED_TEXT_LENGTH = 50

# Sentence boundary patterns for chunking
SENTENCE_ENDINGS = {".", "!", "?", "。", "！", "？"}

# ChromaDB distance metric
CHROMA_DISTANCE_METRIC = "cosine"

# Maximum file size for upload (100 MB)
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024

# API version
API_VERSION = "v1"

# Application metadata
APP_NAME = "RAG OCR System"
APP_DESCRIPTION = "Retrieval-Augmented Generation system with OCR for PDFs and Image files"
APP_VERSION = "0.1.0"
