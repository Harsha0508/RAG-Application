"""
Embedder for local, model-free text vectors.

This wrapper uses scikit-learn's HashingVectorizer so the app does not
need to download any Hugging Face models.
"""

import logging
from typing import ClassVar

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton wrapper for model-free text embeddings.

    The vectors are lexical rather than neural, but they are fully local,
    deterministic, and require no training or downloads.
    """

    _instance: ClassVar["Embedder | None"] = None
    _vectorizer: HashingVectorizer
    _dimension: int

    def __init__(self) -> None:
        """Initialize the hashing vectorizer. Use get_instance() instead."""
        self._dimension = settings.EMBEDDING_DIM
        logger.info(
            "Initializing local HashingVectorizer embeddings (dimension=%d)...",
            self._dimension,
        )
        self._vectorizer = HashingVectorizer(
            n_features=self._dimension,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        logger.info("HashingVectorizer initialized successfully.")

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Get or create the singleton Embedder instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _encode(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Encode text with the local vectorizer and return plain Python lists."""
        matrix = self._vectorizer.transform([texts] if isinstance(texts, str) else texts)
        dense = matrix.astype(np.float32).toarray()
        if isinstance(texts, str):
            return dense[0].tolist()
        return dense.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts into vectors.
        """
        if not texts:
            return []
        return self._encode(texts)  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.
        """
        return self._encode(text)  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
