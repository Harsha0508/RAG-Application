"""
Retriever â€” Handles the query â†’ search â†’ context pipeline.

Connects the embedder and vector store to retrieve relevant
document chunks for a given query.
"""

import logging
from typing import Any

from app.embeddings.embedder import Embedder
from app.vector_store.qdrant_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks for a query.

    Usage:
        retriever = Retriever(vector_store, embedder)
        results = retriever.retrieve("What is machine learning?", top_k=5)
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self._store = vector_store
        self._embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.
        """
        logger.info("Retrieving top-%d chunks for query: '%s'", top_k, query[:80])

        # Embed the query
        query_embedding = self._embedder.embed_query(query)

        # Search the vector store
        results = self._store.search(query_embedding, top_k=top_k)

        logger.info("Retrieved %d chunks", len(results))
        return results
