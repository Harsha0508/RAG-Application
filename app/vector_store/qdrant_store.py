"""
Qdrant Vector Store - local persistent vector database.

Provides incremental indexing, cosine similarity search,
deduplication by chunk ID, and persistence to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.ingestion.text_chunker import TextChunk
from app.storage.sqlite_chunk_store import SQLiteChunkStore
from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around Qdrant for vector storage and retrieval.

    Data is persisted locally to disk. Supports incremental additions
    and similarity search over normalized embeddings.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """
        Initialize the Qdrant vector store.

        Args:
            db_path: Path to the Qdrant storage directory.
                     Defaults to settings.QDRANT_DIR.
        """
        self._db_path = Path(db_path or settings.QDRANT_DIR)
        self._collection_name = settings.QDRANT_COLLECTION_NAME
        self._client: QdrantClient | None = None
        self._dimension = settings.EMBEDDING_DIM
        self._chunk_store = SQLiteChunkStore()

    def _connect(self) -> None:
        """Establish connection to Qdrant."""
        if self._client is None:
            self._db_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._db_path))
            logger.info("Connected to Qdrant at '%s'", self._db_path)

    def _existing_collection_size(self) -> int | None:
        """Return the size of the existing collection, if any."""
        self._connect()
        assert self._client is not None

        if not self._client.collection_exists(self._collection_name):
            return None

        info = self._client.get_collection(self._collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()))
            return int(first.size)
        return int(vectors.size)

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        self._connect()
        assert self._client is not None

        if self._client.collection_exists(self._collection_name):
            existing_size = self._existing_collection_size()
            if existing_size is not None and existing_size != self._dimension:
                logger.warning(
                    "Qdrant collection '%s' has dimension %d but embeddings use %d. "
                    "Recreating the collection.",
                    self._collection_name,
                    existing_size,
                    self._dimension,
                )
                self._client.delete_collection(self._collection_name)
            else:
                return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=rest.VectorParams(
                size=self._dimension,
                distance=rest.Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (dimension=%d)",
            self._collection_name,
            self._dimension,
        )

    def add_documents(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        source_files: list[str],
        page_numbers: list[int],
        chunk_indices: list[int],
        chunks: list[TextChunk] | None = None,
    ) -> int:
        """
        Add document chunks to the vector store.

        Existing chunk IDs are skipped so ingestion remains idempotent.
        """
        if not chunk_ids:
            return 0

        self._ensure_collection()
        assert self._client is not None

        existing_ids: set[str] = set()
        try:
            existing = self._client.retrieve(
                collection_name=self._collection_name,
                ids=chunk_ids,
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {str(point.id) for point in existing}
        except Exception as exc:
            logger.warning("Could not check existing chunk IDs: %s", exc)

        records: list[rest.PointStruct] = []
        sqlite_rows: list[dict[str, Any]] = []
        for i in range(len(chunk_ids)):
            if chunk_ids[i] in existing_ids:
                continue

            records.append(
                rest.PointStruct(
                    id=chunk_ids[i],
                    vector=embeddings[i],
                    payload={
                        "chunk_id": chunk_ids[i],
                        "text": texts[i],
                        "source_file": source_files[i],
                        "page_number": page_numbers[i],
                        "chunk_index": chunk_indices[i],
                    },
                )
            )

            chunk = chunks[i] if chunks and i < len(chunks) else None
            sqlite_rows.append(
                {
                    "chunk_id": chunk_ids[i],
                    "source_file": source_files[i],
                    "page_number": page_numbers[i],
                    "chunk_index": chunk_indices[i],
                    "char_start": chunk.char_start if chunk else 0,
                    "char_end": chunk.char_end if chunk else len(texts[i]),
                    "text": texts[i],
                    "metadata": chunk.metadata if chunk else {},
                }
            )

        if not records:
            logger.info("All %d chunks already exist in store, skipping", len(chunk_ids))
            return 0

        self._client.upsert(
            collection_name=self._collection_name,
            points=records,
        )
        self._chunk_store.upsert_chunks(sqlite_rows)
        logger.info(
            "Added %d new chunks to store (skipped %d duplicates)",
            len(records),
            len(chunk_ids) - len(records),
        )
        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for the most similar chunks to the query embedding.
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL

        if not self._client or not self._client.collection_exists(self._collection_name):
            logger.warning("Search called but no collection exists yet")
            return []

        assert self._client is not None

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            points = getattr(response, "points", response)
            clean_results: list[dict[str, Any]] = []
            for row in points:
                payload = row.payload or {}
                clean_results.append({
                    "chunk_id": payload.get("chunk_id", str(row.id)),
                    "text": payload.get("text", ""),
                    "source_file": payload.get("source_file", ""),
                    "page_number": payload.get("page_number", 0),
                    "chunk_index": payload.get("chunk_index", 0),
                    "_distance": row.score,
                })

            return clean_results

        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def get_document_list(self) -> list[str]:
        """Get a list of all unique source files in the store."""
        try:
            return self._chunk_store.get_document_list()
        except Exception as e:
            logger.error("Failed to get document list: %s", e)
            return []

    def delete_document(self, source_file: str) -> int:
        """
        Delete all chunks belonging to a specific source file.
        """
        sqlite_deleted = 0
        try:
            sqlite_deleted = self._chunk_store.delete_document(source_file)
        except Exception as e:
            logger.error("Failed to delete SQLite chunks for '%s': %s", source_file, e)

        if not self._client or not self._client.collection_exists(self._collection_name):
            return sqlite_deleted

        assert self._client is not None
        try:
            before = self._client.count(
                collection_name=self._collection_name,
                exact=True,
            ).count
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=rest.FilterSelector(
                    filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="source_file",
                                match=rest.MatchValue(value=source_file),
                            )
                        ]
                    )
                ),
            )
            after = self._client.count(
                collection_name=self._collection_name,
                exact=True,
            ).count
            deleted = before - after
            logger.info("Deleted %d chunks for document '%s'", deleted, source_file)
            return deleted
        except Exception as e:
            logger.error("Failed to delete document '%s': %s", source_file, e)
            return sqlite_deleted

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            stats = self._chunk_store.get_stats()
            return {
                "total_chunks": stats.get("total_chunks", 0),
                "total_documents": stats.get("total_documents", 0),
                "documents": stats.get("documents", []),
                "table_exists": stats.get("table_exists", False),
            }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "table_exists": False,
                "error": str(e),
            }

    def close(self) -> None:
        """Clean up resources."""
        self._client = None
        logger.info("VectorStore closed")
