"""
SQLite chunk store for persisting extracted text chunks.

Keeps the chunk text and metadata in a local SQLite database while the
vector embeddings continue to live in Qdrant.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class SQLiteChunkStore:
    """Persist chunk text and metadata in a local SQLite database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path or settings.BASE_DIR / "data" / "chunks.sqlite")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized:
            return

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks(source_file)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_page ON chunks(source_file, page_number)"
            )

        self._initialized = True
        logger.info("Connected to SQLite chunk store at '%s'", self._db_path)

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Insert or replace chunk rows."""
        if not chunks:
            return 0

        self._ensure_schema()
        rows = [
            (
                chunk["chunk_id"],
                chunk["source_file"],
                chunk["page_number"],
                chunk["chunk_index"],
                chunk["char_start"],
                chunk["char_end"],
                chunk["text"],
                json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
            )
            for chunk in chunks
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, source_file, page_number, chunk_index,
                    char_start, char_end, text, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source_file=excluded.source_file,
                    page_number=excluded.page_number,
                    chunk_index=excluded.chunk_index,
                    char_start=excluded.char_start,
                    char_end=excluded.char_end,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json
                """,
                rows,
            )

        logger.info("Stored %d chunks in SQLite", len(rows))
        return len(rows)

    def get_document_list(self) -> list[str]:
        """Return unique source file names stored in SQLite."""
        if not self._db_path.exists():
            return []

        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source_file FROM chunks ORDER BY source_file"
            ).fetchall()
        return [str(row["source_file"]) for row in rows]

    def delete_document(self, source_file: str) -> int:
        """Delete all chunks for a source file."""
        if not self._db_path.exists():
            return 0

        self._ensure_schema()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM chunks WHERE source_file = ?",
                (source_file,),
            )
            deleted = cur.rowcount or 0
        logger.info("Deleted %d SQLite chunks for '%s'", deleted, source_file)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Return chunk and document counts from SQLite."""
        if not self._db_path.exists():
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "documents": [],
                "table_exists": False,
            }

        self._ensure_schema()
        with self._connect() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
            documents = self.get_document_list()
        return {
            "total_chunks": int(total_chunks),
            "total_documents": len(documents),
            "documents": documents,
            "table_exists": True,
        }
