"""
Central configuration for the RAG OCR system.
Uses pydantic-settings for environment variable support and validation.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    """Application settings with sensible defaults."""

    # === Paths ===
    BASE_DIR: Path = Path(__file__).resolve().parent
    INGESTION_DIR: Path = Path(__file__).resolve().parent / "ingestion"
    PROCESSED_DIR: Path = Path(__file__).resolve().parent / "ingestion" / "processed"
    FAILED_DIR: Path = Path(__file__).resolve().parent / "ingestion" / "failed"
    QDRANT_DIR: Path = Path(__file__).resolve().parent / "data" / "qdrant"

    # === Embedding Model ===
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sklearn.feature_extraction.text.HashingVectorizer")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 4096))

    # === Chunking ===
    CHUNK_SIZE: int = os.getenv("CHUNK_SIZE")
    CHUNK_OVERLAP: int = os.getenv("CHUNK_OVERLAP")

    # === Retrieval ===
    TOP_K_RETRIEVAL: int = 5

    # === OCR ===
    OCR_DPI: int = 400
    OCR_LANG: str = "en"

    # === Qdrant ===
    QDRANT_COLLECTION_NAME: str = "rag_documents"

    # === LLM (Groq API) ===
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL")

    # === FastAPI ===
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # === File Watcher ===
    WATCHER_DEBOUNCE_SECONDS: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_path in [
            self.INGESTION_DIR,
            self.PROCESSED_DIR,
            self.FAILED_DIR,
            self.QDRANT_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
