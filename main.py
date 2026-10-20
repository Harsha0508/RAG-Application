"""
RAG OCR System â€” FastAPI Application Entry Point.

Initializes all components on startup and tears them down on shutdown
using FastAPI's lifespan context manager.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_dependencies
from app.core.constants import APP_DESCRIPTION, APP_NAME, APP_VERSION
from config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("=" * 60)
    logger.info("  %s v%s Starting up", APP_NAME, APP_VERSION)
    logger.info("=" * 60)

    # Ensure all directories exist
    settings.ensure_dirs()

    # Initialize components
    from app.embeddings.embedder import Embedder
    from app.vector_store.qdrant_store import VectorStore
    from app.retrieval.retriever import Retriever
    from app.ingestion.file_watcher import FileWatcher

    logger.info("Initializing embedding model...")
    embedder = Embedder.get_instance()

    logger.info("Initializing vector store...")
    vector_store = VectorStore()

    logger.info("Initializing retriever...")
    retriever = Retriever(vector_store, embedder)

    logger.info("Starting file watcher...")
    file_watcher = FileWatcher(vector_store, embedder)
    file_watcher.start()

    # Inject dependencies into routes
    set_dependencies(vector_store, embedder, retriever, file_watcher)

    # Log LLM status
    from app.llm.llm_engine import is_llm_available
    if is_llm_available():
        logger.info("LLM: Groq API enabled (model: %s)", settings.GROQ_MODEL)
    else:
        logger.warning(
            "LLM: Groq API NOT configured. Set RAG_GROQ_API_KEY in .env "
            "to enable LLM-powered answers."
        )

    stats = vector_store.get_stats()
    logger.info(
        "Vector store: %d documents, %d chunks indexed",
        stats.get("total_documents", 0),
        stats.get("total_chunks", 0),
    )

    logger.info("=" * 60)
    logger.info("  System ready! API at http://%s:%d", settings.API_HOST, settings.API_PORT)
    logger.info("  Drop PDFs into: %s", settings.INGESTION_DIR)
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")
    file_watcher.stop()
    vector_store.close()
    logger.info("Shutdown complete")


# â”€â”€ FastAPI App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )
