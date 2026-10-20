"""
API Routes — FastAPI endpoints for the RAG OCR system.

Endpoints:
- POST /api/chat       — Query documents and get an answer
- POST /api/ingest     — Upload and ingest a PDF
- GET  /api/status     — System status
- GET  /api/health     — Health check
- GET  /api/documents  — List indexed documents
- DELETE /api/documents/{doc_name} — Remove a document from the index
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.constants import APP_VERSION, MAX_UPLOAD_SIZE_BYTES, SUPPORTED_EXTENSIONS
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    SourceReference,
    SystemStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG OCR API"])

# These are set during app startup (see main.py)
_vector_store = None
_embedder = None
_retriever = None
_file_watcher = None


def set_dependencies(vector_store, embedder, retriever, file_watcher):
    """Inject runtime dependencies (called from main.py lifespan)."""
    global _vector_store, _embedder, _retriever, _file_watcher
    _vector_store = vector_store
    _embedder = embedder
    _retriever = retriever
    _file_watcher = file_watcher


# ── Chat ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Query the document knowledge base and get an answer.

    Retrieves relevant document chunks via similarity search,
    then generates an answer using the Groq LLM (or returns
    context-only if the LLM is not configured).
    """
    if _retriever is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Retrieve relevant chunks
    chunks = _retriever.retrieve(request.query, top_k=request.top_k)

    # Generate answer
    from app.llm.llm_engine import generate_answer
    result = generate_answer(request.query, chunks)

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        llm_used=result["llm_used"],
        model=result.get("model"),
    )


# ── Ingestion ────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document.

    The PDF is processed through OCR, chunked, embedded, and indexed.
    """
    if _vector_store is None or _embedder is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    # Read and validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB.",
        )

    # Save to temp file and process
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=f"ingest_{file.filename}_",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Rename to preserve original filename
        final_path = tmp_path.parent / file.filename
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)

        from app.ingestion.pipeline import ingest_pdf
        result = ingest_pdf(final_path, _vector_store, _embedder)

        return IngestResponse(
            file_name=result.file_name,
            success=result.success,
            pages_processed=result.pages_processed,
            chunks_created=result.chunks_created,
            chunks_added=result.chunks_added,
            processing_time_seconds=result.processing_time_seconds,
            error=result.error,
        )

    except Exception as e:
        logger.error("Ingestion endpoint error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Status ───────────────────────────────────────────────────────────

@router.get("/status", response_model=SystemStatus)
async def status():
    """Get system status including index stats and configuration."""
    from app.llm.llm_engine import is_llm_available
    from config import settings

    stats = _vector_store.get_stats() if _vector_store else {
        "total_chunks": 0, "total_documents": 0, "documents": []
    }

    return SystemStatus(
        total_chunks=stats.get("total_chunks", 0),
        total_documents=stats.get("total_documents", 0),
        documents=stats.get("documents", []),
        embedding_model=settings.EMBEDDING_MODEL,
        llm_available=is_llm_available(),
        llm_model=settings.GROQ_MODEL if is_llm_available() else None,
        file_watcher_running=_file_watcher.is_running if _file_watcher else False,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version=APP_VERSION)


# ── Documents ────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    docs = _vector_store.get_document_list()
    return DocumentListResponse(
        documents=[DocumentInfo(name=d) for d in docs],
        total=len(docs),
    )


@router.delete("/documents/{doc_name}", response_model=DeleteDocumentResponse)
async def delete_document(doc_name: str):
    """Remove a document and all its chunks from the index."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    deleted = _vector_store.delete_document(doc_name)
    return DeleteDocumentResponse(
        document=doc_name,
        chunks_deleted=deleted,
        success=deleted > 0,
    )
