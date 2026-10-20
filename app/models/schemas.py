"""
Pydantic Schemas — Request/response models for the FastAPI API.
"""

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class SourceReference(BaseModel):
    """A reference to a source document page."""
    file: str
    page: int


class ChatResponse(BaseModel):
    """Response from the /api/chat endpoint."""
    answer: str
    sources: list[SourceReference]
    llm_used: bool
    model: str | None = None


# ── Ingestion ────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response from the /api/ingest endpoint."""
    file_name: str
    success: bool
    pages_processed: int
    chunks_created: int
    chunks_added: int
    processing_time_seconds: float
    error: str | None = None


# ── Documents ────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    name: str


class DocumentListResponse(BaseModel):
    """Response from the /api/documents endpoint."""
    documents: list[DocumentInfo]
    total: int


class DeleteDocumentResponse(BaseModel):
    """Response from DELETE /api/documents/{doc_name}."""
    document: str
    chunks_deleted: int
    success: bool


# ── Status ───────────────────────────────────────────────────────────

class SystemStatus(BaseModel):
    """Response from the /api/status endpoint."""
    total_chunks: int
    total_documents: int
    documents: list[str]
    embedding_model: str
    llm_available: bool
    llm_model: str | None = None
    file_watcher_running: bool


class HealthResponse(BaseModel):
    """Response from the /api/health endpoint."""
    status: str = "ok"
    version: str
