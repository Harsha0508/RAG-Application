"""
LLM Engine — Generates answers using Groq API with retrieved context via LangChain.

Uses ChatGroq from langchain_groq. Falls back to a structured context-only
response if the API key is not configured.
"""

import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings

logger = logging.getLogger(__name__)

# Module-level ChatGroq instance (lazy-initialized)
_chat_model: ChatGroq | None = None


def _get_chat_model() -> ChatGroq | None:
    """Get or initialize the ChatGroq model. Returns None if no API key."""
    global _chat_model

    if _chat_model is not None:
        return _chat_model

    api_key = settings.GROQ_API_KEY
    if not api_key:
        logger.warning(
            "GROQ_API_KEY not set. LLM generation disabled. "
            "Set GROQ_API_KEY environment variable in your .env file."
        )
        return None

    # Initialize ChatGroq using the config settings
    _chat_model = ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )
    logger.info("ChatGroq model initialized (model: %s)", settings.GROQ_MODEL)
    return _chat_model


def is_llm_available() -> bool:
    """Check if the LLM is configured and available."""
    return bool(settings.GROQ_API_KEY)


def _build_context_prompt(query: str, context_chunks: list[dict[str, Any]]) -> str:
    """
    Build the system + user prompt with retrieved context.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks from the vector store.

    Returns:
        Formatted prompt string for the LLM.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{text}"
        )

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text


_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Answer the question using ONLY the information from the provided context.
- If the context doesn't contain enough information to answer, say so clearly.
- Cite the source document and page number when referencing specific information.
- Be concise and accurate.
- If the question is not related to the context, politely redirect the user."""


def generate_answer(
    query: str,
    context_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate an answer using ChatGroq with retrieved context.

    If the Groq API key is not configured, returns a structured
    context-only response instead.

    Args:
        query: The user's question.
        context_chunks: Retrieved chunks from the vector store.

    Returns:
        Dict with keys:
        - answer: The generated answer text.
        - sources: List of source references.
        - llm_used: Whether an LLM was used (vs context-only).
        - model: The model name used (if LLM was used).
    """
    # Build source references
    sources = []
    seen = set()
    for chunk in context_chunks:
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number", 0)
        key = f"{source}:p{page}"
        if key not in seen:
            sources.append({"file": source, "page": page})
            seen.add(key)

    # If no context, return early
    if not context_chunks:
        return {
            "answer": "I don't have any relevant documents to answer your question. "
                      "Please upload some PDF documents first.",
            "sources": [],
            "llm_used": False,
            "model": None,
        }

    context_text = _build_context_prompt(query, context_chunks)

    # Try LLM generation via langchain-groq
    model = _get_chat_model()
    if model is not None:
        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}"),
            ]

            response = model.invoke(messages)
            answer = response.content
            logger.info("ChatGroq generated answer (%d chars)", len(answer))

            return {
                "answer": answer,
                "sources": sources,
                "llm_used": True,
                "model": settings.GROQ_MODEL,
            }

        except Exception as e:
            logger.error("Langchain Groq API call failed: %s", e)
            # Fall through to context-only mode

    # Fallback: context-only response
    logger.info("Using context-only response (no LLM)")
    fallback_parts = ["**Relevant excerpts from your documents:**\n"]
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_file", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        fallback_parts.append(
            f"**[{i}] {source} (Page {page}):**\n> {text}\n"
        )

    return {
        "answer": "\n".join(fallback_parts),
        "sources": sources,
        "llm_used": False,
        "model": None,
    }
