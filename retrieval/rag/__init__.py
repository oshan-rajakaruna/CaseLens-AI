"""Public RAG context assembly contracts."""

from retrieval.rag.context import (
    DEFAULT_MAX_CONTEXT_PASSAGES,
    DEFAULT_MAX_CONTEXT_TOKENS,
    RAGContextAssembler,
    RAGContextSettings,
    estimate_tokens,
    format_context,
    truncate_to_tokens,
)
from retrieval.rag.schemas import (
    ChunkRetrievalScore,
    RAGCitation,
    RAGContext,
    RAGContextPassage,
)

__all__ = [
    "ChunkRetrievalScore",
    "DEFAULT_MAX_CONTEXT_PASSAGES",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "RAGCitation",
    "RAGContext",
    "RAGContextAssembler",
    "RAGContextPassage",
    "RAGContextSettings",
    "estimate_tokens",
    "format_context",
    "truncate_to_tokens",
]
