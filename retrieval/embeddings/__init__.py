"""Embedding interfaces and the Gemini retrieval embedding service."""

from retrieval.embeddings.base import EmbeddingService
from retrieval.embeddings.gemini import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfigurationError,
    EmbeddingResponseError,
    GeminiEmbeddingError,
    GeminiEmbeddingService,
    GeminiEmbeddingSettings,
    MissingGeminiAPIKeyError,
    format_document_for_embedding,
    format_query_for_embedding,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingConfigurationError",
    "EmbeddingResponseError",
    "EmbeddingService",
    "GeminiEmbeddingError",
    "GeminiEmbeddingService",
    "GeminiEmbeddingSettings",
    "MissingGeminiAPIKeyError",
    "format_document_for_embedding",
    "format_query_for_embedding",
]
