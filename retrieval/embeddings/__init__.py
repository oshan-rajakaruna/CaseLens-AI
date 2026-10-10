"""Embedding interfaces, provider selection, and checkpoint support."""

from retrieval.embeddings.base import EmbeddingService
from retrieval.embeddings.checkpoint import (
    EmbeddingCheckpointError,
    EmbeddingCheckpointStore,
)
from retrieval.embeddings.factory import (
    EmbeddingProviderConfigurationError,
    configured_embedding_provider,
    create_embedding_service,
)
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
from retrieval.embeddings.openai import (
    DEFAULT_OPENAI_EMBEDDING_DIMENSION,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    MissingOpenAIAPIKeyError,
    OpenAIEmbeddingConfigurationError,
    OpenAIEmbeddingError,
    OpenAIEmbeddingResponseError,
    OpenAIEmbeddingService,
    OpenAIEmbeddingSettings,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingConfigurationError",
    "EmbeddingCheckpointError",
    "EmbeddingCheckpointStore",
    "EmbeddingProviderConfigurationError",
    "EmbeddingResponseError",
    "EmbeddingService",
    "GeminiEmbeddingError",
    "GeminiEmbeddingService",
    "GeminiEmbeddingSettings",
    "MissingGeminiAPIKeyError",
    "MissingOpenAIAPIKeyError",
    "OpenAIEmbeddingConfigurationError",
    "OpenAIEmbeddingError",
    "OpenAIEmbeddingResponseError",
    "OpenAIEmbeddingService",
    "OpenAIEmbeddingSettings",
    "DEFAULT_OPENAI_EMBEDDING_DIMENSION",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "configured_embedding_provider",
    "create_embedding_service",
    "format_document_for_embedding",
    "format_query_for_embedding",
]
