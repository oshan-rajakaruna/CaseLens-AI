"""Offline tests for environment-driven embedding-provider selection."""

import pytest

from retrieval.embeddings import (
    GeminiEmbeddingService,
    LocalEmbeddingService,
    OpenAIEmbeddingService,
)
from retrieval.embeddings.factory import (
    EmbeddingProviderConfigurationError,
    create_embedding_service,
)


def test_provider_factory_selects_openai_without_a_live_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSION", raising=False)

    service = create_embedding_service()

    assert isinstance(service, OpenAIEmbeddingService)
    assert service.dimension == 1536
    assert service.request_count == 0


def test_provider_factory_preserves_gemini_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

    assert isinstance(create_embedding_service(), GeminiEmbeddingService)


def test_provider_factory_selects_local_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("LOCAL_EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("LOCAL_EMBEDDING_CACHE_DIR", "data/models/fastembed")

    service = create_embedding_service()

    assert isinstance(service, LocalEmbeddingService)
    assert service.model == "BAAI/bge-small-en-v1.5"
    assert service.dimension == 384
    assert service._embedding_model is None


def test_provider_factory_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "unknown")

    with pytest.raises(EmbeddingProviderConfigurationError):
        create_embedding_service()
