"""Tests for Gemini retrieval formatting, configuration, and validation."""

from types import SimpleNamespace

import pytest

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


class _FakeModels:
    def __init__(self, values: list[object] | None = None, error: Exception | None = None) -> None:
        self.values = values
        self.error = error
        self.calls: list[dict[str, object]] = []

    def embed_content(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        embedding = SimpleNamespace(values=self.values)
        return SimpleNamespace(embeddings=[embedding])


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


def test_retrieval_text_formatting_is_exact_and_isolated() -> None:
    assert format_query_for_embedding(" employment termination ") == (
        "task: search result | query: employment termination"
    )
    assert format_document_for_embedding("Original Legal Text", "Case A") == (
        "title: Case A | text: Original Legal Text"
    )
    assert format_document_for_embedding("Original Legal Text") == (
        "title: none | text: Original Legal Text"
    )


@pytest.mark.parametrize("value", ["", "  "])
def test_retrieval_text_formatting_rejects_empty_input(value: str) -> None:
    with pytest.raises(ValueError):
        format_query_for_embedding(value)
    with pytest.raises(ValueError):
        format_document_for_embedding(value)


def test_document_formatting_validates_title_type() -> None:
    with pytest.raises(TypeError, match="title must be a string or None"):
        format_document_for_embedding("Legal text", 42)  # type: ignore[arg-type]


def test_settings_read_environment_with_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EMBEDDING_DIMENSION", raising=False)

    settings = GeminiEmbeddingSettings.from_environment()

    assert settings.api_key == "test-key"
    assert settings.model == DEFAULT_EMBEDDING_MODEL
    assert settings.dimension == DEFAULT_EMBEDDING_DIMENSION


def test_settings_reject_invalid_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSION", "not-an-integer")

    with pytest.raises(EmbeddingConfigurationError, match="must be an integer"):
        GeminiEmbeddingSettings.from_environment()


def test_service_construction_does_not_require_api_key() -> None:
    service = GeminiEmbeddingService(GeminiEmbeddingSettings(api_key=""))

    assert service.model == DEFAULT_EMBEDDING_MODEL
    assert service.dimension == DEFAULT_EMBEDDING_DIMENSION


def test_request_without_api_key_fails_before_client_creation() -> None:
    service = GeminiEmbeddingService(GeminiEmbeddingSettings(api_key=""))

    with pytest.raises(MissingGeminiAPIKeyError, match="GEMINI_API_KEY is required"):
        service.embed_query("employment termination")


def test_service_calls_official_sdk_shape_and_returns_numeric_vector() -> None:
    models = _FakeModels(values=[1, 2.5, 3])
    service = GeminiEmbeddingService(
        GeminiEmbeddingSettings(
            api_key="",
            model="gemini-embedding-2",
            dimension=3,
        ),
        client=_FakeClient(models),
    )

    vector = service.embed_document("Legal text", title="Synthetic Matter")

    assert vector == [1.0, 2.5, 3.0]
    assert models.calls[0]["model"] == "gemini-embedding-2"
    assert models.calls[0]["contents"] == (
        "title: Synthetic Matter | text: Legal text"
    )
    assert models.calls[0]["config"].output_dimensionality == 3  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0],
        [1.0, "invalid", 3.0],
        [1.0, float("inf"), 3.0],
    ],
)
def test_service_validates_embedding_values(values: list[object]) -> None:
    service = GeminiEmbeddingService(
        GeminiEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(_FakeModels(values=values)),
    )

    with pytest.raises(EmbeddingResponseError):
        service.embed_query("employment")


def test_service_converts_api_failures_without_exposing_input() -> None:
    confidential_text = "confidential-evidence-123"
    service = GeminiEmbeddingService(
        GeminiEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(_FakeModels(error=RuntimeError("provider details"))),
    )

    with pytest.raises(GeminiEmbeddingError) as exc_info:
        service.embed_document(confidential_text)

    assert str(exc_info.value) == "Gemini embedding request failed"
    assert confidential_text not in str(exc_info.value)
