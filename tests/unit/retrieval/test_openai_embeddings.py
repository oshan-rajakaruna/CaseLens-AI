"""Offline tests for the OpenAI embedding provider."""

from types import SimpleNamespace

import pytest

from retrieval.embeddings.openai import (
    DEFAULT_OPENAI_EMBEDDING_DIMENSION,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    MissingOpenAIAPIKeyError,
    OpenAIEmbeddingError,
    OpenAIEmbeddingResponseError,
    OpenAIEmbeddingService,
    OpenAIEmbeddingSettings,
)


class _FakeEmbeddings:
    def __init__(
        self,
        data: list[SimpleNamespace] | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        self.data = data or []
        self.errors = list(errors or [])
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(data=self.data)


class _FakeClient:
    def __init__(self, embeddings: _FakeEmbeddings) -> None:
        self.embeddings = embeddings


class _TransientOpenAIError(RuntimeError):
    status_code = 429


def _item(index: int, values: list[object]) -> SimpleNamespace:
    return SimpleNamespace(index=index, embedding=values)


def test_openai_settings_use_documented_small_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSION", raising=False)

    settings = OpenAIEmbeddingSettings.from_environment()

    assert settings.api_key == "test-key"
    assert settings.model == DEFAULT_OPENAI_EMBEDDING_MODEL
    assert settings.dimension == DEFAULT_OPENAI_EMBEDDING_DIMENSION


def test_openai_service_initializes_lazily_without_a_request() -> None:
    service = OpenAIEmbeddingService(OpenAIEmbeddingSettings(api_key=""))

    assert service.provider == "openai"
    assert service.model == "text-embedding-3-small"
    assert service.dimension == 1536
    assert service.request_count == 0


def test_openai_request_requires_api_key_before_client_creation() -> None:
    service = OpenAIEmbeddingService(OpenAIEmbeddingSettings(api_key=""))

    with pytest.raises(MissingOpenAIAPIKeyError, match="OPENAI_API_KEY is required"):
        service.embed_query("state responsibility")


def test_openai_batch_preserves_response_index_order() -> None:
    embeddings = _FakeEmbeddings(
        data=[_item(1, [4, 5, 6]), _item(0, [1, 2, 3])]
    )
    service = OpenAIEmbeddingService(
        OpenAIEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(embeddings),
        batch_size=2,
    )

    vectors = service.embed_documents([("First", "One"), ("Second", "Two")])

    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert embeddings.calls[0]["model"] == "text-embedding-3-small"
    assert embeddings.calls[0]["dimensions"] == 3
    assert embeddings.calls[0]["input"] == [
        "title: One | text: First",
        "title: Two | text: Second",
    ]
    assert service.successful_embedding_count == 2


@pytest.mark.parametrize(
    "values",
    [[1.0, 2.0], [1.0, "invalid", 3.0], [1.0, float("inf"), 3.0]],
)
def test_openai_validates_dimensions_and_values(values: list[object]) -> None:
    service = OpenAIEmbeddingService(
        OpenAIEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(_FakeEmbeddings(data=[_item(0, values)])),
    )

    with pytest.raises(OpenAIEmbeddingResponseError):
        service.embed_query("trade subsidies")


def test_openai_rejects_duplicate_or_missing_response_indexes() -> None:
    service = OpenAIEmbeddingService(
        OpenAIEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(
            _FakeEmbeddings(data=[_item(0, [1, 2, 3]), _item(0, [4, 5, 6])])
        ),
        batch_size=2,
    )

    with pytest.raises(OpenAIEmbeddingResponseError, match="ordering"):
        service.embed_documents([("First", None), ("Second", None)])


def test_openai_retries_transient_failures_with_a_bound() -> None:
    embeddings = _FakeEmbeddings(
        data=[_item(0, [1, 2, 3])],
        errors=[_TransientOpenAIError("rate limited")],
    )
    service = OpenAIEmbeddingService(
        OpenAIEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(embeddings),
        max_retries=1,
        retry_base_seconds=0,
    )

    assert service.embed_query("privacy") == [1.0, 2.0, 3.0]
    assert service.request_count == 2
    assert service.retry_count == 1


def test_openai_converts_provider_errors_without_exposing_input() -> None:
    confidential_text = "confidential-evidence-123"
    service = OpenAIEmbeddingService(
        OpenAIEmbeddingSettings(api_key="", dimension=3),
        client=_FakeClient(
            _FakeEmbeddings(errors=[RuntimeError("provider details")])
        ),
    )

    with pytest.raises(OpenAIEmbeddingError) as exc_info:
        service.embed_document(confidential_text)

    assert str(exc_info.value) == "OpenAI embedding request failed"
    assert confidential_text not in str(exc_info.value)
