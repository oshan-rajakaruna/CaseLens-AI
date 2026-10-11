"""Offline tests for the synchronous Gemini summarization provider."""

from types import SimpleNamespace
from typing import Any

import pytest

from agents.summarization import (
    GeminiSummarizationAuthenticationError,
    GeminiSummarizationConfigurationError,
    GeminiSummarizationModelError,
    GeminiSummarizationProvider,
    GeminiSummarizationRateLimitError,
    GeminiSummarizationResponseError,
    GeminiSummarizationSettings,
    GeminiSummarizationTimeoutError,
    SummarizationAgent,
    SummarizationProvider,
    SummarizationValidationError,
    build_summarization_provider_request,
)
from tests.unit.summarization.test_agent import _payload


class FakeModels:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.models = FakeModels(response, error)


class FakeAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _provider(response: Any = None, error: Exception | None = None):
    client = FakeClient(response, error)
    provider = GeminiSummarizationProvider(
        GeminiSummarizationSettings(
            api_key="offline-test-key",
            model="test-gemini-model",
            max_output_tokens=512,
        ),
        client=client,
    )
    return provider, client


def _valid_response() -> SimpleNamespace:
    return SimpleNamespace(
        parsed={
            "answer": "The supplied authority supports the rule [CTX-001].",
            "citations": ["CTX-001"],
            "confidence": 0.8,
            "limitations": ["Limited to the retrieved evidence"],
        },
        text=None,
    )


def test_provider_is_compatible_and_parses_structured_response() -> None:
    provider, client = _provider(_valid_response())
    typed_provider: SummarizationProvider = provider

    output = typed_provider.generate(build_summarization_provider_request(_payload()))

    assert output.answer.endswith("[CTX-001].")
    assert output.citations == ["CTX-001"]
    assert client.models.calls[0]["model"] == "test-gemini-model"
    assert "RETRIEVED CONTEXT" in client.models.calls[0]["contents"]
    assert client.models.calls[0]["config"].response_mime_type == "application/json"
    assert client.models.calls[0]["config"].response_json_schema is not None


def test_provider_integrates_with_agent_and_preserves_ctx_validation() -> None:
    provider, _ = _provider(_valid_response())

    result = SummarizationAgent(provider).summarize(_payload())

    assert result.used_context_ids == ["CTX-001"]
    assert result.citation_validation.all_citations_valid is True
    assert result.provider_metadata.provider == "gemini"
    assert result.provider_metadata.model == "test-gemini-model"


@pytest.mark.parametrize(
    "answer,citations",
    [
        ("Unknown authority [CTX-999].", ["CTX-999"]),
        ("Malformed authority [CTX-1].", []),
    ],
)
def test_existing_agent_rejects_invalid_ctx_citations(answer, citations) -> None:
    provider, _ = _provider(
        SimpleNamespace(
            parsed={
                "answer": answer,
                "citations": citations,
                "confidence": 0.5,
                "limitations": [],
            }
        )
    )

    with pytest.raises(SummarizationValidationError):
        SummarizationAgent(provider).summarize(_payload())


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(parsed=None, text=""),
        SimpleNamespace(parsed=None, text="not-json"),
        SimpleNamespace(parsed={"citations": ["CTX-001"]}, text=None),
    ],
)
def test_invalid_or_missing_output_fields_are_rejected(response) -> None:
    provider, _ = _provider(response)

    with pytest.raises(GeminiSummarizationResponseError):
        provider.generate(build_summarization_provider_request(_payload()))


def test_missing_api_configuration_is_rejected(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(GeminiSummarizationConfigurationError, match="GEMINI_API_KEY"):
        GeminiSummarizationProvider()


@pytest.mark.parametrize(
    "error,expected",
    [
        (FakeAPIError("credential rejected", 401), GeminiSummarizationAuthenticationError),
        (FakeAPIError("requested model not found", 404), GeminiSummarizationModelError),
        (FakeAPIError("quota exhausted", 429), GeminiSummarizationRateLimitError),
        (
            FakeAPIError("upstream deadline", code=504),
            GeminiSummarizationTimeoutError,
        ),
        (TimeoutError("secret prompt timed out"), GeminiSummarizationTimeoutError),
    ],
)
def test_provider_failures_are_classified_without_sensitive_payloads(error, expected) -> None:
    provider, _ = _provider(error=error)

    with pytest.raises(expected) as caught:
        provider.generate(build_summarization_provider_request(_payload()))

    assert "secret prompt" not in str(caught.value)


def test_sdk_client_has_bounded_timeout_and_one_total_attempt(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    client = FakeClient(_valid_response())

    def fake_client(**kwargs: Any) -> FakeClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("agents.summarization.gemini.genai.Client", fake_client)
    provider = GeminiSummarizationProvider(
        GeminiSummarizationSettings(
            api_key="offline-test-key",
            request_timeout_ms=30_000,
        )
    )

    provider.generate(build_summarization_provider_request(_payload()))

    http_options = captured["http_options"]
    assert http_options.timeout == 30_000
    assert http_options.retry_options.attempts == 1
    assert len(client.models.calls) == 1


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_invalid_request_timeout_configuration_is_rejected(
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_MS", value)

    with pytest.raises(
        GeminiSummarizationConfigurationError,
        match="GEMINI_REQUEST_TIMEOUT_MS",
    ):
        GeminiSummarizationProvider()


def test_insufficient_context_does_not_call_gemini() -> None:
    provider, client = _provider(error=AssertionError("must not call"))

    result = SummarizationAgent(provider).summarize(_payload(empty=True))

    assert result.provider_metadata.provider_called is False
    assert client.models.calls == []
