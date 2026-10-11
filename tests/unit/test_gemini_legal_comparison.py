"""Offline tests for the Gemini legal-comparison provider."""

from types import SimpleNamespace

import pytest

from backend.schemas.legal_comparison import (
    ComparisonAuthorityReference,
    ComparisonEvidenceMetadata,
    LegalComparisonGenerationRequest,
)
from backend.services.gemini_legal_comparison import (
    GeminiLegalComparisonAuthenticationError,
    GeminiLegalComparisonEmptyResponseError,
    GeminiLegalComparisonError,
    GeminiLegalComparisonInvalidJSONError,
    GeminiLegalComparisonInvalidRequestError,
    GeminiLegalComparisonModelError,
    GeminiLegalComparisonProvider,
    GeminiLegalComparisonRateLimitError,
    GeminiLegalComparisonResponseError,
    GeminiLegalComparisonSchemaValidationError,
    GeminiLegalComparisonSettings,
    GeminiLegalComparisonTimeoutError,
    GeminiLegalComparisonTransportError,
    GeminiLegalComparisonTruncatedResponseError,
)


class FakeModels:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None) -> None:
        self.models = FakeModels(response, error)


class FakeAPIError(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__("PRIVATE PROVIDER RESPONSE")
        self.code = code


def _request() -> LegalComparisonGenerationRequest:
    authority = ComparisonAuthorityReference(
        context_id="CTX-001",
        document_id="DOC-A",
        chunk_ids=["chunk-1"],
        pages=[3],
    )
    return LegalComparisonGenerationRequest(
        comparison_id="CMP-001",
        context_id="CTX-001",
        fact_statement="The tenant allegedly did not pay.",
        fact_epistemic_status="alleged",
        fact_confidence=0.8,
        fact_verification_status="not_independently_verified",
        evidence=ComparisonEvidenceMetadata(document_id="PRIVATE-DOC-1", page=2),
        authority_passage="The selected authority discusses payment obligations.",
        authority=authority,
        limitations=["Applicability was not assessed."],
    )


def _response():
    return SimpleNamespace(
        parsed={
            "comparison_id": "CMP-001",
            "context_id": "CTX-001",
            "explanation": "The statements may relate, subject to review.",
            "uncertainties": ["Applicability cannot be determined."],
            "ai_generated": True,
            "human_review_required": True,
        },
        text=None,
    )


def test_structured_request_uses_configured_model_schema_and_prompt_rules() -> None:
    client = FakeClient(_response())
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(
            api_key="offline-key",
            model="test-model",
            max_output_tokens=700,
            request_timeout_ms=12_000,
        ),
        client=client,
    )

    output = provider.generate(_request())

    call = client.models.calls[0]
    assert output.context_id == "CTX-001"
    assert call["model"] == "test-model"
    assert call["config"].max_output_tokens == 700
    assert call["config"].response_json_schema is not None
    assert "untrusted content" in call["contents"]
    assert "PRIVATE-DOC-1" in call["contents"]
    assert "Do not select or change references" in call["contents"]


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(parsed=None, text=""),
        SimpleNamespace(parsed=None, text="not-json"),
        SimpleNamespace(parsed={"comparison_id": "CMP-001"}, text=None),
    ],
)
def test_invalid_json_or_schema_is_rejected(response) -> None:
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(response),
    )

    with pytest.raises(GeminiLegalComparisonResponseError):
        provider.generate(_request())


def test_empty_response_has_safe_specific_category() -> None:
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(SimpleNamespace(parsed=None, text="")),
    )

    with pytest.raises(GeminiLegalComparisonEmptyResponseError) as caught:
        provider.generate(_request())

    assert caught.value.category == "provider_empty_response"


def test_invalid_json_and_schema_have_distinct_categories() -> None:
    invalid_json = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(SimpleNamespace(parsed=None, text="not-json")),
    )
    invalid_schema = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(
            SimpleNamespace(parsed={"comparison_id": "CMP-001"}, text=None)
        ),
    )

    with pytest.raises(GeminiLegalComparisonInvalidJSONError) as json_error:
        invalid_json.generate(_request())
    with pytest.raises(
        GeminiLegalComparisonSchemaValidationError
    ) as schema_error:
        invalid_schema.generate(_request())

    assert json_error.value.category == "provider_invalid_json"
    assert schema_error.value.category == "provider_schema_validation"


@pytest.mark.parametrize(
    ("error", "expected", "category"),
    [
        (
            FakeAPIError(400),
            GeminiLegalComparisonInvalidRequestError,
            "provider_invalid_request",
        ),
        (
            FakeAPIError(401),
            GeminiLegalComparisonAuthenticationError,
            "provider_authentication_error",
        ),
        (
            FakeAPIError(403),
            GeminiLegalComparisonAuthenticationError,
            "provider_authentication_error",
        ),
        (FakeAPIError(404), GeminiLegalComparisonModelError, "provider_model_error"),
        (FakeAPIError(429), GeminiLegalComparisonRateLimitError, "provider_rate_limit"),
        (
            FakeAPIError(500),
            GeminiLegalComparisonTransportError,
            "provider_transport_error",
        ),
        (
            TimeoutError("PRIVATE"),
            GeminiLegalComparisonTimeoutError,
            "provider_timeout",
        ),
    ],
)
def test_sdk_failures_have_safe_categories(error, expected, category) -> None:
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(error=error),
    )

    with pytest.raises(expected) as caught:
        provider.generate(_request())

    assert caught.value.category == category
    assert getattr(caught.value, "status_code", None) == getattr(error, "code", None)
    assert "PRIVATE" not in str(caught.value)


def test_max_tokens_finish_reason_is_rejected_as_truncated() -> None:
    response = _response()
    response.candidates = [SimpleNamespace(finish_reason="MAX_TOKENS")]
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(response),
    )

    with pytest.raises(GeminiLegalComparisonTruncatedResponseError) as caught:
        provider.generate(_request())

    assert caught.value.category == "provider_truncated_response"
    assert caught.value.finish_reason == "MAX_TOKENS"


def test_provider_failure_message_does_not_echo_sensitive_exception() -> None:
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(api_key="offline-key"),
        client=FakeClient(error=RuntimeError("SECRET PRIVATE EVIDENCE")),
    )

    with pytest.raises(GeminiLegalComparisonError) as caught:
        provider.generate(_request())

    assert "SECRET PRIVATE EVIDENCE" not in str(caught.value)


def test_client_timeout_and_single_attempt_are_configured(monkeypatch) -> None:
    captured = {}
    client = FakeClient(_response())

    def fake_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(
        "backend.services.gemini_legal_comparison.genai.Client",
        fake_client,
    )
    provider = GeminiLegalComparisonProvider(
        GeminiLegalComparisonSettings(
            api_key="offline-key",
            request_timeout_ms=12_345,
        )
    )

    provider.generate(_request())

    assert captured["http_options"].timeout == 12_345
    assert captured["http_options"].retry_options.attempts == 1
    assert len(client.models.calls) == 1
