"""Offline tests for the Google Gemini summarization adapter."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agents.summarization import (
    CaseInformation,
    EmptyModelResponseError,
    EvidenceSummary,
    ExtractedFact,
    GeminiAuthenticationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiSummaryGenerator,
    GeminiTimeoutError,
    InvalidModelResponseError,
    ModelUnavailableError,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
    SummarizationResponse,
    UnsupportedSourceReferenceError,
)
from backend.schemas import Citation


SOURCE = SourceReference(document_id="document-1", locator="page 4")
FACT_TEXT = "The supplied evidence states this fact."


def make_request() -> SummarizationRequest:
    return SummarizationRequest(
        case=CaseInformation(case_id="case-1", title="Example case"),
        extracted_facts=[
            ExtractedFact(
                fact_id="fact-1",
                text=FACT_TEXT,
                source_references=[SOURCE],
            )
        ],
    )


def make_response() -> SummarizationResponse:
    return SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Grounded summary.", source_references=[SOURCE])
        ],
        citations=[
            Citation(
                document_id=SOURCE.document_id,
                locator=SOURCE.locator,
                excerpt=FACT_TEXT,
            )
        ],
    )


class FakeModels:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.aio = SimpleNamespace(models=models)


class ProviderFailure(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("sensitive provider details must not escape")


class BlockedResponse:
    parsed = None

    @property
    def text(self) -> str:
        raise ValueError("Response has no content because it was blocked")


def schema_nodes(value: Any) -> list[dict[str, Any]]:
    """Return every object node from a nested JSON Schema."""

    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        nodes.append(value)
        for item in value.values():
            nodes.extend(schema_nodes(item))
    elif isinstance(value, list):
        for item in value:
            nodes.extend(schema_nodes(item))
    return nodes


def test_valid_structured_generation_preserves_context_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_real_client(**kwargs: Any) -> None:
        raise AssertionError("A real Gemini client must not be created")

    monkeypatch.setattr(
        "agents.summarization.gemini.genai.Client", reject_real_client
    )
    models = FakeModels(response=SimpleNamespace(parsed=make_response(), text=None))
    generator = GeminiSummaryGenerator(
        client=FakeClient(models), model="gemini-2.5-flash", max_output_tokens=512
    )

    result = asyncio.run(generator.generate(make_request()))

    assert result.citations[0].document_id == SOURCE.document_id
    assert result.citations[0].locator == SOURCE.locator
    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert '"document_id": "document-1"' in call["contents"]
    assert '"locator": "page 4"' in call["contents"]
    assert "untrusted data" in call["contents"]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is None
    schema = call["config"].response_json_schema
    assert schema is not None
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "$defs" in schema
    nodes = schema_nodes(schema)
    assert any("$ref" in node for node in nodes)
    assert any("anyOf" in node for node in nodes)
    assert any(node.get("enum") == ["draft"] for node in nodes)
    unsupported = {"const", "default", "maxLength", "minLength", "pattern"}
    assert not any(unsupported.intersection(node) for node in nodes)
    assert all(
        len(node) == 1 or all(key.startswith("$") for key in node)
        for node in nodes
        if "$ref" in node
    )
    assert call["config"].max_output_tokens == 512


@pytest.mark.parametrize(
    "text",
    ["not-json", '{"evidence_summaries": [{"text": "missing sources"}]}'],
)
def test_invalid_json_or_schema_is_rejected(text: str) -> None:
    models = FakeModels(response=SimpleNamespace(parsed=None, text=text))
    generator = GeminiSummaryGenerator(client=FakeClient(models))

    with pytest.raises(InvalidModelResponseError):
        asyncio.run(generator.generate(make_request()))


def test_unsupported_source_reference_is_rejected_by_agent() -> None:
    invented = SourceReference(document_id="invented", locator="page 99")
    response = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Unsupported.", source_references=[invented])
        ]
    )
    models = FakeModels(response=SimpleNamespace(parsed=response, text=None))
    generator = GeminiSummaryGenerator(client=FakeClient(models))

    with pytest.raises(UnsupportedSourceReferenceError):
        asyncio.run(SummarizationAgent(generator).summarize(make_request()))


def test_missing_api_key_is_rejected_before_client_creation() -> None:
    with pytest.raises(ModelUnavailableError, match="LLM_API_KEY"):
        GeminiSummaryGenerator(api_key="")


@pytest.mark.parametrize(
    "response",
    [SimpleNamespace(parsed=None, text=""), BlockedResponse()],
)
def test_empty_or_blocked_response_is_rejected(response: Any) -> None:
    models = FakeModels(response=response)
    generator = GeminiSummaryGenerator(client=FakeClient(models))

    with pytest.raises(EmptyModelResponseError):
        asyncio.run(generator.generate(make_request()))


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, GeminiAuthenticationError),
        (400, GeminiProviderError),
        (429, GeminiRateLimitError),
        (500, GeminiProviderError),
    ],
)
def test_provider_failures_are_sanitized(
    status_code: int, expected_error: type[Exception]
) -> None:
    models = FakeModels(error=ProviderFailure(status_code))
    generator = GeminiSummaryGenerator(client=FakeClient(models))

    with pytest.raises(expected_error) as captured:
        asyncio.run(generator.generate(make_request()))

    assert "sensitive provider details" not in str(captured.value)


def test_bad_request_identifies_configuration_rejection() -> None:
    models = FakeModels(error=ProviderFailure(400))
    generator = GeminiSummaryGenerator(client=FakeClient(models))

    with pytest.raises(GeminiProviderError, match="configuration \(HTTP 400\)"):
        asyncio.run(generator.generate(make_request()))


def test_timeout_is_reported_without_retry() -> None:
    models = FakeModels(error=TimeoutError("sensitive timeout details"))
    generator = GeminiSummaryGenerator(client=FakeClient(models))

    with pytest.raises(GeminiTimeoutError) as captured:
        asyncio.run(generator.generate(make_request()))

    assert len(models.calls) == 1
    assert "sensitive timeout details" not in str(captured.value)
