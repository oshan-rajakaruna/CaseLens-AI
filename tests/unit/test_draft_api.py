"""Offline API tests for development-only summary and report endpoints."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agents.summarization import (
    CaseInformation,
    EvidenceSummary,
    ExtractedFact,
    GeminiProviderError,
    ModelUnavailableError,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
    SummarizationResponse,
)
from backend.api.dependencies import get_summarization_agent
from backend.config.settings import Settings, get_settings
from backend.main import app


SOURCE = SourceReference(document_id="document-1", locator="page 3")
FACT_TEXT = "Synthetic evidence supplied for API testing."


def make_settings(
    *, enabled: bool = True, app_env: str = "development"
) -> Settings:
    return Settings(
        app_env=app_env,
        backend_host="127.0.0.1",
        backend_port=8000,
        database_url="",
        llm_api_key="",
        gemini_model="gemini-2.5-flash",
        gemini_max_output_tokens=512,
        secret_key="",
        enable_unauthenticated_dev_api=enabled,
    )


def make_request() -> SummarizationRequest:
    return SummarizationRequest(
        case=CaseInformation(
            case_id="case-1",
            title="Synthetic API matter",
            description="Synthetic data only.",
        ),
        extracted_facts=[
            ExtractedFact(
                fact_id="fact-1",
                text=FACT_TEXT,
                source_references=[SOURCE],
            )
        ],
    )


def make_summary(reference: SourceReference = SOURCE) -> SummarizationResponse:
    return SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(
                text="Synthetic evidence summary.",
                source_references=[reference],
            )
        ],
        warnings=["Requires verification."],
    )


class FakeGenerator:
    def __init__(
        self,
        response: SummarizationResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or SummarizationResponse()
        self.error = error
        self.calls = 0

    async def generate(self, request: SummarizationRequest) -> SummarizationResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def reset_dependency_overrides() -> Iterator[None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def enable_api() -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings()


def use_generator(generator: FakeGenerator) -> None:
    app.dependency_overrides[get_summarization_agent] = lambda: SummarizationAgent(
        generator
    )


def test_successful_summarization_uses_one_fake_generation() -> None:
    enable_api()
    generator = FakeGenerator(make_summary())
    use_generator(generator)

    response = TestClient(app).post(
        "/api/summarization/draft",
        json=make_request().model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json() == make_summary().model_dump(mode="json")
    assert response.json()["warnings"] == ["Requires verification."]
    assert generator.calls == 1


def test_report_returns_pending_json_and_escaped_markdown_without_model() -> None:
    enable_api()

    def reject_model_creation() -> SummarizationAgent:
        raise AssertionError("Report assembly must not resolve the Gemini dependency")

    app.dependency_overrides[get_summarization_agent] = reject_model_creation
    request = make_request()
    payload = {
        "case": request.case.model_dump(mode="json"),
        "source_request": request.model_dump(mode="json"),
        "summarization": make_summary().model_dump(mode="json"),
    }

    response = TestClient(app).post("/api/reports/draft", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["verification_status"] == "pending_verification"
    assert body["report"]["case"]["case_id"] == "case-1"
    assert "# Draft Legal Research Report" in body["markdown"]
    assert "Requires verification\\." in body["markdown"]


def test_request_schema_validation_remains_fastapi_422() -> None:
    enable_api()
    use_generator(FakeGenerator(make_summary()))

    response = TestClient(app).post(
        "/api/summarization/draft",
        json={"case": {"case_id": "case-1"}},
    )

    assert response.status_code == 422


def test_missing_source_evidence_is_rejected_without_generation() -> None:
    enable_api()
    generator = FakeGenerator(make_summary())
    use_generator(generator)
    request = SummarizationRequest(
        case=CaseInformation(case_id="case-1", title="No evidence")
    )

    response = TestClient(app).post(
        "/api/summarization/draft", json=request.model_dump(mode="json")
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "At least one supported evidence source is required."
    assert generator.calls == 0


def test_generated_unsupported_reference_is_rejected() -> None:
    enable_api()
    invented = SourceReference(document_id="invented", locator="page 99")
    use_generator(FakeGenerator(make_summary(invented)))

    response = TestClient(app).post(
        "/api/summarization/draft",
        json=make_request().model_dump(mode="json"),
    )

    assert response.status_code == 422
    assert "invented" not in response.text


def test_report_rejects_unsupported_reference() -> None:
    enable_api()
    request = make_request()
    invented = SourceReference(document_id="invented", locator="page 99")

    response = TestClient(app).post(
        "/api/reports/draft",
        json={
            "case": request.case.model_dump(mode="json"),
            "source_request": request.model_dump(mode="json"),
            "summarization": make_summary(invented).model_dump(mode="json"),
        },
    )

    assert response.status_code == 422
    assert "invented" not in response.text


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ModelUnavailableError("secret configuration detail"), 503),
        (GeminiProviderError("secret provider detail"), 502),
    ],
)
def test_provider_and_configuration_errors_are_sanitized(
    error: Exception, expected_status: int
) -> None:
    enable_api()
    use_generator(FakeGenerator(error=error))

    response = TestClient(app).post(
        "/api/summarization/draft",
        json=make_request().model_dump(mode="json"),
    )

    assert response.status_code == expected_status
    assert "secret" not in response.text


def test_missing_backend_api_key_maps_to_service_unavailable() -> None:
    enable_api()

    response = TestClient(app).post(
        "/api/summarization/draft",
        json=make_request().model_dump(mode="json"),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The summarization model is not configured or available."
    )


@pytest.mark.parametrize(
    "settings",
    [make_settings(enabled=False), make_settings(enabled=True, app_env="production")],
)
def test_draft_endpoints_are_disabled_unless_both_conditions_hold(
    settings: Settings,
) -> None:
    app.dependency_overrides[get_settings] = lambda: settings

    def reject_model_creation() -> SummarizationAgent:
        raise AssertionError("Disabled APIs must not construct a model client")

    app.dependency_overrides[get_summarization_agent] = reject_model_creation
    request = make_request()

    summary_response = TestClient(app).post(
        "/api/summarization/draft",
        json=request.model_dump(mode="json"),
    )
    report_response = TestClient(app).post(
        "/api/reports/draft",
        json={
            "case": request.case.model_dump(mode="json"),
            "source_request": request.model_dump(mode="json"),
            "summarization": make_summary().model_dump(mode="json"),
        },
    )

    assert summary_response.status_code == 403
    assert report_response.status_code == 403


def test_health_endpoint_remains_available_when_draft_api_is_disabled() -> None:
    app.dependency_overrides[get_settings] = lambda: make_settings(enabled=False)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "caselens-backend"}
