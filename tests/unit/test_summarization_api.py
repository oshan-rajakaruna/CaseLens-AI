"""Offline API tests for development-only summary verification."""

from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from agents.summarization import (
    SummarizationAgent,
    SummarizationProviderError,
    SummarizationVerificationBridge,
    SummaryProvenanceError,
)
from agents.verification import VerificationAgent
from backend.api.summarization import get_api_summarization_service
from backend.config import get_settings
from backend.main import app
from backend.schemas import (
    Citation,
    LegalClaim,
    RetrievedEvidence,
    SourceMetadata,
    VerificationRequest,
)
from backend.schemas.summarization import (
    SummarizationVerifyRequest,
    SummarizationVerifyResponse,
)
from backend.services.summarization_service import SummarizationService
from tests.unit.summarization._fakes import FakeSummarizationProvider
from tests.unit.summarization.test_verification_bridge import _passage, _payload
from tests.unit.test_summarization_service import _service

_TOKEN = "offline-development-token"


class FakeAPIService:
    def __init__(
        self,
        response: SummarizationVerifyResponse | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[SummarizationVerifyRequest] = []

    def summarize_and_verify(
        self,
        request: SummarizationVerifyRequest,
    ) -> SummarizationVerifyResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _response(
    evidence_text: str = "The appeal was allowed.",
) -> SummarizationVerifyResponse:
    return _service("The appeal was allowed. [CTX-001]").summarize_and_verify(
        SummarizationVerifyRequest(
            task_id="task-1",
            payload=_payload(_passage(text=evidence_text)),
        )
    )


def _request_json(evidence_text: str = "The appeal was allowed.") -> dict:
    request = SummarizationVerifyRequest(
        task_id="task-1",
        payload=_payload(_passage(text=evidence_text)),
    )
    return request.model_dump(mode="json")


def _enable_development_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("ENABLE_DEV_SUMMARIZATION_API", "true")
    monkeypatch.setenv("DEV_SUMMARIZATION_API_TOKEN", _TOKEN)
    get_settings.cache_clear()


@pytest.fixture
def api_setup(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, FakeAPIService, list[int]]]:
    service = FakeAPIService(_response())
    resolutions: list[int] = []

    def dependency() -> SummarizationService:
        resolutions.append(1)
        return service  # type: ignore[return-value]

    app.dependency_overrides[get_api_summarization_service] = dependency
    get_settings.cache_clear()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, service, resolutions
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {
            "APP_ENV": "development",
            "ENABLE_DEV_SUMMARIZATION_API": "false",
            "DEV_SUMMARIZATION_API_TOKEN": _TOKEN,
        },
        {
            "APP_ENV": "production",
            "ENABLE_DEV_SUMMARIZATION_API": "true",
            "DEV_SUMMARIZATION_API_TOKEN": _TOKEN,
        },
        {
            "APP_ENV": "development",
            "ENABLE_DEV_SUMMARIZATION_API": "true",
            "DEV_SUMMARIZATION_API_TOKEN": "",
        },
    ],
)
def test_access_fails_closed_without_complete_development_configuration(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    client, service, resolutions = api_setup
    monkeypatch.setenv("APP_ENV", environment.get("APP_ENV", "development"))
    monkeypatch.setenv(
        "ENABLE_DEV_SUMMARIZATION_API",
        environment.get("ENABLE_DEV_SUMMARIZATION_API", "false"),
    )
    monkeypatch.setenv(
        "DEV_SUMMARIZATION_API_TOKEN",
        environment.get("DEV_SUMMARIZATION_API_TOKEN", ""),
    )
    get_settings.cache_clear()

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(),
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Development summarization access denied"
    assert service.calls == []
    assert resolutions == []
    assert _TOKEN not in response.text


@pytest.mark.parametrize("header", [None, "incorrect-token"])
def test_missing_or_incorrect_header_is_denied_without_service_call(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
    header: str | None,
) -> None:
    client, service, resolutions = api_setup
    _enable_development_access(monkeypatch)
    headers = {} if header is None else {"X-CaseLens-Dev-Token": header}

    response = client.post(
        "/api/summarization/verify",
        headers=headers,
        json=_request_json(),
    )

    assert response.status_code == 403
    assert service.calls == []
    assert resolutions == []


def test_valid_development_access_returns_typed_output(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, resolutions = api_setup
    _enable_development_access(monkeypatch)

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["output"]["citations"] == ["CTX-001"]
    assert body["verification"]["overall_status"] == "supported"
    assert body["provenance"]["CTX-001"]["document_id"] == "DOC-A"
    assert body["bridge_warnings"]
    assert len(service.calls) == 1
    assert resolutions == [1]


def test_endpoint_completes_with_real_composed_service_and_fake_provider(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = api_setup
    _enable_development_access(monkeypatch)
    service = _service("The appeal was allowed. [CTX-001]")
    app.dependency_overrides[get_api_summarization_service] = lambda: service

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(),
    )

    assert response.status_code == 200
    assert response.json()["verification"]["overall_status"] == "supported"
    assert response.json()["summary"]["provider_metadata"]["provider"] == "fake"


def test_real_composed_service_sanitizes_provider_failure(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = api_setup
    _enable_development_access(monkeypatch)
    secret = "PRIVATE-EVIDENCE-AND-CREDENTIAL-SENTINEL"
    provider = FakeSummarizationProvider(error=RuntimeError(secret))
    service = SummarizationService(
        SummarizationAgent(provider),
        SummarizationVerificationBridge(VerificationAgent()),
    )
    app.dependency_overrides[get_api_summarization_service] = lambda: service

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(secret),
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Summarization provider is unavailable"
    assert secret not in response.text


@pytest.mark.parametrize(
    "evidence_text,expected",
    [
        ("The appeal was allowed.", "supported"),
        ("The appeal was not allowed.", "unsupported"),
        ("The court heard argument.", "uncertain"),
    ],
)
def test_completed_verification_status_is_not_converted_to_http_failure(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
    evidence_text: str,
    expected: str,
) -> None:
    client, service, _ = api_setup
    _enable_development_access(monkeypatch)
    service.response = _response(evidence_text)

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(evidence_text),
    )

    assert response.status_code == 200
    assert response.json()["verification"]["overall_status"] == expected


def test_invalid_citation_outcome_is_preserved_as_successful_workflow(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _ = api_setup
    _enable_development_access(monkeypatch)
    baseline = _response()
    invalid = VerificationAgent().verify(
        VerificationRequest(
            task_id="task-1",
            claims=[
                LegalClaim(
                    id="summary-claim-001",
                    text="The appeal was allowed.",
                    citations=[Citation(document_id="DOC-UNKNOWN")],
                )
            ],
            evidence=[
                RetrievedEvidence(
                    id="CTX-001",
                    source=SourceMetadata(document_id="DOC-A", title="Decision A"),
                    text="The appeal was allowed.",
                    locator="CTX-001",
                )
            ],
        )
    )
    service.response = baseline.model_copy(update={"verification": invalid})

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(),
    )

    assert response.status_code == 200
    assert response.json()["verification"]["overall_status"] == "invalid_citations"
    assert response.json()["verification"]["verified"] is False


def test_response_does_not_echo_original_evidence_text(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _ = api_setup
    _enable_development_access(monkeypatch)
    evidence_text = "PRIVATE-EVIDENCE-SENTINEL"
    service.response = _response(evidence_text)

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(evidence_text),
    )

    assert response.status_code == 200
    assert evidence_text not in response.text
    assert "verification_request" not in response.text


def test_invalid_request_returns_422(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, _ = api_setup
    _enable_development_access(monkeypatch)

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json={"task_id": "", "payload": {}},
    )

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    "error,status_code,detail,secret",
    [
        (
            SummaryProvenanceError("PRIVATE-EVIDENCE-SENTINEL"),
            400,
            "Summary could not be verified from the supplied context",
            "PRIVATE-EVIDENCE-SENTINEL",
        ),
        (
            SummarizationProviderError("secret-provider-detail"),
            503,
            "Summarization provider is unavailable",
            "secret-provider-detail",
        ),
        (
            RuntimeError("D:/private/legal-file.txt"),
            500,
            "Summarization workflow failed",
            "private",
        ),
    ],
)
def test_failures_return_sanitized_http_errors(
    api_setup,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
    detail: str,
    secret: str,
) -> None:
    client, service, _ = api_setup
    _enable_development_access(monkeypatch)
    service.error = error

    response = client.post(
        "/api/summarization/verify",
        headers={"X-CaseLens-Dev-Token": _TOKEN},
        json=_request_json(),
    )

    assert response.status_code == status_code
    assert response.json()["detail"] == detail
    assert secret not in response.text



def test_existing_system_and_retrieval_routes_remain_registered(
    api_setup,
) -> None:
    client, _, _ = api_setup

    # Existing application endpoints
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"

    # Both API endpoints are POST-only.
    # GET should return 405 (Method Not Allowed), not 404.
    assert client.get("/api/retrieval/search").status_code == 405
    assert client.get("/api/summarization/verify").status_code == 405
