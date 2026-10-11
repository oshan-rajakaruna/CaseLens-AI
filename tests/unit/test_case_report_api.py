"""Offline tests for development authentication and owner-scoped APIs."""

from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from backend.api.dependencies import get_persistence_service
from backend.config import get_settings
from backend.main import app
from backend.schemas.legal_comparison import LegalComparisonDraft
from database import PersistenceService, TrustedPrincipal, build_engine, build_session_factory
from database.errors import PersistenceError
from database.models import Base
from tests.unit.test_legal_report_markdown import _complete_report

_ORIGIN = "http://127.0.0.1:5173"
_PASSWORD = "offline-development-password"
_SECRET = "offline-signing-key-with-at-least-32-characters"
_OWNER = TrustedPrincipal(subject="local|operator")


@pytest.fixture
def api_setup(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, PersistenceService]]:
    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'api.sqlite3'}")
    Base.metadata.create_all(engine)
    persistence = PersistenceService(build_session_factory(engine))
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("ENABLE_DEV_CASE_API", "false")
    monkeypatch.setenv("DEV_AUTH_OPERATOR_SUBJECT", _OWNER.subject)
    monkeypatch.setenv("DEV_AUTH_PASSWORD", _PASSWORD)
    monkeypatch.setenv("DEV_AUTH_ALLOWED_ORIGIN", _ORIGIN)
    monkeypatch.setenv("SECRET_KEY", _SECRET)
    get_settings.cache_clear()
    get_persistence_service.cache_clear()
    app.dependency_overrides[get_persistence_service] = lambda: persistence
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, persistence
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        get_persistence_service.cache_clear()
        engine.dispose()


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_DEV_CASE_API", "true")
    get_settings.cache_clear()


def _login(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": _PASSWORD},
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_authentication_is_required_and_login_fails_closed(api_setup) -> None:
    client, _ = api_setup

    assert client.get("/api/cases").status_code == 401
    response = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": _PASSWORD},
    )
    assert response.status_code == 404


def test_invalid_and_successful_development_login(api_setup, monkeypatch) -> None:
    client, _ = api_setup
    _enable(monkeypatch)

    assert client.post(
        "/api/auth/login", json={"password": _PASSWORD}
    ).status_code == 403

    denied = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": "wrong"},
    )
    assert denied.status_code == 401
    assert _PASSWORD not in denied.text

    accepted = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": _PASSWORD},
    )
    assert accepted.status_code == 200
    cookie = accepted.headers["set-cookie"].casefold()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "path=/api" in cookie
    assert accepted.headers["cache-control"] == "no-store"


def test_development_login_is_disabled_in_production(api_setup, monkeypatch) -> None:
    client, _ = api_setup
    _enable(monkeypatch)
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()

    response = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": _PASSWORD},
    )
    assert response.status_code == 404


def test_owner_scoped_case_create_list_and_detail(api_setup, monkeypatch) -> None:
    client, _ = api_setup
    _enable(monkeypatch)
    csrf = _login(client)

    rejected_owner = client.post(
        "/api/cases",
        headers={"Origin": _ORIGIN, "X-CSRF-Token": csrf},
        json={
            "title": "Synthetic case",
            "description": "Authorized content",
            "research_question": "What law applies?",
            "owner_subject": "browser|attacker",
        },
    )
    assert rejected_owner.status_code == 422

    created = client.post(
        "/api/cases",
        headers={"Origin": _ORIGIN, "X-CSRF-Token": csrf},
        json={
            "title": "Synthetic case",
            "description": "Authorized content",
            "research_question": "What law applies?",
        },
    )
    assert created.status_code == 201
    assert "owner_subject" not in created.text
    case_id = created.json()["id"]

    listing = client.get("/api/cases?offset=0&limit=10")
    detail = client.get(f"/api/cases/{case_id}")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["items"]] == [case_id]
    assert detail.status_code == 200
    assert detail.json() == created.json()


def test_cross_owner_case_and_report_are_not_disclosed(api_setup, monkeypatch) -> None:
    client, persistence = api_setup
    other = TrustedPrincipal(subject="local|other")
    persistence.create_case(other, case_id="case-other", title="Other case")
    persistence.create_task(
        other,
        task_id="task-other",
        case_id="case-other",
        request_fingerprint="request-other",
    )
    report = _complete_report().model_copy(deep=True)
    report.case_information.case.id = "case-other"
    report.case_information.case.title = "PRIVATE-OTHER-OWNER-EVIDENCE"
    persistence.save_report(
        other,
        report_id="report-other",
        case_id="case-other",
        task_id="task-other",
        report=report,
        state="completed",
    )
    _enable(monkeypatch)
    _login(client)

    assert client.get("/api/cases/case-other").status_code == 404
    assert client.get("/api/cases/case-other/reports").status_code == 404
    denied_report = client.get("/api/reports/report-other")
    assert denied_report.status_code == 404
    assert "PRIVATE-OTHER-OWNER-EVIDENCE" not in denied_report.text
    assert client.get("/api/cases").json()["items"] == []


def test_report_list_detail_preserve_provenance_and_review_labels(api_setup, monkeypatch) -> None:
    client, persistence = api_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Synthetic case")
    persistence.create_task(
        _OWNER,
        task_id="task-1",
        case_id="case-1",
        request_fingerprint="request-1",
    )
    report = _complete_report(comparison=True)
    report.comparisons[0].ai_draft = LegalComparisonDraft(
        comparison_id="CMP-001",
        context_id="CTX-002",
        explanation="Potential relationship requiring review.",
        uncertainties=["Applicability remains uncertain."],
        ai_generated=True,
        human_review_required=True,
        provider="offline-fake",
    )
    persistence.save_report(
        _OWNER,
        report_id="report-1",
        case_id="case-1",
        task_id="task-1",
        report=report,
        state="completed",
    )
    _enable(monkeypatch)
    _login(client)

    listing = client.get("/api/cases/case-1/reports")
    detail = client.get("/api/reports/report-1")
    assert listing.status_code == detail.status_code == 200
    assert listing.json()["items"] == [detail.json()]
    body = detail.json()["report"]
    assert body["authorities"][0]["context_id"] == "CTX-001"
    assert body["authorities"][0]["pages"] == [3]
    assert body["authorities"][0]["chunk_ids"]
    assert body["verification"] == report.verification.model_dump(mode="json")
    comparison = body["comparisons"][0]
    assert comparison["review_status"] == "requires_human_review"
    assert comparison["ai_draft"]["verification_status"] == "not_verified"
    assert body["human_review_disclaimer"]
    assert "owner_subject" not in detail.text


def test_origin_and_csrf_are_required_for_mutation(api_setup, monkeypatch) -> None:
    client, _ = api_setup
    _enable(monkeypatch)
    csrf = _login(client)
    payload = {"title": "Denied mutation"}

    assert client.post("/api/cases", json=payload).status_code == 403
    assert client.post(
        "/api/cases", headers={"Origin": _ORIGIN, "X-CSRF-Token": "wrong"}, json=payload
    ).status_code == 403
    assert client.post(
        "/api/cases", headers={"Origin": "http://localhost:9999", "X-CSRF-Token": csrf}, json=payload
    ).status_code == 403

    client.cookies.clear()
    client.cookies.set("caselens_dev_session", "tampered", path="/api")
    assert client.get("/api/cases").status_code == 401


def test_invalid_input_does_not_echo_private_values(api_setup, monkeypatch) -> None:
    client, _ = api_setup
    _enable(monkeypatch)
    csrf = _login(client)
    private_value = "PRIVATE-EVIDENCE-SENTINEL" * 1000

    response = client.post(
        "/api/cases",
        headers={"Origin": _ORIGIN, "X-CSRF-Token": csrf},
        json={"title": "Case", "description": private_value},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}
    assert "PRIVATE-EVIDENCE-SENTINEL" not in response.text


def test_database_errors_are_sanitized(api_setup, monkeypatch) -> None:
    client, _ = api_setup
    _enable(monkeypatch)
    _login(client)

    class FailingPersistence:
        def list_cases(self, *args, **kwargs):
            raise PersistenceError("postgresql://user:password@host/private evidence")

    app.dependency_overrides[get_persistence_service] = lambda: FailingPersistence()
    response = client.get("/api/cases")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database is unavailable"}
    assert "password" not in response.text
    assert "private evidence" not in response.text
