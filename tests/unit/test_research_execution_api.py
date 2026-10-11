"""Offline coverage for authorized research execution and persistence."""

from collections.abc import Iterator
from io import BytesIO
from threading import Event

from fastapi.testclient import TestClient
import pytest

from backend.api.dependencies import (
    get_persistence_service,
    get_research_execution_service,
)
from backend.config import get_settings
from backend.main import app
from backend.schemas import AgentResult, AgentTask
from backend.schemas.case_report_api import ResearchStartRequest
from backend.schemas.research_workflow import (
    ResearchWorkflowError,
    ResearchWorkflowResult,
    ResearchWorkflowStages,
)
from backend.services.private_documents import PrivateDocumentStorage, PrivateEvidenceService
from backend.services.research_execution import ResearchExecutionService
from database import PersistenceError, PersistenceService, TrustedPrincipal, build_engine, build_session_factory
from database.models import Base
from tests.unit.test_legal_report_markdown import _complete_report

_ORIGIN = "http://127.0.0.1:5173"
_PASSWORD = "offline-development-password"
_SECRET = "offline-signing-key-with-at-least-32-characters"
_OWNER = TrustedPrincipal(subject="local|operator")


class _Coordinator:
    def __init__(
        self,
        *,
        status: str = "completed",
        with_report: bool = True,
        error_code: str = "retrieval_unavailable",
        verification_status: str = "supported",
    ):
        self.status = status
        self.with_report = with_report
        self.error_code = error_code
        self.verification_status = verification_status
        self.calls = []
        self.results = {}

    async def handle(self, task):
        self.calls.append(task)
        report = (
            _complete_report(verification_status=self.verification_status)
            if self.with_report
            else None
        )
        errors = (
            [
                ResearchWorkflowError(
                    stage="retrieval",
                    code=self.error_code,
                    message="Research could not complete.",
                )
            ]
            if self.status != "completed"
            else []
        )
        stages = ResearchWorkflowStages(
            analysis="partial" if self.status == "partial" else "completed",
            retrieval="failed" if self.status == "failed" else "completed",
            summarization_verification=(
                "not_started" if self.status == "failed" else "completed"
            ),
            report="completed" if report is not None else "not_started",
            comparisons="skipped",
            markdown="skipped",
        )
        self.results[task.id] = ResearchWorkflowResult(
            task_id=task.id,
            status=self.status,
            stages=stages,
            report=report,
            errors=errors,
        )
        return AgentResult(
            task_id=task.id,
            agent="coordinator",
            status=self.status,
            output={
                "workflow_status": self.status,
                "stages": stages.model_dump(mode="json"),
                "errors": [item.model_dump(mode="json") for item in errors],
                "report_available": report is not None,
            },
        )

    def get_private_result(self, task_id):
        return self.results.get(task_id)


class _InlineExecutionService(ResearchExecutionService):
    def _submit(self, principal, case_id, task):
        self._execute(principal, case_id, task)


@pytest.fixture
def persistence(tmp_path) -> Iterator[PersistenceService]:
    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'research.sqlite3'}")
    Base.metadata.create_all(engine)
    yield PersistenceService(build_session_factory(engine))
    engine.dispose()


def _case_and_document(persistence, tmp_path, *, owner=_OWNER):
    persistence.create_case(
        owner,
        case_id="case-1",
        title="Synthetic case",
        description="Authorized case description",
    )
    storage = PrivateDocumentStorage(tmp_path / "private", 1024 * 1024)
    stored = storage.store(BytesIO(b"The tenant received notice."), ".txt")
    persistence.add_document(
        owner,
        document_id="document-1",
        case_id="case-1",
        safe_filename="evidence.txt",
        media_type="text/plain",
        storage_key=stored.storage_key,
    )
    return storage


def _request(**updates):
    values = {
        "query": "What authorities govern notice?",
        "document_ids": ["document-1"],
    }
    values.update(updates)
    return ResearchStartRequest.model_validate(values)


def _service(persistence, storage, coordinator):
    return _InlineExecutionService(
        persistence=persistence,
        evidence=PrivateEvidenceService(persistence, storage),
        coordinator=coordinator,
        private_results=coordinator,
    )


def test_execution_loads_server_evidence_delegates_once_and_persists_report(
    persistence, tmp_path
) -> None:
    storage = _case_and_document(persistence, tmp_path)
    coordinator = _Coordinator()
    service = _service(persistence, storage, coordinator)

    accepted = service.start(_OWNER, case_id="case-1", request=_request())
    task = service.get_task(_OWNER, accepted.id)
    report_id = service.report_id_for(_OWNER, task.id)
    report = persistence.get_report(_OWNER, report_id)

    assert len(coordinator.calls) == 1
    delegated = coordinator.calls[0]
    assert delegated.context["retrieval_mode"] == "bm25"
    assert delegated.context["generate_comparison_drafts"] is False
    assert delegated.context["comparisons"] == []
    assert delegated.context["evidence_documents"][0]["text"] == "The tenant received notice."
    assert delegated.context["evidence_documents"][0]["document"]["id"] == "document-1"
    assert task.status == "completed"
    assert report.task_id == task.id
    assert report.report.authorities[0].context_id == "CTX-001"
    assert report.report.authorities[0].chunk_ids
    assert report.report.verification.response.verified is True
    assert report.report.human_review_disclaimer


@pytest.mark.parametrize("verification_status", ["uncertain", "unsupported"])
def test_false_verification_result_is_preserved_without_promotion(
    persistence, tmp_path, verification_status
) -> None:
    storage = _case_and_document(persistence, tmp_path)
    coordinator = _Coordinator(verification_status=verification_status)
    service = _service(persistence, storage, coordinator)

    accepted = service.start(_OWNER, case_id="case-1", request=_request())
    report_id = service.report_id_for(_OWNER, accepted.id)
    report = persistence.get_report(_OWNER, report_id).report

    assert report.verification.response.overall_status == verification_status
    assert report.verification.response.verified is False


def test_identical_request_is_idempotent_and_not_executed_twice(
    persistence, tmp_path
) -> None:
    storage = _case_and_document(persistence, tmp_path)
    coordinator = _Coordinator()
    service = _service(persistence, storage, coordinator)

    first = service.start(_OWNER, case_id="case-1", request=_request())
    second = service.start(_OWNER, case_id="case-1", request=_request())

    assert first.id == second.id
    assert len(coordinator.calls) == 1


def test_concurrent_duplicate_execution_is_suppressed(persistence, tmp_path) -> None:
    storage = _case_and_document(persistence, tmp_path)
    persistence.create_task(
        _OWNER,
        task_id="task-concurrent",
        case_id="case-1",
        request_fingerprint="concurrent-fingerprint",
        stages={"analysis": "pending"},
    )
    started = Event()
    release = Event()

    class BlockingCoordinator(_Coordinator):
        async def handle(self, task):
            self.calls.append(task)
            started.set()
            release.wait(timeout=2)
            return AgentResult(
                task_id=task.id,
                agent="coordinator",
                status="failed",
                output={"errors": [], "report_available": False},
            )

    coordinator = BlockingCoordinator()
    service = ResearchExecutionService(
        persistence=persistence,
        evidence=PrivateEvidenceService(persistence, storage),
        coordinator=coordinator,
        private_results=coordinator,
    )
    task = AgentTask(
        id="task-concurrent",
        agent="coordinator",
        instruction="test",
        context={},
    )
    service._submit(_OWNER, "case-1", task)
    assert started.wait(timeout=2)
    service._submit(_OWNER, "case-1", task)
    with service._active_lock:
        future = service._active[task.id]
    release.set()
    future.result(timeout=2)

    assert len(coordinator.calls) == 1


@pytest.mark.parametrize(
    ("status", "with_report", "error_code", "expected_status", "has_report"),
    [
        ("partial", True, "analysis_document_failed", "partial", True),
        ("failed", False, "retrieval_unavailable", "failed", False),
        (
            "failed",
            False,
            "summarization_verification_failed",
            "failed",
            False,
        ),
    ],
)
def test_partial_report_and_controlled_workflow_failure(
    persistence,
    tmp_path,
    status,
    with_report,
    error_code,
    expected_status,
    has_report,
) -> None:
    storage = _case_and_document(persistence, tmp_path)
    coordinator = _Coordinator(
        status=status,
        with_report=with_report,
        error_code=error_code,
    )
    service = _service(persistence, storage, coordinator)

    accepted = service.start(_OWNER, case_id="case-1", request=_request())
    task = service.get_task(_OWNER, accepted.id)

    assert task.status == expected_status
    assert (service.report_id_for(_OWNER, task.id) is not None) is has_report
    assert task.error_category == error_code


def test_foreign_cross_case_and_missing_documents_are_rejected(
    persistence, tmp_path
) -> None:
    storage = _case_and_document(persistence, tmp_path)
    other = TrustedPrincipal(subject="local|other")
    persistence.create_case(other, case_id="case-other", title="Other")
    service = _service(persistence, storage, _Coordinator())

    with pytest.raises(PersistenceError):
        service.start(other, case_id="case-1", request=_request())
    with pytest.raises(PersistenceError):
        service.start(_OWNER, case_id="case-1", request=_request(document_ids=["missing"]))


def test_restart_interrupted_task_is_reconciled_to_failed(
    persistence, tmp_path
) -> None:
    storage = _case_and_document(persistence, tmp_path)
    task = persistence.create_task(
        _OWNER,
        task_id="task-interrupted",
        case_id="case-1",
        request_fingerprint="interrupted-fingerprint",
        stages={"analysis": "running", "retrieval": "pending"},
    )
    persistence.update_task(
        _OWNER,
        task.id,
        status="running",
        stages={"analysis": "running", "retrieval": "pending"},
    )
    service = _service(persistence, storage, _Coordinator())

    reconciled = service.get_task(_OWNER, task.id)

    assert reconciled.status == "failed"
    assert reconciled.error_category == "restart_interrupted"
    assert reconciled.stages["analysis"] == "failed"
    with pytest.raises(PersistenceError):
        service.get_task(TrustedPrincipal(subject="local|other"), task.id)


def test_report_persistence_failure_marks_task_failed(persistence, tmp_path) -> None:
    storage = _case_and_document(persistence, tmp_path)

    class FailingPersistence:
        def __getattr__(self, name):
            return getattr(persistence, name)

        def finalize_task_with_report(self, *args, **kwargs):
            raise PersistenceError("database=password private evidence")

    coordinator = _Coordinator()
    service = _InlineExecutionService(
        persistence=FailingPersistence(),
        evidence=PrivateEvidenceService(persistence, storage),
        coordinator=coordinator,
        private_results=coordinator,
    )
    accepted = service.start(_OWNER, case_id="case-1", request=_request())
    task = persistence.get_task(_OWNER, accepted.id)

    assert task.status == "failed"
    assert task.error_category == "report_persistence_failed"
    assert persistence.find_report_by_task(_OWNER, task.id) is None


class _ApiExecution:
    def __init__(self, persistence):
        self.persistence = persistence
        self.starts = []

    def start(self, principal, *, case_id, request):
        self.starts.append((principal, case_id, request))
        return self.persistence.create_task(
            principal,
            task_id="task-api",
            case_id=case_id,
            request_fingerprint="api-fingerprint",
            stages={"analysis": "pending"},
        )

    def get_task(self, principal, task_id):
        return self.persistence.get_task(principal, task_id)

    def report_id_for(self, principal, task_id):
        self.persistence.get_task(principal, task_id)
        return None


@pytest.fixture
def api_setup(
    persistence, monkeypatch
) -> Iterator[tuple[TestClient, _ApiExecution, PersistenceService]]:
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    execution = _ApiExecution(persistence)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("ENABLE_DEV_CASE_API", "true")
    monkeypatch.setenv("DEV_AUTH_OPERATOR_SUBJECT", _OWNER.subject)
    monkeypatch.setenv("DEV_AUTH_PASSWORD", _PASSWORD)
    monkeypatch.setenv("DEV_AUTH_ALLOWED_ORIGIN", _ORIGIN)
    monkeypatch.setenv("SECRET_KEY", _SECRET)
    get_settings.cache_clear()
    app.dependency_overrides[get_persistence_service] = lambda: persistence
    app.dependency_overrides[get_research_execution_service] = lambda: execution
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, execution, persistence
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def _login(client):
    response = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": _PASSWORD},
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def test_research_api_auth_csrf_safe_contract_and_polling(api_setup) -> None:
    client, execution, persistence = api_setup
    body = {"query": "notice", "document_ids": ["document-1"]}

    assert client.post("/api/cases/case-1/research", json=body).status_code == 401
    csrf = _login(client)
    assert client.post(
        "/api/cases/case-1/research", json=body
    ).status_code == 403
    response = client.post(
        "/api/cases/case-1/research",
        headers={"Origin": _ORIGIN, "X-CSRF-Token": csrf},
        json=body,
    )

    assert response.status_code == 202
    assert response.json()["id"] == "task-api"
    assert "notice" not in response.text
    assert "document-1" not in response.text
    assert execution.starts[0][2].document_ids == ["document-1"]
    polled = client.get("/api/research-tasks/task-api")
    assert polled.status_code == 200
    assert "private" not in polled.text.casefold()

    persistence.update_task(
        _OWNER,
        "task-api",
        status="running",
        stages={"analysis": "running"},
    )
    persistence.finalize_task_with_report(
        _OWNER,
        report_id="report-api",
        case_id="case-1",
        task_id="task-api",
        report=_complete_report(),
        state="completed",
        task_status="completed",
        stages={"analysis": "completed", "report": "completed"},
    )
    persisted = client.get("/api/reports/report-api")
    assert persisted.status_code == 200
    assert persisted.json()["report"]["authorities"][0]["context_id"] == "CTX-001"
    assert persisted.json()["report"]["human_review_disclaimer"]


def test_research_api_rejects_raw_evidence_and_invalid_selection(api_setup) -> None:
    client, execution, _ = api_setup
    csrf = _login(client)
    headers = {"Origin": _ORIGIN, "X-CSRF-Token": csrf}

    raw = client.post(
        "/api/cases/case-1/research",
        headers=headers,
        json={
            "query": "notice",
            "document_ids": ["document-1"],
            "evidence_text": "PRIVATE EVIDENCE",
        },
    )
    empty = client.post(
        "/api/cases/case-1/research",
        headers=headers,
        json={"query": "notice", "document_ids": []},
    )

    assert raw.status_code == empty.status_code == 422
    assert "PRIVATE EVIDENCE" not in raw.text
    assert execution.starts == []
