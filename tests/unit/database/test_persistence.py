"""Offline tests for the internal durable persistence boundary."""

from datetime import datetime, timezone
import json

import pytest

from agents.analysis import AnalysisAgent
from agents.summarization import GeminiSummarizationProvider
from agents.verification import VerificationAgent
from backend.schemas.legal_comparison import LegalComparisonDraft
from database import (
    DuplicateRecordError,
    ImmutableRecordError,
    InvalidRecordError,
    MissingRecordError,
    OwnershipError,
    PersistenceService,
    TrustedPrincipal,
    build_engine,
    build_session_factory,
)
from database.models import Base
from database.repository import CaseLensRepository
from tests.unit.test_legal_report_markdown import _complete_report
from tests.unit.test_legal_report import _assemble


@pytest.fixture
def database_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'persistence.sqlite3'}"


@pytest.fixture
def service(database_url) -> PersistenceService:
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    yield PersistenceService(build_session_factory(engine))
    engine.dispose()


@pytest.fixture
def owner() -> TrustedPrincipal:
    return TrustedPrincipal(subject="auth0|owner-1")


def _case(service, owner, case_id="case-1"):
    return service.create_case(
        owner,
        case_id=case_id,
        title="Synthetic tenancy matter",
        description="Private case description",
        research_question="What authorities apply?",
    )


def _task(service, owner, task_id="task-1", fingerprint="sha256-request-1"):
    return service.create_task(
        owner,
        task_id=task_id,
        case_id="case-1",
        request_fingerprint=fingerprint,
        stages={"analysis": "pending"},
    )


def _assert_aware_utc(value: datetime) -> None:
    assert value.tzinfo is not None
    assert value.utcoffset() == timezone.utc.utcoffset(value)


def test_case_creation_retrieval_and_owner_identity(service, owner) -> None:
    created = _case(service, owner)
    loaded = service.get_case(owner, "case-1")

    assert loaded == created
    assert loaded.owner_subject == owner.subject
    _assert_aware_utc(created.created_at)
    _assert_aware_utc(created.updated_at)
    _assert_aware_utc(loaded.created_at)
    _assert_aware_utc(loaded.updated_at)
    with pytest.raises(OwnershipError):
        service.get_case(TrustedPrincipal(subject="auth0|other"), "case-1")


def test_document_association_and_unsafe_relationships(service, owner) -> None:
    _case(service, owner)
    document = service.add_document(
        owner,
        document_id="document-1",
        case_id="case-1",
        safe_filename="evidence.pdf",
        media_type="application/pdf",
        storage_key="owners/owner-1/cases/case-1/document-1",
        source_metadata={"source": "authorized_upload"},
    )

    assert document.case_id == "case-1"
    assert document.owner_subject == owner.subject
    with pytest.raises(MissingRecordError):
        service.add_document(
            owner,
            document_id="document-2",
            case_id="missing",
            safe_filename="missing.pdf",
            media_type="application/pdf",
            storage_key="objects/document-2",
        )
    with pytest.raises(InvalidRecordError):
        service.add_document(
            owner,
            document_id="document-3",
            case_id="case-1",
            safe_filename="unsafe.pdf",
            media_type="application/pdf",
            storage_key="C:/private/unsafe.pdf",
        )


def test_task_creation_update_and_duplicate_identity(service, owner) -> None:
    _case(service, owner)
    created = _task(service, owner)
    running = service.update_task(
        owner,
        created.id,
        status="running",
        stages={"analysis": "running"},
    )
    completed = service.update_task(
        owner,
        created.id,
        status="completed",
        stages={"analysis": "completed"},
    )

    assert running.status == "running"
    assert completed.completed_at is not None
    loaded = service.get_task(owner, created.id)
    for timestamp in (loaded.created_at, loaded.updated_at, loaded.completed_at):
        assert timestamp is not None
        _assert_aware_utc(timestamp)
    with pytest.raises(DuplicateRecordError):
        _task(service, owner, task_id="task-1", fingerprint="another-request")
    with pytest.raises(DuplicateRecordError):
        _task(service, owner, task_id="task-2", fingerprint="sha256-request-1")
    with pytest.raises(OwnershipError):
        service.update_task(
            TrustedPrincipal(subject="auth0|other"),
            created.id,
            status="failed",
        )


@pytest.mark.parametrize("verification_status", ["unsupported", "uncertain"])
def test_structured_report_preserves_provenance_and_verification(
    service, owner, verification_status
) -> None:
    _case(service, owner)
    _task(service, owner)
    report = _assemble(status=verification_status)

    service.save_report(
        owner,
        report_id="report-1",
        case_id="case-1",
        task_id="task-1",
        report=report,
        state="completed",
    )
    loaded = service.get_report(owner, "report-1")

    _assert_aware_utc(loaded.created_at)
    assert loaded.report.authorities[0].context_id == "CTX-001"
    assert loaded.report.authorities[0].document_id == "DOC-A"
    assert loaded.report.authorities[0].pages == [3]
    assert loaded.report.authorities[0].chunk_ids
    assert loaded.report.verification.response.overall_status == verification_status
    assert loaded.report.verification.response.verified is False
    assert service.find_report_by_task(owner, "task-1") == loaded
    assert service.find_reports_by_case(owner, "case-1") == [loaded]


def test_ai_comparison_review_state_is_preserved(service, owner) -> None:
    _case(service, owner)
    _task(service, owner)
    report = _complete_report(comparison=True)
    report.comparisons[0].ai_draft = LegalComparisonDraft(
        comparison_id="CMP-001",
        context_id="CTX-002",
        explanation="Potential relationship requiring legal review.",
        uncertainties=["Applicability remains uncertain."],
        ai_generated=True,
        human_review_required=True,
        provider="offline-fake",
    )

    service.save_report(
        owner,
        report_id="report-1",
        case_id="case-1",
        task_id="task-1",
        report=report,
        state="completed",
    )
    comparison = service.get_report(owner, "report-1").report.comparisons[0]

    assert comparison.review_status == "requires_human_review"
    assert comparison.ai_draft.verification_status == "not_verified"
    assert comparison.ai_draft.human_review_required is True


def test_reports_are_owner_scoped_and_cannot_be_overwritten(service, owner) -> None:
    _case(service, owner)
    _task(service, owner)
    values = dict(
        report_id="report-1",
        case_id="case-1",
        task_id="task-1",
        report=_complete_report(),
        state="completed",
    )
    service.save_report(owner, **values)

    with pytest.raises(OwnershipError):
        service.get_report(TrustedPrincipal(subject="auth0|other"), "report-1")
    with pytest.raises(ImmutableRecordError):
        service.save_report(owner, **values)


def test_agent_logs_are_sanitized_and_reject_raw_evidence(service, owner) -> None:
    _case(service, owner)
    _task(service, owner)
    event = service.record_agent_event(
        owner,
        task_id="task-1",
        agent_name="retrieval",
        stage_name="bm25",
        status="completed",
        safe_metadata={"document_count": 5, "duration_ms": 12},
    )

    assert event.safe_metadata == {"document_count": 5, "duration_ms": 12}
    loaded_event = service.list_agent_events(owner, "task-1")[0]
    _assert_aware_utc(loaded_event.created_at)
    assert "private evidence" not in json.dumps(event.model_dump(mode="json")).casefold()
    with pytest.raises(InvalidRecordError):
        service.record_agent_event(
            owner,
            task_id="task-1",
            agent_name="analysis",
            stage_name="facts",
            status="failed",
            safe_metadata={"evidence": "private evidence"},
        )


def test_missing_records_are_controlled(service, owner) -> None:
    with pytest.raises(MissingRecordError, match="case was not found"):
        service.get_case(owner, "missing-case")
    with pytest.raises(MissingRecordError, match="task was not found"):
        service.get_task(owner, "missing-task")


def test_transaction_rolls_back_related_writes(database_url, owner) -> None:
    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    sessions = build_session_factory(engine)

    with pytest.raises(DuplicateRecordError):
        with sessions.begin() as session:
            repository = CaseLensRepository(session)
            _ = repository.create_case(owner, case_id="case-rollback", title="First")
            _ = repository.create_case(owner, case_id="case-rollback", title="Duplicate")

    persistence = PersistenceService(sessions)
    with pytest.raises(MissingRecordError):
        persistence.get_case(owner, "case-rollback")
    engine.dispose()


def test_persistence_survives_service_reinitialization(database_url, owner) -> None:
    first_engine = build_engine(database_url)
    Base.metadata.create_all(first_engine)
    first = PersistenceService(build_session_factory(first_engine))
    _case(first, owner)
    first_engine.dispose()

    second_engine = build_engine(database_url)
    second = PersistenceService(build_session_factory(second_engine))
    assert second.get_case(owner, "case-1").title == "Synthetic tenancy matter"
    second_engine.dispose()


def test_persistence_never_calls_agents_or_providers(service, owner, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("external execution is forbidden")

    monkeypatch.setattr(AnalysisAgent, "analyze", forbidden)
    monkeypatch.setattr(VerificationAgent, "verify", forbidden)
    monkeypatch.setattr(GeminiSummarizationProvider, "generate", forbidden)

    _case(service, owner)
    _task(service, owner)
    assert service.get_task(owner, "task-1").status == "queued"
