"""Session-scoped repository; callers provide transaction ownership."""

from pathlib import PurePath
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.schemas.legal_report import LegalResearchReport
from database.errors import (
    DuplicateRecordError,
    ImmutableRecordError,
    InvalidRecordError,
    InvalidStateTransitionError,
    MissingRecordError,
    OwnershipError,
)
from database.models import (
    AgentLogModel,
    CaseDocumentModel,
    CaseModel,
    LegalReportModel,
    ResearchTaskModel,
    utc_now,
)
from database.schemas import (
    AgentLogRecord,
    CaseDocumentRecord,
    CaseRecord,
    CaseStatus,
    LegalReportRecord,
    ReportState,
    ResearchTaskRecord,
    TaskStatus,
    TrustedPrincipal,
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_METADATA_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_STAGE_STATUSES = {"pending", "running", "completed", "partial", "failed", "skipped", "unavailable"}
_LOG_STATUSES = _STAGE_STATUSES | {"queued"}
_TERMINAL_TASK_STATUSES = {"partial", "completed", "failed"}
_TASK_TRANSITIONS = {
    "queued": {"running", "partial", "completed", "failed"},
    "running": {"partial", "completed", "failed"},
}
_ALLOWED_LOG_KEYS = {
    "attempt", "context_count", "document_count", "duration_ms", "item_count",
    "model", "provider", "retryable",
}


class CaseLensRepository:
    """CRUD operations constrained by a server-established principal."""

    def __init__(self, session: Session):
        self._session = session

    def create_case(
        self,
        principal: TrustedPrincipal,
        *,
        case_id: str,
        title: str,
        description: str | None = None,
        research_question: str | None = None,
        status: CaseStatus = "active",
    ) -> CaseRecord:
        _validate_identity(case_id, "case_id")
        if self._session.get(CaseModel, case_id) is not None:
            raise DuplicateRecordError("case identity already exists")
        model = CaseModel(
            id=case_id,
            owner_subject=principal.subject,
            title=_required_text(title, "title", 500),
            description=description,
            research_question=research_question,
            status=status,
        )
        self._session.add(model)
        self._session.flush()
        return CaseRecord.model_validate(model)

    def get_case(self, principal: TrustedPrincipal, case_id: str) -> CaseRecord:
        return CaseRecord.model_validate(self._owned_case(principal, case_id))

    def list_cases(
        self,
        principal: TrustedPrincipal,
        *,
        offset: int = 0,
        limit: int = 25,
    ) -> list[CaseRecord]:
        if offset < 0 or limit < 1 or limit > 100:
            raise InvalidRecordError("case pagination is invalid")
        models = self._session.scalars(
            select(CaseModel)
            .where(CaseModel.owner_subject == principal.subject)
            .order_by(CaseModel.created_at.desc(), CaseModel.id)
            .offset(offset)
            .limit(limit)
        )
        return [CaseRecord.model_validate(model) for model in models]

    def add_document(
        self,
        principal: TrustedPrincipal,
        *,
        document_id: str,
        case_id: str,
        safe_filename: str,
        media_type: str,
        storage_key: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> CaseDocumentRecord:
        case = self._owned_case(principal, case_id)
        _validate_identity(document_id, "document_id")
        if self._session.get(CaseDocumentModel, document_id) is not None:
            raise DuplicateRecordError("document identity already exists")
        _validate_filename(safe_filename)
        _validate_storage_key(storage_key)
        model = CaseDocumentModel(
            id=document_id,
            case_id=case.id,
            owner_subject=principal.subject,
            safe_filename=safe_filename,
            media_type=_required_text(media_type, "media_type", 255),
            storage_key=storage_key,
            source_metadata=source_metadata or {},
        )
        self._session.add(model)
        self._session.flush()
        return CaseDocumentRecord.model_validate(model)

    def get_document(
        self,
        principal: TrustedPrincipal,
        case_id: str,
        document_id: str,
    ) -> CaseDocumentRecord:
        self._owned_case(principal, case_id)
        model = self._session.get(CaseDocumentModel, document_id)
        if model is None or model.case_id != case_id:
            raise MissingRecordError("document was not found")
        _require_owner(principal, model.owner_subject)
        return CaseDocumentRecord.model_validate(model)

    def list_documents(
        self,
        principal: TrustedPrincipal,
        case_id: str,
    ) -> list[CaseDocumentRecord]:
        self._owned_case(principal, case_id)
        models = self._session.scalars(
            select(CaseDocumentModel)
            .where(
                CaseDocumentModel.case_id == case_id,
                CaseDocumentModel.owner_subject == principal.subject,
            )
            .order_by(CaseDocumentModel.uploaded_at, CaseDocumentModel.id)
        )
        return [CaseDocumentRecord.model_validate(model) for model in models]

    def create_task(
        self,
        principal: TrustedPrincipal,
        *,
        task_id: str,
        case_id: str,
        request_fingerprint: str,
        stages: dict[str, str] | None = None,
    ) -> ResearchTaskRecord:
        case = self._owned_case(principal, case_id)
        _validate_identity(task_id, "task_id")
        _validate_identity(request_fingerprint, "request_fingerprint")
        _validate_stages(stages or {})
        if self._session.get(ResearchTaskModel, task_id) is not None:
            raise DuplicateRecordError("task identity already exists")
        existing = self._session.scalar(
            select(ResearchTaskModel).where(
                ResearchTaskModel.owner_subject == principal.subject,
                ResearchTaskModel.request_fingerprint == request_fingerprint,
            )
        )
        if existing is not None:
            raise DuplicateRecordError("task request fingerprint already exists")
        model = ResearchTaskModel(
            id=task_id,
            case_id=case.id,
            owner_subject=principal.subject,
            request_fingerprint=request_fingerprint,
            stages=stages or {},
        )
        self._session.add(model)
        self._session.flush()
        return ResearchTaskRecord.model_validate(model)

    def get_task(self, principal: TrustedPrincipal, task_id: str) -> ResearchTaskRecord:
        return ResearchTaskRecord.model_validate(self._owned_task(principal, task_id))

    def update_task(
        self,
        principal: TrustedPrincipal,
        task_id: str,
        *,
        status: TaskStatus,
        stages: dict[str, str] | None = None,
        error_category: str | None = None,
    ) -> ResearchTaskRecord:
        model = self._owned_task(principal, task_id)
        if status == model.status:
            raise InvalidStateTransitionError("task status must advance")
        if status not in _TASK_TRANSITIONS.get(model.status, set()):
            raise InvalidStateTransitionError("task status transition is not permitted")
        if stages is not None:
            _validate_stages(stages)
            model.stages = dict(stages)
        model.error_category = _optional_safe_token(error_category, "error_category")
        model.status = status
        model.updated_at = utc_now()
        if status in _TERMINAL_TASK_STATUSES:
            model.completed_at = model.updated_at
        self._session.flush()
        return ResearchTaskRecord.model_validate(model)

    def save_report(
        self,
        principal: TrustedPrincipal,
        *,
        report_id: str,
        case_id: str,
        task_id: str,
        report: LegalResearchReport,
        state: ReportState,
    ) -> LegalReportRecord:
        case = self._owned_case(principal, case_id)
        task = self._owned_task(principal, task_id)
        _validate_identity(report_id, "report_id")
        if task.case_id != case.id:
            raise InvalidRecordError("task does not belong to the case")
        if report.case_information.case.id != case.id:
            raise InvalidRecordError("report does not belong to the case")
        if self._session.get(LegalReportModel, report_id) is not None:
            raise ImmutableRecordError("report identity already exists")
        existing = self._session.scalar(
            select(LegalReportModel).where(LegalReportModel.task_id == task.id)
        )
        if existing is not None:
            raise ImmutableRecordError("a report already exists for this task")
        model = LegalReportModel(
            id=report_id,
            case_id=case.id,
            task_id=task.id,
            owner_subject=principal.subject,
            state=state,
            report_payload=report.model_dump(mode="json"),
        )
        self._session.add(model)
        self._session.flush()
        return _report_record(model)

    def get_report(self, principal: TrustedPrincipal, report_id: str) -> LegalReportRecord:
        model = self._session.get(LegalReportModel, report_id)
        if model is None:
            raise MissingRecordError("report was not found")
        _require_owner(principal, model.owner_subject)
        case = self._owned_case(principal, model.case_id)
        task = self._owned_task(principal, model.task_id)
        if task.case_id != case.id:
            raise InvalidRecordError("report relationships are inconsistent")
        return _report_record(model)

    def find_reports_by_case(self, principal: TrustedPrincipal, case_id: str) -> list[LegalReportRecord]:
        self._owned_case(principal, case_id)
        models = list(self._session.scalars(
            select(LegalReportModel).where(
                LegalReportModel.case_id == case_id,
                LegalReportModel.owner_subject == principal.subject,
            ).order_by(LegalReportModel.created_at)
        ))
        for model in models:
            task = self._owned_task(principal, model.task_id)
            if task.case_id != case_id:
                raise InvalidRecordError("report relationships are inconsistent")
        return [_report_record(model) for model in models]

    def find_report_by_task(self, principal: TrustedPrincipal, task_id: str) -> LegalReportRecord | None:
        self._owned_task(principal, task_id)
        model = self._session.scalar(
            select(LegalReportModel).where(
                LegalReportModel.task_id == task_id,
                LegalReportModel.owner_subject == principal.subject,
            )
        )
        return None if model is None else _report_record(model)

    def find_task_by_fingerprint(
        self,
        principal: TrustedPrincipal,
        case_id: str,
        request_fingerprint: str,
    ) -> ResearchTaskRecord | None:
        self._owned_case(principal, case_id)
        model = self._session.scalar(
            select(ResearchTaskModel).where(
                ResearchTaskModel.owner_subject == principal.subject,
                ResearchTaskModel.case_id == case_id,
                ResearchTaskModel.request_fingerprint == request_fingerprint,
            )
        )
        return None if model is None else ResearchTaskRecord.model_validate(model)

    def finalize_task_with_report(
        self,
        principal: TrustedPrincipal,
        *,
        report_id: str,
        case_id: str,
        task_id: str,
        report: LegalResearchReport,
        state: ReportState,
        task_status: TaskStatus,
        stages: dict[str, str],
        error_category: str | None = None,
    ) -> tuple[LegalReportRecord, ResearchTaskRecord]:
        if task_status not in {"completed", "partial"}:
            raise InvalidStateTransitionError(
                "a persisted report requires a completed or partial task"
            )
        report_record = self.save_report(
            principal,
            report_id=report_id,
            case_id=case_id,
            task_id=task_id,
            report=report,
            state=state,
        )
        task_record = self.update_task(
            principal,
            task_id,
            status=task_status,
            stages=stages,
            error_category=error_category,
        )
        return report_record, task_record

    def record_agent_event(
        self,
        principal: TrustedPrincipal,
        *,
        task_id: str,
        agent_name: str,
        stage_name: str,
        status: str,
        error_category: str | None = None,
        safe_metadata: dict[str, Any] | None = None,
    ) -> AgentLogRecord:
        task = self._owned_task(principal, task_id)
        if status not in _LOG_STATUSES:
            raise InvalidRecordError("agent log status is not permitted")
        metadata = _validate_safe_metadata(safe_metadata or {})
        model = AgentLogModel(
            task_id=task.id,
            owner_subject=principal.subject,
            agent_name=_safe_token(agent_name, "agent_name"),
            stage_name=_safe_token(stage_name, "stage_name"),
            status=status,
            error_category=_optional_safe_token(error_category, "error_category"),
            safe_metadata=metadata,
        )
        self._session.add(model)
        self._session.flush()
        return AgentLogRecord.model_validate(model)

    def list_agent_events(self, principal: TrustedPrincipal, task_id: str) -> list[AgentLogRecord]:
        self._owned_task(principal, task_id)
        models = self._session.scalars(
            select(AgentLogModel).where(
                AgentLogModel.task_id == task_id,
                AgentLogModel.owner_subject == principal.subject,
            ).order_by(AgentLogModel.id)
        )
        return [AgentLogRecord.model_validate(model) for model in models]

    def _owned_case(self, principal: TrustedPrincipal, case_id: str) -> CaseModel:
        model = self._session.get(CaseModel, case_id)
        if model is None:
            raise MissingRecordError("case was not found")
        _require_owner(principal, model.owner_subject)
        return model

    def _owned_task(self, principal: TrustedPrincipal, task_id: str) -> ResearchTaskModel:
        model = self._session.get(ResearchTaskModel, task_id)
        if model is None:
            raise MissingRecordError("task was not found")
        _require_owner(principal, model.owner_subject)
        return model


def _report_record(model: LegalReportModel) -> LegalReportRecord:
    return LegalReportRecord(
        id=model.id,
        case_id=model.case_id,
        task_id=model.task_id,
        owner_subject=model.owner_subject,
        state=model.state,
        report=LegalResearchReport.model_validate(model.report_payload),
        created_at=model.created_at,
    )


def _require_owner(principal: TrustedPrincipal, owner_subject: str) -> None:
    if principal.subject != owner_subject:
        raise OwnershipError("record is not available to this principal")


def _validate_identity(value: str, field: str) -> None:
    _safe_token(value, field)


def _safe_token(value: str, field: str) -> str:
    if not _SAFE_TOKEN.fullmatch(value):
        raise InvalidRecordError(f"{field} is not a safe identifier")
    return value


def _optional_safe_token(value: str | None, field: str) -> str | None:
    return None if value is None else _safe_token(value, field)


def _required_text(value: str, field: str, maximum: int) -> str:
    if not value.strip() or len(value) > maximum:
        raise InvalidRecordError(f"{field} is invalid")
    return value


def _validate_filename(value: str) -> None:
    if (
        not value
        or len(value) > 255
        or "/" in value
        or "\\" in value
        or PurePath(value).name != value
        or value in {".", ".."}
    ):
        raise InvalidRecordError("safe_filename must be a basename")


def _validate_storage_key(value: str) -> None:
    normalized = value.replace("\\", "/")
    if not value or len(value) > 1024 or normalized.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise InvalidRecordError("storage_key must be a relative opaque key")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise InvalidRecordError("storage_key contains an unsafe path segment")


def _validate_stages(stages: dict[str, str]) -> None:
    for name, status in stages.items():
        _safe_token(name, "stage_name")
        if status not in _STAGE_STATUSES:
            raise InvalidRecordError("stage status is not permitted")


def _validate_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if len(metadata) > 20:
        raise InvalidRecordError("safe_metadata has too many entries")
    checked: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = key.casefold()
        if normalized not in _ALLOWED_LOG_KEYS:
            raise InvalidRecordError("safe_metadata field is not allowlisted")
        _safe_token(key, "safe_metadata key")
        if not isinstance(value, (bool, int, float, str)):
            raise InvalidRecordError("safe_metadata values must be short scalar values")
        if isinstance(value, str) and not _SAFE_METADATA_VALUE.fullmatch(value):
            raise InvalidRecordError("safe_metadata string is not an operational token")
        checked[key] = value
    return checked
