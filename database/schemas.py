"""Validated internal persistence inputs and detached return records."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas.legal_report import LegalResearchReport

CaseStatus = Literal["active", "closed", "archived"]
TaskStatus = Literal["queued", "running", "partial", "completed", "failed"]
ReportState = Literal["draft", "partial", "completed"]


class TrustedPrincipal(BaseModel):
    """Opaque subject established by a future trusted authentication layer."""
    subject: str = Field(min_length=1, max_length=255)

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("principal subject must not contain surrounding whitespace")
        return value


class _Record(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CaseRecord(_Record):
    id: str
    owner_subject: str
    title: str
    description: str | None
    research_question: str | None
    status: CaseStatus
    created_at: datetime
    updated_at: datetime


class CaseDocumentRecord(_Record):
    id: str
    case_id: str
    owner_subject: str
    safe_filename: str
    media_type: str
    storage_key: str
    source_metadata: dict[str, Any]
    uploaded_at: datetime


class ResearchTaskRecord(_Record):
    id: str
    case_id: str
    owner_subject: str
    request_fingerprint: str
    status: TaskStatus
    stages: dict[str, str]
    error_category: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class LegalReportRecord(_Record):
    id: str
    case_id: str
    task_id: str
    owner_subject: str
    state: ReportState
    report: LegalResearchReport
    created_at: datetime


class AgentLogRecord(_Record):
    id: int
    task_id: str
    owner_subject: str
    agent_name: str
    stage_name: str
    status: str
    error_category: str | None
    safe_metadata: dict[str, Any]
    created_at: datetime
