"""Owner-neutral public contracts for authenticated case and report APIs."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from typing import Annotated

from backend.schemas.legal_report import LegalResearchReport
from database.schemas import CaseStatus, ReportState, TaskStatus

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CaseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: NonEmptyText = Field(max_length=500)
    description: str | None = Field(default=None, max_length=20_000)
    research_question: str | None = Field(default=None, max_length=20_000)


class CaseResponse(BaseModel):
    id: str
    title: str
    description: str | None
    research_question: str | None
    status: CaseStatus
    created_at: datetime
    updated_at: datetime


class CaseListResponse(BaseModel):
    items: list[CaseResponse]
    offset: int
    limit: int


class CaseDocumentResponse(BaseModel):
    id: str
    case_id: str
    safe_filename: str
    media_type: str
    uploaded_at: datetime


class CaseDocumentListResponse(BaseModel):
    items: list[CaseDocumentResponse]


class ResearchStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: NonEmptyText = Field(max_length=20_000)
    document_ids: list[str] = Field(min_length=1, max_length=50)
    legal_issue: NonEmptyText | None = Field(default=None, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=20, strict=True)

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 128 for value in values):
            raise ValueError("document IDs must be non-empty safe identifiers")
        if len(values) != len(set(values)):
            raise ValueError("document IDs must be unique")
        return values


class ResearchTaskResponse(BaseModel):
    id: str
    case_id: str
    status: TaskStatus
    stages: dict[str, str]
    error_category: str | None
    report_id: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class LegalReportResponse(BaseModel):
    id: str
    case_id: str
    task_id: str
    state: ReportState
    report: LegalResearchReport
    created_at: datetime


class LegalReportListResponse(BaseModel):
    items: list[LegalReportResponse]
