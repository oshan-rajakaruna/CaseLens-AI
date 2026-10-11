"""Owner-neutral public contracts for authenticated case and report APIs."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated

from backend.schemas.legal_report import LegalResearchReport
from database.schemas import CaseStatus, ReportState

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


class LegalReportResponse(BaseModel):
    id: str
    case_id: str
    task_id: str
    state: ReportState
    report: LegalResearchReport
    created_at: datetime


class LegalReportListResponse(BaseModel):
    items: list[LegalReportResponse]
