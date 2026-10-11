"""Deterministic contracts for an evidence-grounded legal research report."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.analysis.schemas import AnalysisResponse
from agents.coordinator.schemas import RetrievalContextDiagnostics
from agents.summarization.schemas import SummarizationResult
from backend.schemas.common import Case
from backend.schemas.verification import VerificationResponse
from retrieval.rag import ChunkRetrievalScore


AnalysisAvailability = Literal[
    "not_supplied",
    "requires_input",
    "partial",
    "completed",
    "mixed",
    "unavailable",
]


class ReportCaseInformation(BaseModel):
    """Supplied case identity and research question without inferred metadata."""

    case: Case
    query: str
    legal_issue: str | None = None


class ReportAnalysisFindings(BaseModel):
    """Unchanged Analysis Agent results and an explicit availability state."""

    supplied: bool
    availability: AnalysisAvailability
    results: list[AnalysisResponse] = Field(default_factory=list)


class ReportLegalAuthority(BaseModel):
    """Report-safe metadata for one cited prompt-local authority passage."""

    context_id: str
    document_id: str
    title: str | None = None
    court: str | None = None
    source: str | None = None
    source_url: str | None = None
    official_citation: str | None = None
    provenance: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    ranks: list[int] = Field(default_factory=list)
    retrieval_score: float
    retrieval_scores: list[ChunkRetrievalScore] = Field(default_factory=list)
    truncated: bool = False


class ReportVerificationFindings(BaseModel):
    """Unchanged verifier output plus the bridge's scope qualifications."""

    response: VerificationResponse
    bridge_warnings: list[str] = Field(default_factory=list)
    scope: str


class LegalResearchReport(BaseModel):
    """Structured research draft assembled without executing any agent."""

    case_information: ReportCaseInformation
    analysis: ReportAnalysisFindings
    authorities: list[ReportLegalAuthority] = Field(default_factory=list)
    summary: SummarizationResult
    verification: ReportVerificationFindings
    retrieval_diagnostics: RetrievalContextDiagnostics
    legal_comparison_status: Literal["not_assessed"] = "not_assessed"
    limitations: list[str] = Field(default_factory=list)
    human_review_disclaimer: str
