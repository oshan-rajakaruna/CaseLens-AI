"""Internal contracts for one evidence-grounded legal research workflow."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.analysis.schemas import AnalysisResponse
from agents.coordinator.schemas import SummarizationReadyPayload
from backend.schemas.common import Case, Document
from backend.schemas.legal_comparison import FactAuthoritySelection
from backend.schemas.legal_report import LegalResearchReport
from backend.schemas.retrieval import RetrievalFilters, RetrievalMode
from backend.schemas.summarization import SummarizationVerifyResponse

WorkflowStatus = Literal["completed", "partial", "failed"]
WorkflowStageStatus = Literal[
    "not_started", "completed", "partial", "failed", "skipped"
]


class EvidenceDocumentInput(BaseModel):
    """Authorized document text supplied internally; never loaded by the workflow."""

    model_config = ConfigDict(extra="forbid")
    document: Document
    text: str
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    segments: list[Any] = Field(default_factory=list)


class ResearchWorkflowRequest(BaseModel):
    """Validated input for a single internal research execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    task_id: str = Field(min_length=1)
    case: Case
    query: str = Field(min_length=1)
    evidence_documents: list[EvidenceDocumentInput] = Field(default_factory=list)
    legal_issue: str | None = None
    top_k: int = Field(default=5, ge=1, strict=True)
    filters: RetrievalFilters | None = None
    retrieval_mode: RetrievalMode = "bm25"
    diversify: bool = True
    max_chunks_per_document: int | None = Field(default=None, ge=1, strict=True)
    rag_max_context_tokens: int | None = Field(default=None, ge=1, strict=True)
    rag_max_context_passages: int | None = Field(default=None, ge=1, strict=True)
    rag_min_retrieval_score: float | None = None
    comparisons: list[FactAuthoritySelection] = Field(default_factory=list)
    generate_comparison_drafts: bool = False
    render_markdown: bool = False

    @model_validator(mode="after")
    def validate_internal_identities(self) -> "ResearchWorkflowRequest":
        if not self.task_id.strip():
            raise ValueError("task_id must contain text")
        if not self.case.id.strip():
            raise ValueError("case.id must contain text")
        if not self.query.strip():
            raise ValueError("query must contain text")
        if self.legal_issue is not None and not self.legal_issue.strip():
            raise ValueError("legal_issue must contain text when supplied")
        document_ids: list[str] = []
        for item in self.evidence_documents:
            if item.document.case_id != self.case.id:
                raise ValueError("evidence document does not belong to the case")
            document_ids.append(item.document.id)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("evidence document IDs must be unique")
        if self.generate_comparison_drafts and not self.comparisons:
            raise ValueError(
                "comparison drafts require explicit fact-authority selections"
            )
        return self


class ResearchWorkflowStages(BaseModel):
    analysis: WorkflowStageStatus = "not_started"
    retrieval: WorkflowStageStatus = "not_started"
    summarization_verification: WorkflowStageStatus = "not_started"
    report: WorkflowStageStatus = "not_started"
    comparisons: WorkflowStageStatus = "not_started"
    markdown: WorkflowStageStatus = "not_started"


class ResearchWorkflowError(BaseModel):
    """Sanitized failure information without evidence or provider details."""

    stage: Literal[
        "analysis",
        "retrieval",
        "summarization_verification",
        "report",
        "comparisons",
        "markdown",
    ]
    code: str
    message: str


class ResearchWorkflowResult(BaseModel):
    """Complete or partial output from one internal workflow execution."""

    task_id: str
    status: WorkflowStatus
    stages: ResearchWorkflowStages
    analyses: list[AnalysisResponse] = Field(default_factory=list)
    retrieval_payload: SummarizationReadyPayload | None = None
    summary_verification: SummarizationVerifyResponse | None = None
    report: LegalResearchReport | None = None
    markdown: str | None = None
    errors: list[ResearchWorkflowError] = Field(default_factory=list)
