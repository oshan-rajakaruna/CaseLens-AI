"""Coordinator-specific request, retrieval-context, and response envelopes."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agents.analysis.schemas import AnalysisResponse
from backend.schemas import (
    AgentResult,
    LegalClaim,
    RetrievedEvidence,
    VerificationResponse,
    VerificationResult,
)
from backend.schemas.retrieval import RetrievalFilters, RetrievalMode
from nlp.extraction import ExtractionSegment
from retrieval.rag import RAGContextPassage
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    SemanticSearchResult,
)


CoordinatorTask = Literal[
    "analyze_evidence",
    "retrieve_precedents",
    "summarize_evidence",
    "summarize_precedents",
    "verify_claims",
    "full_research",
]
CoordinatorStageStatus = Literal[
    "planned", "not_run", "completed", "partial", "requires_input", "failed"
]


class CoordinatorMessage(BaseModel):
    """A non-sensitive planning warning or prerequisite error."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage: str | None = Field(default=None, min_length=1)


class CoordinatorStage(BaseModel):
    """One proposed stage; Step 1 never records an execution result here."""

    agent: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    status: CoordinatorStageStatus
    result_reference: str | None = Field(default=None, min_length=1)
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)


class CoordinatorEvidenceDocument(BaseModel):
    """Coordinator-owned reference to supplied evidence for Analysis execution."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    text: str
    media_type: str | None = None
    document_metadata: dict[str, Any] = Field(default_factory=dict, max_length=50)
    segments: list[ExtractionSegment] = Field(default_factory=list)

    @field_validator("document_id", "name")
    @classmethod
    def identifier_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must contain non-whitespace text")
        return value


class CoordinatorRequest(BaseModel):
    """Validated input for deterministic, non-executing Coordinator planning."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    case_id: str = Field(min_length=1)
    case_title: str | None = Field(default=None, min_length=1, max_length=500)
    user_query: str = Field(min_length=1, max_length=2000)
    requested_task: CoordinatorTask
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20, strict=True)
    retrieval_filters: RetrievalFilters | None = None
    document_ids: list[str] = Field(default_factory=list, max_length=25)
    evidence_documents: list[CoordinatorEvidenceDocument] = Field(
        default_factory=list, max_length=25
    )
    evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    reference_ids: list[str] = Field(default_factory=list, max_length=100)
    claim_ids: list[str] = Field(default_factory=list, max_length=100)
    verification_claims: list[LegalClaim] = Field(default_factory=list, max_length=100)
    verification_evidence: list[RetrievedEvidence] = Field(default_factory=list, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=50)

    @field_validator("case_id", "user_query", "case_title")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("value must contain non-whitespace text")
        return value

    @field_validator("document_ids", "evidence_ids", "reference_ids", "claim_ids")
    @classmethod
    def identifiers_must_be_unique_and_nonblank(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("identifiers must contain non-whitespace text")
        if len(set(values)) != len(values):
            raise ValueError("identifiers must not contain duplicates")
        return values

    @field_validator("metadata")
    @classmethod
    def metadata_keys_must_be_nonblank(cls, value: dict[str, Any]) -> dict[str, Any]:
        if any(not key.strip() for key in value):
            raise ValueError("metadata keys must not be blank")
        return value

    @model_validator(mode="after")
    def evidence_documents_must_match_declared_identifiers(self) -> "CoordinatorRequest":
        evidence_ids = [document.document_id for document in self.evidence_documents]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence document identifiers must not contain duplicates")
        if evidence_ids and set(evidence_ids) != set(self.document_ids):
            raise ValueError(
                "evidence document identifiers must exactly match document_ids"
            )
        return self

    @model_validator(mode="after")
    def retrieval_filters_require_hybrid_mode(self) -> "CoordinatorRequest":
        if self.retrieval_filters is not None and self.retrieval_mode != "hybrid":
            raise ValueError("retrieval_filters are supported only in hybrid mode")
        return self


class CoordinatorPlanResponse(BaseModel):
    """A proposed route, not evidence that any agent has executed."""

    case_id: str = Field(min_length=1)
    requested_task: CoordinatorTask
    status: Literal["planned", "requires_input"]
    document_ids: list[str] = Field(default_factory=list)
    stages: list[CoordinatorStage] = Field(default_factory=list)
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool = False


CoordinatorRetrievalResult = (
    BM25SearchResult | SemanticSearchResult | HybridSearchResult
)


class CoordinatorRetrievalResponse(BaseModel):
    """Actual native Retrieval results retained for a future RAG handoff."""

    case_id: str = Field(min_length=1)
    requested_task: Literal["retrieve_precedents"]
    status: Literal["completed", "requires_input", "failed"]
    retrieval_mode: RetrievalMode
    top_k: int = Field(ge=1)
    stages: list[CoordinatorStage] = Field(default_factory=list)
    retrieval_results: list[CoordinatorRetrievalResult] = Field(default_factory=list)
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool


class CoordinatorAnalysisResponse(BaseModel):
    """Actual Analysis outputs retained without Coordinator-side transformation."""

    case_id: str = Field(min_length=1)
    requested_task: Literal["analyze_evidence"]
    status: Literal["completed", "partial", "requires_input", "failed"]
    document_ids: list[str] = Field(default_factory=list)
    stages: list[CoordinatorStage] = Field(default_factory=list)
    analysis_results: list[AnalysisResponse] = Field(default_factory=list)
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool


class CoordinatorResponse(BaseModel):
    """Aggregated coordinator output and optional verification result."""

    result: AgentResult
    verification: VerificationResponse | VerificationResult | None = None

    @field_validator("verification", mode="before")
    @classmethod
    def validate_detailed_verification(cls, value: object) -> object:
        if isinstance(value, dict) and any(
            key in value for key in ("task_id", "overall_status", "claim_results", "warnings")
        ):
            return VerificationResponse.model_validate(value)
        return value


class LegalRetrievalContextRequest(BaseModel):
    """Validated Coordinator request for retrieval-backed legal context."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query: str
    legal_issue: str | None = None
    top_k: int = Field(default=5, ge=1, strict=True)
    filters: RetrievalFilters | None = None
    retrieval_mode: RetrievalMode = "hybrid"
    bm25_weight: float | None = Field(default=None, ge=0)
    semantic_weight: float | None = Field(default=None, ge=0)
    diversify: bool = True
    max_chunks_per_document: int | None = Field(default=None, ge=1, strict=True)
    rag_max_context_tokens: int | None = Field(default=None, ge=1, strict=True)
    rag_max_context_passages: int | None = Field(default=None, ge=1, strict=True)
    rag_min_retrieval_score: float | None = None

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value

    @field_validator("legal_issue")
    @classmethod
    def legal_issue_must_contain_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("legal_issue must contain text when provided")
        return value

    @field_validator(
        "bm25_weight",
        "semantic_weight",
        "rag_min_retrieval_score",
        mode="before",
    )
    @classmethod
    def numeric_values_cannot_be_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("numeric configuration cannot be boolean")
        return value

    @model_validator(mode="after")
    def hybrid_weights_must_have_positive_total(
        self,
    ) -> "LegalRetrievalContextRequest":
        if (
            self.retrieval_mode == "hybrid"
            and self.bm25_weight is not None
            and self.semantic_weight is not None
            and self.bm25_weight + self.semantic_weight <= 0
        ):
            raise ValueError(
                "at least one hybrid retrieval weight must be greater than zero"
            )
        return self


class ContextCitationMapEntry(BaseModel):
    """Prompt-local citation identity with original legal provenance."""

    context_id: str
    document_id: str
    title: str | None = None
    court: str | None = None
    source: str | None = None
    chunk_ids: list[str]
    pages: list[int]
    official_citation: str | None = None
    source_url: str | None = None


class RetrievalContextMetadata(BaseModel):
    """Retrieval configuration and result count used for one context build."""

    mode: RetrievalMode
    top_k: int
    result_count: int = Field(ge=0)
    filters: dict[str, Any] | None = None
    bm25_weight: float | None = None
    semantic_weight: float | None = None
    diversified: bool
    max_chunks_per_document: int | None = None


class RetrievalContextDiagnostics(BaseModel):
    """Non-sensitive context assembly diagnostics for orchestration."""

    retrieval_result_count: int = Field(ge=0)
    rag_passage_count: int = Field(ge=0)
    source_document_count: int = Field(ge=0)
    estimated_context_tokens: int = Field(ge=0)
    context_truncated: bool
    empty_retrieval: bool
    no_context: bool
    weak_context_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SummarizationReadyPayload(BaseModel):
    """Evidence package ready for a future Summarization Agent call."""

    query: str
    legal_issue: str | None = None
    context_text: str
    context_passages: list[RAGContextPassage] = Field(default_factory=list)
    citation_map: dict[str, ContextCitationMapEntry] = Field(default_factory=dict)
    retrieval: RetrievalContextMetadata
    diagnostics: RetrievalContextDiagnostics
