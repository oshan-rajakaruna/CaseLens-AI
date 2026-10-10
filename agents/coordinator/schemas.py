"""Coordinator-specific request, retrieval-context, and response envelopes."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.schemas import AgentResult, AgentTask, VerificationResponse, VerificationResult
from backend.schemas.retrieval import RetrievalFilters, RetrievalMode
from retrieval.rag import RAGContextPassage


class CoordinatorRequest(BaseModel):
    """Request accepted by a future coordinator implementation."""

    task: AgentTask


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
