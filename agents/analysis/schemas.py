"""Typed, analysis-specific contracts for the Analysis Agent foundation."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from backend.schemas import Case, Document


def _require_identifier(value: str, field_name: str) -> str:
    """Reject empty identifiers while preserving valid identifier values."""

    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class EvidenceSource(BaseModel):
    """Trace an extracted item to an exact, supplied document location."""

    document_id: str = Field(min_length=1)
    page: int | None = Field(default=None, gt=0)
    paragraph: int | None = Field(default=None, ge=0)
    locator: str | None = Field(default=None, min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    excerpt: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> "EvidenceSource":
        _require_identifier(self.document_id, "document_id")
        if self.end_char is not None and self.start_char is None:
            raise ValueError("start_char is required when end_char is supplied")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must not precede start_char")
        return self


class ExtractedEntity(BaseModel):
    """A future named-entity extraction result."""

    text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    source: EvidenceSource
    method: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class ExtractedFact(BaseModel):
    """A future source-grounded factual proposition."""

    statement: str = Field(min_length=1)
    source: EvidenceSource
    method: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    category: str | None = Field(default=None, min_length=1)
    epistemic_status: str = "stated"
    entity_texts: list[str] = Field(default_factory=list)
    event_date: str | None = None
    money_text: str | None = None
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    verification_status: str = "not_independently_verified"


class ExtractedClause(BaseModel):
    """A future important clause, condition, or obligation."""

    text: str = Field(min_length=1)
    clause_type: str | None = Field(default=None, min_length=1)
    source: EvidenceSource
    method: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    epistemic_status: str = "stated"
    party_texts: list[str] = Field(default_factory=list)
    money_text: str | None = None
    deadline_text: str | None = None
    is_conditional: bool = False


class LegalIssue(BaseModel):
    """A future legal issue label; its taxonomy remains intentionally open."""

    label: str = Field(min_length=1)
    description: str | None = None
    sources: list[EvidenceSource] = Field(default_factory=list)
    method: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    category_id: str | None = None
    review_status: str = "needs_review"
    epistemic_status: str = "uncertain"
    evidence_summary: str | None = None


class TimelineEvent(BaseModel):
    """A future chronological event grounded in supplied evidence."""

    description: str = Field(min_length=1)
    date_text: str | None = Field(default=None, min_length=1)
    normalized_date: str | None = Field(default=None, min_length=1)
    source: EvidenceSource
    method: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    event_id: str | None = None
    category: str | None = None
    date_precision: str | None = None
    temporal_role: str = "event_date"
    epistemic_status: str = "stated"
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    verification_status: str = "not_independently_verified"
    review_warning: str | None = None


class ContradictionCandidate(BaseModel):
    """A future review candidate, never a conclusive legal determination."""

    description: str = Field(min_length=1)
    sources: list[EvidenceSource] = Field(min_length=2)
    confidence: float | None = Field(default=None, ge=0, le=1)
    candidate_id: str | None = None
    category: str | None = None
    review_status: str = "needs_review"
    evidence_context: dict[str, Any] = Field(default_factory=dict)
    method: str = "rule-based-contradiction"


class AnalysisWarning(BaseModel):
    """Structured non-sensitive processing feedback."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    field: str | None = Field(default=None, min_length=1)


class AnalysisRequest(BaseModel):
    """A single supplied case document for local analysis processing."""

    case: Case
    document: Document
    text: str
    document_metadata: dict[str, Any] = Field(default_factory=dict)
    segments: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "AnalysisRequest":
        _require_identifier(self.case.id, "case.id")
        _require_identifier(self.document.id, "document.id")
        _require_identifier(self.document.case_id, "document.case_id")
        if self.document.case_id != self.case.id:
            raise ValueError("document.case_id must match case.id")
        return self


class AnalysisResponse(BaseModel):
    """Structured analysis output, including explicit foundation-stage status."""

    case_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    status: Literal["not_analyzed", "completed", "partial", "requires_input", "failed"]
    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)
    clauses: list[ExtractedClause] = Field(default_factory=list)
    legal_issues: list[LegalIssue] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    contradiction_candidates: list[ContradictionCandidate] = Field(default_factory=list)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    errors: list[AnalysisWarning] = Field(default_factory=list)
    stage_status: dict[str, str] = Field(default_factory=dict)
