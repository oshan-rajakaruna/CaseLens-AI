"""Contracts for explicit, non-substantive fact-to-authority links."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from agents.analysis.schemas import EvidenceSource


class FactAuthoritySelection(BaseModel):
    """One caller-selected fact and authority pair; no relationship is inferred."""

    analysis_document_id: str = Field(min_length=1)
    fact_index: int = Field(ge=0, strict=True)
    context_id: str = Field(pattern=r"^CTX-\d{3,}$")


class ComparisonFactReference(BaseModel):
    """Input-local reference to one exact extracted fact."""

    analysis_document_id: str
    fact_index: int = Field(ge=0)
    statement: str
    source: EvidenceSource
    epistemic_status: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    verification_status: str


class ComparisonAuthorityReference(BaseModel):
    """Validated legal-authority identity without retained passage text."""

    context_id: str = Field(pattern=r"^CTX-\d{3,}$")
    document_id: str
    title: str | None = None
    court: str | None = None
    source: str | None = None
    official_citation: str | None = None
    source_url: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    provenance: str | None = None


class ComparisonEvidenceMetadata(BaseModel):
    """Private-evidence identity supplied to generation without its excerpt."""

    document_id: str
    page: int | None = Field(default=None, gt=0)
    paragraph: int | None = Field(default=None, ge=0)
    locator: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_character_range(self) -> "ComparisonEvidenceMetadata":
        if self.end_char is not None and self.start_char is None:
            raise ValueError("start_char is required when end_char is supplied")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must not precede start_char")
        return self


class LegalComparisonGenerationRequest(BaseModel):
    """One already-selected fact and authority passage for a provider call."""

    comparison_id: str = Field(pattern=r"^CMP-\d{3,}$")
    context_id: str = Field(pattern=r"^CTX-\d{3,}$")
    fact_statement: str = Field(min_length=1)
    fact_epistemic_status: str
    fact_confidence: float | None = Field(default=None, ge=0, le=1)
    fact_verification_status: str
    evidence: ComparisonEvidenceMetadata
    authority_passage: str = Field(min_length=1)
    authority: ComparisonAuthorityReference
    limitations: list[str] = Field(default_factory=list)


class LegalComparisonProviderOutput(BaseModel):
    """Schema-constrained provider output before orchestration checks."""

    comparison_id: str = Field(pattern=r"^CMP-\d{3,}$")
    context_id: str = Field(pattern=r"^CTX-\d{3,}$")
    explanation: str = Field(min_length=1)
    uncertainties: list[str] = Field(min_length=1)
    ai_generated: Literal[True]
    human_review_required: Literal[True]

    @field_validator("explanation")
    @classmethod
    def explanation_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explanation must contain text")
        return value

    @field_validator("uncertainties")
    @classmethod
    def uncertainties_must_contain_text(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("uncertainties must contain text")
        return values


class LegalComparisonDraft(LegalComparisonProviderOutput):
    """Unverified AI interpretation attached to a deterministic source link."""

    verification_status: Literal["not_verified"] = "not_verified"
    provider: str
    model: str | None = None


class LegalComparisonRecord(BaseModel):
    """Explicit source link requiring review, not a legal applicability finding."""

    id: str = Field(pattern=r"^CMP-\d{3,}$")
    fact: ComparisonFactReference
    authority: ComparisonAuthorityReference
    origin: Literal["explicit_selection"] = "explicit_selection"
    relationship: str
    review_status: Literal["requires_human_review"] = "requires_human_review"
    limitations: list[str] = Field(default_factory=list)
    ai_draft: LegalComparisonDraft | None = None
