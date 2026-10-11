"""Contracts for explicit, non-substantive fact-to-authority links."""

from typing import Literal

from pydantic import BaseModel, Field

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


class LegalComparisonRecord(BaseModel):
    """Explicit source link requiring review, not a legal applicability finding."""

    id: str = Field(pattern=r"^CMP-\d{3,}$")
    fact: ComparisonFactReference
    authority: ComparisonAuthorityReference
    origin: Literal["explicit_selection"] = "explicit_selection"
    relationship: str
    review_status: Literal["requires_human_review"] = "requires_human_review"
    limitations: list[str] = Field(default_factory=list)
