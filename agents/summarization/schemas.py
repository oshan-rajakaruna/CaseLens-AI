"""Validated contracts for evidence-grounded legal summarization."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.schemas import Citation


class StrictModel(BaseModel):
    """Reject unexpected fields at the summarization boundary."""

    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    """Stable pointer to supplied source material."""

    document_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)

    @field_validator("document_id", "locator")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source identifiers cannot be blank")
        return value


class CaseInformation(StrictModel):
    """Case metadata supplied to the agent; it is not generated."""

    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None


class ExtractedFact(StrictModel):
    """A fact extracted upstream and its supporting source pointers."""

    fact_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class LegalPassage(StrictModel):
    """Retrieved legal text with an immutable source pointer."""

    text: str = Field(min_length=1)
    source: SourceReference
    authority: str | None = None


class SummarizationRequest(StrictModel):
    """Complete structured input accepted by the Summarization Agent."""

    case: CaseInformation
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    retrieved_passages: list[LegalPassage] = Field(default_factory=list)
    instruction: str | None = None


class EvidenceSummary(StrictModel):
    """Summary text grounded only in supplied evidence references."""

    text: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class PrecedentSummary(StrictModel):
    """Summary of supplied retrieved legal passages."""

    text: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class Comparison(StrictModel):
    """A comparison grounded in supplied evidence and/or precedent sources."""

    subject: str = Field(min_length=1)
    analysis: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)


class DraftClaim(StrictModel):
    """Generated draft text, explicitly marked as non-final."""

    text: str = Field(min_length=1)
    supporting_sources: list[SourceReference] = Field(min_length=1)
    status: Literal["draft"] = "draft"


class SummarizationResponse(StrictModel):
    """Structured model output; all source pointers must originate in input."""

    evidence_summaries: list[EvidenceSummary] = Field(default_factory=list)
    precedent_summaries: list[PrecedentSummary] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)
    draft_claims: list[DraftClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
