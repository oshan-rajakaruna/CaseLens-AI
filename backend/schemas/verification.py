"""Shared contracts for verification inputs and reviewable decisions."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, computed_field, model_validator

from backend.schemas.common import Citation, VerificationResult


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ClaimStatus = Literal["supported", "partially_supported", "unsupported", "uncertain"]
CitationStatus = Literal["valid", "invalid", "missing", "uncertain"]
OverallVerificationStatus = Literal[
    "supported", "partially_supported", "unsupported", "uncertain", "invalid_citations"
]
WarningSeverity = Literal["info", "caution", "critical"]


class LegalClaim(BaseModel):
    """One generated legal assertion and the citations attached to it."""

    id: NonEmptyText
    text: NonEmptyText
    citations: list[Citation] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    """Identity and provenance of a retrieved source document."""

    document_id: NonEmptyText
    title: NonEmptyText | None = None
    court: str | None = None
    jurisdiction: str | None = None
    decision_date: date | None = None
    source_uri: str | None = None


class RetrievedEvidence(BaseModel):
    """A source passage available for checking a claim."""

    id: NonEmptyText
    source: SourceMetadata
    text: NonEmptyText
    locator: str | None = None


class CitationValidationResult(BaseModel):
    """Outcome of checking one citation against retrieved source material."""

    citation: Citation | None
    status: CitationStatus
    evidence_id: NonEmptyText | None = None
    source: SourceMetadata | None = None
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_status_fields(self) -> "CitationValidationResult":
        if self.status == "missing":
            if self.citation is not None or self.evidence_id is not None or self.source is not None:
                raise ValueError("A missing citation cannot have a citation or matched source")
        elif self.citation is None:
            raise ValueError("A non-missing citation result requires a citation")
        if self.status == "valid":
            if self.evidence_id is None or self.source is None:
                raise ValueError("A valid citation requires matched evidence and source metadata")
            if self.source.title is None:
                raise ValueError("A valid citation requires a source title")
            if self.citation.document_id != self.source.document_id:
                raise ValueError("Citation document ID must match the source document ID")
        return self


class ClaimVerificationResult(BaseModel):
    """Evidence support and citation checks for a single claim."""

    claim: LegalClaim
    status: ClaimStatus
    evidence_ids: list[NonEmptyText] = Field(default_factory=list)
    citation_results: list[CitationValidationResult] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def missing_evidence(self) -> bool:
        return not self.evidence_ids

    @computed_field
    @property
    def missing_citations(self) -> bool:
        return not self.claim.citations or any(
            result.status == "missing" for result in self.citation_results
        )

    @model_validator(mode="after")
    def check_evidence_for_status(self) -> "ClaimVerificationResult":
        if not self.evidence_ids and self.status != "uncertain":
            raise ValueError("A claim without reviewed evidence must be uncertain")
        return self


class ResponsibleAIWarning(BaseModel):
    """A user-facing qualification or concern about a verification outcome."""

    code: NonEmptyText
    message: NonEmptyText
    severity: WarningSeverity
    claim_ids: list[NonEmptyText] = Field(default_factory=list)


class VerificationRequest(BaseModel):
    """Claims to check and the actual retrieved passages available to check them."""

    task_id: NonEmptyText
    claims: list[LegalClaim] = Field(min_length=1)
    evidence: list[RetrievedEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_unique_ids(self) -> "VerificationRequest":
        if len({claim.id for claim in self.claims}) != len(self.claims):
            raise ValueError("Claim IDs must be unique")
        if len({item.id for item in self.evidence}) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique")
        return self


class VerificationResponse(VerificationResult):
    """Detailed result that remains usable as the legacy VerificationResult."""

    task_id: NonEmptyText
    overall_status: OverallVerificationStatus
    claim_results: list[ClaimVerificationResult] = Field(default_factory=list)
    warnings: list[ResponsibleAIWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_verified_decision(self) -> "VerificationResponse":
        if not self.verified:
            return self
        if self.overall_status != "supported" or not self.claim_results:
            raise ValueError("A verified response requires supported, reviewed claims")
        for result in self.claim_results:
            if result.status != "supported" or result.missing_evidence or result.missing_citations:
                raise ValueError("Every verified claim needs evidence and citations")
            remaining_citations = result.claim.citations.copy()
            if len(result.citation_results) != len(remaining_citations):
                raise ValueError("Every citation in a verified claim must be validated")
            for check in result.citation_results:
                if (
                    check.status != "valid"
                    or check.citation not in remaining_citations
                    or check.evidence_id not in result.evidence_ids
                ):
                    raise ValueError("Every citation must match reviewed evidence for its claim")
                remaining_citations.remove(check.citation)
            if any(citation not in self.citations for citation in result.claim.citations):
                raise ValueError("Verified claim citations must appear in the shared result")
        return self
