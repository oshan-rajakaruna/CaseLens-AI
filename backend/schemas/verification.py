"""Shared contracts for verification inputs and reviewable decisions."""

from collections.abc import Sequence
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
    """Detailed result of deterministic checks, not a ruling on legal truth or authority."""

    verified: bool = Field(
        description=(
            "True only when every claim and citation passes the current deterministic "
            "checks; this does not establish legal truth, authority, or an outcome."
        )
    )
    task_id: NonEmptyText
    overall_status: OverallVerificationStatus = Field(
        description=(
            "Deterministic aggregate, prioritized as invalid citations, unsupported, "
            "uncertain, partially supported, then supported."
        )
    )
    claim_results: list[ClaimVerificationResult] = Field(default_factory=list)
    warnings: list[ResponsibleAIWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_verified_decision(self) -> "VerificationResponse":
        for result in self.claim_results:
            if result.status != "uncertain" and not _has_complete_valid_citations(result):
                raise ValueError("Decided claims require reviewed evidence and valid citations")
        expected_status = determine_overall_status(self.claim_results)
        if self.overall_status != expected_status:
            raise ValueError(f"Overall status must be {expected_status} for these claim results")
        if self.verified != (expected_status == "supported"):
            raise ValueError("Verified must match the deterministic overall status")
        if self.verified:
            for result in self.claim_results:
                if any(citation not in self.citations for citation in result.claim.citations):
                    raise ValueError("Verified claim citations must appear in the shared result")
        return self


def determine_overall_status(
    claim_results: Sequence[ClaimVerificationResult],
) -> OverallVerificationStatus:
    """Apply the agent's fixed priority to the recorded checks, without legal inference."""

    if any(
        check.status == "invalid"
        for result in claim_results
        for check in result.citation_results
    ):
        return "invalid_citations"
    if any(result.status == "unsupported" for result in claim_results):
        return "unsupported"
    if not claim_results or any(
        result.status == "uncertain" or not _has_complete_valid_citations(result)
        for result in claim_results
    ):
        return "uncertain"
    if any(result.status == "partially_supported" for result in claim_results):
        return "partially_supported"
    return "supported"


def _has_complete_valid_citations(result: ClaimVerificationResult) -> bool:
    if not result.evidence_ids or not result.claim.citations:
        return False
    remaining_citations = result.claim.citations.copy()
    if len(result.citation_results) != len(remaining_citations):
        return False
    for check in result.citation_results:
        if (
            check.status != "valid"
            or check.citation not in remaining_citations
            or check.evidence_id not in result.evidence_ids
        ):
            return False
        remaining_citations.remove(check.citation)
    return True
