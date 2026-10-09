"""Validation and compatibility checks for shared verification contracts."""

import pytest
from pydantic import ValidationError

from agents.coordinator.schemas import CoordinatorResponse
from backend.schemas import (
    AgentResult,
    Citation,
    CitationValidationResult,
    ClaimVerificationResult,
    LegalClaim,
    ResponsibleAIWarning,
    RetrievedEvidence,
    SourceMetadata,
    VerificationRequest,
    VerificationResponse,
    VerificationResult,
)


def test_request_keeps_missing_evidence_and_citations_explicit() -> None:
    request = VerificationRequest(
        task_id="task-1", claims=[LegalClaim(id="claim-1", text="The rule applies.")]
    )

    assert request.evidence == []
    assert request.claims[0].citations == []
    with pytest.raises(ValidationError):
        VerificationRequest(task_id="task-1", claims=[])
    with pytest.raises(ValidationError):
        VerificationRequest(
            task_id="task-1",
            claims=[LegalClaim(id="claim-1", text="First"), LegalClaim(id="claim-1", text="Second")],
        )
    with pytest.raises(ValidationError):
        LegalClaim(id=" ", text="The rule applies.")


def test_citation_validation_requires_a_matched_source() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3")
    source = SourceMetadata(document_id="doc-1", title="Decision")
    result = CitationValidationResult(
        citation=citation, status="valid", evidence_id="passage-1", source=source
    )

    assert result.status == "valid"
    with pytest.raises(ValidationError):
        CitationValidationResult(citation=citation, status="valid")
    with pytest.raises(ValidationError):
        CitationValidationResult(
            citation=citation,
            status="valid",
            evidence_id="passage-1",
            source=SourceMetadata(document_id="doc-2", title="Another decision"),
        )
    with pytest.raises(ValidationError):
        CitationValidationResult(
            citation=citation,
            status="valid",
            evidence_id="passage-1",
            source=SourceMetadata(document_id="doc-1"),
        )
    assert CitationValidationResult(citation=None, status="missing").status == "missing"


def test_missing_evidence_cannot_support_a_claim() -> None:
    claim = LegalClaim(id="claim-1", text="The rule applies.")
    result = ClaimVerificationResult(claim=claim, status="uncertain")

    assert result.model_dump()["missing_evidence"] is True
    assert result.model_dump()["missing_citations"] is True
    with pytest.raises(ValidationError):
        ClaimVerificationResult(claim=claim, status="supported")


def test_valid_citation_does_not_prove_claim_support() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3")
    source = SourceMetadata(document_id="doc-1", title="Decision")
    claim = LegalClaim(id="claim-1", text="The rule applies.", citations=[citation])
    evidence = RetrievedEvidence(id="passage-1", source=source, text="Relevant passage")
    request = VerificationRequest(task_id="task-1", claims=[claim], evidence=[evidence])
    result = ClaimVerificationResult(
        claim=claim,
        status="uncertain",
        evidence_ids=[evidence.id],
        citation_results=[
            CitationValidationResult(
                citation=citation, status="valid", evidence_id=evidence.id, source=source
            )
        ],
    )

    assert request.evidence[0].source.document_id == citation.document_id
    assert VerificationResponse(
        task_id="task-1", verified=False, overall_status="uncertain", claim_results=[result]
    ).verified is False
    with pytest.raises(ValidationError):
        VerificationResponse(
            task_id="task-1", verified=True, overall_status="supported", claim_results=[result]
        )


def test_coordinator_accepts_detailed_and_legacy_results() -> None:
    citation = Citation(document_id="doc-1")
    source = SourceMetadata(document_id="doc-1", title="Decision")
    claim = LegalClaim(id="claim-1", text="The rule applies.", citations=[citation])
    response = VerificationResponse(
        task_id="task-1",
        verified=True,
        overall_status="supported",
        citations=[citation],
        claim_results=[
            ClaimVerificationResult(
                claim=claim,
                status="supported",
                evidence_ids=["passage-1"],
                citation_results=[
                    CitationValidationResult(
                        citation=citation,
                        status="valid",
                        evidence_id="passage-1",
                        source=source,
                    )
                ],
            )
        ],
        warnings=[
            ResponsibleAIWarning(
                code="legal_advice", message="This is not legal advice.", severity="caution"
            )
        ],
    )
    agent_result = AgentResult(task_id="task-1", agent="analysis", status="complete")
    coordinator = CoordinatorResponse(result=agent_result, verification=response)
    round_trip = CoordinatorResponse.model_validate(coordinator.model_dump())

    assert isinstance(response, VerificationResult)
    assert isinstance(round_trip.verification, VerificationResponse)
    assert round_trip.verification.claim_results[0].status == "supported"
    assert round_trip.verification.warnings[0].severity == "caution"
    with pytest.raises(ValidationError):
        CoordinatorResponse.model_validate(
            {
                "result": agent_result.model_dump(),
                "verification": {
                    "task_id": "task-1",
                    "verified": True,
                    "overall_status": "supported",
                    "claim_results": [],
                },
            }
        )
    with pytest.raises(ValidationError):
        VerificationResponse(
            task_id="task-1",
            verified=True,
            overall_status="supported",
            claim_results=response.claim_results,
        )
    assert isinstance(
        CoordinatorResponse(
            result=agent_result, verification=VerificationResult(verified=False, notes=["Pending"])
        ).verification,
        VerificationResult,
    )
