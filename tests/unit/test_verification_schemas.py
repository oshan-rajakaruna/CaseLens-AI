"""Validation and compatibility checks for shared verification contracts."""

import pytest
from pydantic import ValidationError

from agents.coordinator.schemas import CoordinatorResponse
from backend.schemas import (
    AgentResult,
    Citation,
    CitationStatus,
    CitationValidationResult,
    ClaimStatus,
    ClaimVerificationResult,
    LegalClaim,
    ResponsibleAIWarning,
    RetrievedEvidence,
    SourceMetadata,
    VerificationRequest,
    VerificationResponse,
    VerificationResult,
)
from backend.schemas.verification import determine_overall_status


def _claim_result(
    claim_id: str,
    status: ClaimStatus,
    citation_status: CitationStatus = "valid",
) -> ClaimVerificationResult:
    citation = Citation(document_id=f"doc-{claim_id}") if citation_status != "missing" else None
    claim = LegalClaim(
        id=claim_id,
        text="A legal assertion.",
        citations=[citation] if citation is not None else [],
    )
    evidence_id = f"passage-{claim_id}" if citation_status == "valid" else None
    source = (
        SourceMetadata(document_id=citation.document_id, title="Decision")
        if citation_status == "valid"
        else None
    )
    return ClaimVerificationResult(
        claim=claim,
        status=status,
        evidence_ids=[evidence_id] if evidence_id is not None else [],
        citation_results=[
            CitationValidationResult(
                citation=citation,
                status=citation_status,
                evidence_id=evidence_id,
                source=source,
            )
        ],
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


def test_issue_codes_default_for_older_results_and_reject_unknown_codes() -> None:
    payload = {
        "claim": {"id": "claim-1", "text": "The rule applies."},
        "status": "uncertain",
        "issues": ["The available passages do not establish the claim."],
    }

    result = ClaimVerificationResult.model_validate(payload)

    assert result.issue_codes == []
    assert result.issues == payload["issues"]
    with pytest.raises(ValidationError):
        ClaimVerificationResult.model_validate({**payload, "issue_codes": ["other_issue"]})


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


@pytest.mark.parametrize(
    ("claim_status", "citation_status", "expected_status"),
    [
        ("supported", "valid", "supported"),
        ("partially_supported", "valid", "partially_supported"),
        ("unsupported", "valid", "unsupported"),
        ("uncertain", "valid", "uncertain"),
        ("uncertain", "invalid", "invalid_citations"),
        ("uncertain", "missing", "uncertain"),
        ("uncertain", "uncertain", "uncertain"),
    ],
)
def test_response_status_and_verified_follow_claim_and_citation_results(
    claim_status: ClaimStatus,
    citation_status: CitationStatus,
    expected_status: str,
) -> None:
    result = _claim_result("claim-1", claim_status, citation_status)
    citations = result.claim.citations
    verified = expected_status == "supported"

    assert determine_overall_status([result]) == expected_status
    response = VerificationResponse(
        task_id="task-1",
        verified=verified,
        overall_status=expected_status,
        claim_results=[result],
        citations=citations,
    )
    assert response.verified is verified

    with pytest.raises(ValidationError):
        VerificationResponse(
            task_id="task-1",
            verified=not verified,
            overall_status=expected_status,
            claim_results=[result],
            citations=citations,
        )
    with pytest.raises(ValidationError):
        VerificationResponse(
            task_id="task-1",
            verified=False,
            overall_status="uncertain" if expected_status == "supported" else "supported",
            claim_results=[result],
            citations=citations,
        )


@pytest.mark.parametrize(
    ("results", "expected_status"),
    [
        (["supported", "unsupported", "uncertain"], "unsupported"),
        (["partially_supported", "uncertain"], "uncertain"),
        (["supported", "partially_supported"], "partially_supported"),
    ],
)
def test_mixed_claim_status_precedence(results: list[ClaimStatus], expected_status: str) -> None:
    claim_results = [_claim_result(f"claim-{index}", status) for index, status in enumerate(results)]
    response = VerificationResponse(
        task_id="task-1",
        verified=False,
        overall_status=expected_status,
        claim_results=claim_results,
    )

    assert response.overall_status == determine_overall_status(claim_results)


def test_invalid_citation_takes_precedence_over_unsupported_claim() -> None:
    claim_results = [
        _claim_result("refuted", "unsupported"),
        _claim_result("bad-reference", "uncertain", "invalid"),
    ]

    assert determine_overall_status(claim_results) == "invalid_citations"
    response = VerificationResponse(
        task_id="task-1",
        verified=False,
        overall_status="invalid_citations",
        claim_results=claim_results,
    )
    assert response.verified is False


@pytest.mark.parametrize("citation_status", ["invalid", "missing", "uncertain"])
def test_decided_claim_rejects_incomplete_citation_checks(citation_status: CitationStatus) -> None:
    citation = Citation(document_id="doc-1") if citation_status != "missing" else None
    claim = LegalClaim(
        id="claim-1",
        text="A legal assertion.",
        citations=[citation] if citation is not None else [],
    )
    result = ClaimVerificationResult(
        claim=claim,
        status="supported",
        evidence_ids=["passage-1"],
        citation_results=[CitationValidationResult(citation=citation, status=citation_status)],
    )

    with pytest.raises(ValidationError):
        VerificationResponse(
            task_id="task-1",
            verified=False,
            overall_status="invalid_citations" if citation_status == "invalid" else "uncertain",
            claim_results=[result],
        )


def test_no_reviewed_evidence_is_uncertain_and_empty_results_are_not_supported() -> None:
    result = _claim_result("claim-1", "uncertain", "uncertain")
    assert result.missing_evidence is True
    assert determine_overall_status([result]) == "uncertain"
    assert VerificationResponse(
        task_id="task-1", verified=False, overall_status="uncertain", claim_results=[result]
    ).verified is False
    assert determine_overall_status([]) == "uncertain"
    with pytest.raises(ValidationError):
        VerificationResponse(task_id="task-1", verified=False, overall_status="supported")


def test_verified_is_documented_as_a_deterministic_check_only() -> None:
    description = VerificationResponse.model_json_schema()["properties"]["verified"]["description"]

    assert "deterministic" in description
    assert "does not establish legal truth" in description
