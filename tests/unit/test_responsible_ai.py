"""Tests for warnings derived from verification findings."""

from agents.verification.citation_validator import validate_citations
from agents.verification.claim_checker import check_claim
from agents.verification.responsible_ai import generate_warnings
from backend.schemas import (
    Citation,
    CitationValidationResult,
    ClaimStatus,
    ClaimVerificationResult,
    LegalClaim,
    ResponsibleAIWarning,
    RetrievedEvidence,
    SourceMetadata,
)


def _result(
    claim_id: str,
    status: ClaimStatus = "supported",
    *,
    issues: list[str] | None = None,
) -> ClaimVerificationResult:
    citation = Citation(document_id=f"doc-{claim_id}")
    source = SourceMetadata(document_id=citation.document_id, title="Decision")
    evidence_id = f"passage-{claim_id}"
    return ClaimVerificationResult(
        claim=LegalClaim(id=claim_id, text="A legal assertion.", citations=[citation]),
        status=status,
        evidence_ids=[evidence_id],
        citation_results=[
            CitationValidationResult(
                citation=citation, status="valid", evidence_id=evidence_id, source=source
            )
        ],
        issues=issues or [],
    )


def _by_code(results: list[ClaimVerificationResult]) -> dict[str, ResponsibleAIWarning]:
    return {warning.code: warning for warning in generate_warnings(results)}


def test_unsupported_claim_warning_and_legal_review_notice() -> None:
    warnings = _by_code([_result("claim-1", "unsupported")])

    assert warnings["unsupported_claim"].severity == "critical"
    assert warnings["unsupported_claim"].claim_ids == ["claim-1"]
    assert "legal information support" in warnings["legal_information_only"].message
    assert "does not guarantee legal outcomes" in warnings["legal_information_only"].message
    assert "qualified legal professional" in warnings["legal_information_only"].message
    assert warnings["legal_information_only"].claim_ids == []


def test_partially_supported_claim_warning() -> None:
    warnings = _by_code([_result("claim-1", "partially_supported")])

    assert warnings["partially_supported_claim"].claim_ids == ["claim-1"]
    assert "unsupported_claim" not in warnings
    assert "uncertain_claim" not in warnings


def test_uncertain_claim_warning() -> None:
    warnings = _by_code([_result("claim-1", "uncertain")])

    assert warnings["uncertain_claim"].claim_ids == ["claim-1"]
    assert "conflicting_evidence" not in warnings


def test_missing_citation_warning() -> None:
    result = ClaimVerificationResult(
        claim=LegalClaim(id="claim-1", text="A legal assertion."),
        status="uncertain",
        citation_results=[CitationValidationResult(citation=None, status="missing")],
    )

    warnings = _by_code([result])

    assert warnings["missing_citation"].claim_ids == ["claim-1"]
    assert "invalid_citation" not in warnings


def test_invalid_citation_warning() -> None:
    citation = Citation(document_id="unknown-doc")
    result = ClaimVerificationResult(
        claim=LegalClaim(id="claim-1", text="A legal assertion.", citations=[citation]),
        status="uncertain",
        citation_results=[CitationValidationResult(citation=citation, status="invalid")],
    )

    warnings = _by_code([result])

    assert warnings["invalid_citation"].claim_ids == ["claim-1"]
    assert "missing_citation" not in warnings


def test_conflicting_evidence_warning_uses_recorded_claim_issue() -> None:
    claim = LegalClaim(
        id="claim-1",
        text="The appeal was allowed.",
        citations=[
            Citation(document_id="doc-1", locator="p. 3"),
            Citation(document_id="doc-1", locator="p. 4"),
        ],
    )
    source = SourceMetadata(document_id="doc-1", title="Decision")
    evidence = [
        RetrievedEvidence(id="passage-1", source=source, locator="p. 3", text=claim.text),
        RetrievedEvidence(
            id="passage-2",
            source=source,
            locator="p. 4",
            text="The appeal was not allowed.",
        ),
    ]
    result = check_claim(claim, evidence, validate_citations(claim, evidence))

    warnings = _by_code([result])

    assert result.status == "uncertain"
    assert warnings["conflicting_evidence"].claim_ids == ["claim-1"]
    assert warnings["uncertain_claim"].claim_ids == ["claim-1"]


def test_multiple_affected_claims_are_grouped_without_duplicate_ids() -> None:
    first = _result("claim-1", "unsupported")
    second = _result("claim-2", "unsupported")

    warnings = _by_code([first, second, first])

    assert warnings["unsupported_claim"].claim_ids == ["claim-1", "claim-2"]
    assert "partially_supported_claim" not in warnings


def test_supported_claim_has_only_general_notice() -> None:
    warnings = generate_warnings([_result("claim-1")])

    assert [warning.code for warning in warnings] == ["legal_information_only"]


def test_empty_results_have_only_general_notice() -> None:
    warnings = generate_warnings([])

    assert [warning.code for warning in warnings] == ["legal_information_only"]
