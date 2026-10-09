"""End-to-end tests for deterministic VerificationAgent orchestration."""

import pytest
from pydantic import ValidationError

from agents.coordinator.schemas import CoordinatorResponse
from agents.verification import VerificationAgent
from backend.schemas import (
    AgentResult,
    Citation,
    LegalClaim,
    RetrievedEvidence,
    SourceMetadata,
    VerificationRequest,
    VerificationResponse,
)


def _claim(
    text: str = "The appeal was allowed.",
    *,
    claim_id: str = "claim-1",
    citations: list[Citation] | None = None,
) -> LegalClaim:
    if citations is None:
        citations = [Citation(document_id="doc-1", locator="p. 3")]
    return LegalClaim(id=claim_id, text=text, citations=citations)


def _evidence(
    text: str = "The appeal was allowed.",
    *,
    evidence_id: str = "passage-1",
    locator: str = "p. 3",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        id=evidence_id,
        source=SourceMetadata(document_id="doc-1", title="Decision"),
        locator=locator,
        text=text,
    )


def _codes(response: VerificationResponse) -> set[str]:
    return {warning.code for warning in response.warnings}


def test_supported_request_preserves_ids_citations_and_coordinator_contract() -> None:
    claim = _claim()
    evidence = _evidence()
    request = VerificationRequest(task_id="task-1", claims=[claim], evidence=[evidence])

    response = VerificationAgent().verify(request)

    assert isinstance(response, VerificationResponse)
    assert response.task_id == "task-1"
    assert response.verified is True
    assert response.overall_status == "supported"
    assert response.claim_results[0].claim.id == claim.id
    assert response.claim_results[0].evidence_ids == [evidence.id]
    assert response.claim_results[0].citation_results[0].status == "valid"
    assert response.claim_results[0].citation_results[0].source == evidence.source
    assert response.citations == claim.citations
    assert response.notes == []
    assert _codes(response) == {"legal_information_only"}

    coordinator = CoordinatorResponse(
        result=AgentResult(task_id="task-1", agent="verification", status="complete"),
        verification=response,
    )
    restored = CoordinatorResponse.model_validate(coordinator.model_dump())
    assert isinstance(restored.verification, VerificationResponse)
    assert restored.verification.claim_results[0].evidence_ids == [evidence.id]


def test_empty_evidence_is_uncertain_and_not_verified() -> None:
    request = VerificationRequest(task_id="task-1", claims=[_claim()], evidence=[])

    response = VerificationAgent().verify(request)

    assert response.verified is False
    assert response.overall_status == "uncertain"
    assert response.claim_results[0].missing_evidence is True
    assert response.claim_results[0].citation_results[0].status == "uncertain"
    assert "uncertain_claim" in _codes(response)
    assert response.notes[0].startswith("claim-1:")


def test_invalid_citation_overrides_matching_uncited_passage() -> None:
    claim = _claim(citations=[Citation(document_id="doc-unknown")])
    request = VerificationRequest(task_id="task-1", claims=[claim], evidence=[_evidence()])

    response = VerificationAgent().verify(request)

    assert response.verified is False
    assert response.overall_status == "invalid_citations"
    assert response.claim_results[0].status == "uncertain"
    assert response.claim_results[0].evidence_ids == []
    assert response.claim_results[0].citation_results[0].status == "invalid"
    assert response.citations == claim.citations
    assert next(w for w in response.warnings if w.code == "invalid_citation").claim_ids == [
        claim.id
    ]


def test_valid_citation_alone_does_not_verify_claim() -> None:
    request = VerificationRequest(
        task_id="task-1", claims=[_claim()], evidence=[_evidence("The court heard argument.")]
    )

    response = VerificationAgent().verify(request)

    assert response.claim_results[0].citation_results[0].status == "valid"
    assert response.claim_results[0].status == "uncertain"
    assert response.overall_status == "uncertain"
    assert response.verified is False
    assert "uncertain_claim" in _codes(response)


def test_missing_citations_remain_visible() -> None:
    claim = _claim(citations=[])
    request = VerificationRequest(task_id="task-1", claims=[claim], evidence=[_evidence()])

    response = VerificationAgent().verify(request)

    assert response.verified is False
    assert response.overall_status == "uncertain"
    assert response.claim_results[0].citation_results[0].status == "missing"
    assert response.claim_results[0].missing_citations is True
    assert response.citations == []
    assert "missing_citation" in _codes(response)


def test_partially_supported_claim_sets_overall_status_and_warning() -> None:
    claim = _claim("The appeal was allowed and costs were awarded.")
    request = VerificationRequest(task_id="task-1", claims=[claim], evidence=[_evidence()])

    response = VerificationAgent().verify(request)

    assert response.overall_status == "partially_supported"
    assert response.verified is False
    assert response.claim_results[0].status == "partially_supported"
    assert next(w for w in response.warnings if w.code == "partially_supported_claim").claim_ids == [
        claim.id
    ]


def test_unsupported_claim_takes_precedence_over_other_uncertainty() -> None:
    refuted = _claim(claim_id="claim-refuted")
    unresolved = _claim("The court considered the record.", claim_id="claim-uncertain")
    request = VerificationRequest(
        task_id="task-1",
        claims=[refuted, unresolved],
        evidence=[_evidence("The appeal was not allowed.")],
    )

    response = VerificationAgent().verify(request)

    assert [result.claim.id for result in response.claim_results] == [
        "claim-refuted",
        "claim-uncertain",
    ]
    assert [result.status for result in response.claim_results] == ["unsupported", "uncertain"]
    assert response.overall_status == "unsupported"
    assert response.verified is False
    assert {"unsupported_claim", "uncertain_claim"} <= _codes(response)


def test_conflicting_evidence_stays_uncertain_and_emits_warning() -> None:
    claim = _claim(
        citations=[
            Citation(document_id="doc-1", locator="p. 3"),
            Citation(document_id="doc-1", locator="p. 4"),
        ]
    )
    request = VerificationRequest(
        task_id="task-1",
        claims=[claim],
        evidence=[
            _evidence(),
            _evidence("The appeal was not allowed.", evidence_id="passage-2", locator="p. 4"),
        ],
    )

    response = VerificationAgent().verify(request)

    assert response.overall_status == "uncertain"
    assert response.verified is False
    assert response.claim_results[0].evidence_ids == ["passage-1", "passage-2"]
    assert next(w for w in response.warnings if w.code == "conflicting_evidence").claim_ids == [
        claim.id
    ]


def test_empty_claims_are_rejected_by_request_schema() -> None:
    with pytest.raises(ValidationError):
        VerificationRequest(task_id="task-1", claims=[], evidence=[])
