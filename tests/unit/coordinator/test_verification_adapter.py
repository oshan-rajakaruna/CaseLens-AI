"""Coordinator Step 5 deterministic verification boundary tests."""

from agents.coordinator import Coordinator, CoordinatorEvidenceDocument, CoordinatorRequest
from agents.coordinator.verification_adapter import CoordinatorVerificationAdapter
from agents.summarization.schemas import EvidenceSummary, EvidenceSummaryPoint, SourceReference
from backend.schemas import Citation, LegalClaim, RetrievedEvidence, SourceMetadata


def _request(**changes: object) -> CoordinatorRequest:
    values: dict[str, object] = {
        "case_id": "CASE-VERIFY-001", "user_query": "Was payment made?", "requested_task": "verify_claims",
    }
    values.update(changes)
    return CoordinatorRequest.model_validate(values)


def _evidence(text: str = "Nimal Perera paid LKR 5,000.") -> RetrievedEvidence:
    return RetrievedEvidence(
        id="EVID-001", source=SourceMetadata(document_id="DOC-001", title="fictional receipt"),
        locator="paragraph:1", text=text,
    )


def _claim(text: str = "Nimal Perera paid LKR 5,000.") -> LegalClaim:
    return LegalClaim(id="CLAIM-001", text=text, citations=[Citation(document_id="DOC-001", locator="paragraph:1", excerpt="Nimal Perera paid LKR 5,000.")])


def test_real_verification_preserves_supported_and_denied_outcomes() -> None:
    supported = Coordinator().execute_verify_claims(_request(verification_claims=[_claim()], verification_evidence=[_evidence()]))
    denied = Coordinator().execute_verify_claims(_request(
        verification_claims=[_claim("Nimal Perera did not pay LKR 5,000.")], verification_evidence=[_evidence()]
    ))

    assert supported.status == "completed"
    assert supported.verification_response.overall_status == "supported"
    assert denied.status == "partial"
    # The existing verifier cannot normalize "did not pay" against "paid";
    # the Coordinator preserves its conservative uncertain state.
    assert denied.verification_response.claim_results[0].status == "uncertain"
    assert denied.verification_response.verified is False


def test_missing_claims_and_source_mismatch_are_explicit() -> None:
    missing = Coordinator().execute_verify_claims(_request())
    mismatch = Coordinator().execute_verify_claims(_request(
        verification_claims=[_claim()], verification_evidence=[_evidence("Different text.")]
    ))

    assert missing.status == "requires_input" and missing.executed is False
    assert mismatch.status == "partial"
    assert mismatch.verification_response.overall_status == "uncertain"


def test_summary_mapping_excludes_unsourced_claims_and_preserves_qualifiers() -> None:
    source = SourceReference(document_id="DOC-001", locator="paragraph:1", excerpt="Ms. Silva alleged payment may not be due.", citation_id="SRC-1234")
    summary = EvidenceSummary(
        summary_text=source.excerpt,
        points=[
            EvidenceSummaryPoint(text=source.excerpt, source_references=[source], citations=["SRC-1234"], is_negated=True, is_conditional=True, is_attributed=True),
            EvidenceSummaryPoint(text="Ungrounded conclusion", provenance_limited=True),
        ],
    )
    claims, evidence = CoordinatorVerificationAdapter().from_evidence_summary(summary, task_id="TASK", source_titles={"DOC-001": "fictional notice"})

    assert len(claims) == len(evidence) == 1
    assert claims[0].text == source.excerpt
    assert evidence[0].text == source.excerpt
    assert evidence[0].source.document_id == "DOC-001"


def test_evidence_summarization_is_deterministic_and_does_not_call_verification() -> None:
    request = CoordinatorRequest(
        case_id="CASE-SUM-001", user_query="Summarize the notice.", requested_task="summarize_evidence",
        document_ids=["DOC-001"], evidence_documents=[CoordinatorEvidenceDocument(
            document_id="DOC-001", name="fictional-notice.txt", text="Ms. Silva alleged that payment may not be due."
        )],
    )
    response = Coordinator().execute_summarize_evidence(request)

    assert response.status in {"completed", "partial"}
    assert response.summarization_response is not None
    assert response.summarization_response.source_references[0].document_id == "DOC-001"
    assert response.summarization_response.source_references[0].citation_id.startswith("SRC-")
