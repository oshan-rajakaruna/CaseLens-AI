"""Offline tests for the Summarization-to-Verification adapter."""

from collections.abc import Callable
from copy import deepcopy

import pytest

from agents.coordinator.schemas import (
    ContextCitationMapEntry,
    RetrievalContextDiagnostics,
    RetrievalContextMetadata,
    SummarizationReadyPayload,
)
from agents.summarization import (
    CitationValidationResult,
    SummarizationCitationValidation,
    SummarizationOutput,
    SummarizationProviderMetadata,
    SummarizationResult,
    SummarizationVerificationBridge,
    SummaryNotVerifiableError,
    SummaryProvenanceError,
    VerificationExecutionError,
)
from agents.verification import VerificationAgent
from backend.schemas import Citation, LegalClaim, VerificationRequest, VerificationResponse
from retrieval.rag import ChunkRetrievalScore, RAGCitation, RAGContextPassage


class CapturingVerifier:
    def __init__(
        self,
        callback: Callable[[VerificationRequest], VerificationResponse] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.callback = callback or VerificationAgent().verify
        self.error = error
        self.requests: list[VerificationRequest] = []

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.callback(request)


def _passage(
    context_id: str = "CTX-001",
    document_id: str = "DOC-A",
    text: str = "The appeal was allowed.",
    *,
    title: str | None = "Decision A",
    page: int = 3,
) -> RAGContextPassage:
    chunk_id = f"{document_id}-chunk-{page:04d}"
    return RAGContextPassage(
        context_id=context_id,
        text=text,
        document_id=document_id,
        title=title,
        court="Synthetic Court",
        source="Official source",
        citation=RAGCitation(
            document_id=document_id,
            citation=f"CIT-{document_id}",
            source="Official source",
            source_url=f"https://example.test/{document_id}",
            provenance="official_court_source",
            chunk_ids=[chunk_id],
            pages=[page],
        ),
        chunk_ids=[chunk_id],
        pages=[page],
        ranks=[1],
        score=1.0,
        retrieval_scores=[
            ChunkRetrievalScore(
                chunk_id=chunk_id,
                rank=1,
                final_score=1.0,
                hybrid_score=1.0,
            )
        ],
    )


def _payload(*passages: RAGContextPassage) -> SummarizationReadyPayload:
    active = list(passages) or [_passage()]
    citation_map = {
        passage.context_id: ContextCitationMapEntry(
            context_id=passage.context_id,
            document_id=passage.document_id,
            title=passage.title,
            court=passage.court,
            source=passage.source,
            chunk_ids=passage.chunk_ids,
            pages=passage.pages,
            official_citation=passage.citation.citation,
            source_url=passage.citation.source_url,
        )
        for passage in active
    }
    return SummarizationReadyPayload(
        query="What happened?",
        context_text="\n".join(f"[{item.context_id}] {item.text}" for item in active),
        context_passages=active,
        citation_map=citation_map,
        retrieval=RetrievalContextMetadata(
            mode="hybrid",
            top_k=len(active),
            result_count=len(active),
            diversified=True,
        ),
        diagnostics=RetrievalContextDiagnostics(
            retrieval_result_count=len(active),
            rag_passage_count=len(active),
            source_document_count=len({item.document_id for item in active}),
            estimated_context_tokens=20,
            context_truncated=False,
            empty_retrieval=False,
            no_context=False,
        ),
    )


def _result(answer: str, context_ids: list[str]) -> SummarizationResult:
    cited = list(context_ids)
    unique = list(dict.fromkeys(cited))
    return SummarizationResult(
        output=SummarizationOutput(
            answer=answer,
            citations=unique,
            confidence=0.9,
            limitations=[],
        ),
        citation_validation=SummarizationCitationValidation(
            inline=CitationValidationResult(
                cited_context_ids=cited,
                valid_citations=unique,
                invalid_citations=[],
                malformed_citations=[],
                has_citations=bool(cited),
                all_citations_valid=True,
            ),
            declared_citations=unique,
            all_citations_valid=True,
        ),
        provider_metadata=SummarizationProviderMetadata(
            provider="fake",
            model="offline",
            provider_called=True,
        ),
        used_context_ids=unique,
    )


def test_maps_ctx_to_document_evidence_metadata_and_audit_provenance() -> None:
    payload = _payload(_passage())
    verifier = CapturingVerifier()

    bridged = SummarizationVerificationBridge(verifier).verify(
        "task-1", payload, _result("The appeal was allowed. [CTX-001]", ["CTX-001"])
    )

    request = bridged.verification_request
    assert request is verifier.requests[0]
    assert request.claims[0].citations == [
        Citation(document_id="DOC-A", locator="CTX-001", excerpt=None)
    ]
    assert request.evidence[0].id == "CTX-001"
    assert request.evidence[0].text == "The appeal was allowed."
    assert request.evidence[0].source.title == "Decision A"
    assert request.evidence[0].source.source_uri == "https://example.test/DOC-A"
    assert request.evidence[0].source.jurisdiction is None
    assert bridged.provenance["CTX-001"].chunk_ids == ["DOC-A-chunk-0003"]
    assert bridged.provenance["CTX-001"].pages == [3]


def test_same_document_passages_use_distinct_internal_ctx_locators() -> None:
    first = _passage("CTX-001", "DOC-A", page=3)
    second = _passage("CTX-002", "DOC-A", "Costs were awarded.", page=4)
    bridged = SummarizationVerificationBridge(CapturingVerifier()).verify(
        "task-1",
        _payload(first, second),
        _result(
            "The appeal was allowed [CTX-001] and costs were awarded [CTX-002].",
            ["CTX-001", "CTX-002"],
        ),
    )

    assert [item.locator for item in bridged.verification_request.evidence] == [
        "CTX-001",
        "CTX-002",
    ]
    assert [item.locator for item in bridged.verification_request.claims[0].citations] == [
        "CTX-001",
        "CTX-002",
    ]


def test_complete_answer_is_one_claim_and_only_markers_are_removed() -> None:
    answer = "First proposition [CTX-001].\nSecond proposition [CTX-001]."
    bridged = SummarizationVerificationBridge(CapturingVerifier()).verify(
        "task-1", _payload(_passage()), _result(answer, ["CTX-001", "CTX-001"])
    )

    claims = bridged.verification_request.claims
    assert len(claims) == 1
    assert claims[0].id == "summary-claim-001"
    assert claims[0].text == "First proposition.\nSecond proposition."


@pytest.mark.parametrize(
    "payload,result",
    [
        (_payload(_passage()), _result("Unknown [CTX-999].", ["CTX-999"])),
        (
            _payload(_passage()).model_copy(update={"citation_map": {}}),
            _result("Known [CTX-001].", ["CTX-001"]),
        ),
        (
            _payload(_passage()).model_copy(update={"context_passages": []}),
            _result("Known [CTX-001].", ["CTX-001"]),
        ),
    ],
)
def test_missing_ctx_provenance_is_rejected(payload, result) -> None:
    with pytest.raises(SummaryProvenanceError):
        SummarizationVerificationBridge(CapturingVerifier()).verify(
            "task-1", payload, result
        )


def test_duplicate_context_passage_is_rejected_as_ambiguous() -> None:
    passage = _passage()
    payload = _payload(passage).model_copy(
        update={"context_passages": [passage, deepcopy(passage)]}
    )

    with pytest.raises(SummaryProvenanceError, match="exactly one"):
        SummarizationVerificationBridge(CapturingVerifier()).verify(
            "task-1", payload, _result("Claim [CTX-001].", ["CTX-001"])
        )


def test_document_or_optional_provenance_mismatch_is_rejected() -> None:
    payload = _payload(_passage())
    mismatched = payload.citation_map["CTX-001"].model_copy(
        update={"document_id": "DOC-OTHER"}
    )
    payload = payload.model_copy(update={"citation_map": {"CTX-001": mismatched}})

    with pytest.raises(SummaryProvenanceError, match="do not agree"):
        SummarizationVerificationBridge(CapturingVerifier()).verify(
            "task-1", payload, _result("Claim [CTX-001].", ["CTX-001"])
        )


def test_missing_optional_title_is_not_invented_and_remains_uncertain() -> None:
    payload = _payload(_passage(title=None))
    bridged = SummarizationVerificationBridge(VerificationAgent()).verify(
        "task-1",
        payload,
        _result("The appeal was allowed. [CTX-001]", ["CTX-001"]),
    )

    assert bridged.verification_request.evidence[0].source.title is None
    assert bridged.verification_response.overall_status == "uncertain"
    assert bridged.verification_response.verified is False


@pytest.mark.parametrize(
    "result",
    [
        _result("No cited evidence.", []),
        _result("[CTX-001]", ["CTX-001"]),
    ],
)
def test_unverifiable_or_marker_only_summary_is_rejected(result) -> None:
    with pytest.raises(SummaryNotVerifiableError):
        SummarizationVerificationBridge(CapturingVerifier()).verify(
            "task-1", _payload(_passage()), result
        )


def test_inconsistent_summarization_validation_is_rejected() -> None:
    result = _result("Claim [CTX-001].", ["CTX-001"])
    result = result.model_copy(update={"used_context_ids": ["CTX-002"]})

    with pytest.raises(SummaryNotVerifiableError, match="inconsistent"):
        SummarizationVerificationBridge(CapturingVerifier()).verify(
            "task-1", _payload(_passage()), result
        )


@pytest.mark.parametrize(
    "evidence_text,expected",
    [
        ("The appeal was allowed.", "supported"),
        ("The appeal was not allowed.", "unsupported"),
        ("The court heard argument.", "uncertain"),
    ],
)
def test_real_verification_statuses_are_preserved(evidence_text, expected) -> None:
    payload = _payload(_passage(text=evidence_text))
    bridged = SummarizationVerificationBridge(VerificationAgent()).verify(
        "task-1",
        payload,
        _result("The appeal was allowed. [CTX-001]", ["CTX-001"]),
    )

    assert bridged.verification_response.overall_status == expected
    assert bridged.verification_response.claim_results[0].status == expected
    assert bridged.verification_response.verified is (expected == "supported")


def test_verifier_response_and_warnings_are_returned_unchanged() -> None:
    def invalid_response(request: VerificationRequest) -> VerificationResponse:
        return VerificationAgent().verify(
            VerificationRequest(
                task_id=request.task_id,
                claims=[
                    LegalClaim(
                        id="summary-claim-001",
                        text=request.claims[0].text,
                        citations=[Citation(document_id="unknown")],
                    )
                ],
                evidence=request.evidence,
            )
        )

    verifier = CapturingVerifier(invalid_response)
    bridged = SummarizationVerificationBridge(verifier).verify(
        "task-1",
        _payload(_passage()),
        _result("The appeal was allowed. [CTX-001]", ["CTX-001"]),
    )

    assert bridged.verification_response.overall_status == "invalid_citations"
    assert bridged.verification_response.verified is False
    assert "invalid_citation" in {
        warning.code for warning in bridged.verification_response.warnings
    }
    assert "internal verification locators" in bridged.warnings[0]


def test_verification_failure_is_wrapped_without_private_content() -> None:
    verifier = CapturingVerifier(error=RuntimeError("private evidence leaked"))

    with pytest.raises(VerificationExecutionError) as caught:
        SummarizationVerificationBridge(verifier).verify(
            "task-1",
            _payload(_passage()),
            _result("Sensitive answer [CTX-001].", ["CTX-001"]),
        )

    assert "private evidence" not in str(caught.value)


def test_answer_citations_must_match_validated_result() -> None:
    result = _result(
        "The appeal was allowed. [CTX-999]",
        ["CTX-001"],
    )

    with pytest.raises(
        SummaryNotVerifiableError,
        match="inline citations do not match",
    ):
        SummarizationVerificationBridge(CapturingVerifier()).verify(
            "task-1",
            _payload(_passage()),
            result,
        )
