"""Offline Coordinator-to-Retrieval integration over a fictional indexed corpus."""

import asyncio

from agents.coordinator import (
    Coordinator,
    CoordinatorEvidenceDocument,
    CoordinatorRequest,
    CoordinatorRetrievalAdapter,
)
from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _chunk(document_id: str, number: int, text: str, *, category: str) -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Fictional {document_id}",
        court="Fictional International Commercial Tribunal",
        citation=f"[2025] FIT {number}",
        legal_category=category,
        document_type="judgment",
        source="fictional public archive",
        source_url=f"https://example.invalid/{document_id.lower()}",
        page_number=number,
        page_start=number,
        page_end=number,
    )
    return LegalTextChunk(
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-{number:04d}",
        chunk_text=text,
        metadata=metadata,
    )


def _coordinator(chunks: list[LegalTextChunk]) -> Coordinator:
    retrieval = RetrievalAgent(chunks, embedding_service=DeterministicEmbeddingService())
    return Coordinator(retrieval_adapter=CoordinatorRetrievalAdapter(retrieval))


def _request(query: str, **overrides: object) -> CoordinatorRequest:
    values: dict[str, object] = {
        "case_id": "CASE-RET-INT-001",
        "user_query": query,
        "requested_task": "retrieve_precedents",
        "retrieval_mode": "bm25",
        "top_k": 3,
    }
    values.update(overrides)
    return CoordinatorRequest.model_validate(values)


def test_real_bm25_retrieval_preserves_ranked_payment_and_notice_provenance() -> None:
    coordinator = _coordinator(
        [
            _chunk("PAYMENT-001", 1, "contract payment default notice and remedy", category="payment"),
            _chunk("TERMINATION-002", 2, "termination notice condition and contractual remedy", category="termination"),
            _chunk("PROPERTY-003", 3, "boundary title property dispute", category="property"),
        ]
    )

    response = coordinator.execute_retrieval(_request("contract notice remedy"))

    assert response.executed is True
    assert response.status == "completed"
    assert [result.rank for result in response.retrieval_results] == [1, 2]
    assert [result.document_id for result in response.retrieval_results] == [
        "PAYMENT-001", "TERMINATION-002"
    ]
    payment = response.retrieval_results[0]
    assert payment.chunk_id == "PAYMENT-001-chunk-0001"
    assert payment.chunk_text == "contract payment default notice and remedy"
    assert payment.metadata.page_start == 1
    assert payment.metadata.court == "Fictional International Commercial Tribunal"
    assert payment.metadata.source_url == "https://example.invalid/payment-001"
    assert payment.metadata.citation == "[2025] FIT 1"
    assert payment.score > 0
    assert any("do not establish authority" in limit for limit in response.verification_limitations)


def test_real_bm25_no_results_is_executed_without_fabricated_precedent() -> None:
    coordinator = _coordinator([_chunk("PAYMENT-001", 1, "contract payment default", category="payment")])

    response = coordinator.execute_retrieval(_request("maritime salvage"))

    assert response.executed is True
    assert response.status == "completed"
    assert response.retrieval_results == []
    assert response.warnings[0].code == "no_retrieval_results"


def test_real_bm25_is_repeatable_and_missing_index_is_not_executed() -> None:
    chunks = [_chunk("TERMINATION-002", 2, "termination notice condition", category="termination")]
    coordinator = _coordinator(chunks)
    request = _request("termination notice")

    first = coordinator.execute_retrieval(request)
    second = coordinator.execute_retrieval(request)
    missing = _coordinator([]).execute_retrieval(request)

    assert first == second
    assert first.retrieval_results[0].document_id == "TERMINATION-002"
    assert missing.executed is False
    assert missing.status == "requires_input"
    assert missing.errors[0].code == "missing_retrieval_index"


def test_existing_analysis_execution_remains_available() -> None:
    response = asyncio.run(
        Coordinator().handle(
            CoordinatorRequest(
                case_id="CASE-AN-RET-001",
                case_title="Fictional contract dispute",
                user_query="Analyze supplied evidence.",
                requested_task="analyze_evidence",
                document_ids=["DOC-AN-RET-001"],
                evidence_documents=[
                    CoordinatorEvidenceDocument(
                        document_id="DOC-AN-RET-001",
                        name="agreement.txt",
                        text="The agreement was signed on 12 March 2025.",
                    )
                ],
            )
        )
    )

    assert response.executed is True
    assert response.status == "completed"
    assert response.stages[0].agent == "analysis"
