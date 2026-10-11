"""Offline real Retrieval-to-deterministic-Summarization Coordinator integration."""

import asyncio

from agents.coordinator import Coordinator, CoordinatorRequest, CoordinatorRetrievalAdapter
from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _chunk(document_id: str, number: int, text: str, document_type: str = "judgment") -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Fictional {document_id}",
        court="Fictional Court of Appeal",
        citation=f"[2025] FCA {number}",
        document_type=document_type,
        source="fictional archive",
        source_url=f"https://example.invalid/{document_id.lower()}",
        page_number=number,
        page_start=number,
        page_end=number,
    )
    return LegalTextChunk(
        document_id=document_id,
        chunk_id=f"{document_id}-C{number}",
        chunk_text=text,
        metadata=metadata,
    )


def _request(query: str) -> CoordinatorRequest:
    return CoordinatorRequest(
        case_id="CASE-INT-SUM-001",
        user_query=query,
        requested_task="summarize_precedents",
        retrieval_mode="bm25",
        top_k=5,
    )


def _coordinator(chunks: list[LegalTextChunk]) -> Coordinator:
    agent = RetrievalAgent(chunks, embedding_service=DeterministicEmbeddingService())
    return Coordinator(retrieval_adapter=CoordinatorRetrievalAdapter(agent))


def test_real_offline_bm25_to_precedent_summary_keeps_traceable_sources() -> None:
    coordinator = _coordinator([
        _chunk(
            "JUDG-101", 1,
            "The issue before the court was whether notice was timely. The court reasoned that the notice complied with section 12. Accordingly, the appeal is dismissed.",
        ),
        _chunk(
            "JUDG-101", 2,
            "The claimant alleged that payment was delayed.",
        ),
        _chunk("CONTRACT-201", 3, "The supplier must deliver goods by Friday.", "contract"),
    ])

    response = asyncio.run(coordinator.handle(_request("notice payment appeal")))

    assert response.status == "completed"
    assert response.executed is True
    assert len(response.retrieval_results) == 2
    assert len(response.summarization_responses) == 1
    assert response.precedent_summaries[0].holding == "Accordingly, the appeal is dismissed."
    assert [source.document_id for source in response.source_references] == ["JUDG-101", "JUDG-101"]
    assert all(source.citation_id in response.citation_provenance for source in response.source_references)
    assert {
        response.citation_provenance[source.citation_id].official_citation
        for source in response.source_references
    } == {"[2025] FCA 1", "[2025] FCA 2"}


def test_real_offline_empty_retrieval_does_not_create_a_summary() -> None:
    response = _coordinator([
        _chunk("JUDG-101", 1, "The Court dismisses the appeal."),
    ]).execute_summarize_precedents(_request("maritime salvage"))

    assert response.status == "partial"
    assert response.retrieval_results == []
    assert response.precedent_summaries == []
    assert [stage.status for stage in response.stages] == ["completed", "not_run"]
    assert response.warnings[0].code == "no_retrieval_results"
