"""Coordinator-to-context-to-test-summarizer integration without network."""

from agents.coordinator import (
    LegalRetrievalContextRequest,
    LegalRetrievalContextService,
)
from agents.summarization import validate_context_citations
from retrieval.preprocessing.metadata import (
    HybridSearchResult,
    LegalDocumentMetadata,
)
from tests.unit.coordinator._fakes import (
    DeterministicSummarizerStub,
    FakeCoordinatorRetrievalAgent,
)


def _result(document_id: str, chunk_number: int, rank: int) -> HybridSearchResult:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Case {document_id}",
        court="Synthetic Court",
        citation=f"CIT-{document_id}",
        page_number=chunk_number,
        page_start=chunk_number,
        page_end=chunk_number,
    )
    return HybridSearchResult(
        rank=rank,
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
        case_name=metadata.case_name,
        court=metadata.court,
        citation=metadata.citation,
        chunk_text=f"Evidence {document_id} {chunk_number}",
        metadata=metadata,
        raw_bm25_score=5.0,
        normalized_bm25_score=0.8,
        raw_semantic_score=0.7,
        normalized_semantic_score=0.7,
        hybrid_score=0.78,
    )


def test_coordinator_flow_produces_citation_validated_stub_answer() -> None:
    retrieval_agent = FakeCoordinatorRetrievalAgent(
        [
            _result("DOC-A", 1, 1),
            _result("DOC-A", 3, 2),
            _result("DOC-B", 1, 3),
        ]
    )
    payload = LegalRetrievalContextService(retrieval_agent).build(
        LegalRetrievalContextRequest(
            query="state responsibility",
            top_k=3,
            max_chunks_per_document=1,
        )
    )
    output = DeterministicSummarizerStub().summarize(payload)
    validation = validate_context_citations(output.answer, payload.citation_map)

    assert payload.retrieval.diversified is True
    assert payload.retrieval.result_count == 2
    assert list(payload.citation_map) == ["CTX-001", "CTX-002"]
    assert output.citations == ["CTX-001"]
    assert validation.valid_citations == ["CTX-001"]
    assert validation.all_citations_valid is True
