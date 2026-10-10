"""Coordinator payload to validated summarization integration test."""

from agents.coordinator import (
    LegalRetrievalContextRequest,
    LegalRetrievalContextService,
)
from agents.summarization import SummarizationAgent
from retrieval.preprocessing.metadata import HybridSearchResult, LegalDocumentMetadata
from tests.unit.coordinator._fakes import FakeCoordinatorRetrievalAgent
from tests.unit.summarization._fakes import FakeSummarizationProvider


def _result(document_id: str, rank: int) -> HybridSearchResult:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Case {document_id}",
        court="Synthetic Court",
        citation=f"CIT-{document_id}",
        page_number=rank,
    )
    return HybridSearchResult(
        rank=rank,
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-{rank:04d}",
        case_name=metadata.case_name,
        court=metadata.court,
        citation=metadata.citation,
        chunk_text=f"Retrieved evidence for {document_id}",
        metadata=metadata,
        raw_bm25_score=5.0,
        normalized_bm25_score=0.8,
        raw_semantic_score=0.7,
        normalized_semantic_score=0.7,
        hybrid_score=0.78,
    )


def test_coordinator_payload_flows_through_agent_and_citation_validation() -> None:
    retrieval_agent = FakeCoordinatorRetrievalAgent(
        [_result("DOC-A", 1), _result("DOC-B", 2)]
    )
    payload = LegalRetrievalContextService(retrieval_agent).build(
        LegalRetrievalContextRequest(
            query="state responsibility",
            legal_issue="Attribution and breach",
            top_k=2,
        )
    )
    provider = FakeSummarizationProvider(
        {
            "answer": "The retrieved evidence addresses the issue [CTX-001].",
            "citations": ["CTX-001"],
            "confidence": 0.7,
            "limitations": ["Limited to retrieved evidence"],
        }
    )

    result = SummarizationAgent(provider).summarize(payload)

    assert result.output.citations == ["CTX-001"]
    assert result.citation_validation.all_citations_valid is True
    assert result.used_context_ids == ["CTX-001"]
    assert provider.calls[0].query == "state responsibility"
    assert provider.calls[0].legal_issue == "Attribution and breach"
    assert list(provider.calls[0].citation_map) == ["CTX-001", "CTX-002"]
