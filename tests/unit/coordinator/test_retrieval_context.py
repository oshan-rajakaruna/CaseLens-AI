"""Tests for Coordinator retrieval-to-RAG context orchestration."""

from collections.abc import Iterable

from pydantic import ValidationError
import pytest

from agents.coordinator import (
    LegalRetrievalContextRequest,
    LegalRetrievalContextService,
)
from retrieval.preprocessing.metadata import (
    HybridSearchResult,
    LegalDocumentMetadata,
)
from retrieval.rag import RAGContextAssembler, RAGContextSettings
from tests.unit.coordinator._fakes import FakeCoordinatorRetrievalAgent


def _result(
    document_id: str,
    chunk_number: int,
    rank: int,
    *,
    score: float = 0.8,
    page: int = 1,
) -> HybridSearchResult:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Case {document_id}",
        court="Synthetic Court",
        citation=f"CIT-{document_id}",
        source="official source",
        source_url="https://example.invalid/source",
        page_number=page,
        page_start=page,
        page_end=page,
    )
    return HybridSearchResult(
        rank=rank,
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
        case_name=metadata.case_name,
        chunk_text=f"Legal evidence for {document_id} chunk {chunk_number}.",
        citation=metadata.citation,
        source=metadata.source,
        court=metadata.court,
        metadata=metadata,
        raw_bm25_score=score * 10,
        normalized_bm25_score=score,
        raw_semantic_score=score,
        normalized_semantic_score=score,
        hybrid_score=score,
    )


@pytest.mark.parametrize(
    "values",
    [
        {"query": "   "},
        {"query": "law", "legal_issue": "  "},
        {"query": "law", "top_k": 0},
        {"query": "law", "max_chunks_per_document": 0},
        {"query": "law", "rag_max_context_tokens": 0},
        {"query": "law", "rag_max_context_passages": 0},
        {"query": "law", "rag_min_retrieval_score": float("nan")},
        {
            "query": "law",
            "bm25_weight": 0,
            "semantic_weight": 0,
        },
    ],
)
def test_coordinator_request_validation(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LegalRetrievalContextRequest.model_validate(values)


class _RecordingAssembler(RAGContextAssembler):
    def __init__(
        self,
        settings: RAGContextSettings,
        record: dict[str, object],
    ) -> None:
        super().__init__(settings)
        self.record = record

    def assemble(self, results: Iterable[HybridSearchResult]):  # type: ignore[no-untyped-def]
        materialized = list(results)
        self.record["results"] = materialized
        return super().assemble(materialized)


def test_adapter_passes_retrieval_overrides_and_builds_payload() -> None:
    agent = FakeCoordinatorRetrievalAgent(
        [_result("DOC-A", 1, 1, page=7), _result("DOC-B", 1, 2, page=9)]
    )
    record: dict[str, object] = {}

    def factory(settings: RAGContextSettings) -> RAGContextAssembler:
        record["settings"] = settings
        return _RecordingAssembler(settings, record)

    service = LegalRetrievalContextService(agent, assembler_factory=factory)
    request = LegalRetrievalContextRequest(
        query="state responsibility",
        legal_issue="Attribution of conduct",
        top_k=4,
        filters={"court": "Synthetic Court", "year": 2025},
        bm25_weight=0.7,
        semantic_weight=0.3,
        diversify=True,
        max_chunks_per_document=1,
        rag_max_context_tokens=800,
        rag_max_context_passages=3,
        rag_min_retrieval_score=0.2,
    )

    payload = service.build(request)

    assert agent.calls == [
        {
            "mode": "hybrid",
            "query": "state responsibility",
            "top_k": 4,
            "filters": {"court": "Synthetic Court", "year": 2025},
            "bm25_weight": 0.7,
            "semantic_weight": 0.3,
            "diversify": True,
            "max_chunks_per_document": 1,
        }
    ]
    assert len(record["results"]) == 2  # type: ignore[arg-type]
    settings = record["settings"]
    assert isinstance(settings, RAGContextSettings)
    assert settings == RAGContextSettings(
        max_tokens=800,
        max_passages=3,
        min_retrieval_score=0.2,
    )
    assert payload.query == request.query
    assert payload.legal_issue == request.legal_issue
    assert "[CTX-001]" in payload.context_text
    assert [passage.context_id for passage in payload.context_passages] == [
        "CTX-001",
        "CTX-002",
    ]
    assert payload.retrieval.bm25_weight == 0.7
    assert payload.retrieval.semantic_weight == 0.3
    assert payload.retrieval.diversified is True


def test_citation_map_preserves_source_identity() -> None:
    agent = FakeCoordinatorRetrievalAgent([_result("DOC-A", 1, 1, page=7)])
    payload = LegalRetrievalContextService(agent).build(
        LegalRetrievalContextRequest(query="legal issue")
    )

    citation = payload.citation_map["CTX-001"]
    assert citation.context_id == "CTX-001"
    assert citation.document_id == "DOC-A"
    assert citation.title == "Case DOC-A"
    assert citation.court == "Synthetic Court"
    assert citation.source == "official source"
    assert citation.chunk_ids == ["DOC-A-chunk-0001"]
    assert citation.pages == [7]
    assert citation.official_citation == "CIT-DOC-A"
    assert citation.source_url == "https://example.invalid/source"


def test_diversification_can_be_disabled_explicitly() -> None:
    agent = FakeCoordinatorRetrievalAgent(
        [_result("DOC-A", 1, 1), _result("DOC-A", 3, 2)]
    )
    payload = LegalRetrievalContextService(agent).build(
        LegalRetrievalContextRequest(
            query="legal issue",
            diversify=False,
            top_k=2,
        )
    )

    assert agent.calls[0]["diversify"] is False
    assert agent.calls[0]["max_chunks_per_document"] is None
    assert payload.retrieval.diversified is False
    assert len(payload.context_passages) == 2


def test_empty_retrieval_produces_safe_no_context_payload() -> None:
    agent = FakeCoordinatorRetrievalAgent([])

    payload = LegalRetrievalContextService(agent).build(
        LegalRetrievalContextRequest(query="missing issue")
    )

    assert payload.context_text == ""
    assert payload.context_passages == []
    assert payload.citation_map == {}
    assert payload.diagnostics.empty_retrieval is True
    assert payload.diagnostics.no_context is True
    assert payload.diagnostics.estimated_context_tokens == 0


def test_threshold_can_produce_no_context_without_empty_retrieval() -> None:
    agent = FakeCoordinatorRetrievalAgent([_result("DOC-A", 1, 1, score=0.2)])

    payload = LegalRetrievalContextService(agent).build(
        LegalRetrievalContextRequest(
            query="weak issue",
            rag_min_retrieval_score=0.5,
        )
    )

    assert payload.diagnostics.empty_retrieval is False
    assert payload.diagnostics.no_context is True


def test_relative_weak_context_warning_does_not_filter_results() -> None:
    agent = FakeCoordinatorRetrievalAgent(
        [
            _result("DOC-A", 1, 1, score=1.0),
            _result("DOC-B", 1, 2, score=0.05),
        ]
    )

    payload = LegalRetrievalContextService(agent).build(
        LegalRetrievalContextRequest(query="mixed issue")
    )

    assert len(payload.context_passages) == 2
    assert payload.diagnostics.weak_context_ids == ["CTX-002"]
    assert payload.diagnostics.warnings
