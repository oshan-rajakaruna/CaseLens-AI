"""Tests for semantic indexing and structured top-k search."""

import pytest

from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from retrieval.vector_store import SemanticSearchService, VectorDimensionError
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _chunk(document_id: str, text: str) -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Synthetic {document_id.title()} Matter",
        citation=f"SYNTHETIC-{document_id.upper()}",
        source=f"tests/fixtures/retrieval/synthetic_{document_id}.txt",
        legal_category=document_id,
    )
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-0001",
        document_id=document_id,
        chunk_text=text,
        metadata=metadata,
    )


def _chunks() -> list[LegalTextChunk]:
    return [
        _chunk("employment", "Wrongful employment termination and dismissal."),
        _chunk("contract", "Supplier breach of a commercial contract."),
        _chunk("property", "Property title and boundary dispute."),
    ]


def test_semantic_search_ranks_employment_chunk_highest() -> None:
    embeddings = DeterministicEmbeddingService()
    service = SemanticSearchService(embeddings)
    chunks = _chunks()
    service.build_index(chunks)

    results = service.search("employee termination rights", top_k=3)

    assert results[0].rank == 1
    assert results[0].document_id == "employment"
    assert results[0].case_name == "Synthetic Employment Matter"
    assert results[0].citation == "SYNTHETIC-EMPLOYMENT"
    assert results[0].chunk_text == chunks[0].chunk_text
    assert results[0].metadata.legal_category == "employment"
    assert results[0].similarity_score > results[1].similarity_score


def test_semantic_index_embeds_duplicate_chunk_only_once() -> None:
    embeddings = DeterministicEmbeddingService()
    service = SemanticSearchService(embeddings)
    chunk = _chunks()[0]

    service.build_index([chunk, chunk])

    assert len(service.index) == 1
    assert len(embeddings.document_calls) == 1


def test_semantic_search_handles_empty_inputs_and_large_top_k() -> None:
    embeddings = DeterministicEmbeddingService()
    service = SemanticSearchService(embeddings)

    assert service.search("employment", top_k=5) == []
    assert embeddings.query_calls == []

    service.build_index(_chunks())
    assert service.search("   ", top_k=5) == []
    assert service.search("employment", top_k=0) == []
    assert len(service.search("employment", top_k=50)) == 3


def test_failed_index_rebuild_preserves_previous_index() -> None:
    class WrongDimensionEmbeddingService(DeterministicEmbeddingService):
        def embed_document(self, text: str, title: str | None = None) -> list[float]:
            return [1.0, 0.0]

    valid_service = SemanticSearchService(DeterministicEmbeddingService())
    valid_service.build_index([_chunks()[0]])
    previous_index = valid_service.index
    valid_service.embedding_service = WrongDimensionEmbeddingService()

    with pytest.raises(VectorDimensionError):
        valid_service.build_index([_chunks()[1]])

    assert valid_service.index is previous_index
    assert len(valid_service.index) == 1


@pytest.mark.parametrize("top_k", [-1, 1.5, True])
def test_semantic_search_validates_top_k(top_k: object) -> None:
    service = SemanticSearchService(DeterministicEmbeddingService())
    service.build_index(_chunks())

    with pytest.raises((TypeError, ValueError)):
        service.search("employment", top_k=top_k)  # type: ignore[arg-type]
