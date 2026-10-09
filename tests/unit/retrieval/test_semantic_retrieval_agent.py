"""Tests for independent BM25 and semantic Retrieval Agent modes."""

import pytest

from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _chunk(document_id: str, text: str) -> LegalTextChunk:
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-0001",
        document_id=document_id,
        chunk_text=text,
        metadata=LegalDocumentMetadata(
            document_id=document_id,
            case_name=f"Synthetic {document_id.title()} Matter",
        ),
    )


def test_agent_keeps_bm25_and_semantic_modes_independent() -> None:
    chunks = [
        _chunk("employment", "employment termination dismissal"),
        _chunk("property", "property boundary title"),
    ]
    embeddings = DeterministicEmbeddingService()
    agent = RetrievalAgent(chunks, embedding_service=embeddings)

    assert agent.search_bm25("property", top_k=1)[0].document_id == "property"
    assert agent.search("property", top_k=1)[0].document_id == "property"
    assert embeddings.document_calls == []
    assert embeddings.query_calls == []

    agent.index_semantic_chunks(chunks)
    semantic_results = agent.search_semantic("termination rights", top_k=1)

    assert semantic_results[0].document_id == "employment"
    assert embeddings.query_calls == ["termination rights"]


def test_agent_semantic_filters_are_explicitly_not_implemented() -> None:
    agent = RetrievalAgent(embedding_service=DeterministicEmbeddingService())

    with pytest.raises(NotImplementedError, match="Metadata filtering"):
        agent.search_semantic("employment", filters={"court": "Test Court"})
