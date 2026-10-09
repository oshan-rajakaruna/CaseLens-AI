"""Tests for the Phase 1 BM25-only Retrieval Agent."""

import pytest

from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk


def _chunk(document_id: str, text: str) -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Synthetic {document_id.title()} Matter",
    )
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-0001",
        document_id=document_id,
        chunk_text=text,
        metadata=metadata,
    )


def test_agent_exposes_bm25_search_interface() -> None:
    agent = RetrievalAgent(
        [
            _chunk("employment", "wrongful employment termination and dismissal"),
            _chunk("property", "property title and boundary dispute"),
        ]
    )

    results = agent.search("employment dismissal", top_k=1)

    assert len(results) == 1
    assert results[0].document_id == "employment"


def test_agent_can_replace_its_index() -> None:
    agent = RetrievalAgent([_chunk("old", "contract breach")])

    agent.index_chunks([_chunk("new", "employment termination")])

    assert agent.search("contract") == []
    assert agent.search("termination")[0].document_id == "new"


def test_agent_documents_filters_as_not_implemented() -> None:
    agent = RetrievalAgent()

    with pytest.raises(NotImplementedError, match="Metadata filtering"):
        agent.search("termination", filters={"court": "Synthetic Tribunal"})
