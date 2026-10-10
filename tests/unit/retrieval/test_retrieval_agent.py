"""Tests for the Phase 1 BM25-only Retrieval Agent."""

import pytest

from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk


def _chunk(
    document_id: str,
    text: str,
    *,
    chunk_number: int = 1,
) -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Synthetic {document_id.title()} Matter",
    )
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
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


def test_agent_uses_configured_cap_and_allows_override_or_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        _chunk("dominant", "employment dismissal rights", chunk_number=1),
        _chunk("dominant", "employment dismissal process", chunk_number=2),
        _chunk("dominant", "employment dismissal remedy", chunk_number=3),
        _chunk("other", "employment dismissal appeal", chunk_number=1),
    ]
    agent = RetrievalAgent(chunks)
    monkeypatch.setenv("RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", "2")

    capped = agent.search_bm25("employment dismissal", top_k=4)
    overridden = agent.search_bm25(
        "employment dismissal",
        top_k=4,
        max_chunks_per_document=1,
    )
    uncapped = agent.search_bm25(
        "employment dismissal",
        top_k=4,
        diversify=False,
    )

    assert [result.document_id for result in capped].count("dominant") == 2
    assert [result.document_id for result in overridden] == ["dominant", "other"]
    assert [result.document_id for result in uncapped].count("dominant") == 3
