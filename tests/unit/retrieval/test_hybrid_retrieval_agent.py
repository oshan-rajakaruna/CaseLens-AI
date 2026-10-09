"""Tests for the Retrieval Agent hybrid interface and mode independence."""

from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _chunk(
    document_id: str,
    text: str,
    *,
    category: str,
) -> LegalTextChunk:
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-0001",
        document_id=document_id,
        chunk_text=text,
        metadata=LegalDocumentMetadata(
            document_id=document_id,
            case_name=f"Synthetic {document_id.title()} Matter",
            court="Synthetic Labour Court" if category == "employment" else "Test Court",
            date="2025-01-01",
            legal_category=category,
            document_type="synthetic judgment",
        ),
    )


def _agent_and_chunks() -> tuple[RetrievalAgent, list[LegalTextChunk]]:
    chunks = [
        _chunk(
            "employment",
            "Unlawful termination and dismissal compensation remedies.",
            category="employment",
        ),
        _chunk(
            "contract",
            "Commercial contract breach and supplier damages.",
            category="contract",
        ),
        _chunk(
            "property",
            "Land ownership title and boundary survey dispute.",
            category="property",
        ),
    ]
    agent = RetrievalAgent(chunks, embedding_service=DeterministicEmbeddingService())
    agent.index_semantic_chunks(chunks)
    return agent, chunks


def test_agent_hybrid_ranks_employment_above_property() -> None:
    agent, _ = _agent_and_chunks()

    results = agent.search_hybrid(
        "unlawful termination employment contract",
        top_k=3,
    )

    assert results[0].document_id == "employment"
    assert results[0].hybrid_score >= results[-1].hybrid_score
    assert any(result.document_id == "property" for result in results)


def test_semantic_meaning_recovers_result_without_lexical_overlap() -> None:
    agent, _ = _agent_and_chunks()
    query = "worker job loss protections"

    assert agent.search_bm25(query) == []
    semantic_results = agent.search_semantic(query, top_k=3)
    hybrid_results = agent.search_hybrid(query, top_k=3)

    assert semantic_results[0].document_id == "employment"
    assert hybrid_results[0].document_id == "employment"
    assert hybrid_results[0].raw_bm25_score is None
    assert hybrid_results[0].raw_semantic_score is not None


def test_agent_preserves_all_three_modes_and_hybrid_filtering() -> None:
    agent, _ = _agent_and_chunks()

    assert agent.search("boundary", top_k=1)[0].document_id == "property"
    assert agent.search_bm25("boundary", top_k=1)[0].document_id == "property"
    assert agent.search_semantic("boundary", top_k=1)[0].document_id == "property"

    filtered = agent.search_hybrid(
        "legal dispute",
        top_k=5,
        filters={"legal_category": "employment", "year": 2025},
    )
    assert [result.document_id for result in filtered] == ["employment"]
