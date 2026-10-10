"""Integration coverage from diversified retrieval into RAG context."""

from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from retrieval.rag import RAGContextAssembler, RAGContextSettings
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _chunk(document_id: str, chunk_number: int, text: str) -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Synthetic {document_id}",
        court="Synthetic Tribunal",
        page_number=chunk_number,
        page_start=chunk_number,
        page_end=chunk_number,
    )
    return LegalTextChunk(
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
        chunk_text=text,
        metadata=metadata,
    )


def test_diversified_hybrid_results_assemble_into_structured_context() -> None:
    chunks = [
        _chunk("employment-a", 1, "employment termination dismissal worker"),
        _chunk("employment-a", 2, "employment termination dismissal remedy"),
        _chunk("employment-b", 1, "worker job loss and dismissal appeal"),
        _chunk("property", 1, "property boundary and title dispute"),
    ]
    embeddings = DeterministicEmbeddingService()
    agent = RetrievalAgent(chunks, embedding_service=embeddings)
    agent.index_semantic_chunks(chunks)

    retrieved = agent.search_hybrid(
        "employment dismissal",
        top_k=3,
        max_chunks_per_document=1,
    )
    context = RAGContextAssembler(
        RAGContextSettings(max_tokens=500, max_passages=3)
    ).assemble(retrieved)

    assert len(retrieved) == 3
    assert len({result.document_id for result in retrieved}) == 3
    assert context.input_result_count == 3
    assert context.passages
    assert context.passages[0].context_id == "CTX-001"
    assert all(passage.chunk_ids for passage in context.passages)
    assert context.source_documents == [
        passage.document_id for passage in context.passages
    ]
