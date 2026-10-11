"""Offline tests for API-loadable retrieval index snapshots."""

from pathlib import Path

import pytest

from agents.retrieval import RetrievalAgent
from backend.schemas.retrieval import RetrievalSearchRequest
from backend.services.retrieval_service import RetrievalService
from retrieval.embeddings.checkpoint import EmbeddingCheckpointStore
from retrieval.index_store import RetrievalIndexStore, RetrievalIndexStoreError
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk


class _OfflineEmbeddingService:
    provider = "offline-test"
    model = "deterministic-v1"
    dimension = 3
    batch_size = 8

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def document_content_hash(self, text: str, title: str | None = None) -> str:
        return f"{title}|{text}"

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        self.document_calls += 1
        raise AssertionError("snapshot loading must not embed documents")

    def embed_documents(
        self, documents: list[tuple[str, str | None]]
    ) -> list[list[float]]:
        self.document_calls += len(documents)
        raise AssertionError("snapshot loading must not embed documents")

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0, 0.0]


def _chunk() -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id="ICJ-001",
        case_name="Synthetic ICJ authority",
        court="International Court of Justice",
        citation="I.C.J. Reports 2021, p. 206",
        source="Official reports",
        provenance="official_court_source",
        page_number=7,
        page_start=7,
        page_end=7,
    )
    return LegalTextChunk(
        chunk_id="ICJ-001-chunk-0001",
        document_id="ICJ-001",
        chunk_text="The Court considered the relevant legal question.",
        metadata=metadata,
    )


def test_snapshot_round_trip_preserves_ids_citations_pages_without_network(
    tmp_path: Path,
) -> None:
    service = _OfflineEmbeddingService()
    chunk = _chunk()
    store = RetrievalIndexStore(tmp_path / "index")
    EmbeddingCheckpointStore(store.embedding_checkpoint_path).put_many(
        service, [(chunk, [1.0, 0.0, 0.0])]
    )
    store.save([chunk], embedding_service=service, semantic_indexed=True)

    loaded = store.load(service)

    assert service.document_calls == 0
    assert loaded.vectors == [[1.0, 0.0, 0.0]]
    assert loaded.chunks[0].document_id == "ICJ-001"
    assert loaded.chunks[0].chunk_id == "ICJ-001-chunk-0001"
    assert loaded.chunks[0].metadata.citation == "I.C.J. Reports 2021, p. 206"
    assert loaded.chunks[0].metadata.page_number == 7
    assert loaded.chunks[0].metadata.provenance == "official_court_source"


def test_snapshot_rejects_missing_semantic_vectors(tmp_path: Path) -> None:
    service = _OfflineEmbeddingService()
    store = RetrievalIndexStore(tmp_path / "index")
    store.save([_chunk()], embedding_service=service, semantic_indexed=True)

    with pytest.raises(RetrievalIndexStoreError, match="semantic index"):
        store.load(service)


def test_snapshot_rejects_duplicate_chunks(tmp_path: Path) -> None:
    store = RetrievalIndexStore(tmp_path / "index")

    with pytest.raises(RetrievalIndexStoreError, match="duplicate chunk IDs"):
        store.save([_chunk(), _chunk()])


def test_loaded_snapshot_is_searchable_with_public_response_schema(
    tmp_path: Path,
) -> None:
    embedding_service = _OfflineEmbeddingService()
    chunk = _chunk()
    store = RetrievalIndexStore(tmp_path / "index")
    EmbeddingCheckpointStore(store.embedding_checkpoint_path).put_many(
        embedding_service, [(chunk, [1.0, 0.0, 0.0])]
    )
    store.save(
        [chunk], embedding_service=embedding_service, semantic_indexed=True
    )
    loaded = store.load(embedding_service)
    agent = RetrievalAgent(embedding_service=embedding_service)
    agent.index_chunks(loaded.chunks)
    assert loaded.vectors is not None
    agent.load_semantic_index(loaded.chunks, loaded.vectors)

    response = RetrievalService(agent).search(
        RetrievalSearchRequest(query="legal question", mode="hybrid", top_k=1)
    )

    assert response.result_count == 1
    result = response.results[0]
    assert result.document_id == "ICJ-001"
    assert result.chunk_id == "ICJ-001-chunk-0001"
    assert result.citation == "I.C.J. Reports 2021, p. 206"
    assert result.metadata.page_number == 7
    assert embedding_service.document_calls == 0
    assert embedding_service.query_calls == 1
