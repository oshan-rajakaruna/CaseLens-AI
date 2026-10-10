"""Provider-, model-, dimension-, chunk-, and content-safe checkpoint tests."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from retrieval.embeddings.checkpoint import EmbeddingCheckpointStore
from retrieval.embeddings.gemini import GeminiEmbeddingService, GeminiEmbeddingSettings
from retrieval.embeddings.openai import OpenAIEmbeddingService, OpenAIEmbeddingSettings
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from retrieval.vector_store import SemanticSearchService


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        inputs = kwargs["input"]
        self.calls += 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[1.0, float(index + 1), 0.5])
                for index, _ in enumerate(inputs)  # type: ignore[arg-type]
            ]
        )


class _FakeClient:
    def __init__(self, embeddings: _FakeEmbeddings) -> None:
        self.embeddings = embeddings


def _chunk(text: str = "State responsibility for an internationally wrongful act.") -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id="ICJ-TEST",
        case_name="Synthetic State Responsibility Matter",
    )
    return LegalTextChunk(
        chunk_id="ICJ-TEST-chunk-0001",
        document_id="ICJ-TEST",
        chunk_text=text,
        metadata=metadata,
    )


def _openai_service(client: _FakeClient | None = None) -> OpenAIEmbeddingService:
    return OpenAIEmbeddingService(
        OpenAIEmbeddingSettings(api_key="", model="text-embedding-3-small", dimension=3),
        client=client,
        batch_size=10,
    )


@pytest.fixture
def checkpoint_path() -> Path:
    path = Path("data/legal/processed") / f"pytest-checkpoint-{uuid4().hex}.jsonl"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_checkpoint_invalidates_provider_model_dimension_and_content(
    checkpoint_path: Path,
) -> None:
    chunk = _chunk()
    openai_service = _openai_service()
    checkpoint = EmbeddingCheckpointStore(checkpoint_path)
    checkpoint.put_many(openai_service, [(chunk, [1.0, 2.0, 3.0])])

    assert EmbeddingCheckpointStore(checkpoint_path).get(openai_service, chunk) == [1.0, 2.0, 3.0]
    assert EmbeddingCheckpointStore(checkpoint_path).get(
        OpenAIEmbeddingService(
            OpenAIEmbeddingSettings(api_key="", model="text-embedding-3-large", dimension=3)
        ),
        chunk,
    ) is None
    assert EmbeddingCheckpointStore(checkpoint_path).get(
        OpenAIEmbeddingService(
            OpenAIEmbeddingSettings(api_key="", model="text-embedding-3-small", dimension=2)
        ),
        chunk,
    ) is None
    assert EmbeddingCheckpointStore(checkpoint_path).get(
        GeminiEmbeddingService(GeminiEmbeddingSettings(api_key="", dimension=3)),
        chunk,
    ) is None
    assert EmbeddingCheckpointStore(checkpoint_path).get(
        openai_service, _chunk("Changed text")
    ) is None


def test_semantic_rebuild_reuses_checkpoint_without_provider_calls(
    checkpoint_path: Path,
) -> None:
    first_models = _FakeEmbeddings()
    first_service = _openai_service(_FakeClient(first_models))
    first_search = SemanticSearchService(
        first_service,
        checkpoint=EmbeddingCheckpointStore(checkpoint_path),
    )
    first_search.build_index([_chunk()])

    second_models = _FakeEmbeddings()
    second_service = _openai_service(_FakeClient(second_models))
    second_search = SemanticSearchService(
        second_service,
        checkpoint=EmbeddingCheckpointStore(checkpoint_path),
    )
    second_search.build_index([_chunk()])

    assert first_models.calls == 1
    assert first_search.new_embeddings == 1
    assert second_models.calls == 0
    assert second_search.checkpoint_hits == 1
    assert len(second_search.index) == 1
