"""Offline tests for the local FastEmbed provider."""

from pathlib import Path
from uuid import uuid4

import pytest

from retrieval.embeddings.local import (
    DEFAULT_LOCAL_EMBEDDING_DIMENSION,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    LocalEmbeddingResponseError,
    LocalEmbeddingService,
    LocalEmbeddingSettings,
)
from retrieval.ingestion.pipeline import IngestionPipeline

FIXTURES = Path("tests/fixtures/ingestion")


class _FakeTextEmbedding:
    def __init__(
        self,
        passage_vectors: list[list[object]],
        query_vectors: list[list[object]] | None = None,
    ) -> None:
        self.passage_vectors = passage_vectors
        self.query_vectors = query_vectors or passage_vectors
        self.passage_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[tuple[list[str], int]] = []

    def passage_embed(self, texts, *, batch_size: int):
        materialized = list(texts)
        self.passage_calls.append((materialized, batch_size))
        return iter(self.passage_vectors)

    def query_embed(self, texts, *, batch_size: int):
        materialized = list(texts)
        self.query_calls.append((materialized, batch_size))
        return iter(self.query_vectors)


class _FakeModelFactory:
    def __init__(self, model: _FakeTextEmbedding) -> None:
        self.model = model
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _FakeTextEmbedding:
        self.calls.append(kwargs)
        return self.model


def test_local_settings_resolve_relative_cache_against_application_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application_root = Path.cwd().resolve()
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
    monkeypatch.setenv("LOCAL_EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("LOCAL_EMBEDDING_CACHE_DIR", "data/models/fastembed")

    settings = LocalEmbeddingSettings.from_environment(
        application_root=application_root
    )

    assert settings.model == DEFAULT_LOCAL_EMBEDDING_MODEL
    assert settings.dimension == DEFAULT_LOCAL_EMBEDDING_DIMENSION
    assert settings.cache_dir == application_root / "data/models/fastembed"


def test_local_service_creates_cache_and_loads_model_lazily() -> None:
    cache_dir = (
        Path("data/legal/processed") / f"pytest-fastembed-{uuid4().hex}"
    ).resolve()
    model = _FakeTextEmbedding([[1.0, 2.0, 3.0]])
    factory = _FakeModelFactory(model)
    service = LocalEmbeddingService(
        LocalEmbeddingSettings(dimension=3, cache_dir=cache_dir),
        model_factory=factory,
    )

    try:
        assert factory.calls == []
        assert not cache_dir.exists()

        assert service.embed_query("state responsibility") == [1.0, 2.0, 3.0]

        assert cache_dir.is_dir()
        assert factory.calls == [
            {
                "model_name": DEFAULT_LOCAL_EMBEDDING_MODEL,
                "cache_dir": str(cache_dir),
                "lazy_load": True,
            }
        ]
    finally:
        cache_dir.rmdir()


def test_local_batch_embedding_preserves_input_output_order() -> None:
    model = _FakeTextEmbedding([[1, 2, 3], [4, 5, 6]])
    service = LocalEmbeddingService(
        LocalEmbeddingSettings(dimension=3),
        embedding_model=model,
        batch_size=8,
    )

    vectors = service.embed_documents(
        [("First legal text", "First Matter"), ("Second legal text", None)]
    )

    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert model.passage_calls == [
        (
            [
                "title: First Matter | text: First legal text",
                "title: none | text: Second legal text",
            ],
            8,
        )
    ]
    assert service.successful_embedding_count == 2
    assert service.request_count == 0


def test_local_provider_rejects_wrong_vector_count() -> None:
    service = LocalEmbeddingService(
        LocalEmbeddingSettings(dimension=3),
        embedding_model=_FakeTextEmbedding([[1, 2, 3]]),
    )

    with pytest.raises(LocalEmbeddingResponseError, match="number of embeddings"):
        service.embed_documents([("First", None), ("Second", None)])


@pytest.mark.parametrize(
    "values",
    [[1.0, 2.0], [1.0, "invalid", 3.0], [1.0, float("nan"), 3.0]],
)
def test_local_provider_validates_dimensions_and_values(
    values: list[object],
) -> None:
    service = LocalEmbeddingService(
        LocalEmbeddingSettings(dimension=3),
        embedding_model=_FakeTextEmbedding([values]),
    )

    with pytest.raises(LocalEmbeddingResponseError):
        service.embed_query("maritime delimitation")


def test_local_provider_telemetry_flows_through_ingestion_pipeline() -> None:
    model = _FakeTextEmbedding([])

    def passage_embed(texts, *, batch_size: int):
        materialized = list(texts)
        model.passage_calls.append((materialized, batch_size))
        return iter([[1.0] + [0.0] * 383 for _ in materialized])

    model.passage_embed = passage_embed  # type: ignore[method-assign]
    service = LocalEmbeddingService(
        LocalEmbeddingSettings(),
        embedding_model=model,
    )
    pipeline = IngestionPipeline(FIXTURES, chunk_size=10, overlap=2)
    outcome = pipeline.ingest(
        FIXTURES / "valid_manifest.json",
        semantic_requested=True,
    )

    pipeline.build_indexes(outcome, semantic=True, embedding_service=service)

    assert outcome.report.embedding_provider == "local"
    assert outcome.report.embedding_model == "BAAI/bge-small-en-v1.5"
    assert outcome.report.embedding_dimension == 384
    assert outcome.report.semantic_indexed_chunks == len(outcome.chunks)
