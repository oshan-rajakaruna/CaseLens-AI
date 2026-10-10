"""Tests for file validation, batch ingestion, and index orchestration."""

from pathlib import Path

import pytest

from retrieval.ingestion.manifest import LegalDocumentManifestEntry
from retrieval.ingestion.pipeline import IngestionPipeline, StrictIngestionError
from retrieval.ingestion.validator import (
    DocumentValidationError,
    validate_document_path,
)
from tests.unit.retrieval._fakes import DeterministicEmbeddingService

FIXTURES = Path("tests/fixtures/ingestion")


def test_valid_batch_is_cleaned_chunked_and_metadata_is_attached() -> None:
    pipeline = IngestionPipeline(FIXTURES, chunk_size=8, overlap=2)

    outcome = pipeline.ingest(FIXTURES / "valid_manifest.json")

    assert outcome.report.total_documents == 2
    assert outcome.report.successfully_processed == 2
    assert outcome.report.failed_documents == 0
    assert outcome.report.total_chunks == len(outcome.chunks)
    assert outcome.report.total_chunks > 2
    alpha = next(
        chunk for chunk in outcome.chunks if chunk.document_id == "synthetic-alpha"
    )
    assert alpha.metadata.case_name == "Synthetic Alpha Matter"
    assert alpha.metadata.source_url == "https://example.invalid/synthetic-alpha"
    assert alpha.metadata.provenance == "team_curated_sample"
    assert alpha.metadata.file_name == "synthetic_alpha.txt"


def test_non_strict_batch_continues_after_missing_and_unsupported_files() -> None:
    outcome = IngestionPipeline(FIXTURES).ingest(
        FIXTURES / "mixed_manifest.json"
    )

    assert outcome.report.successfully_processed == 1
    assert outcome.report.failed_documents == 2
    assert outcome.report.skipped_documents == 1
    assert [document.status for document in outcome.report.documents] == [
        "processed",
        "failed",
        "failed",
        "skipped",
    ]
    assert outcome.report.documents[1].errors == ["Referenced file does not exist"]
    assert outcome.report.documents[2].errors == [
        "Unsupported file type; supported types: .txt, .pdf"
    ]
    assert outcome.report.documents[2].failure_category == "unsupported_file_type"


def test_strict_batch_stops_at_first_document_failure() -> None:
    pipeline = IngestionPipeline(FIXTURES, strict=True)

    with pytest.raises(StrictIngestionError) as error:
        pipeline.ingest(FIXTURES / "mixed_manifest.json")

    report = error.value.report
    assert report.successfully_processed == 1
    assert report.failed_documents == 1
    assert len(report.documents) == 2


def test_empty_document_is_reported_without_exposing_local_path() -> None:
    outcome = IngestionPipeline(FIXTURES).ingest(
        FIXTURES / "empty_document_manifest.json"
    )

    assert outcome.report.failed_documents == 1
    message = outcome.report.documents[0].errors[0]
    assert message == "Document could not be loaded as non-empty UTF-8 text"
    assert str(FIXTURES.resolve()) not in message


def test_duplicate_and_invalid_metadata_entries_are_reported() -> None:
    duplicate = IngestionPipeline(FIXTURES).ingest(
        FIXTURES / "duplicate_manifest.json"
    )
    invalid = IngestionPipeline(FIXTURES).ingest(
        FIXTURES / "invalid_metadata_manifest.json"
    )

    assert duplicate.report.successfully_processed == 1
    assert duplicate.report.failed_documents == 1
    assert invalid.report.successfully_processed == 1
    assert invalid.report.failed_documents == 3


def test_file_validation_blocks_path_traversal() -> None:
    entry = LegalDocumentManifestEntry(
        document_id="escape",
        file_name="../outside.txt",
        provenance="team_curated_sample",
    )

    with pytest.raises(DocumentValidationError, match="within input-dir"):
        validate_document_path(entry, FIXTURES)


def test_bm25_index_contains_every_successful_chunk() -> None:
    pipeline = IngestionPipeline(FIXTURES, chunk_size=10, overlap=2)
    outcome = pipeline.ingest(FIXTURES / "valid_manifest.json")

    pipeline.build_indexes(outcome)

    assert outcome.bm25_index is not None
    assert len(outcome.bm25_index) == len(outcome.chunks)
    assert outcome.report.bm25_indexed_chunks == len(outcome.chunks)
    assert outcome.semantic_search is None


def test_semantic_indexing_uses_injected_service_only_when_requested() -> None:
    embeddings = DeterministicEmbeddingService()
    pipeline = IngestionPipeline(FIXTURES)
    outcome = pipeline.ingest(
        FIXTURES / "valid_manifest.json",
        semantic_requested=True,
    )

    pipeline.build_indexes(
        outcome,
        semantic=True,
        embedding_service=embeddings,
    )

    assert outcome.report.semantic_chunks_planned == len(outcome.chunks)
    assert outcome.report.semantic_indexed_chunks == len(outcome.chunks)
    assert len(embeddings.document_calls) == len(outcome.chunks)


def test_dry_run_builds_no_indexes_and_makes_zero_embedding_calls() -> None:
    embeddings = DeterministicEmbeddingService()
    pipeline = IngestionPipeline(FIXTURES, chunk_size=9, overlap=1)
    outcome = pipeline.ingest(
        FIXTURES / "valid_manifest.json",
        dry_run=True,
        semantic_requested=True,
    )

    pipeline.build_indexes(
        outcome,
        semantic=True,
        embedding_service=embeddings,
    )

    assert outcome.report.total_chunks > 0
    assert outcome.report.semantic_chunks_planned == outcome.report.total_chunks
    assert outcome.report.bm25_indexed_chunks == 0
    assert outcome.report.semantic_indexed_chunks == 0
    assert outcome.bm25_index is None
    assert outcome.semantic_search is None
    assert embeddings.document_calls == []


def test_missing_openai_key_is_explicit_before_any_provider_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSION", "1536")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = IngestionPipeline(FIXTURES)
    outcome = pipeline.ingest(
        FIXTURES / "valid_manifest.json",
        semantic_requested=True,
    )

    pipeline.build_indexes(outcome, semantic=True)

    assert outcome.report.embedding_provider == "openai"
    assert outcome.report.embedding_model == "text-embedding-3-small"
    assert outcome.report.embedding_dimension == 1536
    assert outcome.report.embedding_requests == 0
    assert outcome.report.semantic_indexed_chunks == 0
    assert outcome.report.errors == [
        "Semantic provider initialization failed: "
        "OPENAI_API_KEY is required for semantic indexing"
    ]


def test_unknown_provider_error_is_explicit_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "not-a-provider")
    pipeline = IngestionPipeline(FIXTURES)
    outcome = pipeline.ingest(
        FIXTURES / "valid_manifest.json",
        semantic_requested=True,
    )

    pipeline.build_indexes(outcome, semantic=True)

    assert outcome.report.embedding_requests == 0
    assert outcome.report.errors == [
        "Semantic provider initialization failed: "
        "EMBEDDING_PROVIDER must be 'gemini' or 'openai'"
    ]


def test_semantic_runtime_error_does_not_leak_secret() -> None:
    secret = "secret-that-must-never-appear"

    class FailingEmbeddingService:
        provider = "openai"
        model = "test-model"
        dimension = 3
        batch_size = 100
        request_count = 0
        retry_count = 0

        def embed_documents(self, documents):
            raise RuntimeError(f"provider failure contained {secret}")

    pipeline = IngestionPipeline(FIXTURES)
    outcome = pipeline.ingest(
        FIXTURES / "valid_manifest.json",
        semantic_requested=True,
    )

    pipeline.build_indexes(
        outcome,
        semantic=True,
        embedding_service=FailingEmbeddingService(),
    )

    serialized_report = outcome.report.model_dump_json()
    assert secret not in serialized_report
    assert outcome.report.errors == [
        "Semantic index build failed for provider 'openai'"
    ]
