"""Tests for PDF manifests using the existing ingestion and index pipeline."""

from pathlib import Path

import pytest

from retrieval.ingestion.pipeline import IngestionPipeline, StrictIngestionError
from tests.unit.retrieval._fakes import DeterministicEmbeddingService

FIXTURES = Path("tests/fixtures/pdf")


def test_pdf_manifest_produces_page_aware_chunks_and_bm25_index(
) -> None:
    pipeline = IngestionPipeline(FIXTURES, chunk_size=5, overlap=1)

    outcome = pipeline.ingest(FIXTURES / "valid_pdf_manifest.json")
    pipeline.build_indexes(outcome)

    assert outcome.report.successfully_processed == 1
    assert outcome.report.failed_documents == 0
    assert outcome.report.documents[0].warnings == [
        "1 PDF page(s) contained no extractable text"
    ]
    assert [chunk.metadata.page_number for chunk in outcome.chunks] == [1, 1, 3, 3]
    assert all(chunk.metadata.source_file_type == "pdf" for chunk in outcome.chunks)
    assert all(
        chunk.metadata.page_start == chunk.metadata.page_end
        for chunk in outcome.chunks
    )
    assert outcome.bm25_index is not None
    assert outcome.report.bm25_indexed_chunks == len(outcome.chunks)


def test_non_strict_pdf_failure_continues_to_next_document() -> None:
    outcome = IngestionPipeline(FIXTURES).ingest(
        FIXTURES / "mixed_pdf_manifest.json"
    )

    assert outcome.report.failed_documents == 1
    assert outcome.report.successfully_processed == 1
    assert outcome.report.documents[0].failure_category == "unreadable_pdf"
    assert outcome.report.documents[1].status == "processed"


def test_strict_pdf_failure_stops_before_next_document() -> None:
    with pytest.raises(StrictIngestionError) as error:
        IngestionPipeline(FIXTURES, strict=True).ingest(
            FIXTURES / "mixed_pdf_manifest.json"
        )

    assert error.value.report.failed_documents == 1
    assert error.value.report.successfully_processed == 0
    assert len(error.value.report.documents) == 1
    assert error.value.report.documents[0].failure_category == "unreadable_pdf"


def test_no_text_pdf_is_categorized_as_possible_scanned() -> None:
    outcome = IngestionPipeline(FIXTURES).ingest(
        FIXTURES / "no_text_pdf_manifest.json"
    )

    failure = outcome.report.documents[0]
    assert failure.failure_category == "possible_scanned_pdf"
    assert "OCR is not supported" in failure.errors[0]
    assert str(FIXTURES.resolve()) not in failure.errors[0]


def test_pdf_semantic_indexing_uses_only_injected_fake() -> None:
    embeddings = DeterministicEmbeddingService()
    pipeline = IngestionPipeline(FIXTURES)
    outcome = pipeline.ingest(
        FIXTURES / "semantic_pdf_manifest.json",
        semantic_requested=True,
    )

    pipeline.build_indexes(
        outcome,
        semantic=True,
        embedding_service=embeddings,
    )

    assert outcome.report.semantic_indexed_chunks == len(outcome.chunks)
    assert len(embeddings.document_calls) == len(outcome.chunks)
