"""Tests for curated dataset summaries, duplicates, and index readiness."""

import json
from pathlib import Path
from unittest.mock import patch

from retrieval.embeddings.gemini import GeminiEmbeddingService
from retrieval.ingestion.dataset import prepare_dataset
from retrieval.ingestion.manifest import load_manifest
from retrieval.ingestion.prepare import main
from retrieval.ingestion.readiness import ReadinessStatus
from retrieval.ingestion.summary import summarize_manifest
from retrieval.ingestion.validator import safe_file_name_for_report

FIXTURES = Path("tests/fixtures/ingestion")
SUMMARY_MANIFEST = FIXTURES / "dataset_summary_manifest.json"
VALID_MANIFEST = FIXTURES / "valid_manifest.json"
MANIFEST_TEMPLATE = Path("data/legal/metadata/manifest.template.json")


def test_final_curated_manifest_template_is_valid_and_empty() -> None:
    manifest = load_manifest(MANIFEST_TEMPLATE)

    assert manifest.data_classification.value == "curated"
    assert manifest.documents == ()


def test_unsafe_file_names_are_sanitized_for_reports() -> None:
    assert safe_file_name_for_report("../private/legal.txt") == "<unsafe-path>"
    assert safe_file_name_for_report("approved/case.txt") == "approved/case.txt"


def test_dataset_summary_counts_categories_types_courts_and_file_types() -> None:
    summary = summarize_manifest(load_manifest(SUMMARY_MANIFEST))

    assert summary.total_documents == 4
    assert summary.valid_manifest_entries == 2
    assert summary.invalid_manifest_entries == 2
    assert summary.documents_by_legal_category == {"contract": 1, "employment": 2}
    assert summary.documents_by_document_type == {"judgment": 2, "statute": 1}
    assert summary.documents_by_court == {"Synthetic Test Court": 2}
    assert summary.txt_count == 2
    assert summary.pdf_count == 1
    assert summary.unsupported_file_count == 0
    assert summary.missing_optional_metadata["date"] == 3
    assert summary.missing_optional_metadata["court"] == 1


def test_dataset_summary_counts_provenance() -> None:
    summary = summarize_manifest(load_manifest(SUMMARY_MANIFEST))

    assert summary.documents_by_provenance == {
        "approved_public_source": 1,
        "official_court_source": 1,
        "team_curated_sample": 1,
    }


def test_duplicate_document_ids_are_reported_without_deletion() -> None:
    summary = summarize_manifest(load_manifest(SUMMARY_MANIFEST))

    duplicate = summary.duplicate_document_ids[0]
    assert duplicate.value == "summary-alpha"
    assert duplicate.manifest_positions == [1, 3]
    assert summary.total_documents == 4


def test_duplicate_files_citations_and_source_urls_are_reported() -> None:
    summary = summarize_manifest(load_manifest(SUMMARY_MANIFEST))

    assert summary.duplicate_file_references[0].value == "synthetic_alpha.txt"
    assert summary.duplicate_file_references[0].manifest_positions == [1, 2]
    assert summary.duplicate_citations[0].manifest_positions == [1, 2]
    assert summary.duplicate_source_urls[0].manifest_positions == [1, 2]


def test_invalid_manifest_entries_include_safe_labels_and_errors() -> None:
    summary = summarize_manifest(load_manifest(SUMMARY_MANIFEST))

    assert len(summary.invalid_entries) == 2
    assert summary.invalid_entries[0].document_id == "summary-alpha"
    assert "duplicate value" in summary.invalid_entries[0].errors[0]
    assert summary.invalid_entries[1].document_id is None
    assert "document_id" in summary.invalid_entries[1].errors[0]


def test_valid_dataset_is_bm25_ready_but_not_semantic_ready_without_key() -> None:
    report = prepare_dataset(
        VALID_MANIFEST,
        FIXTURES,
        chunk_size=10,
        overlap=2,
        gemini_api_key_configured=False,
    )

    assert report.readiness.overall_status is ReadinessStatus.READY_FOR_BM25_INDEXING
    assert report.readiness.bm25_status is ReadinessStatus.READY_FOR_BM25_INDEXING
    assert report.readiness.semantic_status is ReadinessStatus.NOT_READY
    assert report.readiness.gemini_api_key_configured is False
    assert report.readiness.semantic_blocking_errors == [
        "GEMINI_API_KEY is not configured"
    ]


def test_valid_dataset_is_semantic_ready_when_configuration_is_available() -> None:
    report = prepare_dataset(
        VALID_MANIFEST,
        FIXTURES,
        gemini_api_key_configured=True,
    )

    assert (
        report.readiness.overall_status
        is ReadinessStatus.READY_FOR_SEMANTIC_INDEXING
    )
    assert (
        report.readiness.semantic_status
        is ReadinessStatus.READY_FOR_SEMANTIC_INDEXING
    )


def test_openai_readiness_uses_the_selected_provider_key_name() -> None:
    report = prepare_dataset(
        VALID_MANIFEST,
        FIXTURES,
        embedding_provider="openai",
        embedding_api_key_configured=False,
    )

    assert report.readiness.embedding_provider == "openai"
    assert report.readiness.embedding_api_key_configured is False
    assert report.readiness.semantic_blocking_errors == [
        "OPENAI_API_KEY is not configured"
    ]


def test_invalid_and_duplicate_dataset_is_not_ready() -> None:
    report = prepare_dataset(
        SUMMARY_MANIFEST,
        FIXTURES,
        gemini_api_key_configured=True,
    )

    assert report.readiness.overall_status is ReadinessStatus.NOT_READY
    assert report.readiness.bm25_status is ReadinessStatus.NOT_READY
    assert report.readiness.semantic_status is ReadinessStatus.NOT_READY
    assert any("Duplicate document_id" in item for item in report.readiness.blocking_errors)
    assert any("Duplicate file" in item for item in report.readiness.blocking_errors)


def test_preparation_dry_run_builds_no_indexes_and_calls_no_gemini() -> None:
    with patch.object(
        GeminiEmbeddingService,
        "embed_document",
        side_effect=AssertionError("Gemini must not be called"),
    ) as embed_document:
        report = prepare_dataset(
            VALID_MANIFEST,
            FIXTURES,
            gemini_api_key_configured=True,
        )

    assert report.ingestion_dry_run.dry_run is True
    assert report.ingestion_dry_run.total_chunks > 0
    assert report.ingestion_dry_run.bm25_indexed_chunks == 0
    assert report.ingestion_dry_run.semantic_indexed_chunks == 0
    embed_document.assert_not_called()


def test_preparation_cli_prints_structured_summary(capsys) -> None:
    with patch(
        "retrieval.ingestion.dataset._gemini_api_key_configured",
        return_value=False,
    ):
        exit_code = main(
            [
                "--manifest",
                str(VALID_MANIFEST),
                "--input-dir",
                str(FIXTURES),
            ]
        )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["summary"]["total_documents"] == 2
    assert output["ingestion_dry_run"]["dry_run"] is True
    assert output["readiness"]["bm25_status"] == "READY_FOR_BM25_INDEXING"
