"""Reusable curated-dataset preparation orchestration."""

from pathlib import Path

from pydantic import BaseModel

from retrieval.embeddings.gemini import (
    EmbeddingConfigurationError,
    GeminiEmbeddingSettings,
)
from retrieval.ingestion.manifest import load_manifest
from retrieval.ingestion.pipeline import IngestionPipeline
from retrieval.ingestion.readiness import IndexReadinessReport, assess_index_readiness
from retrieval.ingestion.report import IngestionReport
from retrieval.ingestion.summary import DatasetSummary, summarize_manifest


class DatasetPreparationReport(BaseModel):
    """Metadata summary, ingestion dry-run, and readiness in one safe report."""

    summary: DatasetSummary
    ingestion_dry_run: IngestionReport
    readiness: IndexReadinessReport


def _gemini_api_key_configured() -> bool:
    """Inspect backend configuration without printing the key or making a call."""

    try:
        settings = GeminiEmbeddingSettings.from_environment()
    except EmbeddingConfigurationError:
        return False
    return bool(settings.api_key.strip())


def prepare_dataset(
    manifest_path: str | Path,
    input_dir: str | Path,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    gemini_api_key_configured: bool | None = None,
) -> DatasetPreparationReport:
    """Summarize and dry-run a dataset without building either index."""

    manifest = load_manifest(manifest_path)
    summary = summarize_manifest(manifest)
    outcome = IngestionPipeline(
        input_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    ).ingest(
        manifest_path,
        dry_run=True,
        semantic_requested=True,
    )
    key_configured = (
        _gemini_api_key_configured()
        if gemini_api_key_configured is None
        else gemini_api_key_configured
    )
    readiness = assess_index_readiness(
        summary,
        outcome.report,
        gemini_api_key_configured=key_configured,
    )
    return DatasetPreparationReport(
        summary=summary,
        ingestion_dry_run=outcome.report,
        readiness=readiness,
    )
