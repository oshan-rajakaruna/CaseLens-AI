"""Reusable curated-dataset preparation orchestration."""

from pathlib import Path

from pydantic import BaseModel

from retrieval.embeddings.factory import (
    configured_embedding_provider,
    create_embedding_service,
)
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


def _embedding_configuration() -> tuple[str, bool]:
    """Inspect the selected provider without logging keys or making a call."""

    provider = configured_embedding_provider()
    service = create_embedding_service(provider)
    settings = getattr(service, "settings", None)
    return provider, bool(getattr(settings, "api_key", "").strip())


def prepare_dataset(
    manifest_path: str | Path,
    input_dir: str | Path,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    gemini_api_key_configured: bool | None = None,
    embedding_provider: str | None = None,
    embedding_api_key_configured: bool | None = None,
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
    if gemini_api_key_configured is not None:
        selected_provider = "gemini"
        key_configured = gemini_api_key_configured
    elif embedding_provider is not None and embedding_api_key_configured is not None:
        selected_provider = embedding_provider
        key_configured = embedding_api_key_configured
    else:
        selected_provider, key_configured = _embedding_configuration()
    readiness = assess_index_readiness(
        summary,
        outcome.report,
        embedding_provider=selected_provider,
        embedding_api_key_configured=key_configured,
        gemini_api_key_configured=(
            key_configured if selected_provider == "gemini" else None
        ),
    )
    return DatasetPreparationReport(
        summary=summary,
        ingestion_dry_run=outcome.report,
        readiness=readiness,
    )
