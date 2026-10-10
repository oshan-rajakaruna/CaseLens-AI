"""BM25 and semantic index-readiness assessment for curated datasets."""

from enum import StrEnum

from pydantic import BaseModel, Field

from retrieval.ingestion.report import IngestionReport
from retrieval.ingestion.summary import DatasetSummary


class ReadinessStatus(StrEnum):
    """Dataset readiness states used by preparation reports."""

    READY_FOR_BM25_INDEXING = "READY_FOR_BM25_INDEXING"
    READY_FOR_SEMANTIC_INDEXING = "READY_FOR_SEMANTIC_INDEXING"
    NOT_READY = "NOT_READY"


class IndexReadinessReport(BaseModel):
    """Separate readiness results because BM25 does not require Gemini."""

    overall_status: ReadinessStatus
    bm25_status: ReadinessStatus
    semantic_status: ReadinessStatus
    embedding_provider: str
    embedding_api_key_configured: bool
    gemini_api_key_configured: bool
    blocking_errors: list[str] = Field(default_factory=list)
    semantic_blocking_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def assess_index_readiness(
    summary: DatasetSummary,
    ingestion_report: IngestionReport,
    *,
    gemini_api_key_configured: bool | None = None,
    embedding_provider: str = "gemini",
    embedding_api_key_configured: bool | None = None,
) -> IndexReadinessReport:
    """Assess readiness from manifest analysis and a no-index ingestion dry-run."""

    blocking_errors: list[str] = []
    if summary.invalid_manifest_entries:
        blocking_errors.append(
            f"{summary.invalid_manifest_entries} manifest entry or entries are invalid"
        )
    if summary.duplicate_document_ids:
        blocking_errors.append("Duplicate document_id values must be resolved")
    if summary.duplicate_file_references:
        blocking_errors.append("Duplicate file references must be reviewed")
    if ingestion_report.failed_documents:
        blocking_errors.append(
            f"{ingestion_report.failed_documents} document or documents failed ingestion"
        )
    if not ingestion_report.successfully_processed:
        blocking_errors.append("No enabled documents produced searchable chunks")

    warnings: list[str] = []
    if summary.duplicate_citations:
        warnings.append("Duplicate citations require human review")
    if summary.duplicate_source_urls:
        warnings.append("Duplicate source URLs may indicate repeated sources")
    if ingestion_report.skipped_documents:
        warnings.append(
            f"{ingestion_report.skipped_documents} disabled document or documents were skipped"
        )

    bm25_ready = not blocking_errors
    selected_provider = embedding_provider.strip().casefold()
    key_configured = (
        embedding_api_key_configured
        if embedding_api_key_configured is not None
        else bool(gemini_api_key_configured)
    )
    gemini_key_configured = (
        bool(gemini_api_key_configured)
        if gemini_api_key_configured is not None
        else key_configured if selected_provider == "gemini" else False
    )
    semantic_blocking_errors = list(blocking_errors)
    if selected_provider != "local" and not key_configured:
        key_name = (
            "OPENAI_API_KEY" if selected_provider == "openai" else "GEMINI_API_KEY"
        )
        semantic_blocking_errors.append(f"{key_name} is not configured")
    semantic_ready = not semantic_blocking_errors

    bm25_status = (
        ReadinessStatus.READY_FOR_BM25_INDEXING
        if bm25_ready
        else ReadinessStatus.NOT_READY
    )
    semantic_status = (
        ReadinessStatus.READY_FOR_SEMANTIC_INDEXING
        if semantic_ready
        else ReadinessStatus.NOT_READY
    )
    overall_status = (
        semantic_status
        if semantic_ready
        else bm25_status if bm25_ready else ReadinessStatus.NOT_READY
    )
    return IndexReadinessReport(
        overall_status=overall_status,
        bm25_status=bm25_status,
        semantic_status=semantic_status,
        embedding_provider=selected_provider,
        embedding_api_key_configured=key_configured,
        gemini_api_key_configured=gemini_key_configured,
        blocking_errors=blocking_errors,
        semantic_blocking_errors=semantic_blocking_errors,
        warnings=warnings,
    )
