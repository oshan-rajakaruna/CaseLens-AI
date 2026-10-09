"""Manifest-driven curated legal-document ingestion pipeline."""

from dataclasses import dataclass
from pathlib import Path

from retrieval.bm25 import BM25Index
from retrieval.embeddings.base import EmbeddingService
from retrieval.ingestion.indexing import build_bm25_index, build_semantic_index
from retrieval.ingestion.manifest import LegalDatasetManifest, load_manifest
from retrieval.ingestion.report import DocumentIngestionReport, IngestionReport
from retrieval.ingestion.validator import DocumentValidationError, validate_document_path
from retrieval.preprocessing import (
    DocumentLoadError,
    chunk_text,
    clean_text,
    load_text_document,
)
from retrieval.preprocessing.metadata import LegalTextChunk
from retrieval.vector_store import SemanticSearchService


class StrictIngestionError(RuntimeError):
    """Raised on the first failed document or index when strict mode is set."""

    def __init__(self, report: IngestionReport) -> None:
        super().__init__("Strict ingestion stopped after a failure")
        self.report = report


@dataclass(slots=True)
class IngestionOutcome:
    """Processed chunks, structured report, and any in-memory indexes built."""

    chunks: list[LegalTextChunk]
    report: IngestionReport
    bm25_index: BM25Index | None = None
    semantic_search: SemanticSearchService | None = None


class IngestionPipeline:
    """Validate, load, clean, chunk, annotate, and index a legal dataset."""

    def __init__(
        self,
        input_dir: str | Path,
        *,
        chunk_size: int = 300,
        overlap: int = 50,
        strict: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.input_dir = Path(input_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strict = strict

    def ingest(
        self,
        manifest_path: str | Path,
        *,
        dry_run: bool = False,
        semantic_requested: bool = False,
    ) -> IngestionOutcome:
        """Process every valid manifest document without making API calls."""

        manifest = load_manifest(manifest_path)
        report = self._new_report(
            manifest,
            dry_run=dry_run,
            semantic_requested=semantic_requested,
        )
        chunks: list[LegalTextChunk] = []

        for record in manifest.documents:
            if record.errors:
                self._record_failure(
                    report,
                    position=record.position,
                    document_id=record.document_id,
                    file_name=record.file_name,
                    errors=list(record.errors),
                )
                continue

            entry = record.entry
            if entry is None:  # Defensive: load_manifest always supplies one here.
                self._record_failure(
                    report,
                    position=record.position,
                    document_id=record.document_id,
                    file_name=record.file_name,
                    errors=["Manifest entry could not be parsed"],
                )
                continue
            if not entry.enabled:
                report.documents.append(
                    DocumentIngestionReport(
                        manifest_position=record.position,
                        document_id=entry.document_id,
                        file_name=entry.file_name,
                        status="skipped",
                        warnings=["Document is disabled in the manifest"],
                    )
                )
                report.skipped_documents += 1
                continue

            try:
                file_path = validate_document_path(entry, self.input_dir)
                loaded_text = load_text_document(file_path)
                cleaned_text = clean_text(loaded_text)
                document_chunks = chunk_text(
                    cleaned_text,
                    entry.to_metadata(),
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                )
                if not document_chunks:
                    raise DocumentLoadError("Document produced no usable chunks")
            except DocumentValidationError as exc:
                self._record_failure(
                    report,
                    position=record.position,
                    document_id=entry.document_id,
                    file_name=entry.file_name,
                    errors=[str(exc)],
                )
                continue
            except (DocumentLoadError, OSError, UnicodeError):
                self._record_failure(
                    report,
                    position=record.position,
                    document_id=entry.document_id,
                    file_name=entry.file_name,
                    errors=["Document could not be loaded as non-empty UTF-8 text"],
                )
                continue

            chunks.extend(document_chunks)
            report.documents.append(
                DocumentIngestionReport(
                    manifest_position=record.position,
                    document_id=entry.document_id,
                    file_name=entry.file_name,
                    status="processed",
                    chunk_count=len(document_chunks),
                )
            )
            report.successfully_processed += 1
            report.total_chunks += len(document_chunks)

        report.semantic_chunks_planned = (
            report.total_chunks if semantic_requested else 0
        )
        return IngestionOutcome(chunks=chunks, report=report)

    def build_indexes(
        self,
        outcome: IngestionOutcome,
        *,
        semantic: bool = False,
        embedding_service: EmbeddingService | None = None,
    ) -> IngestionOutcome:
        """Build BM25 and optional semantic indexes for an ingestion outcome."""

        report = outcome.report
        if report.dry_run:
            return outcome

        try:
            outcome.bm25_index = build_bm25_index(outcome.chunks)
            report.bm25_indexed_chunks = len(outcome.bm25_index)
        except (TypeError, ValueError):
            report.errors.append("BM25 index build failed")
            if self.strict:
                raise StrictIngestionError(report) from None

        if not semantic:
            return outcome

        try:
            outcome.semantic_search = build_semantic_index(
                outcome.chunks,
                embedding_service=embedding_service,
            )
            report.semantic_indexed_chunks = len(outcome.semantic_search.index)
        except Exception:
            # Provider errors are intentionally converted to a stable message;
            # local paths, credentials, and document contents are not exposed.
            report.errors.append("Semantic index build failed")
            if self.strict:
                raise StrictIngestionError(report) from None
        return outcome

    def _record_failure(
        self,
        report: IngestionReport,
        *,
        position: int,
        document_id: str | None,
        file_name: str | None,
        errors: list[str],
    ) -> None:
        report.documents.append(
            DocumentIngestionReport(
                manifest_position=position,
                document_id=document_id,
                file_name=file_name,
                status="failed",
                errors=errors,
            )
        )
        report.failed_documents += 1
        if self.strict:
            raise StrictIngestionError(report)

    def _new_report(
        self,
        manifest: LegalDatasetManifest,
        *,
        dry_run: bool,
        semantic_requested: bool,
    ) -> IngestionReport:
        return IngestionReport(
            dataset_name=manifest.dataset_name,
            data_classification=manifest.data_classification.value,
            strict=self.strict,
            dry_run=dry_run,
            semantic_requested=semantic_requested,
            total_documents=len(manifest.documents),
        )
