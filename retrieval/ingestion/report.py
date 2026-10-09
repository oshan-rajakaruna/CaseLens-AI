"""Structured reporting contracts for dataset ingestion."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentIngestionReport(BaseModel):
    """Status and diagnostics for one manifest document."""

    manifest_position: int = Field(ge=1)
    document_id: str | None = None
    file_name: str | None = None
    status: Literal["processed", "failed", "skipped"]
    failure_category: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IngestionReport(BaseModel):
    """Machine-readable batch totals, per-document results, and index totals."""

    dataset_name: str
    data_classification: str
    strict: bool
    dry_run: bool
    semantic_requested: bool
    total_documents: int = Field(ge=0)
    successfully_processed: int = Field(default=0, ge=0)
    failed_documents: int = Field(default=0, ge=0)
    skipped_documents: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    bm25_indexed_chunks: int = Field(default=0, ge=0)
    semantic_chunks_planned: int = Field(default=0, ge=0)
    semantic_indexed_chunks: int = Field(default=0, ge=0)
    documents: list[DocumentIngestionReport] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
