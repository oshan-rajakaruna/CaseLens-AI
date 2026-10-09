"""Curated legal-dataset validation, ingestion, and index building."""

from retrieval.ingestion.dataset import DatasetPreparationReport, prepare_dataset
from retrieval.ingestion.indexing import build_bm25_index, build_semantic_index
from retrieval.ingestion.manifest import (
    DatasetClassification,
    LegalDatasetManifest,
    LegalDocumentManifestEntry,
    ManifestLoadError,
    Provenance,
    load_manifest,
)
from retrieval.ingestion.pipeline import (
    IngestionOutcome,
    IngestionPipeline,
    StrictIngestionError,
)
from retrieval.ingestion.readiness import IndexReadinessReport, ReadinessStatus
from retrieval.ingestion.report import DocumentIngestionReport, IngestionReport
from retrieval.ingestion.summary import DatasetSummary, DuplicateGroup, summarize_manifest

__all__ = [
    "DatasetClassification",
    "DatasetPreparationReport",
    "DatasetSummary",
    "DocumentIngestionReport",
    "DuplicateGroup",
    "IngestionOutcome",
    "IngestionPipeline",
    "IngestionReport",
    "IndexReadinessReport",
    "LegalDatasetManifest",
    "LegalDocumentManifestEntry",
    "ManifestLoadError",
    "Provenance",
    "ReadinessStatus",
    "StrictIngestionError",
    "build_bm25_index",
    "build_semantic_index",
    "load_manifest",
    "prepare_dataset",
    "summarize_manifest",
]
