"""Curated legal-dataset validation, ingestion, and index building."""

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
from retrieval.ingestion.report import DocumentIngestionReport, IngestionReport

__all__ = [
    "DatasetClassification",
    "DocumentIngestionReport",
    "IngestionOutcome",
    "IngestionPipeline",
    "IngestionReport",
    "LegalDatasetManifest",
    "LegalDocumentManifestEntry",
    "ManifestLoadError",
    "Provenance",
    "StrictIngestionError",
    "build_bm25_index",
    "build_semantic_index",
    "load_manifest",
]
