"""Dataset-level manifest summaries and non-destructive duplicate detection."""

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from retrieval.ingestion.manifest import (
    LegalDatasetManifest,
    LegalDocumentManifestEntry,
)

OPTIONAL_METADATA_FIELDS = (
    "case_name",
    "court",
    "date",
    "citation",
    "legal_category",
    "document_type",
    "source",
    "source_url",
    "notes",
)


class DuplicateGroup(BaseModel):
    """One repeated manifest value and every entry where it occurs."""

    value: str
    manifest_positions: list[int] = Field(min_length=2)
    document_ids: list[str | None] = Field(min_length=2)


class InvalidManifestEntry(BaseModel):
    """Safe labels and validation messages for one invalid entry."""

    manifest_position: int = Field(ge=1)
    document_id: str | None = None
    file_name: str | None = None
    errors: list[str] = Field(min_length=1)


class DatasetSummary(BaseModel):
    """Metadata-only dataset statistics; no source document text is included."""

    dataset_name: str
    data_classification: str
    total_documents: int = Field(ge=0)
    valid_manifest_entries: int = Field(ge=0)
    invalid_manifest_entries: int = Field(ge=0)
    disabled_documents: int = Field(ge=0)
    documents_by_legal_category: dict[str, int]
    documents_by_document_type: dict[str, int]
    documents_by_court: dict[str, int]
    documents_by_provenance: dict[str, int]
    txt_count: int = Field(ge=0)
    pdf_count: int = Field(ge=0)
    unsupported_file_count: int = Field(ge=0)
    missing_optional_metadata: dict[str, int]
    duplicate_document_ids: list[DuplicateGroup]
    duplicate_file_references: list[DuplicateGroup]
    duplicate_citations: list[DuplicateGroup]
    duplicate_source_urls: list[DuplicateGroup]
    invalid_entries: list[InvalidManifestEntry]


def _normalized_file_name(value: str) -> str:
    return str(PurePosixPath(value.replace("\\", "/"))).casefold()


def _safe_file_name(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return "<unsafe-path>"
    return str(PurePosixPath(value.replace("\\", "/")))


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _normalized_url(value: str) -> str:
    return value.strip().rstrip("/").casefold()


def _raw_text(raw_entry: dict[str, Any], field: str) -> str | None:
    value = raw_entry.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _duplicate_groups(
    manifest: LegalDatasetManifest,
    field: str,
    normalize: Callable[[str], str],
    *,
    safe_file_values: bool = False,
) -> list[DuplicateGroup]:
    grouped: dict[str, list[tuple[int, str | None, str]]] = defaultdict(list)
    for record in manifest.documents:
        raw_entry = record.raw_entry or {}
        value = _raw_text(raw_entry, field)
        if value is None:
            continue
        grouped[normalize(value)].append(
            (record.position, record.document_id, value)
        )

    duplicates: list[DuplicateGroup] = []
    for occurrences in grouped.values():
        if len(occurrences) < 2:
            continue
        display_value = occurrences[0][2]
        if safe_file_values:
            display_value = _safe_file_name(display_value) or "<missing>"
        duplicates.append(
            DuplicateGroup(
                value=display_value,
                manifest_positions=[item[0] for item in occurrences],
                document_ids=[item[1] for item in occurrences],
            )
        )
    return sorted(duplicates, key=lambda duplicate: duplicate.manifest_positions[0])


def _schema_valid_entries(
    manifest: LegalDatasetManifest,
) -> list[LegalDocumentManifestEntry]:
    entries: list[LegalDocumentManifestEntry] = []
    for record in manifest.documents:
        try:
            entries.append(
                LegalDocumentManifestEntry.model_validate(record.raw_entry or {})
            )
        except ValidationError:
            continue
    return entries


def _sorted_counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_manifest(manifest: LegalDatasetManifest) -> DatasetSummary:
    """Summarize metadata and duplicates without opening source documents."""

    entries = _schema_valid_entries(manifest)
    extensions = [Path(entry.file_name).suffix.casefold() for entry in entries]
    invalid_entries = [
        InvalidManifestEntry(
            manifest_position=record.position,
            document_id=record.document_id,
            file_name=_safe_file_name(record.file_name),
            errors=list(record.errors),
        )
        for record in manifest.documents
        if record.errors
    ]

    return DatasetSummary(
        dataset_name=manifest.dataset_name,
        data_classification=manifest.data_classification.value,
        total_documents=len(manifest.documents),
        valid_manifest_entries=len(manifest.documents) - len(invalid_entries),
        invalid_manifest_entries=len(invalid_entries),
        disabled_documents=sum(not entry.enabled for entry in entries),
        documents_by_legal_category=_sorted_counts(
            [entry.legal_category for entry in entries if entry.legal_category]
        ),
        documents_by_document_type=_sorted_counts(
            [entry.document_type for entry in entries if entry.document_type]
        ),
        documents_by_court=_sorted_counts(
            [entry.court for entry in entries if entry.court]
        ),
        documents_by_provenance=_sorted_counts(
            [entry.provenance.value for entry in entries]
        ),
        txt_count=extensions.count(".txt"),
        pdf_count=extensions.count(".pdf"),
        unsupported_file_count=sum(
            extension not in {".txt", ".pdf"} for extension in extensions
        ),
        missing_optional_metadata={
            field: sum(getattr(entry, field) is None for entry in entries)
            for field in OPTIONAL_METADATA_FIELDS
        },
        duplicate_document_ids=_duplicate_groups(
            manifest,
            "document_id",
            lambda value: value.strip(),
        ),
        duplicate_file_references=_duplicate_groups(
            manifest,
            "file_name",
            _normalized_file_name,
            safe_file_values=True,
        ),
        duplicate_citations=_duplicate_groups(
            manifest,
            "citation",
            _normalized_text,
        ),
        duplicate_source_urls=_duplicate_groups(
            manifest,
            "source_url",
            _normalized_url,
        ),
        invalid_entries=invalid_entries,
    )
