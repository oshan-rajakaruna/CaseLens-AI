"""JSON manifest contracts for curated legal documents.

Top-level manifest failures are fatal. Document-entry failures are retained on
their records so non-strict ingestion can report them and continue safely.
"""

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from retrieval.preprocessing.metadata import LegalDocumentMetadata


class ManifestLoadError(ValueError):
    """Raised when a manifest file or its top-level structure is invalid."""


class Provenance(StrEnum):
    """Approved provenance labels for an ingested legal document."""

    OFFICIAL_COURT_SOURCE = "official_court_source"
    OFFICIAL_LEGISLATION_SOURCE = "official_legislation_source"
    APPROVED_PUBLIC_SOURCE = "approved_public_source"
    TEAM_CURATED_SAMPLE = "team_curated_sample"


class DatasetClassification(StrEnum):
    """Distinguish curated material from non-legal example manifests."""

    CURATED = "curated"
    EXAMPLE_ONLY = "example_only"


class LegalDocumentManifestEntry(BaseModel):
    """One source document and the metadata attached to every derived chunk."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    case_name: str | None = None
    court: str | None = None
    date: str | None = None
    citation: str | None = None
    legal_category: str | None = None
    document_type: str | None = None
    source: str | None = None
    source_url: str | None = None
    provenance: Provenance
    notes: str | None = None
    enabled: bool = True

    @field_validator("document_id", "file_name", mode="before")
    @classmethod
    def _strip_required_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator(
        "case_name",
        "court",
        "date",
        "citation",
        "legal_category",
        "document_type",
        "source",
        "notes",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("source_url", mode="before")
    @classmethod
    def _validate_source_url(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            return None
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("source_url cannot contain credentials")
        return normalized

    @model_validator(mode="after")
    def _require_traceable_official_source(self) -> "LegalDocumentManifestEntry":
        if (
            self.provenance is not Provenance.TEAM_CURATED_SAMPLE
            and self.source is None
            and self.source_url is None
        ):
            raise ValueError(
                "official and approved sources require source or source_url"
            )
        return self

    def to_metadata(self) -> LegalDocumentMetadata:
        """Convert manifest metadata to the existing retrieval contract."""

        return LegalDocumentMetadata(
            document_id=self.document_id,
            file_name=self.file_name,
            case_name=self.case_name,
            court=self.court,
            date=self.date,
            citation=self.citation,
            legal_category=self.legal_category,
            document_type=self.document_type,
            source=self.source,
            source_url=self.source_url,
            provenance=self.provenance.value,
            notes=self.notes,
        )


class _ManifestEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    data_classification: DatasetClassification
    description: str | None = None
    documents: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ManifestDocumentRecord:
    """A parsed entry or its safe, document-scoped validation messages."""

    position: int
    entry: LegalDocumentManifestEntry | None
    document_id: str | None
    file_name: str | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LegalDatasetManifest:
    """Loaded manifest with independently validated document records."""

    manifest_version: str
    dataset_name: str
    data_classification: DatasetClassification
    description: str | None
    documents: tuple[ManifestDocumentRecord, ...]


def _entry_label(raw_entry: dict[str, Any], field: str) -> str | None:
    value = raw_entry.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _validation_messages(error: ValidationError) -> tuple[str, ...]:
    messages: list[str] = []
    for detail in error.errors(include_url=False, include_context=False):
        location = ".".join(str(part) for part in detail["loc"])
        messages.append(f"{location or 'entry'}: {detail['msg']}")
    return tuple(messages)


def load_manifest(path: str | Path) -> LegalDatasetManifest:
    """Load a UTF-8 JSON manifest without failing the whole batch per entry."""

    manifest_path = Path(path)
    try:
        raw_data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ManifestLoadError("Manifest file was not found") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestLoadError("Manifest is not readable UTF-8 JSON") from exc

    try:
        envelope = _ManifestEnvelope.model_validate(raw_data)
    except ValidationError as exc:
        raise ManifestLoadError("Manifest top-level structure is invalid") from exc

    records: list[ManifestDocumentRecord] = []
    seen_document_ids: set[str] = set()
    for position, raw_entry in enumerate(envelope.documents, start=1):
        document_id = _entry_label(raw_entry, "document_id")
        file_name = _entry_label(raw_entry, "file_name")
        try:
            entry = LegalDocumentManifestEntry.model_validate(raw_entry)
        except ValidationError as exc:
            records.append(
                ManifestDocumentRecord(
                    position=position,
                    entry=None,
                    document_id=document_id,
                    file_name=file_name,
                    errors=_validation_messages(exc),
                )
            )
            continue

        if entry.document_id in seen_document_ids:
            records.append(
                ManifestDocumentRecord(
                    position=position,
                    entry=None,
                    document_id=entry.document_id,
                    file_name=entry.file_name,
                    errors=("document_id: duplicate value in manifest",),
                )
            )
            continue

        seen_document_ids.add(entry.document_id)
        records.append(
            ManifestDocumentRecord(
                position=position,
                entry=entry,
                document_id=entry.document_id,
                file_name=entry.file_name,
            )
        )

    return LegalDatasetManifest(
        manifest_version=envelope.manifest_version,
        dataset_name=envelope.dataset_name,
        data_classification=envelope.data_classification,
        description=envelope.description,
        documents=tuple(records),
    )
