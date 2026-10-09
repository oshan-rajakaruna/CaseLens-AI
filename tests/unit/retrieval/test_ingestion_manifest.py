"""Tests for curated legal dataset manifest contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from retrieval.ingestion.manifest import (
    DatasetClassification,
    LegalDocumentManifestEntry,
    ManifestLoadError,
    Provenance,
    load_manifest,
)

FIXTURES = Path("tests/fixtures/ingestion")


def test_load_manifest_preserves_valid_metadata() -> None:
    manifest = load_manifest(FIXTURES / "valid_manifest.json")

    assert manifest.data_classification is DatasetClassification.EXAMPLE_ONLY
    assert len(manifest.documents) == 2
    entry = manifest.documents[0].entry
    assert entry is not None
    assert entry.document_id == "synthetic-alpha"
    assert entry.provenance is Provenance.TEAM_CURATED_SAMPLE
    assert entry.source_url == "https://example.invalid/synthetic-alpha"


def test_duplicate_document_id_is_retained_as_entry_failure() -> None:
    manifest = load_manifest(FIXTURES / "duplicate_manifest.json")

    assert manifest.documents[0].entry is not None
    assert manifest.documents[1].entry is None
    assert manifest.documents[1].errors == (
        "document_id: duplicate value in manifest",
    )


def test_invalid_document_metadata_does_not_hide_later_valid_entry() -> None:
    manifest = load_manifest(FIXTURES / "invalid_metadata_manifest.json")

    assert [record.entry is None for record in manifest.documents] == [
        True,
        True,
        True,
        False,
    ]
    assert "document_id" in manifest.documents[0].errors[0]
    assert "provenance" in manifest.documents[1].errors[0]
    assert "source_url" in manifest.documents[2].errors[0]


def test_manifest_entry_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LegalDocumentManifestEntry(
            document_id="synthetic",
            file_name="synthetic.txt",
            provenance="team_curated_sample",
            unreviewed_label="not allowed",
        )


def test_official_provenance_requires_a_traceable_source() -> None:
    with pytest.raises(ValidationError, match="require source or source_url"):
        LegalDocumentManifestEntry(
            document_id="official-without-source",
            file_name="official.txt",
            provenance="official_court_source",
        )

    entry = LegalDocumentManifestEntry(
        document_id="official-with-source",
        file_name="official.txt",
        source="Named official repository",
        provenance="official_court_source",
    )
    assert entry.source == "Named official repository"


def test_missing_manifest_has_safe_error() -> None:
    with pytest.raises(ManifestLoadError, match="not found") as error:
        load_manifest(FIXTURES / "does-not-exist.json")

    assert str(FIXTURES.resolve()) not in str(error.value)
