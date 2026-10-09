"""Tests for retrieval metadata contracts."""

import pytest
from pydantic import ValidationError

from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk


def test_optional_metadata_and_additional_fields_are_supported() -> None:
    metadata = LegalDocumentMetadata(
        document_id="case-001",
        citation="SYNTHETIC-001",
        jurisdiction="Test Jurisdiction",
    )
    chunk = LegalTextChunk(
        chunk_id="case-001-chunk-0001",
        document_id=metadata.document_id,
        chunk_text="Meaningful legal text.",
        metadata=metadata,
    )

    assert metadata.case_name is None
    assert metadata.model_extra == {"jurisdiction": "Test Jurisdiction"}
    assert chunk.metadata.citation == "SYNTHETIC-001"


def test_metadata_requires_non_empty_document_id() -> None:
    with pytest.raises(ValidationError):
        LegalDocumentMetadata(document_id="")


def test_chunk_requires_non_empty_text() -> None:
    metadata = LegalDocumentMetadata(document_id="case-001")

    with pytest.raises(ValidationError):
        LegalTextChunk(
            chunk_id="case-001-chunk-0001",
            document_id="case-001",
            chunk_text="",
            metadata=metadata,
        )
