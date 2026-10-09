"""Tests for deterministic word-based chunking."""

import pytest

from retrieval.preprocessing.chunker import chunk_text
from retrieval.preprocessing.metadata import LegalDocumentMetadata


def test_chunker_uses_size_overlap_and_sequential_ids() -> None:
    metadata = LegalDocumentMetadata(
        document_id="employment-001",
        case_name="Synthetic Employment Matter",
        court="Synthetic Tribunal",
    )

    chunks = chunk_text(
        "one two three four five six seven eight nine ten",
        metadata,
        chunk_size=4,
        overlap=1,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "employment-001-chunk-0001",
        "employment-001-chunk-0002",
        "employment-001-chunk-0003",
    ]
    assert [chunk.chunk_text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert all(chunk.metadata.case_name == "Synthetic Employment Matter" for chunk in chunks)


def test_chunker_avoids_empty_chunks() -> None:
    metadata = LegalDocumentMetadata(document_id="empty")

    assert chunk_text(" \n\t ", metadata, chunk_size=10, overlap=0) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (5, -1), (5, 5), (5, 6)],
)
def test_chunker_validates_configuration(chunk_size: int, overlap: int) -> None:
    metadata = LegalDocumentMetadata(document_id="config")

    with pytest.raises(ValueError):
        chunk_text("some legal text", metadata, chunk_size=chunk_size, overlap=overlap)
