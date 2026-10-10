"""Tests for the local cosine-similarity vector index."""

import math

import pytest

from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from retrieval.vector_store.index import (
    DuplicateChunkError,
    InMemoryVectorIndex,
    InvalidVectorError,
    VectorDimensionError,
)


def _chunk(document_id: str, chunk_number: int = 1) -> LegalTextChunk:
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
        document_id=document_id,
        chunk_text=f"Synthetic {document_id} text",
        metadata=LegalDocumentMetadata(document_id=document_id),
    )


def test_vector_index_calculates_cosine_similarity_and_ranks_results() -> None:
    index = InMemoryVectorIndex(dimension=3)
    index.add([1.0, 0.0, 0.0], _chunk("employment"))
    index.add([0.0, 1.0, 0.0], _chunk("property"))
    index.add([0.8, 0.2, 0.0], _chunk("contract"))

    matches = index.search([1.0, 0.0, 0.0], top_k=2)

    assert [match.chunk.document_id for match in matches] == [
        "employment",
        "contract",
    ]
    assert matches[0].similarity_score == pytest.approx(1.0)
    assert matches[1].similarity_score == pytest.approx(0.8 / math.sqrt(0.68))


def test_vector_index_has_deterministic_tie_ordering() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add([1.0, 0.0], _chunk("property"))
    index.add([1.0, 0.0], _chunk("employment"))

    matches = index.search([1.0, 0.0], top_k=10)

    assert [match.chunk.document_id for match in matches] == [
        "employment",
        "property",
    ]


def test_vector_index_caps_ranked_chunks_by_metadata_document_id() -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add([1.0, 0.0], _chunk("dominant", 1))
    index.add([0.99, 0.01], _chunk("dominant", 2))
    index.add([0.98, 0.02], _chunk("dominant", 3))
    index.add([0.9, 0.1], _chunk("other", 1))

    matches = index.search(
        [1.0, 0.0],
        top_k=3,
        max_chunks_per_document=2,
    )

    assert [match.chunk.chunk_id for match in matches] == [
        "dominant-chunk-0001",
        "dominant-chunk-0002",
        "other-chunk-0001",
    ]


def test_vector_index_validates_dimensions_and_zero_vectors() -> None:
    index = InMemoryVectorIndex(dimension=3)

    with pytest.raises(VectorDimensionError, match="Expected vector dimension 3"):
        index.add([1.0, 2.0], _chunk("short"))
    with pytest.raises(InvalidVectorError, match="Zero vectors"):
        index.add([0.0, 0.0, 0.0], _chunk("zero"))
    with pytest.raises(InvalidVectorError, match="Zero vectors"):
        index.add([1.0, 0.0, 0.0], _chunk("valid"))
        index.search([0.0, 0.0, 0.0])


def test_vector_index_rejects_duplicate_chunk_ids() -> None:
    index = InMemoryVectorIndex(dimension=2)
    chunk = _chunk("duplicate")
    index.add([1.0, 0.0], chunk)

    with pytest.raises(DuplicateChunkError, match="Duplicate chunk_id"):
        index.add([0.0, 1.0], chunk)


@pytest.mark.parametrize("top_k", [-1, 1.5, True])
def test_vector_index_validates_top_k(top_k: object) -> None:
    index = InMemoryVectorIndex(dimension=2)
    index.add([1.0, 0.0], _chunk("employment"))

    with pytest.raises((TypeError, ValueError)):
        index.search([1.0, 0.0], top_k=top_k)  # type: ignore[arg-type]
