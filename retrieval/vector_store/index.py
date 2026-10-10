"""A lightweight, in-memory cosine-similarity vector index."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real

from retrieval.diversification import diversify_ranked_results
from retrieval.preprocessing.metadata import LegalTextChunk


class VectorIndexError(ValueError):
    """Base error for invalid vector index operations."""


class VectorDimensionError(VectorIndexError):
    """Raised when a vector has the wrong dimension."""


class InvalidVectorError(VectorIndexError):
    """Raised when a vector is non-numeric, non-finite, or has zero magnitude."""


class DuplicateChunkError(VectorIndexError):
    """Raised when the same chunk identifier is inserted more than once."""


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """One cosine-similarity match from the vector index."""

    similarity_score: float
    chunk: LegalTextChunk


@dataclass(frozen=True, slots=True)
class _VectorEntry:
    normalized_vector: tuple[float, ...]
    chunk: LegalTextChunk


class InMemoryVectorIndex:
    """Deterministic local index storing normalized vectors and source chunks."""

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = dimension
        self._entries: list[_VectorEntry] = []
        self._chunk_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, vector: Sequence[float], chunk: LegalTextChunk) -> None:
        """Add a validated vector and retain its original chunk metadata."""

        if chunk.chunk_id in self._chunk_ids:
            raise DuplicateChunkError(f"Duplicate chunk_id: {chunk.chunk_id}")
        normalized = self._normalize(vector)
        self._entries.append(_VectorEntry(normalized, chunk))
        self._chunk_ids.add(chunk.chunk_id)

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        *,
        max_chunks_per_document: int | None = None,
    ) -> list[VectorMatch]:
        """Return the highest cosine-similarity matches with stable tie ordering."""

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if top_k == 0 or not self._entries:
            return []

        normalized_query = self._normalize(query_vector)
        matches = [
            VectorMatch(
                similarity_score=max(
                    -1.0,
                    min(
                        1.0,
                        math.fsum(
                            query_value * document_value
                            for query_value, document_value in zip(
                                normalized_query,
                                entry.normalized_vector,
                                strict=True,
                            )
                        ),
                    ),
                ),
                chunk=entry.chunk,
            )
            for entry in self._entries
        ]
        matches.sort(
            key=lambda match: (
                -match.similarity_score,
                match.chunk.document_id,
                match.chunk.chunk_id,
            )
        )
        return diversify_ranked_results(
            matches,
            top_k=top_k,
            max_chunks_per_document=max_chunks_per_document,
            document_id=lambda match: match.chunk.document_id,
        )

    def _normalize(self, vector: Sequence[float]) -> tuple[float, ...]:
        if len(vector) != self.dimension:
            raise VectorDimensionError(
                f"Expected vector dimension {self.dimension}, received {len(vector)}"
            )

        numeric_vector: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise InvalidVectorError("Vector values must be numeric")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise InvalidVectorError("Vector values must be finite")
            numeric_vector.append(numeric_value)

        magnitude = math.sqrt(math.fsum(value * value for value in numeric_vector))
        if magnitude == 0:
            raise InvalidVectorError("Zero vectors cannot be indexed or searched")
        return tuple(value / magnitude for value in numeric_vector)
