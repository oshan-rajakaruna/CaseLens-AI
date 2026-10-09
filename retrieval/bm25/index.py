"""A small, dependency-free Okapi BM25 index."""

from collections import Counter
from collections.abc import Iterable
import math
import re

from retrieval.preprocessing.metadata import LegalTextChunk

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize_for_search(text: str) -> list[str]:
    """Return case-insensitive Unicode word tokens without changing source text."""

    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """In-memory Okapi BM25 index retaining its source chunks."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = k1
        self.b = b
        self.chunks: list[LegalTextChunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        self._inverse_document_frequency: dict[str, float] = {}
        self._average_document_length = 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: Iterable[LegalTextChunk]) -> "BM25Index":
        """Build or replace the index from processed legal text chunks."""

        indexed_chunks = list(chunks)
        chunk_ids = [chunk.chunk_id for chunk in indexed_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id values must be unique")

        tokenized = [tokenize_for_search(chunk.chunk_text) for chunk in indexed_chunks]
        self.chunks = indexed_chunks
        self._term_frequencies = [Counter(tokens) for tokens in tokenized]
        self._document_lengths = [len(tokens) for tokens in tokenized]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())

        document_count = len(self.chunks)
        self._inverse_document_frequency = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return self

    def scores(self, query: str) -> list[float]:
        """Calculate an Okapi BM25 score for every indexed chunk."""

        if not self.chunks:
            return []
        query_terms = tokenize_for_search(query)
        if not query_terms:
            return [0.0] * len(self.chunks)

        scores: list[float] = []
        for frequencies, document_length in zip(
            self._term_frequencies, self._document_lengths, strict=True
        ):
            score = 0.0
            length_ratio = (
                document_length / self._average_document_length
                if self._average_document_length
                else 0.0
            )
            normalization = self.k1 * (1 - self.b + self.b * length_ratio)
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                score += self._inverse_document_frequency[term] * (
                    term_frequency * (self.k1 + 1)
                ) / (term_frequency + normalization)
            scores.append(score)
        return scores
