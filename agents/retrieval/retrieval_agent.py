"""Phase 1 / BM25-only Retrieval Agent foundation.

Semantic search, hybrid ranking, metadata filters, Gemini calls, and
Coordinator integration are intentionally outside this milestone.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from retrieval.bm25 import BM25Index, BM25Searcher
from retrieval.preprocessing.metadata import BM25SearchResult, LegalTextChunk


class RetrievalAgent:
    """Small agent-facing interface around the BM25 retrieval service."""

    def __init__(self, chunks: Iterable[LegalTextChunk] | None = None) -> None:
        self.index = BM25Index().build(chunks or [])
        self._searcher = BM25Searcher(self.index)

    def index_chunks(self, chunks: Iterable[LegalTextChunk]) -> None:
        """Replace the agent's current BM25 index."""

        self.index.build(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[BM25SearchResult]:
        """Search the BM25 index; metadata filters are reserved for later."""

        if filters:
            raise NotImplementedError(
                "Metadata filtering is not implemented in the BM25-only milestone"
            )
        return self._searcher.search(query, top_k=top_k)
