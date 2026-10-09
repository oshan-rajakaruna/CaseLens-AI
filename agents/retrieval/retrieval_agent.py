"""Retrieval Agent with independent BM25 and semantic search modes.

Hybrid ranking, metadata filters, and Coordinator integration remain outside
this milestone.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from retrieval.bm25 import BM25Index, BM25Searcher
from retrieval.embeddings import EmbeddingService, GeminiEmbeddingService
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    LegalTextChunk,
    SemanticSearchResult,
)
from retrieval.vector_store import SemanticSearchService


class RetrievalAgent:
    """Agent-facing interface with deliberately separate retrieval modes."""

    def __init__(
        self,
        chunks: Iterable[LegalTextChunk] | None = None,
        *,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.index = BM25Index().build(chunks or [])
        self._searcher = BM25Searcher(self.index)
        self._semantic = SemanticSearchService(
            embedding_service or GeminiEmbeddingService()
        )

    def index_chunks(self, chunks: Iterable[LegalTextChunk]) -> None:
        """Replace the agent's current BM25 index."""

        self.index.build(chunks)

    def index_semantic_chunks(self, chunks: Iterable[LegalTextChunk]) -> None:
        """Embed chunks and replace only the semantic vector index."""

        self._semantic.build_index(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[BM25SearchResult]:
        """Backward-compatible alias for :meth:`search_bm25`."""

        return self.search_bm25(query, top_k=top_k, filters=filters)

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[BM25SearchResult]:
        """Search only the BM25 index; metadata filters are reserved."""

        if filters:
            raise NotImplementedError(
                "Metadata filtering is not implemented in Retrieval Milestone 2"
            )
        return self._searcher.search(query, top_k=top_k)

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SemanticSearchResult]:
        """Search only the Gemini-backed semantic vector index."""

        if filters:
            raise NotImplementedError(
                "Metadata filtering is not implemented in Retrieval Milestone 2"
            )
        return self._semantic.search(query, top_k=top_k)
