"""Retrieval Agent with independent BM25, semantic, and hybrid search modes.

Advanced reranking and full Coordinator orchestration remain outside this
milestone.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from retrieval.bm25 import BM25Index, BM25Searcher
from retrieval.diversification import resolve_max_chunks_per_document
from retrieval.embeddings import EmbeddingService, create_embedding_service
from retrieval.hybrid import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    HybridSearchService,
)
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
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
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self.index = BM25Index().build(chunks or [])
        self._searcher = BM25Searcher(self.index)
        self._semantic = SemanticSearchService(
            embedding_service or create_embedding_service()
        )
        self._hybrid = HybridSearchService(
            self._searcher,
            self._semantic,
            candidate_multiplier=candidate_multiplier,
        )

    @property
    def bm25_index_size(self) -> int:
        """Return the number of chunks available to lexical retrieval."""

        return len(self.index)

    @property
    def semantic_index_size(self) -> int:
        """Return the number of chunks available to semantic retrieval."""

        return len(self._semantic.index)

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
        *,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> list[BM25SearchResult]:
        """Backward-compatible alias for :meth:`search_bm25`."""

        return self.search_bm25(
            query,
            top_k=top_k,
            filters=filters,
            diversify=diversify,
            max_chunks_per_document=max_chunks_per_document,
        )

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> list[BM25SearchResult]:
        """Search only the BM25 index; metadata filters are reserved."""

        if filters:
            raise NotImplementedError(
                "Metadata filtering is currently supported only by hybrid search"
            )
        cap = resolve_max_chunks_per_document(
            diversify=diversify,
            max_chunks_per_document=max_chunks_per_document,
        )
        return self._searcher.search(
            query,
            top_k=top_k,
            max_chunks_per_document=cap,
        )

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> list[SemanticSearchResult]:
        """Search only the configured-provider semantic vector index."""

        if filters:
            raise NotImplementedError(
                "Metadata filtering is currently supported only by hybrid search"
            )
        cap = resolve_max_chunks_per_document(
            diversify=diversify,
            max_chunks_per_document=max_chunks_per_document,
        )
        return self._semantic.search(
            query,
            top_k=top_k,
            max_chunks_per_document=cap,
        )

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        bm25_weight: float | None = None,
        semantic_weight: float | None = None,
        candidate_multiplier: int | None = None,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> list[HybridSearchResult]:
        """Search BM25 and semantic candidates, then return a fused ranking."""

        cap = resolve_max_chunks_per_document(
            diversify=diversify,
            max_chunks_per_document=max_chunks_per_document,
        )
        return self._hybrid.search(
            query,
            top_k=top_k,
            filters=filters,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            candidate_multiplier=candidate_multiplier,
            max_chunks_per_document=cap,
        )
