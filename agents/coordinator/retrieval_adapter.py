"""Coordinator-owned adapter for prepared, offline Retrieval Agent instances."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from agents.coordinator.schemas import CoordinatorRequest, CoordinatorRetrievalResult
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    SemanticSearchResult,
)


class CoordinatorRetrievalAgent(Protocol):
    """Public Retrieval surface required by Coordinator Step 3."""

    @property
    def bm25_index_size(self) -> int: ...

    @property
    def semantic_index_size(self) -> int: ...

    def search_bm25(
        self, query: str, top_k: int = 5, filters: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Sequence[BM25SearchResult]: ...

    def search_semantic(
        self, query: str, top_k: int = 5, filters: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Sequence[SemanticSearchResult]: ...

    def search_hybrid(
        self, query: str, top_k: int = 5, filters: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Sequence[HybridSearchResult]: ...


class RetrievalUnavailableError(ValueError):
    """Raised when the caller did not supply an index for the requested mode."""


class CoordinatorRetrievalAdapter:
    """Call Retrieval without changing result order, scores, or provenance."""

    def __init__(self, retrieval_agent: CoordinatorRetrievalAgent) -> None:
        self._retrieval_agent = retrieval_agent

    def search(self, request: CoordinatorRequest) -> list[CoordinatorRetrievalResult]:
        """Search a prepared mode-specific index using the supplied query."""

        filters = (
            request.retrieval_filters.model_dump(exclude_none=True)
            if request.retrieval_filters is not None
            else None
        )
        if request.retrieval_mode == "bm25":
            self._require_index(self._retrieval_agent.bm25_index_size, "BM25")
            return list(self._retrieval_agent.search_bm25(request.user_query, top_k=request.top_k))
        if request.retrieval_mode == "semantic":
            self._require_index(self._retrieval_agent.semantic_index_size, "semantic")
            return list(self._retrieval_agent.search_semantic(request.user_query, top_k=request.top_k))
        self._require_index(self._retrieval_agent.bm25_index_size, "BM25")
        self._require_index(self._retrieval_agent.semantic_index_size, "semantic")
        return list(
            self._retrieval_agent.search_hybrid(
                request.user_query,
                top_k=request.top_k,
                filters=filters,
            )
        )

    @staticmethod
    def _require_index(size: int, mode: str) -> None:
        if size <= 0:
            raise RetrievalUnavailableError(
                f"A prepared {mode} index is required before retrieval can run."
            )
