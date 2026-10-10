"""Application service dispatching retrieval modes and mapping public results."""

from functools import lru_cache

from agents.retrieval import RetrievalAgent
from backend.schemas.retrieval import (
    RetrievalResultResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    metadata_to_public,
)
from retrieval.embeddings import GeminiEmbeddingError, MissingGeminiAPIKeyError
from retrieval.hybrid import resolve_hybrid_weights
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    SemanticSearchResult,
)


class RetrievalServiceError(RuntimeError):
    """Base class for safe application-layer retrieval errors."""


class RetrievalRequestError(RetrievalServiceError):
    """Raised when a valid API shape requests an unsupported operation."""


class RetrievalIndexNotReadyError(RetrievalServiceError):
    """Raised when an explicitly requested semantic index is unavailable."""


class RetrievalProviderError(RetrievalServiceError):
    """Raised when the configured embedding provider is unavailable."""


class RetrievalExecutionError(RetrievalServiceError):
    """Raised for unexpected retrieval failures without leaking internals."""


class RetrievalService:
    """Dispatch public requests to an injected Retrieval Agent."""

    def __init__(self, agent: RetrievalAgent) -> None:
        self.agent = agent

    def search(self, request: RetrievalSearchRequest) -> RetrievalSearchResponse:
        """Execute one retrieval mode and translate internal result models."""

        filters = (
            request.filters.model_dump(exclude_none=True)
            if request.filters is not None
            else None
        )
        if filters and request.mode != "hybrid":
            raise RetrievalRequestError(
                "Metadata filters are currently supported only in hybrid mode"
            )

        try:
            if request.mode == "bm25":
                internal_results = self.agent.search_bm25(
                    request.query,
                    top_k=request.top_k,
                    diversify=request.diversify,
                    max_chunks_per_document=request.max_chunks_per_document,
                )
            elif request.mode == "semantic":
                if getattr(self.agent, "semantic_index_size", None) == 0:
                    raise RetrievalIndexNotReadyError(
                        "Semantic retrieval index is not initialized"
                    )
                internal_results = self.agent.search_semantic(
                    request.query,
                    top_k=request.top_k,
                    diversify=request.diversify,
                    max_chunks_per_document=request.max_chunks_per_document,
                )
            else:
                bm25_weight, semantic_weight = resolve_hybrid_weights(
                    request.bm25_weight,
                    request.semantic_weight,
                )
                internal_results = self.agent.search_hybrid(
                    request.query,
                    top_k=request.top_k,
                    filters=filters,
                    bm25_weight=bm25_weight,
                    semantic_weight=semantic_weight,
                    diversify=request.diversify,
                    max_chunks_per_document=request.max_chunks_per_document,
                )
        except RetrievalServiceError:
            raise
        except MissingGeminiAPIKeyError as exc:
            raise RetrievalProviderError(
                "Semantic retrieval is unavailable because Gemini is not configured"
            ) from exc
        except GeminiEmbeddingError as exc:
            raise RetrievalProviderError(
                "Semantic retrieval provider request failed"
            ) from exc
        except (TypeError, ValueError, NotImplementedError) as exc:
            raise RetrievalRequestError(str(exc)) from exc
        except Exception as exc:
            raise RetrievalExecutionError("Retrieval search failed") from exc

        try:
            results = [self._to_public(result) for result in internal_results]
        except RetrievalServiceError:
            raise
        except Exception as exc:
            raise RetrievalExecutionError(
                "Retrieval result translation failed"
            ) from exc
        return RetrievalSearchResponse(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            result_count=len(results),
            results=results,
        )

    @staticmethod
    def _to_public(
        result: BM25SearchResult | SemanticSearchResult | HybridSearchResult,
    ) -> RetrievalResultResponse:
        metadata = result.metadata
        if metadata is None:
            raise RetrievalExecutionError("Retrieval result metadata is unavailable")

        common = {
            "rank": result.rank,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "case_name": result.case_name,
            "chunk_text": result.chunk_text,
            "citation": result.citation,
            "source": result.source,
            "metadata": metadata_to_public(metadata),
        }
        if isinstance(result, BM25SearchResult):
            return RetrievalResultResponse(**common, bm25_score=result.score)
        if isinstance(result, SemanticSearchResult):
            return RetrievalResultResponse(
                **common,
                semantic_score=result.similarity_score,
            )
        return RetrievalResultResponse(
            **common,
            bm25_score=result.raw_bm25_score,
            normalized_bm25_score=result.normalized_bm25_score,
            semantic_score=result.raw_semantic_score,
            normalized_semantic_score=result.normalized_semantic_score,
            hybrid_score=result.hybrid_score,
        )


@lru_cache
def get_retrieval_service() -> RetrievalService:
    """Return the process-local retrieval service without loading test data."""

    return RetrievalService(RetrievalAgent())
