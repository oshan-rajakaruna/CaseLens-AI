"""Coordinator adapter from Retrieval Agent results to RAG context payloads."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from agents.coordinator.schemas import (
    ContextCitationMapEntry,
    LegalRetrievalContextRequest,
    RetrievalContextDiagnostics,
    RetrievalContextMetadata,
    SummarizationReadyPayload,
)
from retrieval.diversification import resolve_max_chunks_per_document
from retrieval.hybrid import resolve_hybrid_weights
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    SemanticSearchResult,
)
from retrieval.rag import (
    RAGContextAssembler,
    RAGContextSettings,
    format_context,
)

_WEAK_CONTEXT_RELATIVE_RATIO = 0.1
RetrievalResult = BM25SearchResult | SemanticSearchResult | HybridSearchResult


class CoordinatorRetrievalAgent(Protocol):
    """Retrieval Agent surface used by the Coordinator adapter."""

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> Sequence[BM25SearchResult]: ...

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> Sequence[SemanticSearchResult]: ...

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        *,
        bm25_weight: float | None = None,
        semantic_weight: float | None = None,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> Sequence[HybridSearchResult]: ...


class LegalRetrievalContextService:
    """Build summarization-ready legal context without invoking an LLM."""

    def __init__(
        self,
        retrieval_agent: CoordinatorRetrievalAgent,
        *,
        assembler_factory: Callable[
            [RAGContextSettings], RAGContextAssembler
        ] = RAGContextAssembler,
    ) -> None:
        self.retrieval_agent = retrieval_agent
        self.assembler_factory = assembler_factory

    def build(
        self,
        request: LegalRetrievalContextRequest,
    ) -> SummarizationReadyPayload:
        """Retrieve, assemble, and package legal evidence for summarization."""

        filters = (
            request.filters.model_dump(exclude_none=True)
            if request.filters is not None
            else None
        )
        effective_cap = resolve_max_chunks_per_document(
            diversify=request.diversify,
            max_chunks_per_document=request.max_chunks_per_document,
        )
        bm25_weight: float | None = None
        semantic_weight: float | None = None
        if request.retrieval_mode == "hybrid":
            bm25_weight, semantic_weight = resolve_hybrid_weights(
                request.bm25_weight,
                request.semantic_weight,
            )

        results = list(
            self._retrieve(
                request,
                filters=filters,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                effective_cap=effective_cap,
            )
        )
        settings = self._context_settings(request)
        context = self.assembler_factory(settings).assemble(results)
        context_text = format_context(context)
        citation_map = {
            passage.context_id: ContextCitationMapEntry(
                context_id=passage.context_id,
                document_id=passage.document_id,
                title=passage.title,
                court=passage.court,
                source=passage.source,
                chunk_ids=passage.chunk_ids,
                pages=passage.pages,
                official_citation=passage.citation.citation,
                source_url=passage.citation.source_url,
            )
            for passage in context.passages
        }
        weak_context_ids = _weak_context_ids(context.passages)
        warnings = (
            [
                "Selected context includes passages scoring below 10% of "
                "the leading passage; review relevance before summarization"
            ]
            if weak_context_ids
            else []
        )
        diagnostics = RetrievalContextDiagnostics(
            retrieval_result_count=len(results),
            rag_passage_count=len(context.passages),
            source_document_count=len(context.source_documents),
            estimated_context_tokens=context.estimated_tokens,
            context_truncated=context.truncated,
            empty_retrieval=not results,
            no_context=not context.passages,
            weak_context_ids=weak_context_ids,
            warnings=warnings,
        )
        return SummarizationReadyPayload(
            query=request.query,
            legal_issue=request.legal_issue,
            context_text=context_text,
            context_passages=context.passages,
            citation_map=citation_map,
            retrieval=RetrievalContextMetadata(
                mode=request.retrieval_mode,
                top_k=request.top_k,
                result_count=len(results),
                filters=filters,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
                diversified=request.diversify,
                max_chunks_per_document=effective_cap,
            ),
            diagnostics=diagnostics,
        )

    def _retrieve(
        self,
        request: LegalRetrievalContextRequest,
        *,
        filters: Mapping[str, Any] | None,
        bm25_weight: float | None,
        semantic_weight: float | None,
        effective_cap: int | None,
    ) -> Sequence[RetrievalResult]:
        common = {
            "top_k": request.top_k,
            "filters": filters,
            "diversify": request.diversify,
            "max_chunks_per_document": effective_cap,
        }
        if request.retrieval_mode == "bm25":
            return self.retrieval_agent.search_bm25(request.query, **common)
        if request.retrieval_mode == "semantic":
            return self.retrieval_agent.search_semantic(request.query, **common)
        return self.retrieval_agent.search_hybrid(
            request.query,
            **common,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

    @staticmethod
    def _context_settings(
        request: LegalRetrievalContextRequest,
    ) -> RAGContextSettings:
        defaults = RAGContextSettings.from_environment()
        threshold_explicit = "rag_min_retrieval_score" in request.model_fields_set
        return RAGContextSettings(
            max_tokens=request.rag_max_context_tokens or defaults.max_tokens,
            max_passages=(
                request.rag_max_context_passages or defaults.max_passages
            ),
            min_retrieval_score=(
                request.rag_min_retrieval_score
                if threshold_explicit
                else defaults.min_retrieval_score
            ),
        )


def _weak_context_ids(passages: Sequence[Any]) -> list[str]:
    if not passages:
        return []
    leading_score = max(float(passage.score) for passage in passages)
    if leading_score <= 0:
        return []
    cutoff = leading_score * _WEAK_CONTEXT_RELATIVE_RATIO
    return [
        passage.context_id
        for passage in passages
        if float(passage.score) < cutoff
    ]
