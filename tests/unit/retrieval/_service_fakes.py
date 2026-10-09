"""Retrieval Agent test double for service, API, and handler tests."""

from typing import Any

from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    LegalDocumentMetadata,
    SemanticSearchResult,
)


def _metadata() -> LegalDocumentMetadata:
    return LegalDocumentMetadata(
        document_id="employment-001",
        case_name="Synthetic Employment Matter",
        court="Synthetic Labour Court",
        date="2025-01-15",
        citation="SYNTHETIC-EMPLOYMENT-001",
        legal_category="employment",
        document_type="synthetic judgment",
        source="synthetic-test-source",
    )


class FakeRetrievalAgent:
    """Return deterministic internal result models and record dispatch calls."""

    def __init__(
        self,
        *,
        semantic_index_size: int = 1,
        failure: Exception | None = None,
        empty: bool = False,
    ) -> None:
        self.semantic_index_size = semantic_index_size
        self.failure = failure
        self.empty = empty
        self.calls: list[dict[str, Any]] = []

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[BM25SearchResult]:
        self._record("bm25", query, top_k, filters)
        if self.empty:
            return []
        metadata = _metadata()
        return [
            BM25SearchResult(
                rank=1,
                document_id=metadata.document_id,
                chunk_id="employment-001-chunk-0001",
                case_name=metadata.case_name,
                score=3.25,
                chunk_text="Synthetic employment termination text.",
                citation=metadata.citation,
                source=metadata.source,
                court=metadata.court,
                date=metadata.date,
                legal_category=metadata.legal_category,
                document_type=metadata.document_type,
                metadata=metadata,
            )
        ][:top_k]

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SemanticSearchResult]:
        self._record("semantic", query, top_k, filters)
        if self.empty:
            return []
        metadata = _metadata()
        return [
            SemanticSearchResult(
                rank=1,
                document_id=metadata.document_id,
                chunk_id="employment-001-chunk-0001",
                case_name=metadata.case_name,
                similarity_score=0.91,
                chunk_text="Synthetic employment termination text.",
                citation=metadata.citation,
                source=metadata.source,
                court=metadata.court,
                date=metadata.date,
                legal_category=metadata.legal_category,
                document_type=metadata.document_type,
                metadata=metadata,
            )
        ][:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        *,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        candidate_multiplier: int | None = None,
    ) -> list[HybridSearchResult]:
        self._record(
            "hybrid",
            query,
            top_k,
            filters,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )
        if self.empty:
            return []
        metadata = _metadata()
        return [
            HybridSearchResult(
                rank=1,
                document_id=metadata.document_id,
                chunk_id="employment-001-chunk-0001",
                case_name=metadata.case_name,
                chunk_text="Synthetic employment termination text.",
                citation=metadata.citation,
                source=metadata.source,
                court=metadata.court,
                date=metadata.date,
                legal_category=metadata.legal_category,
                document_type=metadata.document_type,
                metadata=metadata,
                raw_bm25_score=3.25,
                normalized_bm25_score=1.0,
                raw_semantic_score=0.91,
                normalized_semantic_score=1.0,
                hybrid_score=1.0,
            )
        ][:top_k]

    def _record(
        self,
        mode: str,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        **values: Any,
    ) -> None:
        if self.failure is not None:
            raise self.failure
        self.calls.append(
            {
                "mode": mode,
                "query": query,
                "top_k": top_k,
                "filters": filters,
                **values,
            }
        )
