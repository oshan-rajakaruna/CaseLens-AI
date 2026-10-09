"""Candidate retrieval, merging, filtering, and hybrid ranking."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from retrieval.bm25.search import BM25Searcher
from retrieval.hybrid.fusion import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    fuse_scores,
    normalize_weights,
)
from retrieval.hybrid.normalize import min_max_normalize
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    LegalDocumentMetadata,
    SemanticSearchResult,
)
from retrieval.vector_store.semantic import SemanticSearchService

DEFAULT_CANDIDATE_MULTIPLIER = 2
_SUPPORTED_FILTERS = {"court", "date", "year", "legal_category", "document_type"}


@dataclass(slots=True)
class _MergedCandidate:
    document_id: str
    chunk_id: str
    chunk_text: str
    metadata: LegalDocumentMetadata
    raw_bm25_score: float | None = None
    normalized_bm25_score: float = 0.0
    raw_semantic_score: float | None = None
    normalized_semantic_score: float = 0.0


class HybridSearchService:
    """Fuse independent BM25 and semantic candidate rankings."""

    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        semantic_searcher: SemanticSearchService,
        *,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        self.bm25_searcher = bm25_searcher
        self.semantic_searcher = semantic_searcher
        self.candidate_multiplier = self._validate_candidate_multiplier(
            candidate_multiplier
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        filters: Mapping[str, Any] | None = None,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        candidate_multiplier: int | None = None,
    ) -> list[HybridSearchResult]:
        """Retrieve larger candidate pools and return a fused top-k ranking."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        normalized_weights = normalize_weights(bm25_weight, semantic_weight)
        validated_filters = self._validate_filters(filters)
        multiplier = self._validate_candidate_multiplier(
            candidate_multiplier
            if candidate_multiplier is not None
            else self.candidate_multiplier
        )
        if not query.strip() or top_k == 0:
            return []

        candidate_k = max(top_k * multiplier, top_k)
        bm25_results = self.bm25_searcher.search(query, top_k=candidate_k)
        semantic_results = self.semantic_searcher.search(query, top_k=candidate_k)
        candidates = self._merge_candidates(bm25_results, semantic_results)

        ranked: list[HybridSearchResult] = []
        for candidate in candidates.values():
            if not self._matches_filters(candidate.metadata, validated_filters):
                continue
            hybrid_score = fuse_scores(
                candidate.normalized_bm25_score,
                candidate.normalized_semantic_score,
                bm25_weight=normalized_weights[0],
                semantic_weight=normalized_weights[1],
            )
            ranked.append(
                HybridSearchResult(
                    rank=1,
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    case_name=candidate.metadata.case_name,
                    chunk_text=candidate.chunk_text,
                    citation=candidate.metadata.citation,
                    source=candidate.metadata.source,
                    court=candidate.metadata.court,
                    date=candidate.metadata.date,
                    legal_category=candidate.metadata.legal_category,
                    document_type=candidate.metadata.document_type,
                    metadata=candidate.metadata,
                    raw_bm25_score=candidate.raw_bm25_score,
                    normalized_bm25_score=candidate.normalized_bm25_score,
                    raw_semantic_score=candidate.raw_semantic_score,
                    normalized_semantic_score=candidate.normalized_semantic_score,
                    hybrid_score=hybrid_score,
                )
            )

        ranked.sort(
            key=lambda result: (
                -result.hybrid_score,
                result.document_id,
                result.chunk_id,
            )
        )
        return [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(ranked[:top_k], start=1)
        ]

    @staticmethod
    def _merge_candidates(
        bm25_results: list[BM25SearchResult],
        semantic_results: list[SemanticSearchResult],
    ) -> dict[tuple[str, str], _MergedCandidate]:
        candidates: dict[tuple[str, str], _MergedCandidate] = {}
        normalized_bm25 = min_max_normalize(
            [result.score for result in bm25_results]
        )
        normalized_semantic = min_max_normalize(
            [result.similarity_score for result in semantic_results]
        )

        for result, normalized_score in zip(
            bm25_results, normalized_bm25, strict=True
        ):
            metadata = result.metadata or LegalDocumentMetadata(
                document_id=result.document_id,
                case_name=result.case_name,
                court=result.court,
                date=result.date,
                citation=result.citation,
                legal_category=result.legal_category,
                document_type=result.document_type,
                source=result.source,
            )
            key = (result.document_id, result.chunk_id)
            candidates[key] = _MergedCandidate(
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                chunk_text=result.chunk_text,
                metadata=metadata,
                raw_bm25_score=result.score,
                normalized_bm25_score=normalized_score,
            )

        for result, normalized_score in zip(
            semantic_results, normalized_semantic, strict=True
        ):
            key = (result.document_id, result.chunk_id)
            candidate = candidates.get(key)
            if candidate is None:
                candidate = _MergedCandidate(
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    metadata=result.metadata,
                )
                candidates[key] = candidate
            candidate.raw_semantic_score = result.similarity_score
            candidate.normalized_semantic_score = normalized_score
            candidate.metadata = result.metadata
        return candidates

    @staticmethod
    def _validate_candidate_multiplier(candidate_multiplier: int) -> int:
        if isinstance(candidate_multiplier, bool) or not isinstance(
            candidate_multiplier, int
        ):
            raise TypeError("candidate_multiplier must be an integer")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least one")
        return candidate_multiplier

    @staticmethod
    def _validate_filters(
        filters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if filters is None:
            return {}
        if not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping or None")
        unsupported = set(filters) - _SUPPORTED_FILTERS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported metadata filter(s): {names}")
        validated = {
            key: value for key, value in filters.items() if value is not None
        }
        if "year" in validated:
            year = str(validated["year"])
            if len(year) != 4 or not year.isdigit():
                raise ValueError("year filter must be a four-digit year")
            validated["year"] = year
        return validated

    @staticmethod
    def _matches_filters(
        metadata: LegalDocumentMetadata,
        filters: Mapping[str, Any],
    ) -> bool:
        for field, expected in filters.items():
            if field == "year":
                if metadata.date is None or not metadata.date.startswith(expected):
                    return False
                continue
            actual = getattr(metadata, field)
            if actual is None:
                return False
            if str(actual).casefold() != str(expected).casefold():
                return False
        return True
