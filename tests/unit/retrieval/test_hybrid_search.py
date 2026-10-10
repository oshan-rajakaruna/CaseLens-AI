"""Tests for hybrid candidate merging, filtering, and ranking."""

from typing import Any

import pytest

from retrieval.hybrid import HybridSearchService
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    LegalDocumentMetadata,
    SemanticSearchResult,
)


def _metadata(
    document_id: str,
    *,
    court: str = "Synthetic Court",
    date: str = "2025-06-01",
    category: str = "employment",
    document_type: str = "judgment",
) -> LegalDocumentMetadata:
    return LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Synthetic {document_id.title()} Matter",
        court=court,
        date=date,
        citation=f"SYNTHETIC-{document_id.upper()}",
        legal_category=category,
        document_type=document_type,
        source=f"synthetic/{document_id}.txt",
    )


def _bm25_result(
    document_id: str,
    score: float,
    *,
    metadata: LegalDocumentMetadata | None = None,
) -> BM25SearchResult:
    metadata = metadata or _metadata(document_id)
    return BM25SearchResult(
        rank=1,
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-0001",
        case_name=metadata.case_name,
        score=score,
        chunk_text=f"Synthetic {document_id} text",
        citation=metadata.citation,
        source=metadata.source,
        court=metadata.court,
        date=metadata.date,
        legal_category=metadata.legal_category,
        document_type=metadata.document_type,
        metadata=metadata,
    )


def _semantic_result(
    document_id: str,
    score: float,
    *,
    metadata: LegalDocumentMetadata | None = None,
) -> SemanticSearchResult:
    metadata = metadata or _metadata(document_id)
    return SemanticSearchResult(
        rank=1,
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-0001",
        case_name=metadata.case_name,
        similarity_score=score,
        chunk_text=f"Synthetic {document_id} text",
        citation=metadata.citation,
        source=metadata.source,
        court=metadata.court,
        date=metadata.date,
        legal_category=metadata.legal_category,
        document_type=metadata.document_type,
        metadata=metadata,
    )


class _RecordingSearcher:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.requested_top_k: list[int] = []

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        self.requested_top_k.append(top_k)
        return self.results[:top_k]


def _hybrid(
    bm25_results: list[BM25SearchResult],
    semantic_results: list[SemanticSearchResult],
    *,
    candidate_multiplier: int = 2,
) -> tuple[HybridSearchService, _RecordingSearcher, _RecordingSearcher]:
    bm25 = _RecordingSearcher(bm25_results)
    semantic = _RecordingSearcher(semantic_results)
    service = HybridSearchService(  # type: ignore[arg-type]
        bm25,
        semantic,
        candidate_multiplier=candidate_multiplier,
    )
    return service, bm25, semantic


def test_hybrid_merges_duplicates_and_keeps_single_mode_candidates() -> None:
    service, _, _ = _hybrid(
        [_bm25_result("both", 10.0), _bm25_result("bm25-only", 5.0)],
        [
            _semantic_result("both", 0.9),
            _semantic_result("semantic-only", 0.5),
        ],
    )

    results = service.search("legal query", top_k=10)
    by_id = {result.document_id: result for result in results}

    assert len(results) == 3
    assert by_id["both"].raw_bm25_score == 10.0
    assert by_id["both"].raw_semantic_score == 0.9
    assert by_id["both"].hybrid_score == pytest.approx(1.0)
    assert by_id["bm25-only"].raw_semantic_score is None
    assert by_id["bm25-only"].normalized_semantic_score == 0.0
    assert by_id["semantic-only"].raw_bm25_score is None
    assert by_id["semantic-only"].normalized_bm25_score == 0.0


def test_candidate_pool_is_larger_than_final_top_k_and_configurable() -> None:
    service, bm25, semantic = _hybrid([], [], candidate_multiplier=3)

    assert service.search("legal query", top_k=4) == []
    assert bm25.requested_top_k == [12]
    assert semantic.requested_top_k == [12]

    service.search("legal query", top_k=4, candidate_multiplier=4)
    assert bm25.requested_top_k[-1] == 16
    assert semantic.requested_top_k[-1] == 16


def test_configurable_weights_change_single_mode_ranking() -> None:
    service, _, _ = _hybrid(
        [_bm25_result("lexical", 4.0)],
        [_semantic_result("conceptual", 0.8)],
    )

    lexical_first = service.search(
        "legal query",
        top_k=2,
        bm25_weight=0.8,
        semantic_weight=0.2,
    )
    semantic_first = service.search(
        "legal query",
        top_k=2,
        bm25_weight=0.2,
        semantic_weight=0.8,
    )

    assert lexical_first[0].document_id == "lexical"
    assert semantic_first[0].document_id == "conceptual"


def test_hybrid_top_k_and_ties_are_deterministic() -> None:
    service, _, _ = _hybrid(
        [_bm25_result("zeta", 1.0), _bm25_result("alpha", 1.0)],
        [],
    )

    results = service.search("legal query", top_k=1)

    assert len(results) == 1
    assert results[0].document_id == "alpha"
    assert results[0].rank == 1


def test_hybrid_diversification_runs_after_scoring_and_preserves_scores() -> None:
    first = _bm25_result("dominant", 10.0)
    second = _bm25_result("dominant", 9.0).model_copy(
        update={"chunk_id": "dominant-chunk-0002"}
    )
    third = _bm25_result("dominant", 8.0).model_copy(
        update={"chunk_id": "dominant-chunk-0003"}
    )
    lower = _bm25_result("other", 7.0)
    service, _, _ = _hybrid([first, second, third, lower], [])

    uncapped = service.search("legal query", top_k=3)
    capped = service.search(
        "legal query",
        top_k=3,
        max_chunks_per_document=2,
    )

    assert [result.chunk_id for result in uncapped] == [
        "dominant-chunk-0001",
        "dominant-chunk-0002",
        "dominant-chunk-0003",
    ]
    assert [result.chunk_id for result in capped] == [
        "dominant-chunk-0001",
        "dominant-chunk-0002",
        "other-chunk-0001",
    ]
    assert [result.hybrid_score for result in capped[:2]] == [
        result.hybrid_score for result in uncapped[:2]
    ]
    assert [result.rank for result in capped] == [1, 2, 3]


def test_metadata_filters_support_exact_fields_and_year() -> None:
    employment = _metadata("employment", court="Appeal Court", date="2024-03-15")
    property_metadata = _metadata(
        "property",
        court="District Court",
        date="2023-08-20",
        category="property",
    )
    service, _, _ = _hybrid(
        [
            _bm25_result("employment", 2.0, metadata=employment),
            _bm25_result("property", 1.0, metadata=property_metadata),
        ],
        [],
    )

    results = service.search(
        "legal query",
        top_k=5,
        filters={
            "court": "appeal court",
            "year": 2024,
            "legal_category": "EMPLOYMENT",
            "document_type": "judgment",
        },
    )

    assert [result.document_id for result in results] == ["employment"]


def test_metadata_filter_rejects_unsupported_fields() -> None:
    service, _, _ = _hybrid([], [])

    with pytest.raises(ValueError, match="Unsupported metadata filter"):
        service.search("legal query", filters={"judge": "Example"})


def test_metadata_filter_rejects_invalid_year() -> None:
    service, _, _ = _hybrid([], [])

    with pytest.raises(ValueError, match="four-digit year"):
        service.search("legal query", filters={"year": "25"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bm25_weight": -1.0},
        {"bm25_weight": 0.0, "semantic_weight": 0.0},
        {"candidate_multiplier": 0},
    ],
)
def test_hybrid_search_rejects_invalid_configuration(
    kwargs: dict[str, Any],
) -> None:
    service, _, _ = _hybrid([], [])

    with pytest.raises((TypeError, ValueError)):
        service.search("legal query", **kwargs)
