"""Evaluation orchestration across BM25, semantic, and hybrid retrieval."""

from collections.abc import Sequence
from typing import Any, Protocol

from retrieval.evaluation.metrics import (
    deduplicate_ranked_ids,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.evaluation.models import (
    AggregateAtK,
    EvaluationComparison,
    EvaluationDataset,
    EvaluationMode,
    EvaluationQuery,
    ModeEvaluationResult,
    PerQueryEvaluationResult,
    QueryMetricsAtK,
)
from retrieval.hybrid import DEFAULT_BM25_WEIGHT, DEFAULT_SEMANTIC_WEIGHT


class EvaluatedRetrievalAgent(Protocol):
    """Retrieval Agent surface needed by the evaluator."""

    def search_bm25(self, query: str, top_k: int = 5) -> Sequence[Any]: ...

    def search_semantic(self, query: str, top_k: int = 5) -> Sequence[Any]: ...

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        *,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    ) -> Sequence[Any]: ...


def normalize_k_values(k_values: Sequence[int]) -> list[int]:
    """Validate, sort, and deduplicate evaluation cutoffs."""

    if not k_values:
        raise ValueError("at least one k value is required")
    normalized: set[int] = set()
    for k in k_values:
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k values must be integers")
        if k <= 0:
            raise ValueError("k values must be greater than zero")
        normalized.add(k)
    return sorted(normalized)


class RetrievalEvaluator:
    """Measure retrieval modes against explicit binary relevance sets."""

    def __init__(self, agent: EvaluatedRetrievalAgent) -> None:
        self.agent = agent

    def evaluate_mode(
        self,
        queries: Sequence[EvaluationQuery],
        mode: EvaluationMode,
        *,
        k_values: Sequence[int] = (1, 3, 5),
        retrieval_depth: int | None = None,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    ) -> ModeEvaluationResult:
        """Evaluate one mode and retain per-query rankings and metrics."""

        cutoffs = normalize_k_values(k_values)
        if retrieval_depth is not None:
            if isinstance(retrieval_depth, bool) or not isinstance(
                retrieval_depth, int
            ):
                raise TypeError("retrieval_depth must be an integer")
            if retrieval_depth <= 0:
                raise ValueError("retrieval_depth must be greater than zero")
        effective_depth = max(max(cutoffs), retrieval_depth or 0)
        per_query: list[PerQueryEvaluationResult] = []

        for evaluation_query in queries:
            results = self._search(
                evaluation_query.query,
                mode,
                top_k=effective_depth,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
            )
            id_field = (
                "document_id"
                if evaluation_query.relevance_unit == "document"
                else "chunk_id"
            )
            retrieved_ids = deduplicate_ranked_ids(
                getattr(result, id_field) for result in results
            )
            relevant_ids = list(dict.fromkeys(evaluation_query.relevant_ids))
            per_query.append(
                PerQueryEvaluationResult(
                    query_id=evaluation_query.query_id,
                    query=evaluation_query.query,
                    mode=mode,
                    relevance_unit=evaluation_query.relevance_unit,
                    relevant_ids=relevant_ids,
                    retrieved_ids=retrieved_ids,
                    metrics_at_k=[
                        QueryMetricsAtK(
                            k=k,
                            precision_at_k=precision_at_k(
                                retrieved_ids, relevant_ids, k
                            ),
                            recall_at_k=recall_at_k(
                                retrieved_ids, relevant_ids, k
                            ),
                        )
                        for k in cutoffs
                    ],
                    reciprocal_rank=reciprocal_rank(
                        retrieved_ids, relevant_ids
                    ),
                )
            )

        query_count = len(per_query)
        aggregate_at_k = [
            AggregateAtK(
                k=k,
                mean_precision_at_k=(
                    sum(
                        next(
                            metric.precision_at_k
                            for metric in result.metrics_at_k
                            if metric.k == k
                        )
                        for result in per_query
                    )
                    / query_count
                    if query_count
                    else 0.0
                ),
                mean_recall_at_k=(
                    sum(
                        next(metric.recall_at_k for metric in result.metrics_at_k if metric.k == k)
                        for result in per_query
                    )
                    / query_count
                    if query_count
                    else 0.0
                ),
            )
            for k in cutoffs
        ]
        return ModeEvaluationResult(
            mode=mode,
            query_count=query_count,
            retrieval_depth=effective_depth,
            k_values=cutoffs,
            aggregate_at_k=aggregate_at_k,
            mrr=mean_reciprocal_rank(
                [result.retrieved_ids for result in per_query],
                [result.relevant_ids for result in per_query],
            ),
            per_query=per_query,
        )

    def compare_modes(
        self,
        dataset: EvaluationDataset,
        *,
        modes: Sequence[EvaluationMode] = ("bm25", "semantic", "hybrid"),
        k_values: Sequence[int] = (1, 3, 5),
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    ) -> EvaluationComparison:
        """Evaluate requested modes without choosing or labelling a winner."""

        cutoffs = normalize_k_values(k_values)
        retrieval_depth = max(max(cutoffs), len(dataset.documents))
        mode_results = [
            self.evaluate_mode(
                dataset.queries,
                mode,
                k_values=cutoffs,
                retrieval_depth=retrieval_depth,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
            )
            for mode in modes
        ]
        return EvaluationComparison(
            dataset_id=dataset.dataset_id,
            data_classification=dataset.data_classification,
            k_values=cutoffs,
            modes=mode_results,
        )

    def _search(
        self,
        query: str,
        mode: EvaluationMode,
        *,
        top_k: int,
        bm25_weight: float,
        semantic_weight: float,
    ) -> Sequence[Any]:
        if mode == "bm25":
            return self.agent.search_bm25(query, top_k=top_k)
        if mode == "semantic":
            return self.agent.search_semantic(query, top_k=top_k)
        if mode == "hybrid":
            return self.agent.search_hybrid(
                query,
                top_k=top_k,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight,
            )
        raise ValueError(f"Unsupported evaluation mode: {mode}")
