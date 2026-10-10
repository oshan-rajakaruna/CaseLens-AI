"""Tests for mode comparison, aggregation, and weight experiments."""

from dataclasses import dataclass

import pytest

from retrieval.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
    HumanRelevanceJudgment,
    RetrievalEvaluator,
    evaluate_hybrid_weights,
)


@dataclass(frozen=True)
class _Result:
    document_id: str
    chunk_id: str


class _ControlledAgent:
    def __init__(self) -> None:
        self.rankings = {
            "bm25": {
                "query one": ["a", "x", "b"],
                "query two": ["x", "d"],
            },
            "semantic": {
                "query one": ["x", "b", "a"],
                "query two": ["d", "x"],
            },
            "hybrid": {
                "query one": ["a", "b", "x"],
                "query two": ["x", "d"],
            },
        }
        self.hybrid_weights: list[tuple[float, float]] = []

    def search_bm25(
        self, query: str, top_k: int = 5, *, diversify: bool = True
    ) -> list[_Result]:
        assert diversify is False
        return self._results("bm25", query, top_k)

    def search_semantic(
        self, query: str, top_k: int = 5, *, diversify: bool = True
    ) -> list[_Result]:
        assert diversify is False
        return self._results("semantic", query, top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        *,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        diversify: bool = True,
    ) -> list[_Result]:
        assert diversify is False
        self.hybrid_weights.append((bm25_weight, semantic_weight))
        return self._results("hybrid", query, top_k)

    def _results(self, mode: str, query: str, top_k: int) -> list[_Result]:
        return [
            _Result(document_id=item_id, chunk_id=f"{item_id}-chunk")
            for item_id in self.rankings[mode][query][:top_k]
        ]


def _queries() -> list[EvaluationQuery]:
    return [
        EvaluationQuery(
            query_id="q1",
            query="query one",
            relevant_ids=["a", "b"],
        ),
        EvaluationQuery(
            query_id="q2",
            query="query two",
            relevant_ids=["d"],
        ),
    ]


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        dataset_id="controlled",
        name="Controlled synthetic dataset",
        description="Known rankings for unit tests only.",
        data_classification="synthetic",
        queries=_queries(),
    )


def test_evaluator_collects_per_query_and_aggregate_metrics() -> None:
    evaluator = RetrievalEvaluator(_ControlledAgent())

    result = evaluator.evaluate_mode(_queries(), "bm25", k_values=(1, 3))

    assert result.query_count == 2
    assert result.k_values == [1, 3]
    assert result.per_query[0].retrieved_ids == ["a", "x", "b"]
    assert result.per_query[0].metrics_at_k[0].precision_at_k == 1.0
    assert result.per_query[0].metrics_at_k[0].recall_at_k == 0.5
    assert result.aggregate_at_k[0].mean_precision_at_k == pytest.approx(0.5)
    assert result.aggregate_at_k[0].mean_recall_at_k == pytest.approx(0.25)
    assert result.aggregate_at_k[1].mean_precision_at_k == pytest.approx(0.5)
    assert result.aggregate_at_k[1].mean_recall_at_k == 1.0
    assert result.mrr == pytest.approx(0.75)


def test_mode_comparison_evaluates_bm25_semantic_and_hybrid() -> None:
    evaluator = RetrievalEvaluator(_ControlledAgent())

    comparison = evaluator.compare_modes(_dataset(), k_values=(3, 1, 3))

    assert comparison.k_values == [1, 3]
    assert [result.mode for result in comparison.modes] == [
        "bm25",
        "semantic",
        "hybrid",
    ]
    assert all(result.query_count == 2 for result in comparison.modes)


def test_empty_query_collection_has_zero_aggregates() -> None:
    evaluator = RetrievalEvaluator(_ControlledAgent())

    result = evaluator.evaluate_mode([], "bm25", k_values=(1, 5))

    assert result.query_count == 0
    assert result.mrr == 0.0
    assert all(metric.mean_precision_at_k == 0.0 for metric in result.aggregate_at_k)
    assert all(metric.mean_recall_at_k == 0.0 for metric in result.aggregate_at_k)


def test_chunk_relevance_uses_chunk_ids() -> None:
    agent = _ControlledAgent()
    evaluator = RetrievalEvaluator(agent)
    query = EvaluationQuery(
        query_id="chunk-query",
        query="query one",
        relevance_unit="chunk",
        relevant_ids=["a-chunk"],
    )

    result = evaluator.evaluate_mode([query], "bm25", k_values=(1,))

    assert result.per_query[0].retrieved_ids[0] == "a-chunk"
    assert result.aggregate_at_k[0].mean_precision_at_k == 1.0


def test_mrr_can_use_retrieval_depth_beyond_metric_cutoff() -> None:
    evaluator = RetrievalEvaluator(_ControlledAgent())
    query = EvaluationQuery(
        query_id="deep-relevant-query",
        query="query one",
        relevant_ids=["b"],
    )

    result = evaluator.evaluate_mode(
        [query],
        "bm25",
        k_values=(1,),
        retrieval_depth=3,
    )

    assert result.retrieval_depth == 3
    assert result.aggregate_at_k[0].mean_recall_at_k == 0.0
    assert result.mrr == pytest.approx(1 / 3)


def test_human_labels_do_not_change_binary_relevant_set() -> None:
    query = EvaluationQuery(
        query_id="human-label-query",
        query="query one",
        relevant_ids=["a"],
        human_judgments=[
            HumanRelevanceJudgment(item_id="x", label="partially_relevant")
        ],
    )
    evaluator = RetrievalEvaluator(_ControlledAgent())

    result = evaluator.evaluate_mode([query], "semantic", k_values=(1,))

    assert result.per_query[0].relevant_ids == ["a"]
    assert result.per_query[0].retrieved_ids[0] == "x"
    assert result.aggregate_at_k[0].mean_precision_at_k == 0.0


def test_weight_experiments_preserve_all_configurations_without_winner() -> None:
    agent = _ControlledAgent()
    evaluator = RetrievalEvaluator(agent)

    experiments = evaluate_hybrid_weights(
        evaluator,
        _dataset(),
        configurations=((0.2, 0.8), (2.0, 1.0)),
        k_values=(1,),
    )

    assert [(item.bm25_weight, item.semantic_weight) for item in experiments] == [
        (0.2, 0.8),
        pytest.approx((2 / 3, 1 / 3)),
    ]
    assert len(experiments) == 2
    assert agent.hybrid_weights == [
        (0.2, 0.8),
        (0.2, 0.8),
        pytest.approx((2 / 3, 1 / 3)),
        pytest.approx((2 / 3, 1 / 3)),
    ]


@pytest.mark.parametrize("k_values", [(), (0,), (True,)])
def test_evaluator_rejects_invalid_k_values(k_values: tuple[object, ...]) -> None:
    evaluator = RetrievalEvaluator(_ControlledAgent())

    with pytest.raises((TypeError, ValueError)):
        evaluator.evaluate_mode(  # type: ignore[arg-type]
            _queries(),
            "bm25",
            k_values=k_values,
        )
