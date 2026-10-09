"""Tests for binary retrieval evaluation metrics."""

import pytest

from retrieval.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_uses_k_as_denominator() -> None:
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)


def test_recall_at_k_uses_relevant_set_as_denominator() -> None:
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 5) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_reciprocal_rank_is_mean_across_queries() -> None:
    score = mean_reciprocal_rank(
        [["a", "x"], ["x", "b"], ["x"]],
        [{"a"}, {"b"}, {"c"}],
    )

    assert score == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_empty_retrieval_and_relevance_are_safe() -> None:
    assert precision_at_k([], {"a"}, 3) == 0.0
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert precision_at_k(["a"], set(), 3) == 0.0
    assert recall_at_k(["a"], set(), 3) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert reciprocal_rank(["a"], set()) == 0.0
    assert mean_reciprocal_rank([], []) == 0.0


def test_duplicate_retrieved_ids_are_counted_only_once() -> None:
    retrieved = ["a", "a", "b"]

    assert precision_at_k(retrieved, {"a", "b"}, 2) == 1.0
    assert recall_at_k(retrieved, {"a", "b"}, 2) == 1.0
    assert reciprocal_rank(["x", "x", "a"], {"a"}) == pytest.approx(0.5)


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        precision_at_k(["a"], {"a"}, k)


def test_metrics_reject_non_integer_k() -> None:
    with pytest.raises(TypeError, match="integer"):
        recall_at_k(["a"], {"a"}, True)


def test_mrr_rejects_misaligned_query_collections() -> None:
    with pytest.raises(ValueError, match="must align"):
        mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])
