"""Tests for hybrid retrieval score normalization."""

import pytest

from retrieval.hybrid.normalize import min_max_normalize


def test_min_max_normalization_handles_empty_and_single_scores() -> None:
    assert min_max_normalize([]) == []
    assert min_max_normalize([7.5]) == [1.0]


@pytest.mark.parametrize("scores", [[0.0, 0.0], [-2.0, -2.0], [4.0, 4.0]])
def test_equal_scores_receive_equal_full_normalized_scores(
    scores: list[float],
) -> None:
    assert min_max_normalize(scores) == [1.0, 1.0]


def test_min_max_normalization_handles_zero_and_negative_scores() -> None:
    assert min_max_normalize([-5.0, 0.0, 5.0]) == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("scores", [[1.0, float("inf")], [1.0, float("nan")]])
def test_normalization_rejects_non_finite_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        min_max_normalize(scores)


def test_normalization_rejects_non_numeric_scores() -> None:
    with pytest.raises(TypeError, match="numeric"):
        min_max_normalize([1.0, "bad"])  # type: ignore[list-item]
