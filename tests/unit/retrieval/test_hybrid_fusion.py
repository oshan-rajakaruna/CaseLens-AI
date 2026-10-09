"""Tests for configurable weighted hybrid score fusion."""

import pytest

from retrieval.hybrid.fusion import fuse_scores, normalize_weights


def test_default_fusion_uses_equal_weights() -> None:
    assert fuse_scores(1.0, 0.0) == pytest.approx(0.5)
    assert fuse_scores(0.25, 0.75) == pytest.approx(0.5)


def test_weights_are_normalized_internally() -> None:
    assert normalize_weights(2.0, 1.0) == pytest.approx((2 / 3, 1 / 3))
    assert fuse_scores(1.0, 0.0, bm25_weight=2.0, semantic_weight=1.0) == (
        pytest.approx(2 / 3)
    )


@pytest.mark.parametrize(
    ("bm25_weight", "semantic_weight", "error_type"),
    [
        (-1.0, 1.0, ValueError),
        (0.0, 0.0, ValueError),
        (float("inf"), 1.0, ValueError),
        ("high", 1.0, TypeError),
        (True, 1.0, TypeError),
    ],
)
def test_invalid_weights_are_rejected(
    bm25_weight: object,
    semantic_weight: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        normalize_weights(  # type: ignore[arg-type]
            bm25_weight,
            semantic_weight,
        )


@pytest.mark.parametrize("scores", [(-0.1, 0.5), (0.5, 1.1)])
def test_fusion_requires_normalized_component_scores(
    scores: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        fuse_scores(*scores)
