"""Tests for configurable weighted hybrid score fusion."""

import pytest

from retrieval.hybrid.fusion import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_SEMANTIC_WEIGHT,
    fuse_scores,
    get_default_hybrid_weights,
    normalize_weights,
)


def test_default_fusion_uses_evaluated_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYBRID_BM25_WEIGHT", raising=False)
    monkeypatch.delenv("HYBRID_SEMANTIC_WEIGHT", raising=False)

    assert DEFAULT_BM25_WEIGHT == 0.8
    assert DEFAULT_SEMANTIC_WEIGHT == 0.2
    assert get_default_hybrid_weights() == pytest.approx((0.8, 0.2))
    assert fuse_scores(1.0, 0.0) == pytest.approx(0.8)
    assert fuse_scores(0.25, 0.75) == pytest.approx(0.35)


def test_default_weights_can_be_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", "0.7")
    monkeypatch.setenv("HYBRID_SEMANTIC_WEIGHT", "0.3")

    assert get_default_hybrid_weights() == pytest.approx((0.7, 0.3))
    assert fuse_scores(1.0, 0.0) == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("bm25_weight", "semantic_weight", "message"),
    [
        ("0.8", "0.8", "sum to 1.0"),
        ("-0.1", "1.1", "cannot be negative"),
        ("high", "0.2", "must be numeric"),
    ],
)
def test_invalid_environment_weight_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    bm25_weight: str,
    semantic_weight: str,
    message: str,
) -> None:
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", bm25_weight)
    monkeypatch.setenv("HYBRID_SEMANTIC_WEIGHT", semantic_weight)

    with pytest.raises(ValueError, match=message):
        get_default_hybrid_weights()


def test_weights_are_normalized_internally() -> None:
    assert normalize_weights(2.0, 1.0) == pytest.approx((2 / 3, 1 / 3))
    assert fuse_scores(1.0, 0.0, bm25_weight=2.0, semantic_weight=1.0) == (
        pytest.approx(2 / 3)
    )


def test_explicit_weights_override_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", "invalid")
    monkeypatch.setenv("HYBRID_SEMANTIC_WEIGHT", "invalid")

    assert fuse_scores(
        1.0,
        0.0,
        bm25_weight=0.6,
        semantic_weight=0.4,
    ) == pytest.approx(0.6)


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
