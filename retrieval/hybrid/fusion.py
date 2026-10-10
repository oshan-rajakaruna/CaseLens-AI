"""Configurable weighted fusion for normalized retrieval scores."""

import math
from numbers import Real
from os import getenv

DEFAULT_BM25_WEIGHT = 0.8
DEFAULT_SEMANTIC_WEIGHT = 0.2
HYBRID_BM25_WEIGHT_ENV = "HYBRID_BM25_WEIGHT"
HYBRID_SEMANTIC_WEIGHT_ENV = "HYBRID_SEMANTIC_WEIGHT"


def get_default_hybrid_weights() -> tuple[float, float]:
    """Read and validate the configured default hybrid weight pair."""

    try:
        bm25_weight = float(
            getenv(HYBRID_BM25_WEIGHT_ENV, str(DEFAULT_BM25_WEIGHT))
        )
        semantic_weight = float(
            getenv(HYBRID_SEMANTIC_WEIGHT_ENV, str(DEFAULT_SEMANTIC_WEIGHT))
        )
    except ValueError as exc:
        raise ValueError(
            "hybrid retrieval environment weights must be numeric"
        ) from exc

    normalize_weights(bm25_weight, semantic_weight)
    if not math.isclose(
        bm25_weight + semantic_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "HYBRID_BM25_WEIGHT and HYBRID_SEMANTIC_WEIGHT must sum to 1.0"
        )
    return bm25_weight, semantic_weight


def resolve_hybrid_weights(
    bm25_weight: float | None = None,
    semantic_weight: float | None = None,
) -> tuple[float, float]:
    """Apply configured defaults to omitted values and normalize overrides."""

    if bm25_weight is not None and semantic_weight is not None:
        return normalize_weights(bm25_weight, semantic_weight)
    default_bm25, default_semantic = get_default_hybrid_weights()
    return normalize_weights(
        default_bm25 if bm25_weight is None else bm25_weight,
        default_semantic if semantic_weight is None else semantic_weight,
    )


def normalize_weights(
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> tuple[float, float]:
    """Validate non-negative finite weights and normalize them to sum to one."""

    weights = (bm25_weight, semantic_weight)
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("retrieval weights must be numeric")
        if not math.isfinite(float(weight)):
            raise ValueError("retrieval weights must be finite")
        if weight < 0:
            raise ValueError("retrieval weights cannot be negative")

    total = float(bm25_weight) + float(semantic_weight)
    if total <= 0:
        raise ValueError("at least one retrieval weight must be greater than zero")
    return float(bm25_weight) / total, float(semantic_weight) / total


def fuse_scores(
    normalized_bm25_score: float,
    normalized_semantic_score: float,
    *,
    bm25_weight: float | None = None,
    semantic_weight: float | None = None,
) -> float:
    """Combine already normalized BM25 and semantic scores."""

    for score in (normalized_bm25_score, normalized_semantic_score):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("normalized scores must be numeric")
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            raise ValueError("normalized scores must be between zero and one")

    normalized_bm25_weight, normalized_semantic_weight = resolve_hybrid_weights(
        bm25_weight,
        semantic_weight,
    )
    return (
        normalized_bm25_weight * float(normalized_bm25_score)
        + normalized_semantic_weight * float(normalized_semantic_score)
    )
