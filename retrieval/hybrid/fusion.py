"""Configurable weighted fusion for normalized retrieval scores."""

import math
from numbers import Real

DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5


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
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> float:
    """Combine already normalized BM25 and semantic scores."""

    for score in (normalized_bm25_score, normalized_semantic_score):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("normalized scores must be numeric")
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            raise ValueError("normalized scores must be between zero and one")

    normalized_bm25_weight, normalized_semantic_weight = normalize_weights(
        bm25_weight,
        semantic_weight,
    )
    return (
        normalized_bm25_weight * float(normalized_bm25_score)
        + normalized_semantic_weight * float(normalized_semantic_score)
    )
