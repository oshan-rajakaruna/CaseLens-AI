"""Score normalization helpers for hybrid retrieval."""

from collections.abc import Sequence
import math
from numbers import Real


def min_max_normalize(scores: Sequence[float]) -> list[float]:
    """Normalize finite numeric scores to the inclusive range ``[0, 1]``.

    A non-empty set of identical scores receives ``1.0`` for every item. This
    includes a single result and records equal relevance within that retrieval
    channel without inventing an ordering.
    """

    if len(scores) == 0:
        return []

    numeric_scores: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("scores must contain only numeric values")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("scores must contain only finite values")
        numeric_scores.append(numeric_score)

    minimum = min(numeric_scores)
    maximum = max(numeric_scores)
    if minimum == maximum:
        return [1.0] * len(numeric_scores)

    score_range = maximum - minimum
    return [(score - minimum) / score_range for score in numeric_scores]
