"""Dependency-free binary relevance metrics for ranked retrieval results.

Precision@K is the number of unique relevant IDs in the first K unique results
divided by K. Recall@K divides the same hit count by the number of relevant IDs.
Reciprocal rank is ``1 / rank`` for the first relevant result, or zero when no
relevant result is retrieved. Duplicate retrieved IDs are ignored after their
first occurrence so they cannot inflate scores.
"""

from collections.abc import Collection, Iterable, Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")


def deduplicate_ranked_ids(retrieved: Iterable[str]) -> list[str]:
    """Preserve the first occurrence of every retrieved identifier."""

    unique: list[str] = []
    seen: set[str] = set()
    for item_id in retrieved:
        if item_id not in seen:
            unique.append(item_id)
            seen.add(item_id)
    return unique


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Return binary Precision@K using K as the denominator."""

    _validate_k(k)
    relevant_set = set(relevant)
    top_k = deduplicate_ranked_ids(retrieved)[:k]
    hits = sum(item_id in relevant_set for item_id in top_k)
    return hits / k


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Return binary Recall@K, or zero for an empty relevant set."""

    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = deduplicate_ranked_ids(retrieved)[:k]
    hits = sum(item_id in relevant_set for item_id in top_k)
    return hits / len(relevant_set)


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Collection[str],
) -> float:
    """Return reciprocal rank of the first relevant unique result."""

    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, item_id in enumerate(deduplicate_ranked_ids(retrieved), start=1):
        if item_id in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_by_query: Sequence[Sequence[str]],
    relevant_by_query: Sequence[Collection[str]],
) -> float:
    """Return mean reciprocal rank across aligned query rankings."""

    if len(retrieved_by_query) != len(relevant_by_query):
        raise ValueError("retrieved and relevant query collections must align")
    if not retrieved_by_query:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(
            retrieved_by_query,
            relevant_by_query,
            strict=True,
        )
    ]
    return sum(scores) / len(scores)
