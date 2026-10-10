"""Post-ranking document-level diversification for retrieval results."""

from collections.abc import Callable, Iterable
import os
from typing import TypeVar

DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 2
MAX_CHUNKS_PER_DOCUMENT_ENV = "RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT"

_RankedItem = TypeVar("_RankedItem")


def validate_max_chunks_per_document(value: int) -> int:
    """Validate a per-document result cap supplied by an explicit caller."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_chunks_per_document must be an integer")
    if value < 1:
        raise ValueError("max_chunks_per_document must be at least one")
    return value


def get_default_max_chunks_per_document() -> int:
    """Read and validate the production diversification cap from the environment."""

    configured = os.getenv(MAX_CHUNKS_PER_DOCUMENT_ENV)
    if configured is None or not configured.strip():
        return DEFAULT_MAX_CHUNKS_PER_DOCUMENT
    try:
        value = int(configured)
    except ValueError as exc:
        raise ValueError(
            f"{MAX_CHUNKS_PER_DOCUMENT_ENV} must be an integer greater than or equal to 1"
        ) from exc
    try:
        return validate_max_chunks_per_document(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{MAX_CHUNKS_PER_DOCUMENT_ENV} must be an integer greater than or equal to 1"
        ) from exc


def resolve_max_chunks_per_document(
    *,
    diversify: bool,
    max_chunks_per_document: int | None = None,
) -> int | None:
    """Resolve an explicit cap, configured default, or intentional opt-out."""

    if max_chunks_per_document is not None:
        validated_override = validate_max_chunks_per_document(
            max_chunks_per_document
        )
    else:
        validated_override = None
    if not isinstance(diversify, bool):
        raise TypeError("diversify must be a boolean")
    if not diversify:
        return None
    if validated_override is not None:
        return validated_override
    return get_default_max_chunks_per_document()


def diversify_ranked_results(
    ranked_items: Iterable[_RankedItem],
    *,
    top_k: int,
    max_chunks_per_document: int | None,
    document_id: Callable[[_RankedItem], str],
) -> list[_RankedItem]:
    """Keep ranked items in order while enforcing an optional document cap.

    ``None`` intentionally disables diversification. Callers must provide items
    in final score order; this helper never recalculates or reorders scores.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if max_chunks_per_document is not None:
        validate_max_chunks_per_document(max_chunks_per_document)

    selected: list[_RankedItem] = []
    counts: dict[str, int] = {}
    for item in ranked_items:
        if len(selected) >= top_k:
            break
        stable_document_id = document_id(item)
        if not isinstance(stable_document_id, str) or not stable_document_id:
            raise ValueError(
                "retrieval result document_id must be a non-empty string"
            )
        if (
            max_chunks_per_document is not None
            and counts.get(stable_document_id, 0) >= max_chunks_per_document
        ):
            continue
        selected.append(item)
        counts[stable_document_id] = counts.get(stable_document_id, 0) + 1
    return selected
