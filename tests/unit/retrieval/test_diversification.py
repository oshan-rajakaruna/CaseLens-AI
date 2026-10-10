"""Tests for score-preserving document-level result diversification."""

from dataclasses import dataclass

import pytest

from retrieval.diversification import (
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    diversify_ranked_results,
    get_default_max_chunks_per_document,
    resolve_max_chunks_per_document,
)


@dataclass(frozen=True)
class _RankedItem:
    name: str
    document_id: str


def _select(items: list[_RankedItem], *, top_k: int, cap: int | None) -> list[str]:
    selected = diversify_ranked_results(
        items,
        top_k=top_k,
        max_chunks_per_document=cap,
        document_id=lambda item: item.document_id,
    )
    return [item.name for item in selected]


def test_cap_preserves_order_and_fills_from_lower_ranked_documents() -> None:
    ranked = [
        _RankedItem("a1", "a"),
        _RankedItem("a2", "a"),
        _RankedItem("a3", "a"),
        _RankedItem("b1", "b"),
        _RankedItem("c1", "c"),
    ]

    assert _select(ranked, top_k=4, cap=2) == ["a1", "a2", "b1", "c1"]
    assert _select(ranked, top_k=3, cap=1) == ["a1", "b1", "c1"]


def test_exhausted_candidates_may_return_fewer_than_top_k() -> None:
    ranked = [_RankedItem("a1", "a"), _RankedItem("a2", "a")]

    assert _select(ranked, top_k=5, cap=1) == ["a1"]


def test_none_intentionally_disables_diversification() -> None:
    ranked = [
        _RankedItem("a1", "a"),
        _RankedItem("a2", "a"),
        _RankedItem("b1", "b"),
    ]

    assert _select(ranked, top_k=2, cap=None) == ["a1", "a2"]


def test_configured_default_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", raising=False)
    assert get_default_max_chunks_per_document() == DEFAULT_MAX_CHUNKS_PER_DOCUMENT

    monkeypatch.setenv("RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", "3")
    assert resolve_max_chunks_per_document(diversify=True) == 3
    assert resolve_max_chunks_per_document(
        diversify=True,
        max_chunks_per_document=1,
    ) == 1
    assert resolve_max_chunks_per_document(diversify=False) is None


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_invalid_environment_configuration_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", value)

    with pytest.raises(
        ValueError,
        match="RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT must be an integer",
    ):
        get_default_max_chunks_per_document()


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_explicit_cap_is_rejected(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_max_chunks_per_document(
            diversify=True,
            max_chunks_per_document=value,  # type: ignore[arg-type]
        )
