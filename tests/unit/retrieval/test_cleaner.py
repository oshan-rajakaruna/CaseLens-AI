"""Tests for conservative legal text cleaning."""

import pytest

from retrieval.preprocessing.cleaner import clean_text


def test_cleaner_normalizes_whitespace_without_destroying_legal_text() -> None:
    text = "  Court   Order\r\n\r\n\r\nSection 12(1):  Rights; Duties.\t\t\r\n"

    assert clean_text(text) == "Court Order\n\nSection 12(1): Rights; Duties."


def test_cleaner_preserves_case_and_punctuation() -> None:
    text = "Fernando v. State — Appeal No. 42/2025"

    assert clean_text(text) == text


def test_cleaner_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        clean_text(None)  # type: ignore[arg-type]
