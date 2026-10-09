"""Tests for plain-text document loading."""

from pathlib import Path
from unittest.mock import patch

import pytest

from retrieval.preprocessing.loader import DocumentLoadError, load_text_document

FIXTURES = Path(__file__).parents[2] / "fixtures" / "retrieval"


def test_loader_reads_utf8_fixture() -> None:
    text = load_text_document(FIXTURES / "synthetic_employment.txt")

    assert "SYNTHETIC TEST DATA" in text
    assert "employment termination" in text


def test_loader_handles_utf8_byte_order_mark() -> None:
    assert load_text_document(FIXTURES / "utf8_bom.txt") == "Legal text\n"


def test_loader_reports_missing_document() -> None:
    with pytest.raises(FileNotFoundError, match="Document not found"):
        load_text_document(FIXTURES / "missing.txt")


def test_loader_rejects_empty_text() -> None:
    with pytest.raises(DocumentLoadError):
        load_text_document(FIXTURES / "empty.txt")


def test_loader_rejects_binary_like_text() -> None:
    document = FIXTURES / "synthetic_employment.txt"

    with patch.object(Path, "read_text", return_value="\x00binary-like content"):
        with pytest.raises(DocumentLoadError, match="binary data"):
            load_text_document(document)


def test_loader_rejects_non_utf8_bytes() -> None:
    document = FIXTURES / "synthetic_employment.txt"
    decoding_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")

    with patch.object(Path, "read_text", side_effect=decoding_error):
        with pytest.raises(DocumentLoadError, match="not valid UTF-8"):
            load_text_document(document)
