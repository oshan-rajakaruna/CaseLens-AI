"""Tests for safe, page-aware PDF text extraction."""

from pathlib import Path

import pytest

from retrieval.preprocessing import (
    DocumentLoadError,
    EmptyPdfTextError,
    PossibleScannedPdfError,
    UnreadablePdfError,
    load_document,
    load_pdf_document,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "pdf"


def test_unified_loader_preserves_existing_txt_behavior() -> None:
    path = Path("tests/fixtures/retrieval/synthetic_employment.txt")

    loaded = load_document(path)

    assert loaded.document_type == "txt"
    assert loaded.pages == ()
    assert "SYNTHETIC TEST DATA" in loaded.text


def test_pdf_loader_extracts_text_from_a_valid_pdf() -> None:
    loaded = load_pdf_document(FIXTURES / "synthetic_valid.pdf")

    assert loaded.document_type == "pdf"
    assert "SYNTHETIC PDF TEST DATA" in loaded.text
    assert loaded.pages[0].page_number == 1


def test_pdf_loader_preserves_page_order_and_empty_page_record() -> None:
    loaded = load_document(FIXTURES / "synthetic_multipage.pdf")

    assert [page.page_number for page in loaded.pages] == [1, 2, 3]
    assert loaded.pages[1].text == ""
    assert loaded.text.index("FIRST PAGE") < loaded.text.index("THIRD PAGE")


def test_zero_page_pdf_is_rejected_as_empty_extracted_text() -> None:
    with pytest.raises(EmptyPdfTextError) as error:
        load_pdf_document(FIXTURES / "synthetic_zero_pages.pdf")

    assert error.value.category == "empty_extracted_text"


def test_no_text_pdf_reports_ocr_boundary() -> None:
    with pytest.raises(PossibleScannedPdfError) as error:
        load_pdf_document(FIXTURES / "synthetic_no_text.pdf")

    assert error.value.category == "possible_scanned_pdf"
    assert str(error.value) == (
        "PDF contains no extractable text. OCR is not supported in the current MVP."
    )


def test_corrupted_pdf_has_safe_error_without_local_path() -> None:
    with pytest.raises(UnreadablePdfError) as error:
        load_pdf_document(FIXTURES / "synthetic_corrupted.pdf")

    assert error.value.category == "unreadable_pdf"
    assert str(FIXTURES.resolve()) not in str(error.value)
    assert "SYNTHETIC CORRUPTED" not in str(error.value)


def test_unified_loader_rejects_unsupported_extension() -> None:
    with pytest.raises(DocumentLoadError) as error:
        load_document("synthetic.docx")

    assert error.value.category == "unsupported_file_type"
