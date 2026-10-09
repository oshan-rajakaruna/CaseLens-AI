"""Safe TXT and PDF loading for curated legal documents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader


class DocumentLoadError(ValueError):
    """Base error with a safe ingestion category and public message."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "unreadable_document",
        public_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.public_message = public_message or message


class UnreadablePdfError(DocumentLoadError):
    """Raised when pypdf cannot safely parse or extract a PDF."""

    def __init__(self) -> None:
        super().__init__(
            "PDF could not be read or parsed",
            category="unreadable_pdf",
        )


class EmptyPdfTextError(DocumentLoadError):
    """Raised when a PDF has no pages from which text can be extracted."""

    def __init__(self) -> None:
        super().__init__(
            "PDF contains no pages with extractable text",
            category="empty_extracted_text",
        )


class PossibleScannedPdfError(DocumentLoadError):
    """Raised when a PDF has pages but no embedded usable text."""

    def __init__(self) -> None:
        super().__init__(
            "PDF contains no extractable text. OCR is not supported in the current MVP.",
            category="possible_scanned_pdf",
        )


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one physical PDF page in source order."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Normalized loader output with optional page-aware provenance."""

    text: str
    document_type: Literal["txt", "pdf"]
    pages: tuple[ExtractedPage, ...] = ()


def load_text_document(file_path: str | Path) -> str:
    """Load a non-empty UTF-8 plain-text document.

    UTF-8 with an optional byte-order mark is accepted. Missing paths and
    directories use the standard filesystem exceptions; unreadable, invalid,
    or empty text is reported as :class:`DocumentLoadError`.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Document path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"Document is not valid UTF-8 text: {path}",
            category="unreadable_text",
            public_message="Document could not be loaded as non-empty UTF-8 text",
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"Could not read document: {path}",
            category="unreadable_text",
            public_message="Document could not be loaded as non-empty UTF-8 text",
        ) from exc

    if not text.strip():
        raise DocumentLoadError(
            f"Document contains no usable text: {path}",
            category="empty_extracted_text",
            public_message="Document could not be loaded as non-empty UTF-8 text",
        )
    if "\x00" in text:
        raise DocumentLoadError(
            f"Document appears to contain binary data: {path}",
            category="unreadable_text",
            public_message="Document could not be loaded as non-empty UTF-8 text",
        )

    return text


def load_pdf_document(file_path: str | Path) -> LoadedDocument:
    """Extract embedded text from a PDF while retaining ordered page records.

    This function only asks pypdf for textual page content. It does not execute
    embedded actions, JavaScript, attachments, macros, or OCR.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Document not found")
    if not path.is_file():
        raise IsADirectoryError("Document path is not a file")

    try:
        with path.open("rb") as source:
            reader = PdfReader(source, strict=True)
            if not reader.pages:
                raise EmptyPdfTextError()
            pages = tuple(
                ExtractedPage(
                    page_number=page_number,
                    text=extracted if isinstance(extracted, str) else "",
                )
                for page_number, page in enumerate(reader.pages, start=1)
                for extracted in (page.extract_text(),)
            )
    except DocumentLoadError:
        raise
    except Exception:
        # The parser can surface several low-level exceptions for malformed
        # object graphs. None should escape with paths or document fragments.
        raise UnreadablePdfError() from None

    usable_page_text = [page.text.strip() for page in pages if page.text.strip()]
    if not usable_page_text:
        raise PossibleScannedPdfError()
    return LoadedDocument(
        text="\n\n".join(usable_page_text),
        document_type="pdf",
        pages=pages,
    )


def load_document(file_path: str | Path) -> LoadedDocument:
    """Load a supported document through one extension-based interface."""

    path = Path(file_path)
    extension = path.suffix.casefold()
    if extension == ".txt":
        return LoadedDocument(
            text=load_text_document(path),
            document_type="txt",
        )
    if extension == ".pdf":
        return load_pdf_document(path)
    raise DocumentLoadError(
        "Unsupported document type; supported types: .txt, .pdf",
        category="unsupported_file_type",
    )
