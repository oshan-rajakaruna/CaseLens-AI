"""Deterministic, local text extraction for TXT, PDF, and DOCX evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel, Field, model_validator

from agents.analysis.schemas import AnalysisWarning, EvidenceSource

DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


class ExtractionSegment(BaseModel):
    """An exact source segment with Python start-inclusive/end-exclusive offsets."""

    segment_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str
    source: EvidenceSource
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "ExtractionSegment":
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("segment offsets must span the exact segment text")
        return self


class DocumentExtractionResult(BaseModel):
    """A local extraction result; failures always contain structured feedback."""

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    file_type: Literal["pdf", "docx", "txt", "unknown"]
    status: Literal["extracted", "no_text", "failed"]
    text: str = ""
    segments: list[ExtractionSegment] = Field(default_factory=list)
    method: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    errors: list[AnalysisWarning] = Field(default_factory=list)


def extract_document(
    file_path: str | Path,
    document_id: str,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> DocumentExtractionResult:
    """Extract evidence text locally without OCR, network access, or content logging."""

    path = Path(file_path)
    filename = path.name or str(path)
    if not document_id or not document_id.strip():
        return _failure(filename, "unknown", document_id or "unknown", "invalid_document_id", "document_id must not be blank.")
    if max_file_size_bytes <= 0:
        return _failure(filename, "unknown", document_id, "invalid_size_limit", "max_file_size_bytes must be positive.")
    try:
        if not path.exists():
            return _failure(filename, "unknown", document_id, "file_not_found", "Input file does not exist.")
        if path.is_dir():
            return _failure(filename, "unknown", document_id, "directory_not_supported", "Input path must be a file, not a directory.")
        size = path.stat().st_size
    except OSError:
        return _failure(filename, "unknown", document_id, "file_unreadable", "Input file cannot be accessed.")

    extension = path.suffix.lower()
    file_type = extension.removeprefix(".") if extension in SUPPORTED_EXTENSIONS else "unknown"
    if extension not in SUPPORTED_EXTENSIONS:
        return _failure(filename, "unknown", document_id, "unsupported_file_type", "Only .txt, .pdf, and .docx files are supported.")
    if size > max_file_size_bytes:
        return _failure(
            filename,
            file_type,
            document_id,
            "file_too_large",
            f"Input file exceeds the configured {max_file_size_bytes}-byte limit.",
            metadata={"file_size_bytes": size, "max_file_size_bytes": max_file_size_bytes},
        )

    try:
        if extension == ".txt":
            return _extract_txt(path, document_id, size)
        if extension == ".pdf":
            return _extract_pdf(path, document_id, size)
        return _extract_docx(path, document_id, size)
    except OSError:
        return _failure(filename, file_type, document_id, "file_unreadable", "Input file cannot be read.")
    except Exception:
        # Library exceptions can expose implementation details; retain a safe result.
        return _failure(filename, file_type, document_id, "extraction_failed", "Document could not be extracted safely.")


def _extract_txt(path: Path, document_id: str, size: int) -> DocumentExtractionResult:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        return _failure(path.name, "txt", document_id, "invalid_text_encoding", "TXT files must be valid UTF-8.")

    segments = _make_segments(document_id, [(text, EvidenceSource(document_id=document_id, locator="text:1", excerpt=text))])
    return _success_or_no_text(path.name, "txt", document_id, segments, "utf-8", size)


def _extract_pdf(path: Path, document_id: str, size: int) -> DocumentExtractionResult:
    if not path.read_bytes().startswith(b"%PDF-"):
        return _failure(path.name, "pdf", document_id, "invalid_pdf", "File does not have a valid PDF header.")

    import fitz

    try:
        pdf = fitz.open(path)
    except (fitz.FileDataError, RuntimeError):
        return _failure(path.name, "pdf", document_id, "invalid_pdf", "PDF is corrupted or unreadable.")

    try:
        if pdf.needs_pass:
            return _failure(path.name, "pdf", document_id, "encrypted_pdf", "Encrypted PDFs are not supported.")
        raw_segments = []
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")
            raw_segments.append(
                (
                    page_text,
                    EvidenceSource(
                        document_id=document_id,
                        page=page_number,
                        locator=f"page:{page_number}",
                        excerpt=page_text,
                    ),
                )
            )
    except (fitz.FileDataError, RuntimeError):
        return _failure(path.name, "pdf", document_id, "invalid_pdf", "PDF is corrupted or unreadable.")
    finally:
        pdf.close()

    segments = _make_segments(document_id, raw_segments)
    result = _success_or_no_text(path.name, "pdf", document_id, segments, "pymupdf", size)
    result.metadata["page_count"] = len(raw_segments)
    if result.status == "no_text":
        result.warnings.append(
            AnalysisWarning(
                code="ocr_required",
                message="PDF has no extractable text and may be scanned or image-only; OCR was not performed.",
            )
        )
    return result


def _extract_docx(path: Path, document_id: str, size: int) -> DocumentExtractionResult:
    if not _is_docx_package(path):
        return _failure(path.name, "docx", document_id, "invalid_docx", "File is not a valid DOCX package.")

    from docx import Document as DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = DocxDocument(path)
    except (BadZipFile, KeyError, ValueError):
        return _failure(path.name, "docx", document_id, "invalid_docx", "DOCX is corrupted or unreadable.")

    raw_segments: list[tuple[str, EvidenceSource]] = []
    paragraph_index = 0
    table_index = 0
    try:
        for element in document.element.body.iterchildren():
            if isinstance(element, CT_P):
                paragraph_index += 1
                paragraph = Paragraph(element, document)
                text = paragraph.text
                raw_segments.append(
                    (
                        text,
                        EvidenceSource(
                            document_id=document_id,
                            paragraph=paragraph_index,
                            locator=f"paragraph:{paragraph_index}",
                            excerpt=text,
                        ),
                    )
                )
            elif isinstance(element, CT_Tbl):
                table_index += 1
                table = Table(element, document)
                text = "\n".join("\t".join(cell.text for cell in row.cells) for row in table.rows)
                raw_segments.append(
                    (
                        text,
                        EvidenceSource(
                            document_id=document_id,
                            locator=f"table:{table_index}",
                            excerpt=text,
                        ),
                    )
                )
    except (KeyError, ValueError):
        return _failure(path.name, "docx", document_id, "invalid_docx", "DOCX is corrupted or unreadable.")

    segments = _make_segments(document_id, raw_segments)
    result = _success_or_no_text(path.name, "docx", document_id, segments, "python-docx", size)
    result.metadata.update({"paragraph_count": paragraph_index, "table_count": table_index})
    return result


def _is_docx_package(path: Path) -> bool:
    """Check the package signature before allowing python-docx to parse it."""

    try:
        with ZipFile(path) as package:
            return "[Content_Types].xml" in package.namelist() and "word/document.xml" in package.namelist()
    except (BadZipFile, OSError):
        return False


def _make_segments(
    document_id: str, raw_segments: list[tuple[str, EvidenceSource]]
) -> list[ExtractionSegment]:
    segments: list[ExtractionSegment] = []
    offset = 0
    for index, (text, source) in enumerate(raw_segments, start=1):
        if index > 1:
            offset += 1  # A single newline separates adjacent source segments.
        segments.append(
            ExtractionSegment(
                segment_id=f"{document_id}:segment:{index}",
                document_id=document_id,
                text=text,
                source=source,
                start_char=offset,
                end_char=offset + len(text),
            )
        )
        offset += len(text)
    return segments


def _success_or_no_text(
    filename: str,
    file_type: Literal["pdf", "docx", "txt"],
    document_id: str,
    segments: list[ExtractionSegment],
    method: str,
    size: int,
) -> DocumentExtractionResult:
    text = "\n".join(segment.text for segment in segments)
    metadata = {
        "file_size_bytes": size,
        "segment_count": len(segments),
        "offset_convention": "Python string indices; start-inclusive and end-exclusive.",
    }
    if text:
        return DocumentExtractionResult(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            status="extracted",
            text=text,
            segments=segments,
            method=method,
            metadata=metadata,
        )
    return DocumentExtractionResult(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        status="no_text",
        text=text,
        segments=segments,
        method=method,
        metadata=metadata,
        warnings=[AnalysisWarning(code="no_extractable_text", message="Document contains no readable text.")],
    )


def _failure(
    filename: str,
    file_type: Literal["pdf", "docx", "txt", "unknown"],
    document_id: str,
    code: str,
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> DocumentExtractionResult:
    return DocumentExtractionResult(
        document_id=document_id or "unknown",
        filename=filename or "unknown",
        file_type=file_type,
        status="failed",
        method="none",
        metadata=metadata or {},
        errors=[AnalysisWarning(code=code, message=message)],
    )
