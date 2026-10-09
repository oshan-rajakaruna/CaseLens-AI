"""Filesystem validation for manifest-referenced legal source files."""

from pathlib import Path

from retrieval.ingestion.manifest import LegalDocumentManifestEntry

SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".txt"})


class DocumentValidationError(ValueError):
    """A safe validation error that never includes a local absolute path."""


def validate_document_path(
    entry: LegalDocumentManifestEntry,
    input_dir: str | Path,
) -> Path:
    """Resolve and validate one file while preventing path traversal."""

    root = Path(input_dir).resolve()
    candidate_name = Path(entry.file_name)
    if candidate_name.is_absolute():
        raise DocumentValidationError("file_name must be relative to input-dir")
    if candidate_name.suffix.casefold() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentValidationError("Unsupported file type; supported types: .txt")

    candidate = (root / candidate_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DocumentValidationError("file_name must stay within input-dir") from exc

    if not candidate.exists():
        raise DocumentValidationError("Referenced file does not exist")
    if not candidate.is_file():
        raise DocumentValidationError("Referenced path is not a file")
    try:
        if candidate.stat().st_size == 0:
            raise DocumentValidationError("Referenced file is empty")
    except OSError as exc:
        raise DocumentValidationError("Referenced file could not be inspected") from exc
    return candidate
