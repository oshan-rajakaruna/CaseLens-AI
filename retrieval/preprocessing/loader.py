"""Plain-text legal document loading.

The narrow function in this module deliberately supports UTF-8 text only.
Additional document formats can be added later without changing preprocessing
or retrieval callers.
"""

from pathlib import Path


class DocumentLoadError(ValueError):
    """Raised when a document exists but cannot be used as UTF-8 text."""


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
            f"Document is not valid UTF-8 text: {path}"
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(f"Could not read document: {path}") from exc

    if not text.strip():
        raise DocumentLoadError(f"Document contains no usable text: {path}")
    if "\x00" in text:
        raise DocumentLoadError(f"Document appears to contain binary data: {path}")

    return text
