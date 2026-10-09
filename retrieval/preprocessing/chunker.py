"""Deterministic word-based chunking for legal text."""

from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk


def chunk_text(
    text: str,
    metadata: LegalDocumentMetadata,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[LegalTextChunk]:
    """Split text into chunks measured in words.

    The source words and their case/punctuation are retained. Whitespace between
    words is normalized because loading and cleaning remain separate concerns.
    Chunk IDs are stable for the same document and input order.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[LegalTextChunk] = []
    step = chunk_size - overlap
    for index, start in enumerate(range(0, len(words), step), start=1):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(
            LegalTextChunk(
                chunk_id=f"{metadata.document_id}-chunk-{index:04d}",
                document_id=metadata.document_id,
                chunk_text=" ".join(chunk_words),
                metadata=metadata,
            )
        )
        if start + chunk_size >= len(words):
            break

    return chunks
