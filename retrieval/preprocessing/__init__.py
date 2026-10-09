"""Legal-document loading, cleaning, chunking, and metadata helpers."""

from retrieval.preprocessing.chunker import chunk_text
from retrieval.preprocessing.cleaner import clean_text
from retrieval.preprocessing.loader import DocumentLoadError, load_text_document
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    LegalDocumentMetadata,
    LegalTextChunk,
)

__all__ = [
    "BM25SearchResult",
    "DocumentLoadError",
    "LegalDocumentMetadata",
    "LegalTextChunk",
    "chunk_text",
    "clean_text",
    "load_text_document",
]
