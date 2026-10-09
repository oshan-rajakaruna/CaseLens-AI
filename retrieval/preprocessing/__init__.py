"""Legal-document loading, cleaning, chunking, and metadata helpers."""

from retrieval.preprocessing.chunker import chunk_text
from retrieval.preprocessing.cleaner import clean_text
from retrieval.preprocessing.loader import (
    DocumentLoadError,
    EmptyPdfTextError,
    ExtractedPage,
    LoadedDocument,
    PossibleScannedPdfError,
    UnreadablePdfError,
    load_document,
    load_pdf_document,
    load_text_document,
)
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    LegalDocumentMetadata,
    LegalTextChunk,
    SemanticSearchResult,
)

__all__ = [
    "BM25SearchResult",
    "DocumentLoadError",
    "EmptyPdfTextError",
    "ExtractedPage",
    "HybridSearchResult",
    "LegalDocumentMetadata",
    "LegalTextChunk",
    "LoadedDocument",
    "PossibleScannedPdfError",
    "SemanticSearchResult",
    "UnreadablePdfError",
    "chunk_text",
    "clean_text",
    "load_document",
    "load_pdf_document",
    "load_text_document",
]
