"""Information-retrieval foundations for CaseLens."""

from retrieval.bm25 import BM25Index, BM25Searcher
from retrieval.preprocessing import (
    BM25SearchResult,
    LegalDocumentMetadata,
    LegalTextChunk,
    chunk_text,
    clean_text,
    load_text_document,
)

__all__ = [
    "BM25Index",
    "BM25SearchResult",
    "BM25Searcher",
    "LegalDocumentMetadata",
    "LegalTextChunk",
    "chunk_text",
    "clean_text",
    "load_text_document",
]
