"""Information-retrieval foundations for CaseLens."""

from retrieval.bm25 import BM25Index, BM25Searcher
from retrieval.embeddings import (
    GeminiEmbeddingService,
    OpenAIEmbeddingService,
    create_embedding_service,
)
from retrieval.hybrid import HybridSearchService
from retrieval.preprocessing import (
    BM25SearchResult,
    HybridSearchResult,
    LegalDocumentMetadata,
    LegalTextChunk,
    SemanticSearchResult,
    chunk_text,
    clean_text,
    load_text_document,
)
from retrieval.vector_store import InMemoryVectorIndex, SemanticSearchService

__all__ = [
    "BM25Index",
    "BM25SearchResult",
    "BM25Searcher",
    "GeminiEmbeddingService",
    "OpenAIEmbeddingService",
    "create_embedding_service",
    "HybridSearchResult",
    "HybridSearchService",
    "InMemoryVectorIndex",
    "LegalDocumentMetadata",
    "LegalTextChunk",
    "SemanticSearchResult",
    "SemanticSearchService",
    "chunk_text",
    "clean_text",
    "load_text_document",
]
