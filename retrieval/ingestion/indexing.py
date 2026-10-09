"""Reusable builders that compose existing lexical and semantic indexes."""

from collections.abc import Iterable

from retrieval.bm25 import BM25Index
from retrieval.embeddings.base import EmbeddingService
from retrieval.embeddings.gemini import GeminiEmbeddingService
from retrieval.preprocessing.metadata import LegalTextChunk
from retrieval.vector_store import SemanticSearchService


def build_bm25_index(chunks: Iterable[LegalTextChunk]) -> BM25Index:
    """Build the existing BM25 index from successfully ingested chunks."""

    return BM25Index().build(chunks)


def build_semantic_index(
    chunks: Iterable[LegalTextChunk],
    *,
    embedding_service: EmbeddingService | None = None,
) -> SemanticSearchService:
    """Build semantic search only when an explicit caller invokes this helper."""

    service = SemanticSearchService(
        embedding_service or GeminiEmbeddingService()
    )
    service.build_index(chunks)
    return service
