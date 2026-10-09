"""Semantic indexing and search independent from BM25 retrieval."""

from collections.abc import Iterable

from retrieval.embeddings.base import EmbeddingService
from retrieval.preprocessing.metadata import (
    LegalTextChunk,
    SemanticSearchResult,
)
from retrieval.vector_store.index import InMemoryVectorIndex


class SemanticSearchService:
    """Build and search a replaceable vector index using an embedding service."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self.index = InMemoryVectorIndex(embedding_service.dimension)

    def build_index(self, chunks: Iterable[LegalTextChunk]) -> None:
        """Replace the semantic index, embedding each unique chunk once.

        Index replacement is transactional: a failed embedding leaves the
        previously built index available.
        """

        unique_chunks: dict[str, LegalTextChunk] = {}
        for chunk in chunks:
            existing = unique_chunks.get(chunk.chunk_id)
            if existing is not None and existing != chunk:
                raise ValueError(
                    f"Conflicting chunks share chunk_id: {chunk.chunk_id}"
                )
            unique_chunks.setdefault(chunk.chunk_id, chunk)

        replacement = InMemoryVectorIndex(self.embedding_service.dimension)
        for chunk in unique_chunks.values():
            vector = self.embedding_service.embed_document(
                chunk.chunk_text,
                title=chunk.metadata.case_name,
            )
            replacement.add(vector, chunk)
        self.index = replacement

    def search(self, query: str, top_k: int = 5) -> list[SemanticSearchResult]:
        """Embed a query and return structured cosine-similarity results."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not query.strip() or top_k == 0 or not self.index:
            return []

        query_vector = self.embedding_service.embed_query(query)
        matches = self.index.search(query_vector, top_k=top_k)
        return [
            SemanticSearchResult(
                rank=rank,
                document_id=match.chunk.document_id,
                chunk_id=match.chunk.chunk_id,
                case_name=match.chunk.metadata.case_name,
                similarity_score=match.similarity_score,
                chunk_text=match.chunk.chunk_text,
                citation=match.chunk.metadata.citation,
                source=match.chunk.metadata.source,
                court=match.chunk.metadata.court,
                date=match.chunk.metadata.date,
                legal_category=match.chunk.metadata.legal_category,
                document_type=match.chunk.metadata.document_type,
                metadata=match.chunk.metadata,
            )
            for rank, match in enumerate(matches, start=1)
        ]
