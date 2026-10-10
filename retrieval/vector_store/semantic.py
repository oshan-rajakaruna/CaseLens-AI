"""Semantic indexing and search independent from BM25 retrieval."""

from collections.abc import Iterable

from retrieval.embeddings.base import EmbeddingService
from retrieval.embeddings.checkpoint import EmbeddingCheckpointStore
from retrieval.preprocessing.metadata import (
    LegalTextChunk,
    SemanticSearchResult,
)
from retrieval.vector_store.index import InMemoryVectorIndex


class SemanticSearchService:
    """Build and search a replaceable vector index using an embedding service."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        *,
        checkpoint: EmbeddingCheckpointStore | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.index = InMemoryVectorIndex(embedding_service.dimension)
        self.checkpoint = checkpoint
        self.checkpoint_hits = 0
        self.new_embeddings = 0

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

        chunks_to_embed = list(unique_chunks.values())
        vectors_by_chunk_id: dict[str, list[float]] = {}
        missing_chunks: list[LegalTextChunk] = []
        if self.checkpoint is not None:
            for chunk in chunks_to_embed:
                cached = self.checkpoint.get(self.embedding_service, chunk)
                if cached is None:
                    missing_chunks.append(chunk)
                else:
                    vectors_by_chunk_id[chunk.chunk_id] = cached
        else:
            missing_chunks = chunks_to_embed

        batch_embedder = getattr(self.embedding_service, "embed_documents", None)
        if callable(batch_embedder):
            batch_size = getattr(
                self.embedding_service,
                "batch_size",
                len(missing_chunks) or 1,
            )
            for start in range(0, len(missing_chunks), batch_size):
                chunk_batch = missing_chunks[start : start + batch_size]
                vectors = batch_embedder(
                    [
                        (chunk.chunk_text, chunk.metadata.case_name)
                        for chunk in chunk_batch
                    ]
                )
                if len(vectors) != len(chunk_batch):
                    raise ValueError(
                        "Embedding provider returned an unexpected batch size"
                    )
                for vector, chunk in zip(vectors, chunk_batch, strict=True):
                    vectors_by_chunk_id[chunk.chunk_id] = vector
                if self.checkpoint is not None:
                    self.checkpoint.put_many(
                        self.embedding_service,
                        zip(chunk_batch, vectors, strict=True),
                    )
        else:
            for chunk in missing_chunks:
                vector = self.embedding_service.embed_document(
                    chunk.chunk_text,
                    title=chunk.metadata.case_name,
                )
                vectors_by_chunk_id[chunk.chunk_id] = vector
                if self.checkpoint is not None:
                    self.checkpoint.put_many(
                        self.embedding_service,
                        [(chunk, vector)],
                    )

        replacement = InMemoryVectorIndex(self.embedding_service.dimension)
        for chunk in chunks_to_embed:
            replacement.add(vectors_by_chunk_id[chunk.chunk_id], chunk)
        self.index = replacement
        self.checkpoint_hits = len(chunks_to_embed) - len(missing_chunks)
        self.new_embeddings = len(missing_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        max_chunks_per_document: int | None = None,
    ) -> list[SemanticSearchResult]:
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
        matches = self.index.search(
            query_vector,
            top_k=top_k,
            max_chunks_per_document=max_chunks_per_document,
        )
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
