"""Reusable builders that compose existing lexical and semantic indexes."""

from collections.abc import Iterable
from pathlib import Path

from retrieval.bm25 import BM25Index
from retrieval.embeddings import (
    EmbeddingCheckpointStore,
    EmbeddingService,
    create_embedding_service,
)
from retrieval.preprocessing.metadata import LegalTextChunk
from retrieval.vector_store import SemanticSearchService


def build_bm25_index(chunks: Iterable[LegalTextChunk]) -> BM25Index:
    """Build the existing BM25 index from successfully ingested chunks."""

    return BM25Index().build(chunks)


def build_semantic_index(
    chunks: Iterable[LegalTextChunk],
    *,
    embedding_service: EmbeddingService | None = None,
    checkpoint_path: str | Path | None = None,
) -> SemanticSearchService:
    """Build semantic search only when an explicit caller invokes this helper."""

    service = SemanticSearchService(
        embedding_service or create_embedding_service(),
        checkpoint=(
            EmbeddingCheckpointStore(checkpoint_path)
            if checkpoint_path is not None
            else None
        ),
    )
    service.build_index(chunks)
    return service
