"""Provider-independent embedding interface used by semantic retrieval."""

from typing import Protocol


class EmbeddingService(Protocol):
    """Minimal interface required by the semantic retrieval service."""

    provider: str
    model: str
    dimension: int
    batch_size: int

    def document_content_hash(self, text: str, title: str | None = None) -> str:
        """Return a provider-specific hash for checkpoint compatibility."""

        ...

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        """Return an embedding for a document chunk."""

        ...

    def embed_query(self, query: str) -> list[float]:
        """Return an embedding for a retrieval query."""

        ...

    def embed_documents(
        self,
        documents: list[tuple[str, str | None]],
    ) -> list[list[float]]:
        """Return ordered embeddings for one bounded document batch."""

        ...
