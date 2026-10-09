"""Provider-independent embedding interface used by semantic retrieval."""

from typing import Protocol


class EmbeddingService(Protocol):
    """Minimal interface required by the semantic retrieval service."""

    dimension: int

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        """Return an embedding for a document chunk."""

        ...

    def embed_query(self, query: str) -> list[float]:
        """Return an embedding for a retrieval query."""

        ...
