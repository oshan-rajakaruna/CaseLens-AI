"""Deterministic test doubles for semantic retrieval tests."""


class DeterministicEmbeddingService:
    """Map synthetic legal topics to simple, deterministic vectors."""

    dimension = 3

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, str | None]] = []
        self.query_calls: list[str] = []

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        self.document_calls.append((text, title))
        return self._vector_for(text)

    def embed_query(self, query: str) -> list[float]:
        self.query_calls.append(query)
        return self._vector_for(query)

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        normalized = text.casefold()
        if any(term in normalized for term in ("employment", "termination", "dismissal")):
            return [1.0, 0.05, 0.0]
        if any(term in normalized for term in ("property", "boundary", "title")):
            return [0.05, 1.0, 0.0]
        if any(term in normalized for term in ("contract", "breach", "supplier")):
            return [0.0, 0.1, 1.0]
        return [0.2, 0.2, 0.2]
