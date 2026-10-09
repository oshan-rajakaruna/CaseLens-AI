"""Ranked search over a :class:`BM25Index`."""

from retrieval.bm25.index import BM25Index
from retrieval.preprocessing.metadata import BM25SearchResult


class BM25Searcher:
    """Return structured, deterministically ordered BM25 matches."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, top_k: int = 5) -> list[BM25SearchResult]:
        """Return up to ``top_k`` positive-scoring results."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not query.strip() or top_k == 0 or not self.index:
            return []

        candidates = [
            (score, chunk)
            for score, chunk in zip(
                self.index.scores(query), self.index.chunks, strict=True
            )
            if score > 0
        ]
        candidates.sort(
            key=lambda item: (-item[0], item[1].document_id, item[1].chunk_id)
        )

        results: list[BM25SearchResult] = []
        for rank, (score, chunk) in enumerate(candidates[:top_k], start=1):
            metadata = chunk.metadata
            results.append(
                BM25SearchResult(
                    rank=rank,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    case_name=metadata.case_name,
                    score=score,
                    chunk_text=chunk.chunk_text,
                    citation=metadata.citation,
                    source=metadata.source,
                    court=metadata.court,
                    date=metadata.date,
                    legal_category=metadata.legal_category,
                    document_type=metadata.document_type,
                    metadata=metadata,
                )
            )
        return results
