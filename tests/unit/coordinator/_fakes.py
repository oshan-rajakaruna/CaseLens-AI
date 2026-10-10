"""Network-free Coordinator and Summarization test doubles."""

from typing import Any

from agents.coordinator.schemas import SummarizationReadyPayload
from agents.summarization import SummarizationOutput
from retrieval.diversification import diversify_ranked_results
from retrieval.preprocessing.metadata import HybridSearchResult


class FakeCoordinatorRetrievalAgent:
    """Return supplied hybrid rankings and apply the production cap helper."""

    def __init__(self, results: list[HybridSearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        *,
        bm25_weight: float | None = None,
        semantic_weight: float | None = None,
        diversify: bool = True,
        max_chunks_per_document: int | None = None,
    ) -> list[HybridSearchResult]:
        self.calls.append(
            {
                "mode": "hybrid",
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "bm25_weight": bm25_weight,
                "semantic_weight": semantic_weight,
                "diversify": diversify,
                "max_chunks_per_document": max_chunks_per_document,
            }
        )
        cap = max_chunks_per_document if diversify else None
        return diversify_ranked_results(
            self.results,
            top_k=top_k,
            max_chunks_per_document=cap,
            document_id=lambda result: result.document_id,
        )

    def search_bm25(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.calls.append({"mode": "bm25", "args": args, "kwargs": kwargs})
        return []

    def search_semantic(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.calls.append({"mode": "semantic", "args": args, "kwargs": kwargs})
        return []


class DeterministicSummarizerStub:
    """Test-only summarizer that never loads or calls an external model."""

    def summarize(self, payload: SummarizationReadyPayload) -> SummarizationOutput:
        context_ids = list(payload.citation_map)
        if not context_ids:
            return SummarizationOutput(
                answer="No retrieved context was available.",
                limitations=["No context passages were supplied"],
            )
        first = context_ids[0]
        return SummarizationOutput(
            answer=f"Deterministic test answer [{first}]",
            citations=[first],
            confidence=0.5,
            limitations=["Test stub; no model was called"],
        )
