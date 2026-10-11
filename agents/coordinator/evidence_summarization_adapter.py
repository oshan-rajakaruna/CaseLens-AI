"""Coordinator-owned deterministic conversion of supplied evidence documents."""

from hashlib import sha256

from agents.coordinator.schemas import CoordinatorRequest
from agents.summarization import SummarizationAgent
from agents.summarization.schemas import EvidenceItem, SourceReference, SummarizationRequest, SummarizationResponse


class CoordinatorEvidenceSummarizationAdapter:
    """Summarize supplied evidence only; never invokes Analysis, Retrieval, or providers."""

    def __init__(self, summarization_agent: SummarizationAgent | None = None) -> None:
        self._summarization_agent = summarization_agent or SummarizationAgent()

    def summarize(self, request: CoordinatorRequest) -> SummarizationResponse:
        items = [
            EvidenceItem(
                text=document.text,
                source=SourceReference(
                    document_id=document.document_id,
                    excerpt=document.text,
                    citation_id=_citation_id(document.document_id, document.text),
                ),
            )
            for document in request.evidence_documents
            if document.text.strip()
        ]
        if not items:
            raise ValueError("No non-blank supplied evidence documents are available.")
        response = self._summarization_agent.summarize(SummarizationRequest(
            case_id=request.case_id,
            document_ids=[item.source.document_id for item in items if item.source],
            query=request.user_query,
            summary_type="evidence_summary",
            mode="deterministic",
            evidence_items=items,
        ))
        assert isinstance(response, SummarizationResponse)
        return response


def _citation_id(document_id: str, text: str) -> str:
    return "SRC-" + sha256(f"{document_id}\0{text}".encode("utf-8")).hexdigest()[:16]
