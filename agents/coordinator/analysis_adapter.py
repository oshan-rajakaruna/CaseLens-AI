"""Coordinator-owned adapter for one-document-at-a-time Analysis execution."""

from collections.abc import Sequence

from agents.analysis import AnalysisAgent, AnalysisRequest, AnalysisResponse
from agents.coordinator.schemas import CoordinatorEvidenceDocument, CoordinatorRequest
from backend.schemas import Case, Document


class AnalysisCoordinatorAdapter:
    """Invoke Analysis while preserving its typed output without rewriting findings."""

    def __init__(self, analysis_agent: AnalysisAgent | None = None) -> None:
        self._analysis_agent = analysis_agent or AnalysisAgent()

    def analyze_documents(self, request: CoordinatorRequest) -> Sequence[AnalysisResponse]:
        """Analyze each supplied evidence document in declared request order."""

        if request.case_title is None:
            raise ValueError("case_title is required for Analysis execution")
        return [
            self._analysis_agent.analyze(self._to_analysis_request(request, document))
            for document in request.evidence_documents
        ]

    @staticmethod
    def _to_analysis_request(
        request: CoordinatorRequest,
        document: CoordinatorEvidenceDocument,
    ) -> AnalysisRequest:
        """Map Coordinator-owned supplied inputs without creating provenance."""

        return AnalysisRequest(
            case=Case(id=request.case_id, title=request.case_title),
            document=Document(
                id=document.document_id,
                case_id=request.case_id,
                name=document.name,
                media_type=document.media_type,
            ),
            text=document.text,
            document_metadata=document.document_metadata,
            segments=document.segments,
        )
