"""Coordinator agent contracts and state models."""

from agents.coordinator.coordinator import Coordinator
from agents.coordinator.analysis_adapter import AnalysisCoordinatorAdapter
from agents.coordinator.retrieval_adapter import CoordinatorRetrievalAdapter
from agents.coordinator.retrieval_context import LegalRetrievalContextService
from agents.coordinator.schemas import (
    CoordinatorAnalysisResponse,
    CoordinatorEvidenceDocument,
    CoordinatorMessage,
    CoordinatorPlanResponse,
    CoordinatorRequest,
    CoordinatorRetrievalResponse,
    CoordinatorRetrievalResult,
    CoordinatorStage,
    CoordinatorStageStatus,
    CoordinatorTask,
    LegalRetrievalContextRequest,
    SummarizationReadyPayload,
)

__all__ = [
    "Coordinator",
    "AnalysisCoordinatorAdapter",
    "CoordinatorRetrievalAdapter",
    "CoordinatorAnalysisResponse",
    "CoordinatorEvidenceDocument",
    "CoordinatorMessage",
    "CoordinatorPlanResponse",
    "CoordinatorRequest",
    "CoordinatorRetrievalResponse",
    "CoordinatorRetrievalResult",
    "CoordinatorStage",
    "CoordinatorStageStatus",
    "CoordinatorTask",
    "LegalRetrievalContextRequest",
    "LegalRetrievalContextService",
    "SummarizationReadyPayload",
]
