"""Coordinator agent contracts and state models."""

from agents.coordinator.coordinator import Coordinator
from agents.coordinator.retrieval_context import LegalRetrievalContextService
from agents.coordinator.research_workflow import ResearchWorkflowCoordinatorAdapter
from agents.coordinator.schemas import (
    LegalRetrievalContextRequest,
    SummarizationReadyPayload,
)

__all__ = [
    "Coordinator",
    "LegalRetrievalContextRequest",
    "LegalRetrievalContextService",
    "ResearchWorkflowCoordinatorAdapter",
    "SummarizationReadyPayload",
]
