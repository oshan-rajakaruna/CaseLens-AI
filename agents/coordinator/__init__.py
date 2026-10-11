"""Coordinator agent contracts and state models."""

from agents.coordinator.coordinator import Coordinator
from agents.coordinator.retrieval_context import LegalRetrievalContextService
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


def __getattr__(name: str):
    if name == "ResearchWorkflowCoordinatorAdapter":
        from agents.coordinator.research_workflow import (
            ResearchWorkflowCoordinatorAdapter,
        )

        return ResearchWorkflowCoordinatorAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
