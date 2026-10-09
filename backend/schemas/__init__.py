"""Reusable API and inter-agent data contracts."""

from backend.schemas.common import (
    AgentResult,
    AgentTask,
    Case,
    Citation,
    Document,
    HealthResponse,
    RootResponse,
    VerificationResult,
)
from backend.schemas.retrieval import (
    RetrievalFilters,
    RetrievalMetadataResponse,
    RetrievalMode,
    RetrievalResultResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievalTaskContext,
)

__all__ = [
    "AgentResult",
    "AgentTask",
    "Case",
    "Citation",
    "Document",
    "HealthResponse",
    "RetrievalFilters",
    "RetrievalMetadataResponse",
    "RetrievalMode",
    "RetrievalResultResponse",
    "RetrievalSearchRequest",
    "RetrievalSearchResponse",
    "RetrievalTaskContext",
    "RootResponse",
    "VerificationResult",
]
