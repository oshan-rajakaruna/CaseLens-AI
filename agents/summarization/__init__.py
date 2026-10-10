"""Public interface for the CaseLens Summarization Agent foundation."""

from agents.summarization.agent import SummarizationAgent, SummaryGenerator
from agents.summarization.exceptions import (
    InvalidSummarizationInputError,
    MissingEvidenceError,
    ModelUnavailableError,
    UnsupportedSourceReferenceError,
)
from agents.summarization.schemas import (
    CaseInformation,
    Comparison,
    DraftClaim,
    EvidenceSummary,
    ExtractedFact,
    LegalPassage,
    PrecedentSummary,
    SourceReference,
    SummarizationRequest,
    SummarizationResponse,
)

__all__ = [
    "CaseInformation",
    "Comparison",
    "DraftClaim",
    "EvidenceSummary",
    "ExtractedFact",
    "InvalidSummarizationInputError",
    "LegalPassage",
    "MissingEvidenceError",
    "ModelUnavailableError",
    "PrecedentSummary",
    "SourceReference",
    "SummarizationAgent",
    "SummarizationRequest",
    "SummarizationResponse",
    "SummaryGenerator",
    "UnsupportedSourceReferenceError",
]
