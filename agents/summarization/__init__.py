"""Public interface for the CaseLens Summarization Agent foundation."""

from agents.summarization.agent import SummarizationAgent, SummaryGenerator
from agents.summarization.exceptions import (
    EmptyModelResponseError,
    GeminiAuthenticationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    InvalidSummarizationInputError,
    InvalidModelResponseError,
    MissingEvidenceError,
    ModelUnavailableError,
    UnsupportedSourceReferenceError,
)
from agents.summarization.gemini import (
    GeminiSummaryGenerator,
    build_summarization_prompt,
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
    "EmptyModelResponseError",
    "EvidenceSummary",
    "ExtractedFact",
    "GeminiAuthenticationError",
    "GeminiProviderError",
    "GeminiRateLimitError",
    "GeminiSummaryGenerator",
    "GeminiTimeoutError",
    "InvalidModelResponseError",
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
    "build_summarization_prompt",
]
