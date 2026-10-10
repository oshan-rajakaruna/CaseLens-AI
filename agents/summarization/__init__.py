"""Provider-neutral, citation-validating Summarization Agent contracts."""

from agents.summarization.agent import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SummarizationAgent,
    SummarizationAgentError,
    SummarizationProviderError,
    SummarizationValidationError,
)
from agents.summarization.citations import (
    CitationValidationResult,
    validate_context_citations,
)
from agents.summarization.prompt import build_summarization_provider_request
from agents.summarization.provider import SummarizationProvider
from agents.summarization.provenance import (
    UnknownContextCitationError,
    get_citation_provenance,
    render_citation_provenance,
)
from agents.summarization.schemas import (
    SummarizationCitationValidation,
    SummarizationOutput,
    SummarizationProviderMetadata,
    SummarizationProviderRequest,
    SummarizationProviderResponse,
    SummarizationResult,
)

__all__ = [
    "CitationValidationResult",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "SummarizationAgent",
    "SummarizationAgentError",
    "SummarizationCitationValidation",
    "SummarizationOutput",
    "SummarizationProvider",
    "SummarizationProviderError",
    "SummarizationProviderMetadata",
    "SummarizationProviderRequest",
    "SummarizationProviderResponse",
    "SummarizationResult",
    "SummarizationValidationError",
    "UnknownContextCitationError",
    "build_summarization_provider_request",
    "get_citation_provenance",
    "render_citation_provenance",
    "validate_context_citations",
]
