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
from agents.summarization.gemini import (
    DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
    DEFAULT_GEMINI_MODEL,
    GeminiSummarizationAuthenticationError,
    GeminiSummarizationConfigurationError,
    GeminiSummarizationError,
    GeminiSummarizationModelError,
    GeminiSummarizationProvider,
    GeminiSummarizationRateLimitError,
    GeminiSummarizationResponseError,
    GeminiSummarizationSettings,
    GeminiSummarizationTimeoutError,
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
    "DEFAULT_GEMINI_MAX_OUTPUT_TOKENS",
    "DEFAULT_GEMINI_MODEL",
    "GeminiSummarizationAuthenticationError",
    "GeminiSummarizationConfigurationError",
    "GeminiSummarizationError",
    "GeminiSummarizationModelError",
    "GeminiSummarizationProvider",
    "GeminiSummarizationRateLimitError",
    "GeminiSummarizationResponseError",
    "GeminiSummarizationSettings",
    "GeminiSummarizationTimeoutError",
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
