"""Domain errors raised by the Summarization Agent boundary."""


class SummarizationError(Exception):
    """Base class for expected summarization failures."""


class InvalidSummarizationInputError(SummarizationError):
    """Raised when an AgentTask cannot be parsed as a request."""


class MissingEvidenceError(SummarizationError):
    """Raised when no extracted facts or legal passages were supplied."""


class ModelUnavailableError(SummarizationError):
    """Raised when no model-generation implementation is configured."""


class UnsupportedSourceReferenceError(SummarizationError):
    """Raised when generated output references a source absent from input."""


class InvalidModelResponseError(SummarizationError):
    """Raised when model output is not valid structured summarization data."""


class EmptyModelResponseError(SummarizationError):
    """Raised when the provider returns no usable model content."""


class GeminiAuthenticationError(SummarizationError):
    """Raised when Gemini rejects the configured credentials."""


class GeminiRateLimitError(SummarizationError):
    """Raised when Gemini declines a request because of a usage limit."""


class GeminiTimeoutError(SummarizationError):
    """Raised when a Gemini request times out."""


class GeminiProviderError(SummarizationError):
    """Raised for sanitized, non-specific Gemini provider failures."""
