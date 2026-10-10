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
