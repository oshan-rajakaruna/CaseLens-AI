"""Injectable provider boundary for the Summarization Agent."""

from typing import Protocol

from agents.summarization.schemas import (
    SummarizationProviderRequest,
    SummarizationProviderResponse,
)


class SummarizationProvider(Protocol):
    """Generate structured output without tying the agent to one SDK."""

    provider_name: str
    model_name: str | None

    def generate(
        self,
        request: SummarizationProviderRequest,
    ) -> SummarizationProviderResponse:
        """Return a structured candidate response for agent validation."""
        ...
