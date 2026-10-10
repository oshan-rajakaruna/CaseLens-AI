"""Deterministic, network-free summarization provider test doubles."""

from collections.abc import Callable
from typing import Any

from agents.summarization import (
    SummarizationProviderRequest,
    SummarizationProviderResponse,
)


class FakeSummarizationProvider:
    """Return a configured response or raise a configured exception."""

    provider_name = "fake"
    model_name = "deterministic-test"

    def __init__(
        self,
        response: SummarizationProviderResponse
        | Callable[[SummarizationProviderRequest], SummarizationProviderResponse]
        | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[SummarizationProviderRequest] = []

    def generate(
        self,
        request: SummarizationProviderRequest,
    ) -> SummarizationProviderResponse:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        if callable(self.response):
            return self.response(request)
        if self.response is None:
            return {
                "answer": "Grounded deterministic answer [CTX-001]",
                "citations": ["CTX-001"],
                "confidence": 0.5,
                "limitations": ["Deterministic test provider"],
            }
        return self.response
