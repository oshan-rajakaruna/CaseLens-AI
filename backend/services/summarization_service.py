"""Application service composing summarization and deterministic verification."""

from functools import lru_cache

from agents.summarization import (
    GeminiSummarizationProvider,
    SummarizationAgent,
    SummarizationVerificationBridge,
)
from agents.verification import VerificationAgent
from backend.schemas.summarization import (
    SummarizationVerifyRequest,
    SummarizationVerifyResponse,
)


class SummarizationService:
    """Generate one grounded summary and preserve its verification outcome."""

    def __init__(
        self,
        summarization_agent: SummarizationAgent,
        verification_bridge: SummarizationVerificationBridge,
    ) -> None:
        self.summarization_agent = summarization_agent
        self.verification_bridge = verification_bridge

    def summarize_and_verify(
        self,
        request: SummarizationVerifyRequest,
    ) -> SummarizationVerifyResponse:
        summary = self.summarization_agent.summarize(request.payload)
        bridged = self.verification_bridge.verify(
            request.task_id,
            request.payload,
            summary,
        )
        return SummarizationVerifyResponse(
            summary=summary,
            verification=bridged.verification_response,
            provenance=bridged.provenance,
            bridge_warnings=bridged.warnings,
        )


@lru_cache
def get_summarization_service() -> SummarizationService:
    """Build the process-local Gemini-backed development service."""

    provider = GeminiSummarizationProvider()
    return SummarizationService(
        SummarizationAgent(provider),
        SummarizationVerificationBridge(VerificationAgent()),
    )
