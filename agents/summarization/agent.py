"""Provider-neutral Summarization Agent with strict citation validation."""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from agents.coordinator.schemas import SummarizationReadyPayload
from agents.summarization.citations import validate_context_citations
from agents.summarization.prompt import build_summarization_provider_request
from agents.summarization.provider import SummarizationProvider
from agents.summarization.schemas import (
    SummarizationCitationValidation,
    SummarizationOutput,
    SummarizationProviderMetadata,
    SummarizationResult,
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "Insufficient retrieved legal context to answer reliably."
)


class SummarizationAgentError(RuntimeError):
    """Base class for controlled summarization failures."""


class SummarizationProviderError(SummarizationAgentError):
    """Raised when an injected provider cannot produce a usable response."""


class SummarizationValidationError(SummarizationAgentError):
    """Raised when generated output violates grounding or citation rules."""

    def __init__(
        self,
        message: str,
        *,
        citation_validation: SummarizationCitationValidation | None = None,
    ) -> None:
        super().__init__(message)
        self.citation_validation = citation_validation


class SummarizationAgent:
    """Generate and validate evidence-grounded legal summaries."""

    def __init__(self, provider: SummarizationProvider) -> None:
        self.provider = provider

    def summarize(
        self,
        payload: SummarizationReadyPayload | Mapping[str, Any],
    ) -> SummarizationResult:
        """Return validated output or raise a controlled domain error."""

        validated_payload = SummarizationReadyPayload.model_validate(payload)
        if not validated_payload.context_passages or not validated_payload.citation_map:
            return self._no_context_result(validated_payload)

        provider_request = build_summarization_provider_request(validated_payload)
        try:
            raw_output = self.provider.generate(provider_request)
        except Exception as exc:
            raise SummarizationProviderError(
                "Summarization provider failed"
            ) from exc
        try:
            output = SummarizationOutput.model_validate(raw_output)
        except ValidationError as exc:
            raise SummarizationProviderError(
                "Summarization provider returned invalid structured output"
            ) from exc

        citation_validation = _validate_output_citations(
            output,
            validated_payload.citation_map,
        )
        if not citation_validation.inline.has_citations:
            raise SummarizationValidationError(
                "Summarization output must cite retrieved context inline",
                citation_validation=citation_validation,
            )
        if not citation_validation.all_citations_valid:
            raise SummarizationValidationError(
                "Summarization output contains invalid or inconsistent citations",
                citation_validation=citation_validation,
            )
        if validated_payload.diagnostics.warnings and not _mentions_uncertainty(
            output.limitations
        ):
            raise SummarizationValidationError(
                "Summarization limitations must acknowledge weak context",
                citation_validation=citation_validation,
            )

        return SummarizationResult(
            output=output,
            citation_validation=citation_validation,
            provider_metadata=self._provider_metadata(provider_called=True),
            used_context_ids=citation_validation.inline.valid_citations,
            warnings=list(validated_payload.diagnostics.warnings),
        )

    def _no_context_result(
        self,
        payload: SummarizationReadyPayload,
    ) -> SummarizationResult:
        output = SummarizationOutput(
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            citations=[],
            limitations=["No retrieved legal context was available"],
        )
        citation_validation = _validate_output_citations(output, {})
        return SummarizationResult(
            output=output,
            citation_validation=citation_validation,
            provider_metadata=self._provider_metadata(provider_called=False),
            used_context_ids=[],
            warnings=[*payload.diagnostics.warnings, "Provider was not called"],
        )

    def _provider_metadata(
        self,
        *,
        provider_called: bool,
    ) -> SummarizationProviderMetadata:
        provider_name = getattr(
            self.provider,
            "provider_name",
            type(self.provider).__name__,
        )
        model_name = getattr(self.provider, "model_name", None)
        return SummarizationProviderMetadata(
            provider=str(provider_name),
            model=str(model_name) if model_name is not None else None,
            provider_called=provider_called,
        )


def _validate_output_citations(
    output: SummarizationOutput,
    citation_map: Mapping[str, Any],
) -> SummarizationCitationValidation:
    allowed = set(citation_map)
    inline = validate_context_citations(output.answer, allowed)
    declared_unique = list(dict.fromkeys(output.citations))
    invalid_declared = [item for item in declared_unique if item not in allowed]
    inline_unique = list(dict.fromkeys(inline.cited_context_ids))
    declared_missing_inline = [
        item for item in declared_unique if item not in inline_unique
    ]
    inline_missing_declared = [
        item for item in inline_unique if item not in declared_unique
    ]
    all_valid = (
        inline.all_citations_valid
        and not invalid_declared
        and not declared_missing_inline
        and not inline_missing_declared
    )
    return SummarizationCitationValidation(
        inline=inline,
        declared_citations=output.citations,
        invalid_declared_citations=invalid_declared,
        declared_missing_inline=declared_missing_inline,
        inline_missing_declared=inline_missing_declared,
        all_citations_valid=all_valid,
    )


def _mentions_uncertainty(limitations: list[str]) -> bool:
    markers = ("weak", "uncertain", "limited", "insufficient", "relevance")
    return any(
        marker in limitation.casefold()
        for limitation in limitations
        for marker in markers
    )
