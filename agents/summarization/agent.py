"""Summarization Agent interface with an injectable generation dependency."""

from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from agents.summarization.exceptions import (
    InvalidSummarizationInputError,
    MissingEvidenceError,
    ModelUnavailableError,
    UnsupportedSourceReferenceError,
)
from agents.summarization.schemas import (
    SourceReference,
    SummarizationRequest,
    SummarizationResponse,
)
from backend.schemas import AgentResult, AgentTask


@runtime_checkable
class SummaryGenerator(Protocol):
    """Port implemented later by an LLM adapter or now by a test double."""

    async def generate(self, request: SummarizationRequest) -> SummarizationResponse:
        """Generate a structured, evidence-grounded response."""
        ...


class SummarizationAgent:
    """Validate inputs and generated source references around a generator."""

    name = "summarization"

    def __init__(self, generator: SummaryGenerator | None = None) -> None:
        self._generator = generator

    async def summarize(self, request: SummarizationRequest) -> SummarizationResponse:
        """Generate a summary without inventing fallback content."""

        if not request.extracted_facts and not request.retrieved_passages:
            raise MissingEvidenceError(
                "At least one extracted fact or retrieved legal passage is required."
            )
        if self._generator is None:
            raise ModelUnavailableError(
                "No summary generator is configured; summarization was not attempted."
            )

        response = SummarizationResponse.model_validate(
            await self._generator.generate(request)
        )
        self._validate_source_references(request, response)
        return response

    async def handle(self, task: AgentTask) -> AgentResult:
        """Adapt a shared Coordinator task to the summarization contract."""

        if task.agent != self.name:
            raise InvalidSummarizationInputError(
                f"Task agent must be '{self.name}', received '{task.agent}'."
            )
        try:
            request = SummarizationRequest.model_validate(task.context)
        except ValidationError as exc:
            raise InvalidSummarizationInputError(
                "Task context is not a valid SummarizationRequest."
            ) from exc
        if request.instruction is None:
            request = request.model_copy(update={"instruction": task.instruction})

        response = await self.summarize(request)
        return AgentResult(
            task_id=task.id,
            agent=self.name,
            status="completed",
            output=response.model_dump(mode="json"),
        )

    @staticmethod
    def _validate_source_references(
        request: SummarizationRequest, response: SummarizationResponse
    ) -> None:
        source_text: dict[tuple[str, str], list[str]] = {}
        for fact in request.extracted_facts:
            for reference in fact.source_references:
                source_text.setdefault(
                    (reference.document_id, reference.locator), []
                ).append(fact.text)
        for passage in request.retrieved_passages:
            source_text.setdefault(
                (passage.source.document_id, passage.source.locator), []
            ).append(passage.text)
        allowed = set(source_text)

        output_references: list[SourceReference] = []
        for item in (
            *response.evidence_summaries,
            *response.precedent_summaries,
            *response.comparisons,
        ):
            output_references.extend(item.source_references)
        for claim in response.draft_claims:
            output_references.extend(claim.supporting_sources)

        unsupported = {
            (reference.document_id, reference.locator)
            for reference in output_references
            if (reference.document_id, reference.locator) not in allowed
        }
        unsupported.update(
            (citation.document_id, citation.locator)
            for citation in response.citations
            if citation.locator is None
            or (citation.document_id, citation.locator) not in allowed
            or (
                citation.excerpt is not None
                and not any(
                    citation.excerpt in text
                    for text in source_text.get(
                        (citation.document_id, citation.locator), []
                    )
                )
            )
        )
        if unsupported:
            formatted = ", ".join(
                f"{document_id}:{locator or '<missing locator>'}"
                for document_id, locator in sorted(
                    unsupported, key=lambda item: (item[0], item[1] or "")
                )
            )
            raise UnsupportedSourceReferenceError(
                f"Generated output contains unsupported source references: {formatted}"
            )
