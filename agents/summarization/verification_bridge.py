"""Deterministic adapter from grounded summaries to the Verification Agent."""

import re

from pydantic import BaseModel, Field

from agents.coordinator.schemas import ContextCitationMapEntry, SummarizationReadyPayload
from agents.summarization.schemas import SummarizationResult
from agents.verification import VerificationAgent
from backend.schemas import (
    Citation,
    LegalClaim,
    RetrievedEvidence,
    SourceMetadata,
    VerificationRequest,
    VerificationResponse,
)
from retrieval.rag import RAGContextPassage

_INLINE_CONTEXT_CITATION = re.compile(r"[ \t]*\[(CTX-\d{3,})\]")
_SUMMARY_CLAIM_ID = "summary-claim-001"
_INTERNAL_LOCATOR_WARNING = (
    "CTX identifiers are internal verification locators, not original legal "
    "page, paragraph, or official citation locators."
)
_WHOLE_ANSWER_WARNING = (
    "Verification evaluated the complete generated summary as one claim."
)


class SummarizationVerificationBridgeError(RuntimeError):
    """Base class for controlled, non-sensitive bridge failures."""


class SummaryNotVerifiableError(SummarizationVerificationBridgeError):
    """Raised when a summary has no safe claim-and-evidence representation."""


class SummaryProvenanceError(SummarizationVerificationBridgeError):
    """Raised when CTX provenance is missing, ambiguous, or inconsistent."""


class VerificationExecutionError(SummarizationVerificationBridgeError):
    """Raised when the injected Verification Agent cannot complete."""


class SummarizationVerificationBridgeResult(BaseModel):
    """Auditable request, unchanged verifier response, and original provenance."""

    verification_request: VerificationRequest
    verification_response: VerificationResponse
    provenance: dict[str, ContextCitationMapEntry]
    warnings: list[str] = Field(default_factory=list)


class SummarizationVerificationBridge:
    """Convert a validated grounded summary into one verification request."""

    def __init__(self, verification_agent: VerificationAgent) -> None:
        self.verification_agent = verification_agent

    def verify(
        self,
        task_id: str,
        payload: SummarizationReadyPayload,
        result: SummarizationResult,
    ) -> SummarizationVerificationBridgeResult:
        context_ids = _validated_context_ids(result)
        claim_text = _claim_text(result.output.answer)
        passages = _passages_by_context_id(payload)

        citations: list[Citation] = []
        evidence: list[RetrievedEvidence] = []
        provenance: dict[str, ContextCitationMapEntry] = {}
        for context_id in context_ids:
            entry = payload.citation_map.get(context_id)
            if entry is None or entry.context_id != context_id:
                raise SummaryProvenanceError(
                    "Summary citation has no consistent citation-map entry"
                )
            matching_passages = passages.get(context_id, [])
            if len(matching_passages) != 1:
                raise SummaryProvenanceError(
                    "Summary citation does not resolve to exactly one context passage"
                )
            passage = matching_passages[0]
            _validate_provenance(entry, passage)

            citations.append(
                Citation(document_id=entry.document_id, locator=context_id)
            )
            evidence.append(
                RetrievedEvidence(
                    id=context_id,
                    source=SourceMetadata(
                        document_id=entry.document_id,
                        title=entry.title,
                        court=entry.court,
                        source_uri=entry.source_url,
                    ),
                    text=passage.text,
                    locator=context_id,
                )
            )
            provenance[context_id] = entry

        if not evidence:
            raise SummaryNotVerifiableError(
                "Summary has no usable cited evidence for verification"
            )
        request = VerificationRequest(
            task_id=task_id,
            claims=[
                LegalClaim(
                    id=_SUMMARY_CLAIM_ID,
                    text=claim_text,
                    citations=citations,
                )
            ],
            evidence=evidence,
        )
        try:
            response = self.verification_agent.verify(request)
        except Exception as exc:
            raise VerificationExecutionError(
                "Verification Agent failed while checking the summary"
            ) from exc

        return SummarizationVerificationBridgeResult(
            verification_request=request,
            verification_response=response,
            provenance=provenance,
            warnings=[_INTERNAL_LOCATOR_WARNING, _WHOLE_ANSWER_WARNING],
        )


def _validated_context_ids(result: SummarizationResult) -> list[str]:
    validation = result.citation_validation
    inline_ids = validation.inline.valid_citations
    actual_ids = list(
        dict.fromkeys(_INLINE_CONTEXT_CITATION.findall(result.output.answer))
    )
    if actual_ids != inline_ids:
        raise SummaryNotVerifiableError(
            "Summary inline citations do not match validated citations"
        )
    declared_ids = list(dict.fromkeys(result.output.citations))
    if (
        not validation.all_citations_valid
        or not validation.inline.all_citations_valid
        or validation.invalid_declared_citations
        or validation.declared_missing_inline
        or validation.inline_missing_declared
        or validation.inline.invalid_citations
        or validation.inline.malformed_citations
        or result.used_context_ids != inline_ids
        or set(declared_ids) != set(inline_ids)
    ):
        raise SummaryNotVerifiableError(
            "Summary citation validation is inconsistent"
        )
    if not inline_ids:
        raise SummaryNotVerifiableError(
            "Summary has no validated context citations"
        )
    return inline_ids


def _claim_text(answer: str) -> str:
    if not isinstance(answer, str) or not answer.strip():
        raise SummaryNotVerifiableError("Summary answer is empty")
    claim_text = _INLINE_CONTEXT_CITATION.sub("", answer).strip()
    if not claim_text:
        raise SummaryNotVerifiableError(
            "Summary contains no substantive text after citation markers"
        )
    return claim_text


def _passages_by_context_id(
    payload: SummarizationReadyPayload,
) -> dict[str, list[RAGContextPassage]]:
    passages: dict[str, list[RAGContextPassage]] = {}
    for passage in payload.context_passages:
        passages.setdefault(passage.context_id, []).append(passage)
    return passages


def _validate_provenance(
    entry: ContextCitationMapEntry,
    passage: RAGContextPassage,
) -> None:
    citation = passage.citation
    consistent = (
        entry.document_id == passage.document_id == citation.document_id
        and entry.title == passage.title
        and entry.court == passage.court
        and entry.source == passage.source == citation.source
        and entry.chunk_ids == passage.chunk_ids == citation.chunk_ids
        and entry.pages == passage.pages == citation.pages
        and entry.official_citation == citation.citation
        and entry.source_url == citation.source_url
    )
    if not consistent:
        raise SummaryProvenanceError(
            "Citation-map and context-passage provenance do not agree"
        )
