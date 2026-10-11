"""Opt-in orchestration for unverified AI legal-comparison drafts."""

from collections.abc import Sequence
import re

from agents.analysis.schemas import AnalysisResponse, ExtractedFact
from agents.coordinator.schemas import SummarizationReadyPayload
from backend.schemas.legal_comparison import (
    ComparisonEvidenceMetadata,
    LegalComparisonDraft,
    LegalComparisonGenerationRequest,
    LegalComparisonRecord,
)
from backend.services.legal_comparison import authority_provenance_agrees
from backend.services.legal_comparison_provider import LegalComparisonProvider
from retrieval.rag import RAGContextPassage

_CTX_TOKEN = re.compile(r"\bCTX-\d+\b")
_BRACKETED_CITATION_TOKEN = re.compile(r"\[[A-Z][A-Z0-9_-]*-\d+\]")


class LegalComparisonGenerationError(RuntimeError):
    """Raised when generation inputs or provider references are inconsistent."""

    def __init__(self, message: str, *, category: str = "validation_error") -> None:
        super().__init__(message)
        self.category = category


class LegalComparisonDraftService:
    """Generate drafts only for caller-supplied, previously validated links."""

    def __init__(self, provider: LegalComparisonProvider) -> None:
        self.provider = provider

    def generate(
        self,
        comparisons: Sequence[LegalComparisonRecord],
        analyses: Sequence[AnalysisResponse],
        payload: SummarizationReadyPayload,
    ) -> list[LegalComparisonRecord]:
        records = list(comparisons)
        if not records:
            return []
        _reject_duplicate_ids(records)
        generated: list[LegalComparisonRecord] = []
        for record in records:
            fact = _resolve_fact(record, analyses)
            passage = _resolve_passage(record, payload)
            request = _request(record, fact, passage)
            try:
                output = self.provider.generate(request)
            except LegalComparisonGenerationError:
                raise
            except Exception as exc:
                category = getattr(exc, "category", "provider_transport_error")
                raise LegalComparisonGenerationError(
                    f"Comparison provider execution failed: {category}",
                    category=category,
                ) from None
            _validate_output(record, output.model_dump())
            generated.append(
                record.model_copy(
                    deep=True,
                    update={
                        "ai_draft": LegalComparisonDraft(
                            **output.model_dump(),
                            verification_status="not_verified",
                            provider=self.provider.provider_name,
                            model=self.provider.model_name,
                        )
                    },
                )
            )
        return generated


def _resolve_fact(
    record: LegalComparisonRecord,
    analyses: Sequence[AnalysisResponse],
) -> ExtractedFact:
    matches = [
        item
        for item in analyses
        if item.document_id == record.fact.analysis_document_id
    ]
    if len(matches) != 1:
        raise LegalComparisonGenerationError(
            "Comparison Analysis document is missing or ambiguous"
        )
    analysis = matches[0]
    index = record.fact.fact_index
    if index >= len(analysis.facts):
        raise LegalComparisonGenerationError("Comparison fact was not found")
    fact = analysis.facts[index]
    expected = record.fact.model_dump()
    actual = {
        "analysis_document_id": analysis.document_id,
        "fact_index": index,
        "statement": fact.statement,
        "source": fact.source.model_dump(),
        "epistemic_status": fact.epistemic_status,
        "confidence": fact.confidence,
        "verification_status": fact.verification_status,
    }
    if actual != expected:
        raise LegalComparisonGenerationError(
            "Comparison fact no longer matches the original Analysis input"
        )
    return fact


def _resolve_passage(
    record: LegalComparisonRecord,
    payload: SummarizationReadyPayload,
) -> RAGContextPassage:
    context_id = record.authority.context_id
    matches = [
        passage
        for passage in payload.context_passages
        if passage.context_id == context_id
    ]
    entry = payload.citation_map.get(context_id)
    if entry is None or len(matches) != 1:
        raise LegalComparisonGenerationError(
            "Comparison authority is missing or ambiguous"
        )
    passage = matches[0]
    expected = record.authority.model_dump()
    actual = {
        "context_id": context_id,
        "document_id": entry.document_id,
        "title": entry.title,
        "court": entry.court,
        "source": entry.source,
        "official_citation": entry.official_citation,
        "source_url": entry.source_url,
        "chunk_ids": list(entry.chunk_ids),
        "pages": list(entry.pages),
        "provenance": passage.citation.provenance,
    }
    if (
        entry.context_id != context_id
        or actual != expected
        or not authority_provenance_agrees(entry, passage)
    ):
        raise LegalComparisonGenerationError(
            "Comparison authority provenance is inconsistent"
        )
    return passage


def _request(
    record: LegalComparisonRecord,
    fact: ExtractedFact,
    passage: RAGContextPassage,
) -> LegalComparisonGenerationRequest:
    source = fact.source
    return LegalComparisonGenerationRequest(
        comparison_id=record.id,
        context_id=record.authority.context_id,
        fact_statement=fact.statement,
        fact_epistemic_status=fact.epistemic_status,
        fact_confidence=fact.confidence,
        fact_verification_status=fact.verification_status,
        evidence=ComparisonEvidenceMetadata(
            document_id=source.document_id,
            page=source.page,
            paragraph=source.paragraph,
            locator=source.locator,
            start_char=source.start_char,
            end_char=source.end_char,
        ),
        authority_passage=passage.text,
        authority=record.authority.model_copy(deep=True),
        limitations=list(record.limitations),
    )


def _validate_output(record: LegalComparisonRecord, output: dict) -> None:
    if output["comparison_id"] != record.id:
        raise LegalComparisonGenerationError(
            "Provider returned an unexpected comparison identifier",
            category="provider_reference_mismatch",
        )
    if output["context_id"] != record.authority.context_id:
        raise LegalComparisonGenerationError(
            "Provider returned an unexpected CTX identifier",
            category="provider_reference_mismatch",
        )
    allowed = {record.authority.context_id}
    prose = "\n".join([output["explanation"], *output["uncertainties"]])
    if set(_CTX_TOKEN.findall(prose)) - allowed:
        raise LegalComparisonGenerationError(
            "Provider output contains an unexpected CTX identifier",
            category="provider_reference_mismatch",
        )
    allowed_token = f"[{record.authority.context_id}]"
    if set(_BRACKETED_CITATION_TOKEN.findall(prose)) - {allowed_token}:
        raise LegalComparisonGenerationError(
            "Provider output contains a fabricated citation token",
            category="provider_reference_mismatch",
        )


def _reject_duplicate_ids(records: Sequence[LegalComparisonRecord]) -> None:
    identities = [record.id for record in records]
    if len(identities) != len(set(identities)):
        raise LegalComparisonGenerationError(
            "Duplicate comparison identifiers are not allowed"
        )
