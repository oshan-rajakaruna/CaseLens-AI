"""Deterministic assembly of explicitly selected fact-to-authority links."""

from collections.abc import Sequence

from agents.analysis.schemas import AnalysisResponse, ExtractedFact
from agents.coordinator.schemas import (
    ContextCitationMapEntry,
    SummarizationReadyPayload,
)
from backend.schemas.legal_comparison import (
    ComparisonAuthorityReference,
    ComparisonFactReference,
    FactAuthoritySelection,
    LegalComparisonRecord,
)
from retrieval.rag import RAGContextPassage

_RELATIONSHIP = (
    "This fact and authority were explicitly selected for comparative human review; "
    "legal applicability was not assessed."
)
_BASE_LIMITATIONS = [
    "Source identities were validated, but legal applicability was not assessed.",
    "This association was not substantively verified by the Verification Agent.",
]


class LegalComparisonValidationError(RuntimeError):
    """Raised when a selected comparison cannot be resolved safely."""


class LegalComparisonAssembler:
    """Resolve caller selections without inference, agent calls, or generation."""

    def build(
        self,
        analyses: Sequence[AnalysisResponse],
        payload: SummarizationReadyPayload,
        selections: Sequence[FactAuthoritySelection],
    ) -> list[LegalComparisonRecord]:
        selected = list(selections)
        if not selected:
            return []
        analysis_results = list(analyses)
        if not analysis_results:
            raise LegalComparisonValidationError(
                "Analysis results are required for selected comparisons"
            )
        _reject_duplicate_selections(selected)
        analyses_by_document = _analyses_by_document(analysis_results)
        passages_by_context = _passages_by_context(payload.context_passages)

        records: list[LegalComparisonRecord] = []
        for position, selection in enumerate(selected, start=1):
            analysis = _resolve_analysis(
                analyses_by_document,
                selection.analysis_document_id,
            )
            fact = _resolve_fact(analysis, selection.fact_index)
            passage, entry = _resolve_authority(
                payload,
                passages_by_context,
                selection.context_id,
            )
            limitations = list(_BASE_LIMITATIONS)
            if fact.epistemic_status != "stated":
                limitations.append(
                    "The selected fact retains a non-stated epistemic status and "
                    "must not be treated as established."
                )
            if fact.verification_status != "verified":
                limitations.append(
                    "The selected fact is not recorded as independently verified."
                )
            if fact.confidence is not None:
                limitations.append(
                    "Fact extraction confidence does not establish factual truth."
                )
            records.append(
                LegalComparisonRecord(
                    id=f"CMP-{position:03d}",
                    fact=ComparisonFactReference(
                        analysis_document_id=analysis.document_id,
                        fact_index=selection.fact_index,
                        statement=fact.statement,
                        source=fact.source,
                        epistemic_status=fact.epistemic_status,
                        confidence=fact.confidence,
                        verification_status=fact.verification_status,
                    ),
                    authority=ComparisonAuthorityReference(
                        context_id=selection.context_id,
                        document_id=entry.document_id,
                        title=entry.title,
                        court=entry.court,
                        source=entry.source,
                        official_citation=entry.official_citation,
                        source_url=entry.source_url,
                        chunk_ids=list(entry.chunk_ids),
                        pages=list(entry.pages),
                        provenance=passage.citation.provenance,
                    ),
                    relationship=_RELATIONSHIP,
                    limitations=limitations,
                )
            )
        return records


def authority_provenance_agrees(
    entry: ContextCitationMapEntry,
    passage: RAGContextPassage,
) -> bool:
    """Check one citation-map entry against its original RAG passage."""

    citation = passage.citation
    return (
        entry.document_id == passage.document_id == citation.document_id
        and entry.title == passage.title
        and entry.court == passage.court
        and entry.source == passage.source == citation.source
        and entry.chunk_ids == passage.chunk_ids == citation.chunk_ids
        and entry.pages == passage.pages == citation.pages
        and entry.official_citation == citation.citation
        and entry.source_url == citation.source_url
    )


def _reject_duplicate_selections(
    selections: Sequence[FactAuthoritySelection],
) -> None:
    identities = [
        (item.analysis_document_id, item.fact_index, item.context_id)
        for item in selections
    ]
    if len(set(identities)) != len(identities):
        raise LegalComparisonValidationError(
            "Duplicate fact-to-authority selections are not allowed"
        )


def _analyses_by_document(
    analyses: Sequence[AnalysisResponse],
) -> dict[str, list[AnalysisResponse]]:
    grouped: dict[str, list[AnalysisResponse]] = {}
    for analysis in analyses:
        grouped.setdefault(analysis.document_id, []).append(analysis)
    return grouped


def _resolve_analysis(
    analyses: dict[str, list[AnalysisResponse]],
    document_id: str,
) -> AnalysisResponse:
    matches = analyses.get(document_id, [])
    if not matches:
        raise LegalComparisonValidationError(
            "Selected Analysis document was not found"
        )
    if len(matches) != 1:
        raise LegalComparisonValidationError(
            "Selected Analysis document is ambiguous"
        )
    return matches[0]


def _resolve_fact(analysis: AnalysisResponse, fact_index: int) -> ExtractedFact:
    if fact_index < 0 or fact_index >= len(analysis.facts):
        raise LegalComparisonValidationError(
            "Selected fact index is outside the Analysis result"
        )
    fact = analysis.facts[fact_index]
    if fact.source.document_id != analysis.document_id:
        raise LegalComparisonValidationError(
            "Selected fact source does not match its Analysis document"
        )
    return fact


def _passages_by_context(
    passages: Sequence[RAGContextPassage],
) -> dict[str, list[RAGContextPassage]]:
    grouped: dict[str, list[RAGContextPassage]] = {}
    for passage in passages:
        grouped.setdefault(passage.context_id, []).append(passage)
    return grouped


def _resolve_authority(
    payload: SummarizationReadyPayload,
    passages: dict[str, list[RAGContextPassage]],
    context_id: str,
) -> tuple[RAGContextPassage, ContextCitationMapEntry]:
    matches = passages.get(context_id, [])
    entry = payload.citation_map.get(context_id)
    if entry is None or not matches:
        raise LegalComparisonValidationError(
            "Selected authority context was not found"
        )
    if len(matches) != 1:
        raise LegalComparisonValidationError(
            "Selected authority context is ambiguous"
        )
    passage = matches[0]
    if entry.context_id != context_id or not authority_provenance_agrees(entry, passage):
        raise LegalComparisonValidationError(
            "Selected authority provenance is inconsistent"
        )
    return passage, entry
