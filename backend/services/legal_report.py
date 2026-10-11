"""Deterministic assembly of existing agent outputs into a legal report."""

from collections.abc import Sequence

from agents.analysis.schemas import AnalysisResponse
from agents.coordinator.schemas import (
    ContextCitationMapEntry,
    SummarizationReadyPayload,
)
from backend.schemas.common import Case
from backend.schemas.legal_report import (
    AnalysisAvailability,
    LegalResearchReport,
    ReportAnalysisFindings,
    ReportCaseInformation,
    ReportLegalAuthority,
    ReportVerificationFindings,
)
from backend.schemas.summarization import SummarizationVerifyResponse
from retrieval.rag import RAGContextPassage

_VERIFICATION_SCOPE = (
    "The Verification Agent evaluated the complete generated summary as one claim; "
    "it did not independently verify every report statement."
)
_DISCLAIMER = (
    "CaseLens produces AI-assisted legal research drafts. Extracted facts and "
    "retrieved authorities require human review. Citation-ID validity does not "
    "establish substantive legal correctness. This report is not a substitute for "
    "professional legal advice or a court determination."
)


class LegalResearchReportValidationError(RuntimeError):
    """Raised when required report identities or provenance are inconsistent."""


class LegalResearchReportAssembler:
    """Assemble validated inputs without invoking agents, providers, or retrieval."""

    def assemble(
        self,
        case: Case,
        payload: SummarizationReadyPayload,
        summary_verification: SummarizationVerifyResponse,
        analyses: Sequence[AnalysisResponse] | None = None,
    ) -> LegalResearchReport:
        analysis_results = list(analyses) if analyses is not None else []
        self._validate_analysis_case_ids(case, analysis_results)
        authorities = self._authorities(payload, summary_verification)
        self._validate_verification_citations(authorities, summary_verification)

        return LegalResearchReport(
            case_information=ReportCaseInformation(
                case=case,
                query=payload.query,
                legal_issue=payload.legal_issue,
            ),
            analysis=ReportAnalysisFindings(
                supplied=analyses is not None,
                availability=_analysis_availability(analyses, analysis_results),
                results=analysis_results,
            ),
            authorities=authorities,
            summary=summary_verification.summary,
            verification=ReportVerificationFindings(
                response=summary_verification.verification,
                bridge_warnings=list(summary_verification.bridge_warnings),
                scope=_VERIFICATION_SCOPE,
            ),
            retrieval_diagnostics=payload.diagnostics,
            limitations=_limitations(
                analyses,
                analysis_results,
                payload,
                summary_verification,
                authorities,
            ),
            human_review_disclaimer=_DISCLAIMER,
        )

    @staticmethod
    def _validate_analysis_case_ids(
        case: Case,
        analyses: Sequence[AnalysisResponse],
    ) -> None:
        if any(item.case_id != case.id for item in analyses):
            raise LegalResearchReportValidationError(
                "Analysis result does not belong to the requested case"
            )

    @staticmethod
    def _authorities(
        payload: SummarizationReadyPayload,
        response: SummarizationVerifyResponse,
    ) -> list[ReportLegalAuthority]:
        used_ids = list(dict.fromkeys(response.summary.used_context_ids))
        declared_ids = list(dict.fromkeys(response.summary.output.citations))
        validated_ids = list(
            dict.fromkeys(
                response.summary.citation_validation.inline.valid_citations
            )
        )
        if (
            not response.summary.citation_validation.all_citations_valid
            or set(declared_ids) != set(used_ids)
            or validated_ids != used_ids
        ):
            raise LegalResearchReportValidationError(
                "Summary context references are inconsistent"
            )
        if set(response.provenance) != set(used_ids):
            raise LegalResearchReportValidationError(
                "Response provenance does not match the summary context references"
            )
        passages: dict[str, list[RAGContextPassage]] = {}
        for passage in payload.context_passages:
            passages.setdefault(passage.context_id, []).append(passage)

        authorities: list[ReportLegalAuthority] = []
        for context_id in used_ids:
            entry = payload.citation_map.get(context_id)
            response_entry = response.provenance.get(context_id)
            matches = passages.get(context_id, [])
            if entry is None or response_entry is None or len(matches) != 1:
                raise LegalResearchReportValidationError(
                    "Summary context provenance is missing or ambiguous"
                )
            if entry.context_id != context_id or response_entry != entry:
                raise LegalResearchReportValidationError(
                    "Summary context provenance does not match the original payload"
                )
            passage = matches[0]
            if not _provenance_agrees(entry, passage):
                raise LegalResearchReportValidationError(
                    "Authority identity does not match the original payload"
                )
            authorities.append(
                ReportLegalAuthority(
                    context_id=context_id,
                    document_id=entry.document_id,
                    title=entry.title,
                    court=entry.court,
                    source=entry.source,
                    source_url=entry.source_url,
                    official_citation=entry.official_citation,
                    provenance=passage.citation.provenance,
                    chunk_ids=list(entry.chunk_ids),
                    pages=list(entry.pages),
                    ranks=list(passage.ranks),
                    retrieval_score=passage.score,
                    retrieval_scores=list(passage.retrieval_scores),
                    truncated=passage.truncated,
                )
            )
        return authorities

    @staticmethod
    def _validate_verification_citations(
        authorities: Sequence[ReportLegalAuthority],
        response: SummarizationVerifyResponse,
    ) -> None:
        by_context = {item.context_id: item.document_id for item in authorities}
        citations = [
            *response.verification.citations,
            *(
                citation
                for result in response.verification.claim_results
                for citation in result.claim.citations
            ),
        ]
        for citation in citations:
            if by_context.get(citation.locator) != citation.document_id:
                raise LegalResearchReportValidationError(
                    "Verification citation does not match report provenance"
                )


def _provenance_agrees(
    entry: ContextCitationMapEntry,
    passage: RAGContextPassage,
) -> bool:
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


def _analysis_availability(
    supplied: Sequence[AnalysisResponse] | None,
    analyses: Sequence[AnalysisResponse],
) -> AnalysisAvailability:
    if supplied is None:
        return "not_supplied"
    if not analyses:
        return "unavailable"
    statuses = {item.status for item in analyses}
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"requires_input"}:
        return "requires_input"
    if statuses == {"partial"}:
        return "partial"
    if statuses <= {"failed", "not_analyzed"}:
        return "unavailable"
    return "mixed"


def _limitations(
    supplied: Sequence[AnalysisResponse] | None,
    analyses: Sequence[AnalysisResponse],
    payload: SummarizationReadyPayload,
    response: SummarizationVerifyResponse,
    authorities: Sequence[ReportLegalAuthority],
) -> list[str]:
    limitations = [
        "Substantive applicability of retrieved legal authorities was not assessed.",
        "Generated summary verification was performed as one claim, not as atomic claims.",
    ]
    if supplied is None:
        limitations.append("Analysis results were not supplied.")
    elif not analyses:
        limitations.append("Analysis input was supplied without any Analysis results.")
    if any(item.status in {"partial", "failed", "requires_input", "not_analyzed"} for item in analyses):
        limitations.append("One or more Analysis results are partial or unavailable.")
    if any(item.errors for item in analyses):
        limitations.append("One or more Analysis stages reported errors.")
    if any(
        item.title is None
        or item.court is None
        or item.source is None
        or item.official_citation is None
        or item.source_url is None
        or not item.pages
        for item in authorities
    ):
        limitations.append("One or more cited authorities have incomplete optional metadata.")
    if payload.diagnostics.no_context or payload.diagnostics.empty_retrieval:
        limitations.append("No usable retrieved legal context was available.")
    if payload.diagnostics.context_truncated:
        limitations.append("Retrieved legal context was truncated.")
    if payload.diagnostics.weak_context_ids:
        limitations.append("Retrieved legal context includes weakly ranked passages.")
    if response.verification.overall_status != "supported":
        limitations.append(
            f"Summary verification status is {response.verification.overall_status}."
        )
    if any(
        "conflicting_evidence" in result.issue_codes
        for result in response.verification.claim_results
    ):
        limitations.append("Conflicting evidence requires human review.")
    limitations.extend(
        f"Retrieval warning: {warning}" for warning in payload.diagnostics.warnings
    )
    return list(dict.fromkeys(limitations))
