"""Deterministic assembly and Markdown rendering of draft legal reports."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.summarization.schemas import (
    CaseInformation,
    Comparison,
    DraftClaim,
    EvidenceSummary,
    PrecedentSummary,
    SourceReference,
    SummarizationRequest,
    SummarizationResponse,
)
from backend.schemas import Citation


REPORT_DISCLAIMER = (
    "This draft legal research report is assembled from supplied materials and "
    "AI-generated summaries. It is not legal advice and has not been verified."
)


class ReportAssemblyError(Exception):
    """Base class for expected report assembly failures."""


class InvalidReportInputError(ReportAssemblyError):
    """Raised when report inputs do not match the validated contracts."""


class InconsistentCaseInformationError(ReportAssemblyError):
    """Raised when explicit case information differs from the source request."""


class UnsupportedReportSourceError(ReportAssemblyError):
    """Raised when report content refers to a source absent from the request."""


class ReportCitation(BaseModel):
    """A report citation with a required, preserved source locator."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    excerpt: str | None = None


class LegalResearchReport(BaseModel):
    """Structured, unverified legal research report suitable for later review."""

    model_config = ConfigDict(extra="forbid")

    case: CaseInformation
    evidence_summaries: list[EvidenceSummary] = Field(default_factory=list)
    precedent_summaries: list[PrecedentSummary] = Field(default_factory=list)
    comparisons: list[Comparison] = Field(default_factory=list)
    draft_findings: list[DraftClaim] = Field(default_factory=list)
    citations: list[ReportCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str = REPORT_DISCLAIMER
    verification_status: Literal["pending_verification"] = "pending_verification"


class LegalReportAssembler:
    """Assemble reports without generation, persistence, or external calls."""

    def assemble(
        self,
        case: CaseInformation,
        source_request: SummarizationRequest,
        summary: SummarizationResponse,
    ) -> LegalResearchReport:
        """Validate source consistency and copy summary sections into a report."""

        try:
            case = CaseInformation.model_validate(case)
            source_request = SummarizationRequest.model_validate(source_request)
            summary = SummarizationResponse.model_validate(summary)
        except ValidationError as exc:
            raise InvalidReportInputError(
                "Report inputs do not match the required structured contracts."
            ) from exc

        if case != source_request.case:
            raise InconsistentCaseInformationError(
                "Case information does not match the summarization source request."
            )

        self._validate_sources(source_request, summary)
        limitations = self._limitations(case, summary)

        return LegalResearchReport(
            case=case.model_copy(deep=True),
            evidence_summaries=[
                item.model_copy(
                    update={
                        "source_references": self._unique_references(
                            item.source_references
                        )
                    },
                    deep=True,
                )
                for item in summary.evidence_summaries
            ],
            precedent_summaries=[
                item.model_copy(
                    update={
                        "source_references": self._unique_references(
                            item.source_references
                        )
                    },
                    deep=True,
                )
                for item in summary.precedent_summaries
            ],
            comparisons=[
                item.model_copy(
                    update={
                        "source_references": self._unique_references(
                            item.source_references
                        )
                    },
                    deep=True,
                )
                for item in summary.comparisons
            ],
            draft_findings=[
                item.model_copy(
                    update={
                        "supporting_sources": self._unique_references(
                            item.supporting_sources
                        )
                    },
                    deep=True,
                )
                for item in summary.draft_claims
            ],
            citations=self._unique_citations(summary.citations),
            warnings=self._unique_text(summary.warnings),
            limitations=limitations,
        )

    @staticmethod
    def _validate_sources(
        source_request: SummarizationRequest,
        summary: SummarizationResponse,
    ) -> None:
        source_text: dict[tuple[str, str], list[str]] = {}
        for fact in source_request.extracted_facts:
            for reference in fact.source_references:
                source_text.setdefault(
                    (reference.document_id, reference.locator), []
                ).append(fact.text)
        for passage in source_request.retrieved_passages:
            source_text.setdefault(
                (passage.source.document_id, passage.source.locator), []
            ).append(passage.text)

        references = [
            reference
            for item in (
                *summary.evidence_summaries,
                *summary.precedent_summaries,
                *summary.comparisons,
            )
            for reference in item.source_references
        ]
        references.extend(
            reference
            for finding in summary.draft_claims
            for reference in finding.supporting_sources
        )

        if any(
            (reference.document_id, reference.locator) not in source_text
            for reference in references
        ):
            raise UnsupportedReportSourceError(
                "Report content contains a source reference absent from its input."
            )

        for citation in summary.citations:
            if citation.locator is None:
                raise UnsupportedReportSourceError(
                    "Report citations require a supported source locator."
                )
            texts = source_text.get((citation.document_id, citation.locator))
            if texts is None or (
                citation.excerpt is not None
                and not any(citation.excerpt in text for text in texts)
            ):
                raise UnsupportedReportSourceError(
                    "Report contains an unsupported citation or excerpt."
                )

    @staticmethod
    def _limitations(
        case: CaseInformation, summary: SummarizationResponse
    ) -> list[str]:
        limitations: list[str] = []
        if case.description is None or not case.description.strip():
            limitations.append("No case description was supplied.")
        if not summary.evidence_summaries:
            limitations.append("No evidence summaries were supplied.")
        if not summary.precedent_summaries:
            limitations.append("No legal precedent summaries were supplied.")
        if not summary.comparisons:
            limitations.append("No case-to-precedent comparisons were supplied.")
        if not summary.draft_claims:
            limitations.append("No draft findings were supplied.")
        if not summary.citations:
            limitations.append("No citations were supplied.")
        return limitations

    @staticmethod
    def _unique_references(
        references: list[SourceReference],
    ) -> list[SourceReference]:
        seen: set[tuple[str, str]] = set()
        unique: list[SourceReference] = []
        for reference in references:
            key = (reference.document_id, reference.locator)
            if key not in seen:
                seen.add(key)
                unique.append(reference.model_copy(deep=True))
        return unique

    @staticmethod
    def _unique_citations(citations: list[Citation]) -> list[ReportCitation]:
        seen: set[tuple[str, str | None, str | None]] = set()
        unique: list[ReportCitation] = []
        for citation in citations:
            key = (citation.document_id, citation.locator, citation.excerpt)
            if key not in seen:
                seen.add(key)
                unique.append(
                    ReportCitation(
                        document_id=citation.document_id,
                        locator=citation.locator or "",
                        excerpt=citation.excerpt,
                    )
                )
        return unique

    @staticmethod
    def _unique_text(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value.strip()))


class MarkdownLegalReportRenderer:
    """Render a report while neutralizing Markdown from untrusted fields."""

    _markdown_character = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|>])")

    def render(self, report: LegalResearchReport) -> str:
        """Return deterministic Markdown without interpreting supplied text."""

        try:
            report = LegalResearchReport.model_validate(report)
        except ValidationError as exc:
            raise InvalidReportInputError(
                "Report does not match the required structured contract."
            ) from exc
        lines = [
            "# Draft Legal Research Report",
            "",
            f"**Verification status:** {report.verification_status}",
            "",
            "## Case Information",
            "",
            f"- **Case ID:** {self._plain(report.case.case_id)}",
            f"- **Title:** {self._plain(report.case.title)}",
            f"- **Description:** {self._plain(report.case.description) if report.case.description else 'Not provided.'}",
        ]
        self._append_grounded_section(
            lines, "Evidence Summaries", report.evidence_summaries, "text"
        )
        self._append_grounded_section(
            lines, "Relevant Legal Precedents", report.precedent_summaries, "text"
        )

        lines.extend(["", "## Case-to-Precedent Comparisons", ""])
        if report.comparisons:
            for item in report.comparisons:
                lines.append(f"### {self._plain(item.subject)}")
                lines.append(self._plain(item.analysis))
                lines.append(self._references(item.source_references))
        else:
            lines.append("Not provided.")

        lines.extend(["", "## Draft Findings", ""])
        if report.draft_findings:
            for item in report.draft_findings:
                lines.append(f"- **DRAFT:** {self._plain(item.text)}")
                lines.append(f"  {self._references(item.supporting_sources)}")
        else:
            lines.append("Not provided.")

        lines.extend(["", "## Citations", ""])
        if report.citations:
            for citation in report.citations:
                location = self._source(citation.document_id, citation.locator or "")
                excerpt = (
                    f" — {self._plain(citation.excerpt)}"
                    if citation.excerpt is not None
                    else ""
                )
                lines.append(f"- {location}{excerpt}")
        else:
            lines.append("Not provided.")

        self._append_text_section(lines, "Warnings", report.warnings)
        self._append_text_section(lines, "Limitations", report.limitations)
        lines.extend(
            ["", "## Disclaimer", "", self._plain(report.disclaimer), ""]
        )
        return "\n".join(lines)

    def _append_grounded_section(
        self,
        lines: list[str],
        title: str,
        items: list[EvidenceSummary] | list[PrecedentSummary],
        text_field: str,
    ) -> None:
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append("Not provided.")
            return
        for item in items:
            lines.append(f"- {self._plain(getattr(item, text_field))}")
            lines.append(f"  {self._references(item.source_references)}")

    def _append_text_section(
        self, lines: list[str], title: str, values: list[str]
    ) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.extend(
            [f"- {self._plain(value)}" for value in values] or ["None recorded."]
        )

    def _references(self, references: list[SourceReference]) -> str:
        return "Sources: " + "; ".join(
            self._source(reference.document_id, reference.locator)
            for reference in references
        )

    def _source(self, document_id: str, locator: str) -> str:
        return f"{self._plain(document_id)} — {self._plain(locator)}"

    def _plain(self, value: str | None) -> str:
        normalized = " ".join((value or "").split())
        return self._markdown_character.sub(r"\\\1", normalized)
