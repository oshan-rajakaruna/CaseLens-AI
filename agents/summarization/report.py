"""Deterministic assembly of source-grounded legal research reports."""

from __future__ import annotations

from agents.summarization.schemas import (
    AnalysisFinding, CaseComparison, EvidenceSummaryPoint, GroundingWarning,
    LegalResearchReport, LegalResearchReportRequest, PrecedentSummary,
    ReportPoint, ReportSection, SourceReference,
)

DISCLAIMER = (
    "This report provides AI-assisted legal research support based on the supplied "
    "materials. It is not legal advice. Source accuracy, legal applicability, and "
    "professional conclusions require qualified human legal review."
)


class LegalResearchReportGenerator:
    """Assemble supplied findings without research, inference, or generation."""

    def generate(self, request: LegalResearchReportRequest) -> LegalResearchReport:
        sections: list[ReportSection] = []
        limitations: list[str] = []
        warnings = list(request.warnings)

        evidence_points = request.evidence_summary.points if request.evidence_summary else []
        sections.append(_section("Executive Summary", evidence_points, "No safely selected evidence summary was supplied.", "executive_summary"))
        sections.append(_section("Case Background and Evidence Overview", evidence_points, "No source-grounded current-case background was supplied.", "case_background"))
        sections.append(_section("Material Facts and Disputed Statements", evidence_points + _findings(request.analysis_findings), "No material facts or disputed statements were supplied.", "material_fact"))
        issue_points = [point for finding in request.analysis_findings if finding.finding_type.lower() in {"legal_issue", "issue"} for point in _findings([finding], prefix="Potential legal issue — requires review: ")]
        sections.append(_section("Potential Legal Issues", issue_points, "No explicitly supplied potential legal issue was available.", "legal_issue"))
        sections.append(_section("Contractual Terms and Conditions", request.clauses, "No source-grounded contractual clauses were supplied.", "contractual_clause"))
        sections.append(_section("Chronological Timeline", request.timeline_events, "No source-grounded timeline was supplied.", "timeline"))
        precedent_points = _precedent_points(request.precedent_summaries)
        sections.append(_section("Relevant Precedent Research", precedent_points, "No precedent summary was supplied.", "precedent"))
        comparison_points, comparison_limits = _comparison_points(request.case_comparisons)
        limitations.extend(comparison_limits)
        sections.append(_section("Case-to-Precedent Comparison", comparison_points, "No source-grounded case-to-precedent comparison was supplied.", "comparison"))
        sections.append(_section("Evidence Inconsistencies and Review Items", request.inconsistency_candidates, "No explicit inconsistency candidate was supplied.", "inconsistency"))
        sections.append(_section("Research Limitations and Verification Requirements", [], _limitations_text(request, limitations), "limitations"))

        sources = _unique_sources(source for section in sections for point in section.points for source in point.source_references)
        citations = list(dict.fromkeys(source.citation_id for source in sources if source.citation_id))
        sections.append(ReportSection(title="Sources and Citations", content="Supplied source reference inventory.", citations=citations))
        sections.append(ReportSection(title="Human Legal Review Disclaimer", content=DISCLAIMER))
        has_material = bool(evidence_points or request.analysis_findings or request.clauses or request.timeline_events or request.precedent_summaries or request.case_comparisons)
        status = "completed" if has_material and not _has_missing_sections(sections) else ("partial" if has_material else "requires_input")
        return LegalResearchReport(
            report_id=f"REPORT-{request.case_id}", case_id=request.case_id,
            research_question=request.research_question,
            title=request.report_title or f"Legal Research Report — {request.case_id}",
            status=status, sections=sections, source_references=sources, citations=citations,
            limitations=list(dict.fromkeys(limitations + ["All report content requires qualified human legal review."])),
            warnings=warnings, human_review_required=True,
        )


def render_report_markdown(report: LegalResearchReport) -> str:
    """Render report data safely as plain Markdown; source text is not interpreted."""
    lines = [f"# {_escape(report.title)}", "", f"**Case ID:** {_escape(report.case_id)}", f"**Research question:** {_escape(report.research_question)}", ""]
    for section in report.sections:
        lines.extend([f"## {_escape(section.title)}", _escape(section.content)])
        for point in section.points:
            cited = " ".join(f"[{_escape(c)}]" for c in point.citations)
            lines.append(f"- {_escape(point.statement)} {cited}".rstrip())
        lines.append("")
    return "\n".join(lines)


def _section(title: str, raw_points: list[object], missing: str, category: str) -> ReportSection:
    points = [_point(item, category) for item in raw_points]
    return ReportSection(title=title, content="Source-grounded supplied statements." if points else missing, points=points, citations=list(dict.fromkeys(c for point in points for c in point.citations)), warnings=[] if points else [GroundingWarning(code="SECTION_NOT_SUPPLIED", description=missing)])


def _point(item: object, category: str) -> ReportPoint:
    if isinstance(item, ReportPoint): return item
    if isinstance(item, EvidenceSummaryPoint):
        return ReportPoint(statement=item.text, category=category, source_references=item.source_references, citations=item.citations, is_negated=item.is_negated, is_conditional=item.is_conditional, is_attributed=item.is_attributed, epistemic_status=item.epistemic_status, verification_status=item.verification_status, provenance_limited=item.provenance_limited)
    if isinstance(item, AnalysisFinding):
        return ReportPoint(statement=item.statement, category=category, source_references=item.sources, citations=[s.citation_id for s in item.sources if s.citation_id], is_negated=item.is_negated, is_conditional=item.is_conditional, is_attributed=item.is_attributed, epistemic_status=item.epistemic_status, verification_status=item.verification_status, provenance_limited=not item.sources)
    return ReportPoint(statement=str(item), category=category, provenance_limited=True)


def _findings(findings: list[AnalysisFinding], prefix: str = "") -> list[ReportPoint]:
    return [_point(AnalysisFinding(**{**finding.model_dump(), "statement": prefix + finding.statement}), "analysis_finding") for finding in findings]


def _precedent_points(precedents: list[PrecedentSummary]) -> list[ReportPoint]:
    result: list[ReportPoint] = []
    for precedent in precedents:
        metadata = [
            *([f"Case identifier: {precedent.case_identifier}"] if precedent.case_identifier.casefold() != "unknown" else []),
            *([f"Court: {precedent.court}"] if precedent.court else []),
            *([f"Jurisdiction: {precedent.jurisdiction}"] if precedent.jurisdiction else []),
            *([f"Decision date: {precedent.judgment_date}"] if precedent.judgment_date else []),
        ]
        if metadata:
            result.append(ReportPoint(statement="; ".join(metadata), category="precedent_metadata", source_references=precedent.source_references, citations=precedent.citations, provenance_limited=not precedent.source_references))
        for point in precedent.points:
            result.append(ReportPoint(statement=point.text, category=f"precedent_{point.category}", source_references=point.source_references, citations=point.citations, is_negated=point.is_negated, is_conditional=point.is_conditional, is_attributed=point.is_attributed, provenance_limited=point.provenance_limited))
    return result


def _comparison_points(comparisons: list[CaseComparison]) -> tuple[list[ReportPoint], list[str]]:
    points: list[ReportPoint] = []; limits: list[str] = []
    for comparison in comparisons:
        limits.extend(comparison.limitations)
        for item in comparison.points:
            sources = [*item.current_case_sources, *item.precedent_sources]
            points.append(ReportPoint(statement=item.label, category=f"comparison_{item.comparison_type}", source_references=sources, citations=[s.citation_id for s in sources if s.citation_id], provenance_limited=item.provenance_limited))
    return points, limits


def _limitations_text(request: LegalResearchReportRequest, limits: list[str]) -> str:
    missing = [section for section, value in (("evidence summary", request.evidence_summary), ("precedent research", request.precedent_summaries), ("timeline", request.timeline_events)) if not value]
    return "; ".join([*(f"No {item} was supplied" for item in missing), *limits, "All claims remain not independently verified."])


def _has_missing_sections(sections: list[ReportSection]) -> bool:
    return any(any(warning.code == "SECTION_NOT_SUPPLIED" for warning in section.warnings) for section in sections[:-2])


def _unique_sources(sources: object) -> list[SourceReference]:
    result: list[SourceReference] = []
    for source in sources:  # type: ignore[union-attr]
        if source not in result: result.append(source)
    return result


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("`", "\\`").replace("<", "&lt;").replace(">", "&gt;")
