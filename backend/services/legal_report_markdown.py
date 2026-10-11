"""Safe deterministic Markdown rendering for structured legal research reports."""

from urllib.parse import quote, urlsplit

from agents.analysis.schemas import EvidenceSource
from backend.schemas.legal_report import LegalResearchReport


class LegalResearchReportMarkdownRenderer:
    """Render an assembled report without executing agents or reading external data."""

    def render(self, report: LegalResearchReport) -> str:
        lines: list[str] = ["# Legal Research Report", ""]
        _render_case_information(lines, report)
        _render_analysis(lines, report)
        _render_authorities(lines, report)
        _render_comparisons(lines, report)
        _render_summary(lines, report)
        _render_verification(lines, report)
        _render_limitations(lines, report)
        return "\n".join(lines).rstrip() + "\n"


def _render_case_information(lines: list[str], report: LegalResearchReport) -> None:
    information = report.case_information
    case = information.case
    lines.extend(
        [
            "## Case Information",
            "",
            _bullet("Case ID", case.id),
            _bullet("Case title", case.title),
        ]
    )
    if case.description is not None:
        lines.append(_bullet("Case description", case.description))
    lines.append(_bullet("Research query", information.query))
    if information.legal_issue is not None:
        lines.append(_bullet("Legal issue", information.legal_issue))
    lines.append("")


def _render_analysis(lines: list[str], report: LegalResearchReport) -> None:
    analysis = report.analysis
    lines.extend(
        [
            "## Analysis Findings",
            "",
            _bullet("Availability", analysis.availability),
        ]
    )
    if not analysis.supplied:
        lines.extend(["", "Analysis results were not supplied.", ""])
        return
    if not analysis.results:
        lines.extend(["", "Analysis input contained no Analysis results.", ""])
        return

    for number, result in enumerate(analysis.results, start=1):
        lines.extend(
            [
                "",
                f"### Analysis Document {number}",
                "",
                _bullet("Document ID", result.document_id),
                _bullet("Status", result.status),
            ]
        )
        _render_facts(lines, result.facts)
        _render_clauses(lines, result.clauses)
        _render_issues(lines, result.legal_issues)
        _render_timeline(lines, result.timeline)
        _render_contradictions(lines, result.contradiction_candidates)
        _render_stage_status(lines, result.stage_status)
        _render_messages(lines, "Analysis Warnings", result.warnings)
        _render_messages(lines, "Analysis Errors", result.errors)
        if not any(
            (
                result.facts,
                result.clauses,
                result.legal_issues,
                result.timeline,
                result.contradiction_candidates,
            )
        ):
            lines.extend(["", "No analysis findings were recorded."])
    lines.append("")


def _render_facts(lines: list[str], facts: list) -> None:
    if not facts:
        return
    lines.extend(["", "#### Extracted Facts", ""])
    for number, fact in enumerate(facts, start=1):
        lines.append(f"**Fact {number}**")
        lines.extend(_blockquote(fact.statement))
        lines.append(_bullet("Epistemic status", fact.epistemic_status))
        lines.append(_bullet("Verification status", fact.verification_status))
        _append_optional(lines, "Confidence", fact.confidence)
        _append_optional(lines, "Category", fact.category)
        lines.extend(_evidence_lines(fact.source))
        lines.append("")


def _render_clauses(lines: list[str], clauses: list) -> None:
    if not clauses:
        return
    lines.extend(["", "#### Extracted Clauses", ""])
    for number, clause in enumerate(clauses, start=1):
        lines.append(f"**Clause {number}**")
        lines.extend(_blockquote(clause.text))
        _append_optional(lines, "Clause type", clause.clause_type)
        lines.append(_bullet("Epistemic status", clause.epistemic_status))
        _append_optional(lines, "Confidence", clause.confidence)
        lines.extend(_evidence_lines(clause.source))
        lines.append("")


def _render_issues(lines: list[str], issues: list) -> None:
    if not issues:
        return
    lines.extend(["", "#### Legal Issue Candidates", ""])
    for number, issue in enumerate(issues, start=1):
        lines.append(f"**Issue {number}:** {_inline(issue.label)}")
        _append_optional(lines, "Description", issue.description)
        lines.append(_bullet("Review status", issue.review_status))
        lines.append(_bullet("Epistemic status", issue.epistemic_status))
        _append_optional(lines, "Confidence", issue.confidence)
        for source in issue.sources:
            lines.extend(_evidence_lines(source))
        lines.append("")


def _render_timeline(lines: list[str], events: list) -> None:
    if not events:
        return
    lines.extend(["", "#### Timeline Events", ""])
    for number, event in enumerate(events, start=1):
        lines.append(f"**Event {number}**")
        lines.extend(_blockquote(event.description))
        _append_optional(lines, "Date text", event.date_text)
        _append_optional(lines, "Normalized date", event.normalized_date)
        lines.append(_bullet("Epistemic status", event.epistemic_status))
        lines.append(_bullet("Verification status", event.verification_status))
        _append_optional(lines, "Confidence", event.confidence)
        lines.extend(_evidence_lines(event.source))
        lines.append("")


def _render_contradictions(lines: list[str], candidates: list) -> None:
    if not candidates:
        return
    lines.extend(["", "#### Contradiction Candidates", ""])
    for number, candidate in enumerate(candidates, start=1):
        lines.append(f"**Candidate {number}**")
        lines.extend(_blockquote(candidate.description))
        lines.append(_bullet("Review status", candidate.review_status))
        _append_optional(lines, "Confidence", candidate.confidence)
        for source in candidate.sources:
            lines.extend(_evidence_lines(source))
        lines.append("")


def _render_stage_status(lines: list[str], stages: dict[str, str]) -> None:
    if not stages:
        return
    lines.extend(["", "#### Analysis Stage Status", ""])
    for name, status in stages.items():
        lines.append(_bullet(name, status))


def _render_messages(lines: list[str], heading: str, messages: list) -> None:
    if not messages:
        return
    lines.extend(["", f"#### {heading}", ""])
    for message in messages:
        text = f"{message.code}: {message.message}"
        if message.field is not None:
            text += f" (field: {message.field})"
        lines.append(f"- {_inline(text)}")


def _render_authorities(lines: list[str], report: LegalResearchReport) -> None:
    lines.extend(["## Retrieved Legal Authorities", ""])
    if not report.authorities:
        lines.extend(["No summary-cited legal authorities were recorded.", ""])
        return
    for number, authority in enumerate(report.authorities, start=1):
        lines.extend(
            [
                f"### Authority {number}: {_inline(authority.context_id)}",
                "",
                _bullet("Document ID", authority.document_id),
            ]
        )
        _append_optional(lines, "Title", authority.title)
        _append_optional(lines, "Court", authority.court)
        _append_optional(lines, "Source", authority.source)
        _append_optional(lines, "Official citation", authority.official_citation)
        _append_optional(lines, "Pages", _join(authority.pages))
        _append_optional(lines, "Chunk IDs", _join(authority.chunk_ids))
        if authority.source_url is not None:
            lines.append(_url_line(authority.source_url))
        lines.append("")


def _render_comparisons(lines: list[str], report: LegalResearchReport) -> None:
    lines.extend(
        [
            "## Explicit Fact-to-Authority Links",
            "",
            _bullet("Comparison status", report.legal_comparison_status),
        ]
    )
    if not report.comparisons:
        lines.extend(
            [
                "",
                "Substantive legal comparison was not assessed.",
                "",
            ]
        )
        return
    lines.extend(
        [
            "",
            "Only explicit reference associations were prepared; legal applicability "
            "was not assessed.",
            "",
        ]
    )
    for record in report.comparisons:
        lines.extend(
            [
                f"### Link {_inline(record.id)}",
                "",
                _bullet("Review status", record.review_status),
                _bullet("Origin", record.origin),
                "**Selected fact**",
            ]
        )
        lines.extend(_blockquote(record.fact.statement))
        lines.append(_bullet("Epistemic status", record.fact.epistemic_status))
        lines.append(_bullet("Fact verification status", record.fact.verification_status))
        _append_optional(lines, "Confidence", record.fact.confidence)
        lines.extend(_evidence_lines(record.fact.source))
        lines.append(_bullet("Selected authority CTX", record.authority.context_id))
        lines.append(_bullet("Authority document ID", record.authority.document_id))
        _append_optional(lines, "Authority title", record.authority.title)
        _append_optional(lines, "Authority court", record.authority.court)
        _append_optional(lines, "Authority source", record.authority.source)
        _append_optional(lines, "Official citation", record.authority.official_citation)
        _append_optional(lines, "Authority pages", _join(record.authority.pages))
        _append_optional(lines, "Authority chunk IDs", _join(record.authority.chunk_ids))
        if record.authority.source_url is not None:
            lines.append(_url_line(record.authority.source_url, label="Authority source URL"))
        lines.append("**Recorded relationship**")
        lines.extend(_blockquote(record.relationship))
        _render_string_list(lines, "Link limitations", record.limitations)
        if record.ai_draft is not None:
            draft = record.ai_draft
            lines.extend(
                [
                    "",
                    "**AI-generated comparative explanation**",
                    "",
                    "**Not independently verified — Human review required**",
                    "",
                    _bullet("Verification status", draft.verification_status),
                    _bullet("Comparison ID", draft.comparison_id),
                    _bullet("Selected authority CTX", draft.context_id),
                    _bullet("Provider", draft.provider),
                ]
            )
            _append_optional(lines, "Model", draft.model)
            lines.extend(["", "**Draft explanation**"])
            lines.extend(_blockquote(draft.explanation))
            _render_string_list(
                lines,
                "Draft uncertainties and limitations",
                draft.uncertainties,
            )
        lines.append("")


def _render_summary(lines: list[str], report: LegalResearchReport) -> None:
    summary = report.summary
    lines.extend(["## Gemini-Generated Legal Summary", "", "**Original answer**"])
    lines.extend(_blockquote(summary.output.answer))
    lines.append(_bullet("Used CTX citations", _join(summary.used_context_ids) or "None"))
    _append_optional(lines, "Confidence", summary.output.confidence)
    lines.append(_bullet("Provider", summary.provider_metadata.provider))
    _append_optional(lines, "Model", summary.provider_metadata.model)
    _render_string_list(lines, "Summary limitations", summary.output.limitations)
    _render_string_list(lines, "Summary warnings", summary.warnings)
    lines.append("")


def _render_verification(lines: list[str], report: LegalResearchReport) -> None:
    findings = report.verification
    response = findings.response
    lines.extend(
        [
            "## Verification Findings",
            "",
            _bullet("Overall status", response.overall_status),
            _bullet("Summary verified", str(response.verified).lower()),
        ]
    )
    for number, result in enumerate(response.claim_results, start=1):
        lines.extend(["", f"### Verification Claim {number}", ""])
        lines.append(_bullet("Claim ID", result.claim.id))
        lines.extend(_blockquote(result.claim.text))
        lines.append(_bullet("Status", result.status))
        lines.append(_bullet("Evidence IDs", _join(result.evidence_ids) or "None"))
        for citation in result.citation_results:
            identity = "missing"
            if citation.citation is not None:
                identity = citation.citation.document_id
                if citation.citation.locator is not None:
                    identity += f" / {citation.citation.locator}"
            if citation.evidence_id is not None:
                identity += f" -> evidence {citation.evidence_id}"
            lines.append(
                f"- Citation {_inline(identity)}: {_inline(citation.status)}"
            )
        _render_string_list(lines, "Claim issues", result.issues)
    _render_string_list(lines, "Verification notes", response.notes)
    if response.warnings:
        lines.extend(["", "### Verification Warnings", ""])
        for warning in response.warnings:
            warning_line = (
                f"- {_inline(warning.severity)} / {_inline(warning.code)}: "
                f"{_inline(warning.message)}"
            )
            if warning.claim_ids:
                warning_line += f" (claims: {_inline(_join(warning.claim_ids))})"
            lines.append(warning_line)
    _render_string_list(lines, "Bridge warnings", findings.bridge_warnings)
    lines.extend(["", "**Verification scope**"])
    lines.extend(_blockquote(findings.scope))
    lines.append("")


def _render_limitations(lines: list[str], report: LegalResearchReport) -> None:
    lines.extend(["## Report Limitations", ""])
    if report.limitations:
        lines.extend(f"- {_inline(item)}" for item in report.limitations)
    else:
        lines.append("- No additional report limitations were recorded.")
    lines.extend(
        [
            "",
            "Source identity validation is not substantive legal verification. "
            "Extracted facts and explicit comparison links require human review.",
            "",
            "## Human Review Disclaimer",
            "",
        ]
    )
    lines.extend(_blockquote(report.human_review_disclaimer))


def _evidence_lines(source: EvidenceSource) -> list[str]:
    lines = [_bullet("Evidence document ID", source.document_id)]
    _append_optional(lines, "Evidence page", source.page)
    _append_optional(lines, "Evidence paragraph", source.paragraph)
    _append_optional(lines, "Evidence locator", source.locator)
    return lines


def _render_string_list(lines: list[str], heading: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend(["", f"### {heading}", ""])
    lines.extend(f"- {_inline(value)}" for value in values)


def _append_optional(lines: list[str], label: str, value: object | None) -> None:
    if value is not None and value != "":
        lines.append(_bullet(label, value))


def _bullet(label: str, value: object) -> str:
    return f"- **{_inline(label)}:** {_inline(str(value))}"


def _blockquote(value: str) -> list[str]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return [f"> {_escape(line)}" if line else ">" for line in normalized.split("\n")]


def _inline(value: object) -> str:
    return _escape(" ".join(str(value).splitlines()))


def _escape(value: str) -> str:
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in "\\`*_{}[]()#+-!|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _join(values: list) -> str:
    return ", ".join(str(value) for value in values)


def _url_line(value: str, *, label: str = "Source URL") -> str:
    target = _safe_http_url(value)
    if target is None:
        return f"- **{_inline(label)}:** Unsafe or unsupported URL omitted."
    return f"- **{_inline(label)}:** [Open source]({target})"


def _safe_http_url(value: str) -> str | None:
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        return None
    if any(character in value for character in '<>"\\'):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        parsed.port
    except ValueError:
        return None
    return quote(value, safe=":/?#[]@!$&'*,;=%+._~-")
