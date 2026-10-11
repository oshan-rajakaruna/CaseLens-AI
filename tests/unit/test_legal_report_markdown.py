"""Offline tests for safe deterministic legal-report Markdown rendering."""

from copy import deepcopy

import pytest

from agents.analysis import AnalysisAgent
from agents.analysis.schemas import AnalysisWarning
from agents.summarization import GeminiSummarizationProvider
from agents.verification import VerificationAgent
from backend.schemas import Case
from backend.schemas.legal_comparison import (
    FactAuthoritySelection,
    LegalComparisonDraft,
)
from backend.services.legal_report import LegalResearchReportAssembler
from backend.services.legal_report_markdown import (
    LegalResearchReportMarkdownRenderer,
)
from tests.unit.summarization.test_verification_bridge import _passage, _payload
from tests.unit.test_legal_report import _analysis, _assemble, _response


def _complete_report(
    *,
    comparison: bool = False,
    verification_status: str = "supported",
):
    payload = _payload(
        _passage("CTX-001", "DOC-A", page=3),
        _passage("CTX-002", "DOC-B", "Comparison-only authority.", page=4),
    )
    selections = (
        [
            FactAuthoritySelection(
                analysis_document_id="PRIVATE-DOC-1",
                fact_index=0,
                context_id="CTX-002",
            )
        ]
        if comparison
        else None
    )
    return LegalResearchReportAssembler().assemble(
        Case(id="case-1", title="Synthetic case", description="Research draft"),
        payload,
        _response(payload, status=verification_status),
        analyses=[_analysis()],
        selections=selections,
    )


def test_complete_report_renders_all_required_sections() -> None:
    markdown = LegalResearchReportMarkdownRenderer().render(
        _complete_report(comparison=True)
    )

    expected_headings = [
        "# Legal Research Report",
        "## Case Information",
        "## Analysis Findings",
        "## Retrieved Legal Authorities",
        "## Explicit Fact-to-Authority Links",
        "## Gemini-Generated Legal Summary",
        "## Verification Findings",
        "## Report Limitations",
        "## Human Review Disclaimer",
    ]
    for heading in expected_headings:
        assert heading in markdown
    assert "The tenant did not pay." in markdown
    assert "Evidence page:** 2" in markdown
    assert "Evidence page:** 3" in markdown
    assert r"CIT\-DOC\-A" in markdown
    assert "professional legal advice" in markdown
    assert r"Citation\-ID validity" in markdown


def test_missing_and_empty_analysis_are_distinguished() -> None:
    missing = LegalResearchReportMarkdownRenderer().render(_assemble())
    empty = LegalResearchReportMarkdownRenderer().render(_assemble(analyses=[]))

    assert "Analysis results were not supplied." in missing
    assert "Analysis input contained no Analysis results." in empty


@pytest.mark.parametrize(
    ("status", "rendered_status"),
    [
        ("requires_input", r"requires\_input"),
        ("partial", "partial"),
        ("failed", "failed"),
    ],
)
def test_incomplete_analysis_status_is_rendered_exactly(
    status: str,
    rendered_status: str,
) -> None:
    report = _assemble(analyses=[_analysis(status, findings=False)])
    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert f"Status:** {rendered_status}" in markdown
    assert "No analysis findings were recorded." in markdown


def test_completed_empty_analysis_is_not_rendered_as_a_finding() -> None:
    markdown = LegalResearchReportMarkdownRenderer().render(
        _assemble(analyses=[_analysis(findings=False)])
    )

    assert "Status:** completed" in markdown
    assert "No analysis findings were recorded." in markdown


def test_analysis_warnings_errors_stages_and_evidence_are_rendered_without_excerpt() -> None:
    analysis = _analysis()
    analysis.facts[0].source.excerpt = "PRIVATE FULL EXCERPT"
    analysis.stage_status = {"facts": "completed", "clauses": "failed"}
    analysis.warnings = [AnalysisWarning(code="review", message="Review facts")]
    analysis.errors = [AnalysisWarning(code="clauses_failed", message="Stage failed")]

    markdown = LegalResearchReportMarkdownRenderer().render(
        _assemble(analyses=[analysis])
    )

    assert "facts:** completed" in markdown
    assert "clauses:** failed" in markdown
    assert "review: Review facts" in markdown
    assert r"clauses\_failed: Stage failed" in markdown
    assert r"Evidence document ID:** PRIVATE\-DOC\-1" in markdown
    assert "PRIVATE FULL EXCERPT" not in markdown


def test_authority_order_ctx_pages_chunks_and_safe_url_are_preserved() -> None:
    report = _complete_report()
    second = report.authorities[0].model_copy(
        deep=True,
        update={
            "context_id": "CTX-009",
            "document_id": "DOC-Z",
            "pages": [4, 8],
            "chunk_ids": ["chunk-a", "chunk-b"],
            "source_url": "https://example.test/source?q=legal",
        },
    )
    report.authorities.append(second)

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert markdown.index(r"CTX\-001") < markdown.index(r"CTX\-009")
    assert "Pages:** 4, 8" in markdown
    assert r"Chunk IDs:** chunk\-a, chunk\-b" in markdown
    assert "[Open source](https://example.test/source?q=legal)" in markdown
    assert "Retrieved legal evidence" not in markdown


def test_missing_authority_metadata_is_omitted_without_invention() -> None:
    report = _complete_report()
    authority = report.authorities[0]
    authority.title = None
    authority.court = None
    authority.official_citation = None
    authority.source_url = None

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert "Title:**" not in markdown
    assert "Court:**" not in markdown
    assert "Official citation:**" not in markdown
    assert "not available" not in markdown


def test_summary_multiline_answer_and_used_citations_are_preserved() -> None:
    report = _complete_report()
    report.summary.output.answer = "First line [CTX-001].\nSecond line."

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert "> First line \\[CTX\\-001\\].\n> Second line." in markdown
    assert r"Used CTX citations:** CTX\-001" in markdown
    assert "Limited to retrieved material" in markdown
    assert "Summary warning" in markdown


@pytest.mark.parametrize("status", ["unsupported", "uncertain"])
def test_non_supported_verification_status_is_not_promoted(status: str) -> None:
    markdown = LegalResearchReportMarkdownRenderer().render(_assemble(status=status))

    assert f"Overall status:** {status}" in markdown
    assert "Summary verified:** false" in markdown
    assert "Status:** supported" not in markdown


def test_bridge_warnings_scope_and_report_limitations_are_rendered() -> None:
    markdown = LegalResearchReportMarkdownRenderer().render(_complete_report())

    assert "Whole summary was verified as one claim." in markdown
    assert "complete generated summary as one claim" in markdown
    assert "Substantive applicability" in markdown


def test_explicit_link_is_human_review_only_and_comparison_ctx_stays_separate() -> None:
    report = _complete_report(comparison=True)
    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert r"Link CMP\-001" in markdown
    assert r"Review status:** requires\_human\_review" in markdown
    assert "legal applicability was not assessed" in markdown
    assert r"Selected authority CTX:** CTX\-002" in markdown
    assert r"Used CTX citations:** CTX\-001" in markdown
    assert r"Used CTX citations:** CTX\-001, CTX\-002" not in markdown
    assert "verified comparison" not in markdown.casefold()


def test_ai_comparison_draft_is_rendered_as_unverified_and_human_reviewed() -> None:
    report = _complete_report(comparison=True)
    report.comparisons[0].ai_draft = LegalComparisonDraft(
        comparison_id="CMP-001",
        context_id="CTX-002",
        explanation="The statements may relate, but applicability is uncertain.",
        uncertainties=["The allegation has not been independently verified."],
        ai_generated=True,
        human_review_required=True,
        provider="fake",
        model="offline-model",
    )

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert "AI-generated comparative explanation" in markdown
    assert "Not independently verified — Human review required" in markdown
    assert r"Verification status:** not\_verified" in markdown
    assert "The statements may relate" in markdown
    assert r"Selected authority CTX:** CTX\-002" in markdown


def test_ai_comparison_draft_escapes_generated_markdown_and_html() -> None:
    report = _complete_report(comparison=True)
    report.comparisons[0].ai_draft = LegalComparisonDraft(
        comparison_id="CMP-001",
        context_id="CTX-002",
        explanation="## Forged heading\n<script>alert(1)</script>",
        uncertainties=["[unsafe](javascript:alert(1))"],
        ai_generated=True,
        human_review_required=True,
        provider="fake",
    )

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert r"> \#\# Forged heading" in markdown
    assert "<script>" not in markdown
    assert r"&lt;script&gt;alert\(1\)&lt;/script&gt;" in markdown
    assert r"\[unsafe\]\(javascript:alert\(1\)\)" in markdown


def test_absent_comparisons_render_not_assessed() -> None:
    markdown = LegalResearchReportMarkdownRenderer().render(_complete_report())

    assert r"Comparison status:** not\_assessed" in markdown
    assert "Substantive legal comparison was not assessed." in markdown


def test_hostile_markdown_and_html_are_escaped_without_losing_text() -> None:
    report = _complete_report()
    report.case_information.case.title = "# Forged heading | [link](x) <script>alert(1)</script>"
    report.summary.output.answer = "## Fake section\n`code` | [click](javascript:alert(1))"

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert "\n# Forged heading" not in markdown
    assert r"&lt;script&gt;alert\(1\)&lt;/script&gt;" in markdown
    assert r"\[link\]\(x\)" in markdown
    assert r"> \#\# Fake section" in markdown
    assert r"\`code\` \| \[click\]\(javascript:alert\(1\)\)" in markdown
    assert "<script>" not in markdown


def test_unsafe_url_is_not_rendered_as_an_active_link() -> None:
    report = _complete_report()
    report.authorities[0].source_url = "javascript:alert(1)"

    markdown = LegalResearchReportMarkdownRenderer().render(report)

    assert "Unsafe or unsupported URL omitted." in markdown
    assert "javascript:" not in markdown


def test_rendering_is_deterministic_and_does_not_mutate_report() -> None:
    report = _complete_report(comparison=True)
    before = deepcopy(report.model_dump())
    renderer = LegalResearchReportMarkdownRenderer()

    first = renderer.render(report)
    second = renderer.render(report)

    assert first == second
    assert report.model_dump() == before
    assert first.endswith("\n")


def test_renderer_does_not_execute_agents_provider_or_filesystem(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("external execution is forbidden")

    monkeypatch.setattr(AnalysisAgent, "analyze", forbidden)
    monkeypatch.setattr(VerificationAgent, "verify", forbidden)
    monkeypatch.setattr(GeminiSummarizationProvider, "generate", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)

    markdown = LegalResearchReportMarkdownRenderer().render(_complete_report())

    assert markdown.startswith("# Legal Research Report\n")
