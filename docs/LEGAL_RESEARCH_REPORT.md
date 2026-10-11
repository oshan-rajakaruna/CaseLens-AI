# Structured Legal Research Report

`LegalResearchReportAssembler` deterministically combines an existing `Case`,
`SummarizationReadyPayload`, and `SummarizationVerifyResponse`. A sequence of
existing `AnalysisResponse` objects is optional. The assembler does not run
Gemini, retrieval, analysis, verification, network operations, or persistence.

The report contains supplied case/query information, unchanged Analysis
results, metadata for CTX authorities actually used by the summary, the
existing generated summary, the exact Verification Agent response, retrieval
diagnostics, limitations, and a mandatory human-review disclaimer. Authority
entries retain document, chunk, page, score, source, citation, URL, and
provenance metadata but omit full passage text by default.

Missing Analysis is represented as `not_supplied`; empty supplied results,
partial processing, stage errors, weak context, missing optional authority
metadata, and non-supported verification outcomes remain visible. The report
does not infer relationships between case facts and legal authorities. CTX
validity is not treated as substantive legal correctness, and summary
verification is explicitly scoped to the bridge's single whole-summary claim.

```python
from backend.services.legal_report import LegalResearchReportAssembler

report = LegalResearchReportAssembler().assemble(
    case,
    summarization_ready_payload,
    summarization_verify_response,
    analyses=analysis_responses,
)
```

Generated legal comparisons, PDF/HTML export, API endpoints, downloads,
database persistence, and AgentLogs are future milestones.

## Explicit fact-to-authority links

Milestone 4B can attach caller-selected reference links without generating a
legal conclusion. A selection identifies one Analysis document, a zero-based
fact index within that exact `AnalysisResponse`, and one CTX passage:

```python
from backend.schemas.legal_comparison import FactAuthoritySelection

report = LegalResearchReportAssembler().assemble(
    case,
    summarization_ready_payload,
    summarization_verify_response,
    analyses=analysis_responses,
    selections=[
        FactAuthoritySelection(
            analysis_document_id="case-evidence-1",
            fact_index=0,
            context_id="CTX-001",
        )
    ],
)
```

Fact indices are input-local and remain meaningful only for the exact supplied
Analysis result version. The assembler validates the fact source, CTX passage,
citation map, document identity, chunks, pages, citation, URL, and provenance.
It stores no full authority passage text in the comparison record.

`links_prepared_for_review` means only that explicit source links were resolved
successfully. It does not mean the authority applies, the fact is true, or the
comparison was substantively verified. Every link uses fixed neutral wording
and `requires_human_review`. Gemini-generated comparative explanations remain
postponed until their generation and verification contracts are agreed.

## Safe Markdown rendering

`LegalResearchReportMarkdownRenderer` converts an already assembled report to
one deterministic Markdown string. It performs no generation, retrieval,
analysis, verification, legal reasoning, comparison verification, file access,
or network operation:

```python
from backend.services.legal_report_markdown import (
    LegalResearchReportMarkdownRenderer,
)

markdown = LegalResearchReportMarkdownRenderer().render(report)
```

The output includes case information, Analysis availability and findings,
summary-cited authorities, explicit comparison links, the original generated
summary, exact Verification Agent statuses, report limitations, and the
mandatory human-review disclaimer. Missing Analysis, empty findings, absent
comparisons, and unavailable optional authority metadata are stated or omitted
without inventing replacement values.

Report-derived text is treated as untrusted: Markdown control characters and
raw HTML are escaped, multiline narrative is placed in escaped block quotes,
and only validated HTTP or HTTPS source URLs become clickable links. Unsafe
schemes are omitted. Full evidence excerpts and authority passages are not
rendered by default.

The returned Markdown can contain sensitive case facts. Rendering does not
authorize publication, establish legal correctness, prove applicability, or
turn summary verification into report-wide verification. The renderer returns
text only; HTML/PDF export, file saving, download APIs, and persistence do not
exist in this milestone.
