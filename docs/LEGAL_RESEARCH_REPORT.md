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

Generated legal comparisons, Markdown/PDF rendering, API endpoints, downloads,
database persistence, and AgentLogs are future milestones.
