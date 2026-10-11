# Summarization Agent integration contract

## Scope

This document describes the standalone, source-grounded contracts currently
implemented in `agents.summarization`. It does not claim that Coordinator,
Analysis, Retrieval, or Verification integration has occurred. All material is
caller supplied; the standalone workflow performs no live retrieval, database
access, export, or network call.

## Public interfaces

```python
SummarizationAgent(provider: SummarizationProvider | None = None)
agent.summarize(request: SummarizationRequest) -> SummarizationResponse
agent.summarize(payload: SummarizationReadyPayload | Mapping[str, Any]) -> SummarizationResult
agent.generate_report(request: LegalResearchReportRequest) -> LegalResearchReport
agent.run_research_workflow(request: SummarizationWorkflowRequest) -> SummarizationWorkflowResponse
```

`SummarizationRequest` requires `case_id`, one or more of `document_id` or
`document_ids`, and `query`. `summary_type` is `evidence_summary` by default;
the other permitted values are `precedent_summary` and `case_comparison`.
`mode` defaults to `deterministic`; `llm_assisted` is supported only for
`evidence_summary` and requires an injected provider. It is not enabled by the
workflow unless the caller explicitly selects it.

`LegalResearchReportRequest` requires `case_id` and `research_question`.
`SummarizationWorkflowRequest` requires `case_id`, `research_question`, and a
current-case `SummarizationRequest` whose `summary_type` is `evidence_summary`.
Every `precedent_requests` item must use `precedent_summary`. The nested current
request `case_id` must equal the workflow `case_id`.

## Workflow request and response

```json
{
  "case_id": "CASE-001",
  "research_question": "What requires review?",
  "current_case_request": {
    "case_id": "CASE-001",
    "document_id": "DOC-001",
    "query": "What requires review?",
    "summary_type": "evidence_summary",
    "mode": "deterministic",
    "evidence_items": []
  },
  "precedent_requests": [],
  "current_case_jurisdiction": "Sri Lanka",
  "render_markdown": false
}
```

`SummarizationWorkflowResponse` contains `evidence_summary`,
`precedent_summaries`, `case_comparisons`, `report`, optional `markdown`,
ordered `stages`, actual `source_references`/`citations`, `warnings`, and
`verification_status`. `WorkflowStageResult.status` is one of `completed`,
`partial`, `requires_input`, `failed`, or `not_run`. Top-level response and
report statuses are `completed`, `partial`, `requires_input`, or `failed`.
`completed` means processing completed for supplied material; it does not mean
legally verified, comprehensive, or correct.

Stage failures are isolated where a safe independent result exists. For
example, a precedent lacking an explicit holding is retained as a partial
precedent summary; no holding is invented. With no current-case evidence,
comparison is `requires_input` and no comparison claim is created.

## Evidence, precedent, sources, and citations

Use `evidence_summary` only for current-case evidence. A judgment requires an
explicit `summary_type="precedent_summary"`; arbitrary legal text is not
silently treated as a judgment. `EvidenceItem`, `AnalysisFinding`, and
`RetrievedPassage` carry `SourceReference` values.

`SourceReference` fields are `document_id`, optional `locator`, `excerpt`,
`page`, `paragraph`, `start_char`, `end_char`, and `citation_id`. Offsets are
preserved, never recalculated. `citation_id`, when present, must match
`CTX-`, `SRC-`, or `CIT-` followed by a permitted identifier. Within one
workflow, the same citation ID cannot identify different document IDs. The
report citation inventory is deduplicated from used source references only.

Citation-ID validation establishes request-local identifier integrity only. It
does not establish that a claim is entailed, true, legally applicable, or
independently verified.

## Semantic and verification fields

Evidence and report points preserve `is_negated`, `is_conditional`,
`is_attributed`, `epistemic_status`, and `verification_status`. Defaults must
not be read as factual verification. An allegation, denial, condition, or
uncertainty must not be converted to an established event. Precedent holdings
are precedent material only, never current-case outcomes. Cross-jurisdiction
comparison uses `authority_status="not_determined"` and requires legal review.

## Analysis mapping (proposed, read-only confirmation required)

The current Analysis schemas expose `EvidenceSource`, `ExtractedFact`,
`ExtractedClause`, `LegalIssue`, `TimelineEvent`, and
`ContradictionCandidate`. A future adapter should map `EvidenceSource` fields
directly into `SourceReference` (`document_id`, `page`, `paragraph`, `locator`,
`start_char`, `end_char`, `excerpt`) and must assign a caller-controlled
request-local citation ID. It must map:

- `ExtractedFact` to `EvidenceItem` or `AnalysisFinding`, preserving statement,
  source, `is_negated`, `is_conditional`, `is_attributed`, epistemic and
  verification status.
- `ExtractedClause` to a `ReportPoint` for `clauses`, preserving text, source,
  epistemic status, and conditionality. Its schema has no attribution or
  negation field; none must be inferred.
- `LegalIssue` to an `AnalysisFinding` only when the adapter explicitly labels
  it as provisional and retains `sources`, `review_status`, and epistemic
  status. No legal diagnosis is added by Summarization.
- `TimelineEvent` to a timeline `ReportPoint`, preserving description/source,
  date text/precision as supported text, negation, conditionality, attribution,
  epistemic and verification status.
- `ContradictionCandidate` to `inconsistency_candidates`, retaining all sources
  and the human-review meaning; it is never resolved by Summarization.

This mapping is documentation only. No Analysis runtime dependency or adapter
is implemented here.

## Retrieval and legacy Coordinator pathway

`SummarizationReadyPayload` contains `query`, optional `legal_issue`,
`context_text`, `context_passages`, `citation_map`, `retrieval`, and
`diagnostics`. `RAGContextPassage.context_id` uses the upstream `CTX-nnn`
scheme; `ContextCitationMapEntry` retains document, chunk, page, court, title,
official citation, and URL metadata. The legacy `summarize(payload)` route can
call an injected provider, validates inline/declared CTX IDs, and returns a
`SummarizationResult`. With no context it returns a controlled insufficient
context result without calling the provider.

The standalone workflow does not consume `SummarizationReadyPayload` or fetch
RAG context. A future adapter must convert selected retrieval passages to
`RetrievedPassage` plus `SourceReference` without changing IDs or claiming
that CTX validation proves semantic support.

## Verification handoff (proposed)

A future Verification Agent needs each `ReportPoint`/generated point text,
its source references/citation IDs, semantic flags, and comparison source pairs.
The existing verification component accepts claims and retrieved evidence and
checks citation identity plus simple claim wording. An adapter is still needed
to map Summarization report points to its `LegalClaim`/`RetrievedEvidence`
contracts. It should assess claim-to-source entailment, attribution and
negation preservation, conflicting evidence, invalid citations, judgment
attribution, and foreign-authority limitations. Report assembly and LLM draft
format checks are not independent verification.

## Limits and safe failure

- Evidence selection: 20,000 source characters, 100 candidates, six selected
  points, 1,800 displayed characters.
- Judgment extraction: 20,000 source characters, 120 candidates, three points
  per category.
- LLM evidence context: six selected points and 10,000 characters; optional.
- Workflow: at most 12 precedent requests and 100 each of clauses, timeline
  events, and inconsistency candidates.

Components omit only complete statements at safety boundaries and return
warnings where implemented. Provider failures, malformed drafts, and rejected
citations use neutral error codes; raw provider exceptions and evidence text
are not surfaced as errors. Markdown rendering escapes selected characters but
is not certified safe for arbitrary HTML renderers.

## Team review checklist

- Confirm citation assignment and cross-document uniqueness upstream.
- Confirm Analysis-field semantic mappings before implementing adapters.
- Approve privacy, retention, provider, timeout, cost, and monitoring policy
  before enabling a live provider.
- Define Verification adapter and claim-entailment policy.
- Review jurisdiction/authority policy with qualified legal reviewers.
