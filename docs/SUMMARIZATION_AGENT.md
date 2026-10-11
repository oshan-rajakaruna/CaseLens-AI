# Summarization Agent

## Purpose and scope

The Summarization Agent produces deterministic, extractive summaries of
supplied legal evidence. It does not retrieve documents, decide facts, predict
legal outcomes, provide legal advice, or call an external model. It preserves
the difference between an allegation, denial, conditional statement, and an
independently verified fact.

## Step 2 architecture

`SummarizationAgent.summarize(SummarizationRequest)` delegates to the
independently testable `EvidenceSummarizer`.

1. Pydantic validates case/document IDs, query, source references, offsets,
   citation formats, and source-to-document consistency.
2. Evidence items, Analysis findings, retrieval passages, and raw text become
   sentence candidates. Whitespace is normalized for comparison only.
3. Transparent deterministic rules identify statements concerning money,
   dates, agreements, obligations, notices, termination, procedural history,
   allegations, denials, uncertainty, and conditions.
4. The service selects up to six relevant statements, suppresses duplicate
   displayed text, and merges only the real sources of identical evidence.
5. It returns `EvidenceSummaryPoint` objects and a concise concatenation of
   those exact selected statements.

The service does not paraphrase. Therefore terms such as `not`, `alleged`,
`denied`, `may`, `if`, and `unless` remain in the output whenever present in
the selected source statement.

## Inputs and outputs

`SummarizationRequest` supports a case ID, one or more document IDs, a query,
raw evidence text, `EvidenceItem` inputs, `AnalysisFinding` inputs,
`RetrievedPassage` inputs, and metadata.

`SourceReference` retains supplied document ID, locator, excerpt, page,
paragraph, offsets, and citation ID. It does not invent missing values.

`EvidenceSummary` includes summary text, text-only key points for simple
clients, and source-grounded `EvidenceSummaryPoint` items. A point contains
the exact selected statement, all actual supporting sources, their supplied
citations, semantic flags, and a limited-provenance marker.

The agent returns:

- `completed` when at least one safe significant extractive point is produced;
- `partial` if evidence exists but no safe significant point is selected;
- `requires_input` if no evidence, finding, passage, or raw text is supplied.

It never manufactures a citation, source locator, page, offset, confidence
score, legal finding, or legal conclusion.

## Source grounding and semantic safeguards

Structured evidence is preferred because it supplies explicit provenance.
Analysis flags for negation, conditionality, attribution, epistemic status,
and verification status are copied to selected points. Raw text can be
selected, but its point is marked `provenance_limited` and no locator or
citation is fabricated.

Selection is extractive, so citation IDs are taken only from selected points'
actual sources. Duplicate statements from different sources merge those source
references rather than attributing either source to unrelated text.

## Safety limits

The component processes at most 20,000 source characters and 100 sentence
candidates, selects at most six points, and keeps displayed summary text under
1,800 characters. Processing stops only before an additional complete
sentence; a truncation warning is emitted. It does not claim to have reviewed
unprocessed evidence.

## Compatibility

The older `SummarizationReadyPayload` / provider pathway is preserved for
Coordinator-facing compatibility. It remains separate from the deterministic
`SummarizationRequest` path and is not used for Step 2 evidence summaries.

## Step 3: deterministic judgment and precedent extraction

Set `summary_type="precedent_summary"` to route an explicitly supplied
judgment to `PrecedentSummarizer`. The default remains `evidence_summary`.
The precedent path accepts source-grounded retrieved passages, source-grounded
evidence items, or raw text only when the task selector explicitly identifies
it as judgment material. It does not retrieve, research, or treat an ordinary
contract document as a judgment.

`JudgmentMetadata` preserves supplied case title/number, court, jurisdiction,
decision date, official citation, and URL. Missing metadata remains `None` or
the explicit case identifier `unknown`; it is never inferred from prose.

`PrecedentSummary` contains exact extracted background, legal issues,
reasoning, holding, statute and authority references, plus source-grounded
`PrecedentSummaryPoint` items. Each point has its actual references and
citations, semantic flags, and a limited-provenance marker.

The deterministic classifier recognizes explicit issue, reasoning, and
disposition formulations. It deliberately excludes party submissions and
requested relief from judicial reasoning and holdings. It also excludes
earlier-court outcomes and quoted authorities from the current court's holding.
A missing issue, reasoning, or holding creates an explicit warning rather than
a plausible-looking invention. The component makes no decision on whether an
international authority is binding or persuasive in Sri Lanka.

Judgment processing is bounded to 20,000 characters and 120 candidate
sentences, retaining complete sentence boundaries and issuing a truncation
warning. It does not claim a complete judgment review when material was
excluded.

## Step 4: current-case to precedent comparison

Set `summary_type="case_comparison"` and provide a `CaseComparisonRequest`.
It contains a source-grounded current-case `EvidenceSummary` or evidence
points, one or more already-extracted `PrecedentSummary` objects, and optional
current-case jurisdiction and research question. Each precedent is compared
separately; CaseLens does not combine judgments or rank a "best" precedent.

`CasePrecedentComparator` produces `CaseComparison` results containing
`ComparisonPoint` records. A point retains the original current-case and
precedent statement, sources from each side, and a review limitation. The
baseline matches only controlled explicit topics: contract execution, payment,
notice, delivery, termination conditions, and procedural issues. It identifies
an explicit amount/currency or stated-date difference only when both sides
actually state conflicting values.

Missing information is reported as missing; it is never called a contradiction.
For example, if the current case contains a notice requirement but the supplied
precedent does not, the output says precedent notice information was not
supplied. A current allegation and a precedent court finding can be topic-
related, but their different attribution status is retained as a limitation.

When current and precedent jurisdictions differ, CaseLens records the supplied
metadata and returns `authority_not_determined`. It does not decide whether a
foreign judgment is binding, persuasive, applicable to Sri Lankan law, or
predictive of the current case outcome. All comparisons require human legal
review.

Example, using fictional material: a Sri Lankan source saying "Ceylon Trading
Ltd must pay LKR 500,000 by 15 March 2025" and a fictional Singapore judgment
stating a payment obligation can produce a topic-level payment similarity. The
result is not a conclusion that Singapore law applies or that either party
will succeed.

## Example

```python
from agents.summarization import (
    EvidenceItem, SourceReference, SummarizationAgent, SummarizationRequest,
)

request = SummarizationRequest(
    case_id="CASE-001",
    document_id="DOC-001",
    query="What does the evidence state?",
    evidence_items=[EvidenceItem(
        text="ABC Holdings alleged that payment was not made.",
        source=SourceReference(document_id="DOC-001", citation_id="SRC-001"),
        is_negated=True,
        is_attributed=True,
        epistemic_status="alleged",
    )],
)
response = SummarizationAgent().summarize(request)
assert response.status == "completed"
assert response.evidence_summary.points[0].text == (
    "ABC Holdings alleged that payment was not made."
)
```

## Limitations and future work

This is not abstractive AI generation, a legal opinion, authoritative precedent
comparison, or claim-entailment verification. The deterministic comparison is a
source-grounded research aid, not an authority or outcome determination.
Keyword indicators are
deliberately conservative and may omit relevant evidence expressed in unusual
language. Citation-ID validation does not prove a claim is supported by its
cited passage. Future approved work should add a privacy-controlled provider,
claim-to-source verification, independent quality evaluation, and structured
precedent comparison.

## Tests

```powershell
python -m pytest tests/unit/summarization -q
python -m pytest tests/integration/test_coordinator_summarization.py -q
python -m pytest -q
```

## Step 5: offline synthetic evaluation

`tests/evaluation/summarization/fixtures/scenarios.json` and
`ground_truth.json` define a fictional, English-only benchmark of 30 cases:
12 evidence-summary cases, 10 judgment-summary cases, and 8 current-case to
precedent comparison cases. The expected labels were authored from the
fictional scenarios and include positive and negative cases; they are not
derived from component output. They remain provisional and require independent
legal-review validation before any research claim is made.

Run the evaluation with:

```powershell
python tests/evaluation/summarization/evaluate_summarization.py
python tests/evaluation/summarization/evaluate_summarization.py --write-report
python -m pytest tests/evaluation/summarization/test_evaluation.py -q
```

On the current 30-case synthetic corpus, the deterministic runner measured
evidence point extraction at 11 TP, 0 FP, 0 FN; precedent issue extraction at
4 TP, 0 FP, 0 FN; precedent reasoning extraction at 4 TP, 0 FP, 0 FN; holding
extraction at 5 TP, 0 FP, 0 FN; and controlled comparison detection at 7 TP,
0 FP, 0 FN. These are 100% precision, recall, and F1 for the labelled support
in this small development benchmark only. Provenance checks validated 35 of 35
evaluated references and found no fabricated source identifiers. This does not
measure real-world legal accuracy, retrieval quality, authority, or outcome
correctness.

The runner measures only deterministic component execution using
`time.perf_counter()` and reports its local timing with the result. It is not a
production performance benchmark. The generated JSON report is intentionally
written under `data/processed/` and is not source-controlled.

## Step 6: optional LLM-assisted evidence drafting

`SummarizationRequest.mode` defaults to `"deterministic"`. Set
`mode="llm_assisted"` explicitly to request a provider-assisted draft, and
only with `summary_type="evidence_summary"`. Precedent extraction and
case-to-precedent comparison remain deterministic. The established
`SummarizationReadyPayload` Coordinator interface is unchanged.

The public agent accepts only an injected `SummarizationProvider`; CaseLens
ships no live OpenAI, Gemini, or other network adapter in this component. A
missing provider, timeout, provider exception, malformed response, insufficient
citable context, or rejected draft returns a structured `failed` response with
a neutral error code. Provider exception text and source text are not exposed
in those errors. The deterministic route is not an automatic fallback for a
failed LLM draft, so users can distinguish a safe extractive result from a
rejected generated result.

### Grounding contract

The agent first runs `EvidenceSummarizer`; only its selected complete evidence
statements can enter the bounded LLM request. Each item is represented as
untrusted, delimited source data alongside supplied citation IDs and explicit
negation, attribution, conditionality, epistemic, and verification markers.
The prompt separates system instructions, the research question, allowed
citation IDs, and `UNTRUSTED EVIDENCE`. Instructions written inside evidence
are not trusted instructions. This reduces, but cannot guarantee resistance to,
prompt injection.

Provider output must validate as `LLMSummaryDraft`: a display-bounded summary,
one to six key points, one or more citation IDs for every point, optional
limitations, and a human-review flag. Citation IDs are validated against the
request-local selected source set; malformed, missing, duplicate-independent,
or unknown IDs are rejected. Accepted duplicate IDs are deduplicated, never
replaced. A valid ID proves only that a reference is request-local; it does not
prove the claim is entailed, true, independently verified, or legally correct.

Basic conservative checks reject low-overlap unsupported points, dropped
negation, allegation/attribution, or conditions, and changed or invented
recognisable dates and monetary values. Differing affirmative and negative
source semantics add a human-review warning rather than resolving the conflict.
Every accepted draft is explicitly marked `not_independently_verified` and
requires human legal review. It cannot determine authority, liability,
applicable law, or case outcome.

The LLM context uses at most six deterministically selected points and 10,000
characters. It omits whole statements rather than truncating a legal statement;
an omission warning is returned. The structured summary display limit is 1,800
characters.

### Offline test and evaluation coverage

`tests/unit/summarization/test_llm_assisted.py` uses fake providers only. It
covers opt-in mode, unchanged deterministic default, malformed output, missing
or wrong-document citations, timeout/provider failures, empty input, duplicate
citations, bounded context, prompt-injection text in fictional evidence,
negation/attribution/conditional loss, changed amounts/dates, unsupported court
holdings, conflict flags, JSON serialization, and no secret-bearing provider
errors.

The evaluation runner additionally passes six fictional fake-provider drafts
through the public agent: one safe accepted draft, one malformed draft, one
invented citation, and three basic semantic mismatches. It reports detected
format/citation/semantic violations and remaining unverified accepted claims.
These checks are bounded heuristics, not claim-entailment verification; the
future Verification Agent remains responsible for deeper source-to-claim
assessment.

Before any real provider is activated, the team must explicitly approve the
provider and model, rotate exposed credentials, complete privacy/data-retention
and confidential-document handling reviews, set cost/request limits plus
timeout/retry policy, and establish production monitoring. No live inference
or external model call is performed by this implementation.

## Step 7: deterministic legal research reports

`LegalResearchReportGenerator.generate(LegalResearchReportRequest)` assembles
already supplied evidence summaries, analysis findings, clauses, timeline
points, precedent summaries, and case comparisons into ordered professional
review sections. `SummarizationAgent.generate_report()` exposes the same typed
entry point without changing `summarize()` or invoking a provider.

The structured `LegalResearchReport` is primary. It records status, sections,
point-level sources and citation inventory, limitations, warnings,
`generation_method="deterministic"`, `not_independently_verified`, and mandatory
human review. Empty input yields `requires_input`; incomplete material yields
`partial`. Missing sections use explicit disclosures, not fabricated content.

Section ordering is: executive summary; background; facts and disputed
statements; potential issues; contractual terms; timeline; precedent research;
comparison; inconsistency review; limitations; sources; and the final legal
review disclaimer. Every substantive `ReportPoint` preserves its supplied
negation, attribution, conditionality, epistemic/verification status, source
references, and citation IDs. Comparison points retain sources from both sides.
Foreign authority remains undetermined.

`render_report_markdown(report)` is a pure optional renderer. It escapes
untrusted Markdown-sensitive source text and performs no export, filesystem,
network, or provider call. Citation inventory validation preserves actual IDs
only; it does not prove claim entailment, factual truth, legal applicability,
or a current-case outcome.

## Step 8: standalone research workflow

`SummarizationAgent.run_research_workflow(SummarizationWorkflowRequest)` is the
standalone, caller-supplied orchestration entry point. It accepts one explicit
current-case `evidence_summary` request, zero to twelve explicitly labelled
`precedent_summary` requests, a research question, optional source-grounded
findings/clauses/timeline/inconsistency points, and a rendering flag. It does
not retrieve new material: international judgments and all source references
must be supplied by the caller.

The workflow runs evidence selection, each precedent summary independently,
controlled case-to-precedent comparison, report assembly, and optional Markdown
rendering. `SummarizationWorkflowResponse` retains intermediate outputs,
source inventory, citations, per-stage `completed`/`partial`/`failed`/
`requires_input` status, and verification limitations. A missing holding leaves
the usable precedent material intact and yields a partial stage. With no
evidence, comparison is not performed and no similarity is invented. A report
assembly failure retains valid earlier results and marks the workflow failed or
partial as appropriate.

Default workflow summarization remains deterministic; an LLM-assisted current
evidence request remains explicit and requires the existing injected provider.
It cannot draft precedent holdings, comparisons, or report conclusions. The
workflow bounds precedent requests to 12 and report-supplied clauses, timeline
events, and inconsistency candidates to 100 each. Component limits continue to
apply, with existing truncation warnings rather than silent partial coverage.

The five-scenario fictional workflow evaluation checks section presence,
semantic flag preservation, request-local citation inventory, missing-material
disclosure, zero introduced report claims, and repeatability. It is a small
engineering check, not validation on real Sri Lankan or international legal
documents. Human legal review, approved privacy controls, live-retrieval
integration, and a Verification Agent remain external prerequisites.

## Step 9: contract readiness and final hardening

The public interface and cross-team handoff requirements are documented in
`docs/SUMMARIZATION_INTEGRATION_CONTRACT.md`. The document distinguishes the
standalone workflow from the legacy Coordinator retrieval pathway, records exact
schema/status/citation conventions, and describes proposed—not implemented—
Analysis and Verification adapters.

Workflow request validation rejects a citation ID that identifies different
documents in the same workflow, preventing ambiguous cross-document citation
inventory. Repeated references to the same document remain valid. Workflow-wide
warnings now include missing-evidence and missing-precedent comparison warnings
rather than retaining them only at stage level.

Final integration coverage additionally tests partial judgment extraction,
source-limited raw evidence, missing comparison inputs, cross-document citation
reuse, hostile Markdown/HTML source text, input truncation warnings, and
explicit LLM mode with no configured provider. These are fictional engineering
tests; they do not prove legal accuracy, complete prompt-injection resistance,
or real-document robustness.
