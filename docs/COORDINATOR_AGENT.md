# Coordinator Agent — Step 1

## Purpose

The Coordinator is the shared CaseLens workflow boundary. Step 1 provides
validated, deterministic task planning only. It does not invoke Analysis,
Retrieval, Summarization, or Verification, and it does not produce legal
advice, retrieval results, summaries, or verification decisions.

## Public entry points

`Coordinator.plan(request)` returns a `CoordinatorPlanResponse` synchronously.
`await Coordinator.handle(request)` returns the same planning response for
future async workflow compatibility.

`CoordinatorRequest` requires a non-blank `case_id`, bounded non-blank
`user_query`, and one supported `requested_task`. It accepts bounded,
duplicate-free `document_ids`, `evidence_ids`, `reference_ids`, `claim_ids`,
and caller metadata. The planner preserves supplied identifiers but never
creates a document, citation, source excerpt, or legal finding.

## Supported routes

| Requested task | Planned agents, in order |
| --- | --- |
| `analyze_evidence` | Analysis |
| `retrieve_precedents` | Retrieval |
| `summarize_evidence` | Summarization |
| `summarize_precedents` | Retrieval → Summarization |
| `verify_claims` | Verification |
| `full_research` | Analysis → Retrieval → Summarization → Verification |

These are plans only. A route does not prove that its inputs can yet be mapped
between agents, that an agent ran, or that a result is verified.

## Statuses and prerequisites

Stages use `planned` when the planner has enough identifiers to propose a
route, or `not_run` when prerequisites are missing. The response uses
`status="planned"` or `status="requires_input"`; it never says `completed`
in Step 1. Every response has `executed=false` and records a limitation that
no agent output, citation validation, or legal claim verification occurred.

`analyze_evidence`, `summarize_evidence`, and `full_research` require a
document or evidence identifier. `verify_claims` requires at least one claim
identifier. Missing inputs are exposed as structured, non-sensitive errors.

## Fictional example

```python
from agents.coordinator import Coordinator, CoordinatorRequest

request = CoordinatorRequest(
    case_id="CASE-DEMO-001",
    user_query="What international precedents may be relevant?",
    requested_task="full_research",
    document_ids=["DOC-DEMO-001"],
)

response = Coordinator().plan(request)
assert response.status == "planned"
assert response.executed is False
assert [stage.agent for stage in response.stages] == [
    "analysis", "retrieval", "summarization", "verification",
]
```

## Existing compatibility

The existing `LegalRetrievalContextService` and `SummarizationReadyPayload`
remain unchanged. They provide a Retrieval-to-Summarization context adapter,
but are not called by Step 1 planning. The existing `CoordinatorResponse`
remains available for Verification-result serialization.

## Current limitations and Step 2

## Step 2: Analysis execution

`Coordinator.execute_analysis(request)` and
`await Coordinator.handle(request)` execute the existing `AnalysisAgent` only
when `requested_task="analyze_evidence"`. Other task types still use planning
only. The Coordinator-owned `CoordinatorEvidenceDocument` contains supplied
document ID, name, text, optional media type, metadata, and optional existing
`ExtractionSegment` values. Its document IDs must exactly match `document_ids`.

Execution additionally requires a `case_title` and at least one supplied
evidence document. Missing inputs return `requires_input`, one `not_run`
Analysis stage, and structured prerequisite errors; Analysis is not invoked.

For valid execution, `AnalysisCoordinatorAdapter` constructs one
`AnalysisRequest` per supplied document, preserving supplied text, metadata,
segments, and identifiers. `CoordinatorAnalysisResponse.analysis_results`
retains the original typed `AnalysisResponse` objects without reformatting
facts, clauses, entities, timelines, contradiction candidates, sources, or
semantic qualifiers. Its stage record mirrors each Analysis result status and
uses its document ID only as a result reference.

This remains a local, offline integration. It does not retrieve precedents,
summarize, verify, aggregate across documents, create source records, or
decide legal truth. In particular, citation identity validation and semantic or
legal verification are not performed by this route.

An inherited Analysis limitation remains: some abbreviated honorific forms
(for example `Ms.` or `Mr.`) can be split by its deterministic sentence
baseline before downstream fact semantics are calculated. The Coordinator
preserves that existing output and does not correct or reinterpret it.

## Remaining next steps

## Step 3: Retrieval execution

`Coordinator.execute_retrieval(request)` and
`await Coordinator.handle(request)` execute Retrieval only when
`requested_task="retrieve_precedents"`. The Coordinator receives a prepared
`CoordinatorRetrievalAdapter`, which wraps the existing `RetrievalAgent` public
methods: `search_bm25`, `search_semantic`, and `search_hybrid`. The Coordinator
does not parse documents, build an index, download a model, or create a
Retrieval Agent from environment configuration.

`CoordinatorRequest` adds `retrieval_mode` (`bm25`, `semantic`, or `hybrid`),
bounded `top_k` (1–20), and optional `retrieval_filters`. Filters are accepted
only for hybrid mode because that is the existing Retrieval Agent contract.
The injected agent must have the index required for its mode: BM25 for `bm25`,
semantic for `semantic`, and both for `hybrid`.

For a real search, `CoordinatorRetrievalResponse.retrieval_results` retains the
native `BM25SearchResult`, `SemanticSearchResult`, or `HybridSearchResult` in
their original order. This preserves document ID, chunk ID, original passage
text, rank, native relevance scores, case/court/citation fields, and attached
metadata such as page ranges and source URL. Scores remain retrieval scores;
they are not authority, applicability, or binding-status assessments.

No adapter or required index returns `requires_input`, `executed=false`, and a
`not_run` Retrieval stage. A valid empty search returns `completed`,
`executed=true`, an empty results list, and `no_retrieval_results`; it does not
manufacture a precedent. Controlled Retrieval failures return a failed stage
without exposing exception contents.

Example fictional offline search:

```python
request = CoordinatorRequest(
    case_id="CASE-RET-001",
    user_query="contract payment notice",
    requested_task="retrieve_precedents",
    retrieval_mode="bm25",
    top_k=3,
)
response = coordinator.execute_retrieval(request)
# response.retrieval_results[0].chunk_id is an upstream chunk identifier.
```

The existing `LegalRetrievalContextService` remains the future
Retrieval-to-Summarization path. It accepts native Retrieval result objects and
builds its own request-local `CTX-*` map. Step 3 therefore keeps raw results
and their metadata available, but does not assign CTX identifiers or invoke
Summarization.

## Remaining next steps

## Step 4: Retrieval-to-precedent summarization

`Coordinator.execute_summarize_precedents(request)` and `await
Coordinator.handle(request)` now execute the narrow, offline route for
`requested_task="summarize_precedents"`: prepared Retrieval → deterministic
Summarization. It uses `CoordinatorSummarizationAdapter` and the public
`SummarizationRequest(summary_type="precedent_summary", mode="deterministic")`
contract. It never calls the legacy `SummarizationReadyPayload` interface or a
provider-backed drafting path.

Native results remain available in `CoordinatorPrecedentResponse.retrieval_results`.
For each result whose caller-provided document type is explicitly `judgment` or
`precedent`, the adapter groups passages by document ID and creates one
deterministic summary request per judgment. Results without that explicit type
are retained as Retrieval output but are not summarized as court material.
This prevents a contract, correspondence, or miscellaneous document from being
represented as a precedent merely because it mentions a court.

Each mapped `SourceReference` preserves the upstream document ID, unchanged
chunk text, and a locator containing the upstream chunk ID and supplied page
range. A request-local `SRC-<sha256-prefix>` citation ID is derived from the
document ID plus chunk ID, never from result position. `citation_provenance`
retains the matching chunk ID, pages, title, court, jurisdiction when supplied,
official citation, and source URL. An attempt to reuse a chunk ID for different
text is rejected as a controlled summarization failure; neither source is
silently merged.

Multiple chunks from one judgment become one `SummarizationResponse`; distinct
document IDs become separate responses and separate precedent summaries. The
Coordinator does not invent a title, court, holding, page number, official
citation, or source URL. The actual deterministic Summarization output,
warnings, errors, semantic qualifiers, and source references are retained.

Status behavior is explicit: a missing prepared Retrieval dependency produces
`requires_input` with both stages `not_run`; an empty search produces a
completed Retrieval stage and a `not_run` Summarization stage; unsupported
document types result in `partial`; and a Summarization failure retains native
Retrieval results and marks only that stage failed. A completed processing stage
is not independent verification, authority, applicability, binding status, or
a legal conclusion.

Fictional offline example:

```python
request = CoordinatorRequest(
    case_id="CASE-SUM-001",
    user_query="Was contractual notice timely?",
    requested_task="summarize_precedents",
    retrieval_mode="bm25",
)
response = coordinator.execute_summarize_precedents(request)
# response.retrieval_results retains native chunks.
# response.precedent_summaries contains only explicitly typed judgment material.
```

`summarize_evidence`, Verification execution, and `full_research` remain
planning-only. A future step must introduce each adapter separately, preserving
the source identifiers and semantic safeguards already present in the
specialized-agent contracts.

## Steps 5–7: evidence, verification, and full research

`summarize_evidence` now runs the deterministic Summarization Agent only over
caller-supplied `CoordinatorEvidenceDocument` values. It does not silently run
Analysis, Retrieval, Verification, a provider, or a network request.
`verify_claims` runs the existing deterministic Verification Agent only when
the caller supplies its public `LegalClaim` and `RetrievedEvidence` records.
Its supported/unsupported/uncertain vocabulary is preserved and is not a legal
truth, authority, or outcome determination.

`full_research` executes the currently integrated offline sequence: Analysis,
evidence summarization, prepared-index Retrieval, deterministic precedent
summarization, report workflow when sufficient materials exist, then narrow
claim verification of traceable extractive evidence points. Earlier results
remain in the response if a later stage is partial or fails. It requires a case
title, source documents, and an injected prepared Retrieval adapter. Verification,
retrieval relevance, and foreign precedent applicability remain non-binding and
require human legal review. Reproduce tests on Windows with
`\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_cache\caselens-test-temp\full_run`.
