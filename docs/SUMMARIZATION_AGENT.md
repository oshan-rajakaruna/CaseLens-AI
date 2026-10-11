# Summarization Agent Contract

## Responsibility and boundary

`SummarizationAgent` consumes the Coordinator's `SummarizationReadyPayload`
and returns a `SummarizationResult` containing a validated
`SummarizationOutput`. It is responsible for grounded answer generation,
prompt-local citation checks, safe provider failure handling, and preservation
of retrieval warnings. It does not perform retrieval, alter rankings, or run
Verification Agent checks.

Generation remains accessed through the injectable `SummarizationProvider`
protocol. `GeminiSummarizationProvider` is the production implementation and
uses the official synchronous Google Gen AI SDK. Its client can be injected so
unit tests remain deterministic and make no Gemini or network calls.

## Gemini configuration

Set these backend-only environment variables:

- `GEMINI_API_KEY` (required): Gemini API credential;
- `GEMINI_MODEL` (optional): generation model, initially
  `gemini-3.8-flash` in `.env.example`;
- `GEMINI_MAX_OUTPUT_TOKENS` (optional): positive integer, default `2048`.

Model availability varies by account and over time. If the configured model is
unavailable, change `GEMINI_MODEL` without modifying application code. Keep the
credential in the ignored backend `.env`; never expose it to Vite variables or
frontend code.

The Gemini provider receives the existing deterministic prompt, requests JSON
constrained by `SummarizationOutput`, and validates the result with that same
Pydantic model. `SummarizationAgent` then performs its existing CTX citation
validation. Provider schema validation is not substantive legal verification.

Gemini generation is independent from retrieval embeddings. Semantic retrieval
continues to use the configured local FastEmbed model
`BAAI/bge-small-en-v1.5`; selecting Gemini here does not change embedding or
retrieval behavior.

Run the offline summarization tests from the repository root with:

```powershell
python -m pytest tests/unit/summarization tests/integration/test_coordinator_summarization.py
```

These tests use injected clients/providers. A later milestone will adapt
summarization results to the independently implemented Verification Agent.

## Provider and prompt contract

A provider receives a `SummarizationProviderRequest` with the user query,
optional legal issue, formatted RAG context, structured citation map,
generation instructions, and the complete deterministic prompt. A provider
must return structured data matching `SummarizationOutput`:

- `answer` (non-empty text);
- `citations` (declared `CTX-nnn` identifiers);
- optional `confidence` from 0 to 1;
- `limitations`.

The prompt separates system instructions, user query, legal issue, retrieved
context, allowed citations, and retrieval limitations. It requires the answer
to use only supplied evidence, distinguish evidence from interpretation, avoid
invented authorities/cases/pages/citations, avoid comprehensive-coverage and
legal-advice claims, and cite factual or legal claims inline as `[CTX-nnn]`.

## Citation validation

After generation, the agent validates both the inline answer citations and the
declared citation list against the Coordinator citation map. The contract
requires:

- every inline citation is allowed and well formed;
- every declared citation is allowed;
- every declared citation appears inline;
- every inline citation appears in the declared list.

Repeated inline citations are valid. Unknown, malformed, missing, or
inconsistent citations raise a controlled `SummarizationValidationError`.
The agent never rewrites an answer or substitutes a citation to hide a
generation error.

Validated citation IDs can be rendered with their stored document ID, title,
court/source, pages, official citation, source URL, and chunk IDs. Missing
provenance is omitted rather than invented.

## No-context and weak-context behavior

When context passages or the citation map are empty, the provider is not
called. The agent returns the fixed answer:

> Insufficient retrieved legal context to answer reliably.

If retrieval diagnostics contain weak-context warnings, those warnings are
included in the provider prompt and carried into the result. The structured
output must acknowledge weak, uncertain, limited, insufficient, or
relevance-sensitive evidence in its limitations. Weak context is not rejected
when usable evidence remains.

## Verification bridge

`SummarizationVerificationBridge.verify(task_id, payload, result)` requires
both the original `SummarizationReadyPayload` and its `SummarizationResult`.
The result supplies the generated answer and validated CTX references; the
payload supplies the authoritative evidence text and source provenance that is
not duplicated in the generated result.

Each used CTX reference becomes both the evidence ID and an internal verifier
locator. The bridge maps it to the original document ID and passage text, while
retaining the complete `ContextCitationMapEntry` separately for audit and
display. A CTX locator is not represented as an original page, paragraph, or
official legal citation, and generated wording is never used as a citation
excerpt.

The current summary contract contains one answer rather than atomic claims, so
the bridge removes only valid inline CTX markers and submits the complete
answer as one `LegalClaim`. The existing verifier then decides support using
its conservative rules. Valid CTX identity does not prove substantive legal
correctness, and generated paraphrases are expected to produce `uncertain`
results when they do not directly match cited evidence.

Missing, ambiguous, or inconsistent provenance and summaries without usable
cited evidence raise controlled bridge errors; no successful verification
result is fabricated. Downstream Coordinator or API code should call the
bridge only after summarization and should preserve the returned Verification
Agent response without promoting its statuses or `verified` flag.
