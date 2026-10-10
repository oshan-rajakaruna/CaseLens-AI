# Summarization Agent Contract

## Responsibility and boundary

`SummarizationAgent` consumes the Coordinator's `SummarizationReadyPayload`
and returns a `SummarizationResult` containing a validated
`SummarizationOutput`. It is responsible for grounded answer generation,
prompt-local citation checks, safe provider failure handling, and preservation
of retrieval warnings. It does not perform retrieval, alter rankings, or run
Verification Agent checks.

No external model is configured in this phase. Generation is accessed only
through the injectable `SummarizationProvider` protocol. Tests and smoke tests
use deterministic fake providers and make no OpenAI, Gemini, or network calls.

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

## Future verification handoff

`SummarizationResult` exposes the validated output, detailed citation
validation, safe provider/model identity, used context IDs, and warnings. A
future Verification Agent should accept that wrapper and independently check
claim support, legal-authority fidelity, citation-to-passage entailment, and
responsible-AI safeguards. The Verification Agent must not be folded into the
Summarization Agent.
