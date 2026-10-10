# Coordinator Retrieval-to-RAG Contract

## Current flow

```text
User query
  -> Coordinator retrieval-context adapter
  -> existing Retrieval Agent
  -> diversified ranked results
  -> RAGContextAssembler
  -> summarization-ready payload
  -> provider-neutral Summarization Agent
  -> citation-validated SummarizationResult
  -> future Verification Agent
```

The Coordinator phase ends at the summarization-ready payload. The current
Summarization Agent accepts an injected provider, validates its structured
answer and citations, and prepares a result for later verification. No external
provider is configured; current tests and smoke paths use deterministic fakes
and make no LLM, network, OpenAI, or Gemini request.

## Request contract

`LegalRetrievalContextRequest` supports:

- `query` and optional `legal_issue`;
- `top_k` and existing metadata `filters`;
- `retrieval_mode` (`hybrid` by default);
- optional BM25 and semantic weight overrides;
- `diversify` and an optional per-document chunk-cap override;
- optional RAG token, passage, and minimum-score overrides.

Omitted retrieval and RAG settings use the existing environment-backed
defaults. An explicitly supplied null RAG score threshold disables an
environment threshold for that request.

## Summarization-ready payload

`SummarizationReadyPayload` contains:

- the query and legal issue;
- the formatted prompt context;
- structured RAG passages;
- a citation map keyed by `CTX-nnn`;
- effective retrieval mode, weights, filters, and diversification settings;
- counts, token usage, truncation state, and weak-context diagnostics.

Each citation-map entry preserves its context ID, document ID and title,
court/source, original chunk IDs, pages, official citation, and source URL.
The `CTX` identifier supplements rather than replaces this provenance.

## Summarization and citation validation

Generated text must cite evidence with bracketed IDs such as
`[CTX-001]`. `validate_context_citations` reports:

- cited IDs in occurrence order;
- unique valid and unknown IDs;
- malformed bracketed or bare CTX-like references;
- whether all cited references are valid.

The validator never edits generated text. No citations is represented
separately from invalid citations through `has_citations`. The
`SummarizationAgent` additionally compares inline IDs with the structured
declared-citation list and rejects unknown, malformed, missing, or inconsistent
references through a controlled domain error. Repeated valid inline citations
are allowed.

`SummarizationOutput` defines the answer, declared citation IDs, optional
confidence, and limitations. With no usable context the provider is skipped
and a fixed insufficient-context answer is returned. Weak-context diagnostics
are included in the prompt and must be acknowledged in output limitations.
See [SUMMARIZATION_AGENT.md](SUMMARIZATION_AGENT.md) for the provider, prompt,
validation, provenance, and future Verification Agent handoff contracts.

## Diagnostics and limitations

The adapter does not introduce a global retrieval-score threshold. It emits a
non-filtering warning when a selected passage scores below 10% of the leading
passage for that same request. This relative warning helps surface severe
within-query score gaps, including the known sanitary and phytosanitary
measures smoke-test case, but it is not a relevance judgment.

Before enabling an external summarization provider, CaseLens still needs a
model-specific token counter, provider selection/configuration, held-out
evaluation of score thresholds, and a Verification Agent that independently
checks claim support after the current citation-identity validation.
