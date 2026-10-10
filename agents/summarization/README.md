# Summarization Agent foundation

`SummarizationAgent` accepts a validated `SummarizationRequest` and delegates
generation to an injected asynchronous `SummaryGenerator`. It never produces a
fallback summary. With no evidence or no configured generator, it raises an
explicit domain error.

Inputs contain case metadata, extracted facts, and retrieved legal passages.
Every fact and passage carries a `document_id` and source `locator`. Outputs
separate grounded evidence/precedent summaries and comparisons from explicitly
marked `draft` claims. The agent rejects any output citation or source pointer
that was not present in the request.

The shared Coordinator can call `handle(AgentTask)`, placing the serialized
`SummarizationRequest` in `AgentTask.context`. The result is returned through
the existing `AgentResult` contract. When the request does not contain its own
instruction, `AgentTask.instruction` is passed to the generator.

A later milestone can implement `SummaryGenerator` with an LLM adapter that
converts structured prompts and structured model responses. Configuration,
provider calls, retry behavior, and prompt design deliberately remain outside
this milestone. Tests should inject a deterministic fake generator; no network
access is required.
