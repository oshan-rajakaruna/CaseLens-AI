# Summarization Agent

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

## Gemini adapter

`GeminiSummaryGenerator` implements `SummaryGenerator` with the official
`google-genai` SDK. It makes one asynchronous structured-output request and
validates the result as `SummarizationResponse`. It does not retry or issue a
second request to repair invalid output. Provider errors are converted to
sanitized domain errors without including prompts, evidence, or credentials.

Configure these backend environment variables:

```text
LLM_API_KEY=<Gemini API key>
GEMINI_MODEL=gemini-3.8-flash
GEMINI_MAX_OUTPUT_TOKENS=4096
```

Create the production adapter with `GeminiSummaryGenerator()`, then inject it
into `SummarizationAgent`. Tests can inject a fake client into
`GeminiSummaryGenerator(client=fake_client)` and require neither credentials
nor network access.

Install dependencies and run the focused offline tests from the repository
root:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/unit/test_summarization_agent.py tests/unit/test_gemini_summary_generator.py
```

The adapter prompt treats source text as untrusted data, prohibits invented
facts and citations, and asks for warnings when evidence is insufficient. The
`SummarizationAgent` still enforces source-reference and citation-excerpt
preservation after generation. All model-generated claims remain drafts and
must be checked by the Verification Agent before use.

## Draft legal research reports

`LegalReportAssembler` in `backend.services` accepts the validated case,
original `SummarizationRequest`, and `SummarizationResponse`. It deterministically
copies evidence summaries, precedents, comparisons, draft findings, citations,
and warnings into `LegalResearchReport`. The original request is required so
direct callers cannot introduce unsupported references or citation excerpts.
Missing optional sections become explicit limitations; reports always have
`pending_verification` status.

```python
from backend.services import LegalReportAssembler, MarkdownLegalReportRenderer

report = LegalReportAssembler().assemble(request.case, request, summary)
markdown = MarkdownLegalReportRenderer().render(report)
```

The Markdown renderer makes no model calls and escapes supplied text so it is
displayed as plain content rather than report structure. A later milestone may
connect the assembler to the Coordinator and an API, and the Verification Agent
may review the draft. Persistence, verified-status transitions, and PDF/DOCX
export are not implemented.
