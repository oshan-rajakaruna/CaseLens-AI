# Development Summarization API

`POST /api/summarization/verify` generates an evidence-grounded summary and
passes it through the existing Verification Agent. It is a local development
endpoint, not a production authentication design or full Coordinator flow.

Access is denied by default. It is permitted only when `APP_ENV` is exactly
`development`, `ENABLE_DEV_SUMMARIZATION_API=true`, and a non-empty
`DEV_SUMMARIZATION_API_TOKEN` matches the `X-CaseLens-Dev-Token` request
header. Keep both this token and `GEMINI_API_KEY` in the ignored `.env` file;
never place either in frontend code or committed configuration. Serve manual
development checks on the configured loopback address.

The JSON request contains a non-empty `task_id` and a previously assembled
`SummarizationReadyPayload` under `payload`. The caller must currently build
this payload; the endpoint does not initialize retrieval indexes or run full
Coordinator orchestration. Tests can submit synthetic RAG passages and use an
overridden fake service.

The response contains `summary`, `verification`, `provenance`, and
`bridge_warnings`. It deliberately omits the bridge's internal
`VerificationRequest`, which contains original evidence text. CTX validation
establishes prompt-local citation identity and consistency; the Verification
Agent separately reports conservative claim-support outcomes.

Gemini is used only for text generation. Retrieval remains separate and its
configured local embedding model remains `BAAI/bge-small-en-v1.5`.

Gemini requests use `GEMINI_REQUEST_TIMEOUT_MS` (30,000 milliseconds by
default) and one total SDK attempt. Set
`SUMMARIZATION_TIMING_DIAGNOSTICS=true` temporarily to log only stage names,
monotonic durations, outcomes, and exception class names. These diagnostics
never log prompts, evidence, generated summaries, request bodies, or secrets.
