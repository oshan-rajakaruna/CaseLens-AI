# CaseLens Development Roadmap

This roadmap records implemented milestones and known dependencies without
implying that planned APIs, persistence, authentication, or legal validation
already exist.

## Completed and committed

- Milestones 1–3: Gemini summarization, verification bridge, and development-only
  FastAPI summarization integration.
- Milestone 4A: deterministic structured Legal Research Report assembly.
- Milestone 4B-1: explicit fact-to-authority links.
- Milestone 4C: safe Markdown report rendering.
- Milestone 4B-2: opt-in, unverified Gemini comparative explanations.
- Milestone 5A: reusable Legal Research Report frontend components, synthetic
  development preview, and frontend testing foundation.
- Milestone 6B-1: internal, dependency-injected ResearchWorkflow composing
  Analysis, retrieval context, summarization, Verification, report assembly,
  optional comparisons, and Markdown rendering.

## Current

- Milestone 5B: dashboard and case/report workspace implemented; manual testing
  and visual verification are pending.
- Milestone 6B-2: shared Coordinator activated through an internal adapter that
  validates AgentTask context, runs ResearchWorkflow off the event loop, maps
  completed/partial/failed outcomes, and exposes only sanitized status metadata.
  Manual regression testing is pending.
- Milestone 6C-1: PostgreSQL-compatible Cases, Case Documents, Research Tasks,
  Legal Reports, and sanitized Agent Logs persistence is implemented behind an
  owner-scoped internal repository/service boundary, with Alembic initialization
  and isolated offline tests. Dependency installation, migration execution, and
  manual test verification are pending.

## Still pending

- 5C: secure frontend API integration.
- Durable Coordinator persistence integration, cancellation, and restart recovery.
- Production authentication and authorization.
- Real case and report APIs.
- Frontend integration with authenticated production APIs.
- Semantic corpus indexing and hybrid production validation. The current
  imported 36-document international-law corpus and persistent BM25 index have
  been manually exercised, but dataset approval and broader evaluation remain.
- Member 2, 3, and 4 contract coordination.
- Human review and substantive comparison-verification design.
- Real-document end-to-end integration, security review, and deployment.

Frontend previews and tests may use explicitly labelled synthetic fixtures.
Production screens must not present synthetic data or simulated agent success as
real case processing.
