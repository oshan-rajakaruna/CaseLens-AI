# CaseLens

CaseLens is an academic project for an AI-powered legal case analysis, evidence
review, and precedent retrieval assistant. The repository is currently in
**Phase 1: Foundation**.

## Implemented in Phase 1

- A minimal FastAPI application with `/` and `/health` endpoints.
- Shared Pydantic contracts for cases, documents, agent tasks/results,
  citations, and verification results.
- Package boundaries and interfaces for the five planned agents.
- A minimal React and Vite application shell.
- Environment, testing, and team ownership documentation.

## Planned for future phases

- Agent orchestration and specialized agent behavior.
- Production retrieval persistence, evaluation, and advanced reranking.
- NLP, evidence analysis, and report generation.
- LLM provider integrations and model prompts.
- Authentication, authorization, security controls, and responsible-AI checks.
- PostgreSQL persistence and production storage.

No generative case-analysis, authentication, or database feature is implemented
in this phase. The retrieval module currently provides lexical, semantic, and
hybrid search foundations plus offline evaluation tooling.

## Retrieval Module - Current Status

Implemented:

- Plain UTF-8 text loading foundation.
- Conservative text cleaning.
- Deterministic word-based chunking with configurable size and overlap.
- Reusable legal document and chunk metadata.
- In-memory Okapi BM25 indexing.
- Structured BM25 top-k search.
- Gemini embedding service using the official Google GenAI SDK.
- Retrieval-specific query and document embedding formatting.
- In-memory semantic vector indexing with cosine similarity.
- Structured semantic top-k search.
- Separate BM25 and semantic Retrieval Agent interfaces.
- Hybrid BM25 + semantic ranking with explicit min-max normalization.
- Configurable weighted score fusion and candidate-pool expansion.
- Optional exact-match filters for court, date/year, legal category, and
  document type.
- FastAPI `POST /api/retrieval/search` endpoint with BM25, semantic, and hybrid
  dispatch.
- Retrieval-side Coordinator task/result adapter using shared agent envelopes.
- Retrieval evaluation framework with per-query and aggregate Precision@K,
  Recall@K, and MRR.
- BM25, semantic, and hybrid mode comparison.
- Configurable hybrid-weight experiment support without automatic winner
  selection.
- Optional manual relevance labels kept separate from binary metric ground
  truth.

Not implemented yet:

- Final fusion-weight tuning with retrieval evaluation data.
- Final real-world evaluation using a curated legal dataset and reviewed
  relevance judgements.
- Full Coordinator integration.
- Production legal dataset ingestion and index initialization.
- Production vector database persistence.
- Advanced reranking.

Semantic indexing and queries call Gemini only when their explicit methods are
used. BM25 remains independent and does not call Gemini. Configure backend-only
embedding access with:

```dotenv
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSION=768
```

Never place a real Gemini key in source control or frontend configuration.

Hybrid retrieval defaults to equal BM25 and semantic weights (`0.5` / `0.5`)
and retrieves twice the requested result count from each mode before fusion.
Both settings are configurable. The default weights are an initial academic
prototype choice, not an empirically proven optimum. The evaluation framework
can compare weights using Precision@K, Recall@K, and MRR, but final tuning must
wait for a curated legal dataset.

### Retrieval API

The backend exposes `POST /api/retrieval/search`. Production indexes must be
initialized separately; synthetic test fixtures are never loaded by the real
application. Example request:

```json
{
  "query": "unlawful termination",
  "mode": "hybrid",
  "top_k": 5,
  "filters": {
    "legal_category": "employment"
  }
}
```

Metadata filters currently apply to hybrid mode. The Retrieval-side Coordinator
adapter accepts the same structured fields plus an optional `legal_issue`; it
does not implement or modify the shared Coordinator workflow.

### Retrieval evaluation

Run the bundled deterministic framework check from the repository root:

```powershell
python -m retrieval.evaluation.run
python -m retrieval.evaluation.run --weights --json
```

The bundled dataset is explicitly synthetic and the offline embedding service
exists only to make framework execution reproducible without Gemini, internet,
or an API key. Its output is not a claim about production accuracy or legal
retrieval quality. Final evaluation results remain pending a curated legal
dataset and reviewed relevance judgements. See
[RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md) for the reporting
template.

## Planned architecture

The React frontend will communicate with the FastAPI backend over HTTP using
structured JSON. The backend will validate shared API contracts and, in later
phases, delegate work through a coordinator to specialized agents. Retrieval,
NLP, persistence, storage, and security remain separate modules so team members
can develop them in parallel.

## Five-agent overview

1. **Coordinator Agent** — accepts requests, determines workflows, shares
   structured context, invokes specialist agents, aggregates results, and
   returns verified output. It is jointly owned by all team members.
2. **Retrieval Agent** — will find relevant legal material and precedents.
3. **Analysis Agent** — will analyze case material and evidence.
4. **Summarization Agent** — will prepare summaries and reports.
5. **Verification Agent** — will validate outputs, citations, and responsible-AI
   safeguards.

## Repository structure

```text
frontend/      React and Vite application shell
backend/       FastAPI app, API routing, configuration, and shared schemas
agents/        Coordinator and specialist-agent package boundaries
retrieval/     Text preprocessing, BM25, semantic, and hybrid retrieval
nlp/           Future language and evidence-processing work
database/      Future PostgreSQL persistence boundary
security/      Future security and responsible-AI controls
storage/       Future document and generated-artifact storage
tests/         Unit, integration, and evaluation test areas
docs/          Project and ownership documentation
data/          Local raw and processed data locations (contents are ignored)
```

## Prerequisites

- Python 3.11 or newer
- Node.js 20.19+, 22.12+, or a newer Vite-supported release
- npm

PostgreSQL is planned but is not required during Phase 1.

## Environment setup

From the repository root, copy `.env.example` to `.env` and adjust only local
development values. Never commit `.env` or real credentials.

```powershell
Copy-Item .env.example .env
```

## Backend setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API root is available at `http://127.0.0.1:8000/` and health status at
`http://127.0.0.1:8000/health`.

## Frontend setup and run

```powershell
Set-Location frontend
npm install
npm run dev
```

The development server defaults to `http://127.0.0.1:5173`.

## Tests

After installing the backend requirements:

```powershell
python -m pytest
```

To verify the frontend production build after installing frontend dependencies:

```powershell
Set-Location frontend
npm run build
```

## Team ownership

Primary ownership is divided among summarization/backend, retrieval/knowledge
base, analysis/NLP, and verification/security. The coordinator and cross-cutting
project work are shared. See [TEAM_OWNERSHIP.md](docs/TEAM_OWNERSHIP.md) for the
full allocation.

## Academic context

CaseLens is being developed as a four-member academic project. The current code
is intentionally a foundation only: it establishes contracts and boundaries
without presenting unfinished research or placeholder behavior as a working
legal AI system.
