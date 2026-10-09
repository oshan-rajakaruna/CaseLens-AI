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
- Semantic embedding, vector-store, and hybrid precedent retrieval pipelines.
- NLP, evidence analysis, and report generation.
- LLM provider integrations and model prompts.
- Authentication, authorization, security controls, and responsible-AI checks.
- PostgreSQL persistence and production storage.

No semantic AI, authentication, or database feature is implemented in this
phase. The retrieval module currently provides the lexical foundation below.

## Retrieval Module - Current Status

Implemented:

- Plain UTF-8 text loading foundation.
- Conservative text cleaning.
- Deterministic word-based chunking with configurable size and overlap.
- Reusable legal document and chunk metadata.
- In-memory Okapi BM25 indexing.
- Structured BM25 top-k search.
- BM25-only Retrieval Agent interface.

Not implemented yet:

- Gemini embeddings.
- Semantic vector search.
- Vector database/index.
- Hybrid BM25 + semantic ranking.
- Metadata filtering.
- Full Coordinator integration.
- Retrieval evaluation metrics.

The retrieval implementation uses lexical BM25 only and does not call Gemini
or any other external AI service.

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
retrieval/     Text preprocessing and BM25 retrieval; semantic work is future
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
