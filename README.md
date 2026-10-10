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

- UTF-8 TXT and embedded-text PDF loading with ordered page provenance.
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
- Human-reviewed real-corpus evaluation covering 38 query-document judgments
  across seven queries and 36 international-law documents.
- Environment-configurable default hybrid fusion using the best observed
  real-corpus compromise of BM25 `0.8` and semantic `0.2`.
- Manifest-driven curated legal dataset validation and ingestion.
- Reusable BM25 index construction with explicit, optional semantic indexing.
- Structured strict/non-strict ingestion reports and a no-API dry-run mode.
- Metadata-only dataset summaries, duplicate detection, and index-readiness
  reporting.

Not implemented yet:

- Larger-scale retrieval evaluation with a held-out validation query set.
- Full Coordinator integration.
- Persistent production index initialization and lifecycle management.
- Production vector database persistence.
- Advanced reranking.

Semantic indexing and queries call the selected provider only when their
explicit methods are used. BM25 remains independent. Configure backend-only
OpenAI embedding access with:

```dotenv
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION=1536
```

Gemini remains supported with:

```dotenv
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSION=768
```

Never place a real provider key in source control or frontend configuration.

Hybrid retrieval defaults to BM25 `0.8` and semantic `0.2`, configured with
`HYBRID_BM25_WEIGHT` and `HYBRID_SEMANTIC_WEIGHT`. The configured defaults must
be finite, nonnegative, and sum to `1.0`; individual callers may still supply
explicit weights, which are normalized by the existing fusion layer. Hybrid
retrieval takes twice the requested result count from each mode before fusion.

Production retrieval also applies document-level diversification after final
score ordering. `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT=2` limits the number of
returned chunks sharing the same metadata `document_id`; scores and their
ordering are not recalculated. API and Retrieval Agent callers can override
the cap with `max_chunks_per_document`, or intentionally disable it with
`diversify=false`. The configured value must be an integer of at least `1`;
invalid values fail clearly instead of silently selecting another cap. This is
a redundancy control for downstream RAG context, not an accuracy claim.

The default was selected as the best observed hybrid compromise in the current
human-reviewed real-corpus evaluation. It is not a universally optimal weight:
the evaluation contains only 36 documents, seven queries, and 38 reviewed
query-document judgments, with no held-out validation set. Pure BM25 tied or
exceeded hybrid on some metrics. For this evaluation only,
`partially_relevant` judgments were mapped to binary relevant alongside
`relevant` judgments.

### Retrieval API

The backend exposes `POST /api/retrieval/search`. Production indexes must be
initialized separately; synthetic test fixtures are never loaded by the real
application. Example request:

```json
{
  "query": "unlawful termination",
  "mode": "hybrid",
  "top_k": 5,
  "diversify": true,
  "max_chunks_per_document": 2,
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
exists only to make framework execution reproducible without an external
provider, internet, or an API key. Its output is not a claim about production
accuracy. A separate human-reviewed evaluation was completed on the curated
36-document corpus; see
[RETRIEVAL_EVALUATION.md](docs/RETRIEVAL_EVALUATION.md) for its policy,
decision, and limitations.

The evaluation runner explicitly disables production diversification so its
historical chunk rankings and weight comparison remain reproducible. Future
evaluation of the cap should be reported as a separate experiment.

### RAG context assembly

`retrieval.rag.RAGContextAssembler` converts already-ranked retrieval results
into citation-preserving passages for future Coordinator and Summarization
work. Retrieval results are individual scored chunks; RAG context passages may
conservatively merge adjacent chunks, remove document-scoped duplicate text,
and truncate the final passage to remain within a prompt-context budget.

Context assembly defaults to `RAG_MAX_CONTEXT_TOKENS=4000` and
`RAG_MAX_CONTEXT_PASSAGES=8`. `RAG_MIN_RETRIEVAL_SCORE` is optional and has no
numeric default: empty or unset keeps all retrieved results. Context IDs such
as `CTX-001` are prompt-local references and supplement rather than replace
document IDs, chunk IDs, page numbers, citations, sources, and retrieval
scores. Token counts use a deterministic dependency-free estimate that counts
Unicode words and punctuation as units, including formatted provenance
headers. See [RAG_CONTEXT.md](docs/RAG_CONTEXT.md) for the complete contract and
limitations.

The Coordinator-side `LegalRetrievalContextService` now composes the existing
Retrieval Agent with this assembler and returns a network-free,
summarization-ready payload. It includes formatted context, structured
passages, a `CTX` citation map, retrieval configuration, and diagnostics. No
Summarization or Verification Agent is invoked in this phase. See
[COORDINATOR_RAG_FLOW.md](docs/COORDINATOR_RAG_FLOW.md).

### Legal Dataset Ingestion

Retrieval Milestones 6–7 provide a local, manifest-driven pipeline for legal
documents that the team has obtained and approved. Place source files in
`data/legal/raw/`, reviewed JSON manifests in `data/legal/metadata/`, and any
future derived local artifacts in `data/legal/processed/`. Raw and processed
contents are Git-ignored; their README files remain tracked. Reviewed
evaluation data can later be placed in `data/evaluation/`.

The manifest is a UTF-8 JSON object with `manifest_version`, `dataset_name`,
`data_classification` (`curated` or `example_only`), an optional `description`,
and a `documents` array. Every document requires a stable `document_id`, a
relative `file_name`, and one of these provenance values:

- `official_court_source`
- `official_legislation_source`
- `approved_public_source`
- `team_curated_sample`

Supported optional document metadata is `case_name`, `court`, `date`,
`citation`, `legal_category`, `document_type`, `source`, HTTP(S) `source_url`,
and `notes`. `enabled` defaults to `true`; disabled entries are reported as
skipped. Leave unavailable metadata null or omit it—never invent it. Provenance,
source, and URL metadata are retained on every chunk for later citation and
verification. Entries marked as an official or approved public source must
provide either `source` or `source_url`; team-curated samples may omit both.
See `data/legal/metadata/manifest.example.json`; it is explicitly
disabled and labelled **EXAMPLE ONLY — NOT REAL CASE LAW**.

Supported ingestion types:

| Type | Status | Behavior |
| --- | --- | --- |
| TXT | ✅ | Loads non-empty UTF-8 text, including an optional byte-order mark. |
| PDF | ✅ | Extracts embedded page text with `pypdf` in physical page order. |
| OCR | ❌ | Not supported in the current MVP. |

PDF extraction retains page records internally and chunks each usable page
through the same cleaner and chunker used for TXT. PDF chunks include
`page_number`, `page_start`, and `page_end` metadata for future citation work.
Pages without embedded text are skipped with a report warning when other pages
remain usable. A PDF with no extractable text is rejected as a possible scanned
or image-only document with the explicit message that OCR is unsupported.
CaseLens does not claim that all PDFs are extractable.

Reference a PDF from the existing manifest exactly like a TXT file:

```json
{
  "document_id": "stable-reviewed-id",
  "file_name": "reviewed-document.pdf",
  "document_type": "judgment",
  "source": "Reviewed source name",
  "source_url": "https://approved.example/document",
  "provenance": "official_court_source"
}
```

Only textual page content is requested from the PDF parser. Ingestion does not
execute embedded JavaScript, actions, attachments, macros, or other PDF
content. Path traversal remains blocked and source files must resolve beneath
the configured input directory.

Validate files and calculate chunk counts without building indexes or making
Gemini requests:

```powershell
python -m retrieval.ingestion.run `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw `
  --dry-run
```

Build the in-memory BM25 index (the default, with no embedding-provider requirement):

```powershell
python -m retrieval.ingestion.run `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw
```

Semantic indexing is opt-in. It uses `EMBEDDING_PROVIDER`, the existing
in-memory vector index, and an optional resumable checkpoint. Checkpoint keys
include provider, model, dimension, chunk ID, and formatted-content hash:

```powershell
python -m retrieval.ingestion.run `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw `
  --semantic `
  --strict `
  --embedding-checkpoint data/legal/indexes/semantic-embeddings.jsonl `
  --chunk-size 300 `
  --chunk-overlap 50
```

Add `--strict` to stop on the first invalid document; otherwise the batch
continues and reports each processed, failed, or skipped entry. The command
emits a structured JSON report with document and chunk totals plus safe
per-document errors. It does not log source text. Keep provider API keys only
in the ignored backend `.env`, never in manifests, frontend configuration,
logs, or source control. Do not commit confidential, copyrighted, or
unreviewed documents.

The team must review source rights, provenance, extraction quality, and dataset
scope before relying on an index. **CaseLens does NOT claim complete coverage
of Sri Lankan law.**

### Curated Legal Dataset Workflow

The initial academic prototype target is approximately 30–45 manually curated
documents: 10–15 each for Employment Law, Contract Law, and Property Law. This
is a planning target only; a real dataset has not yet been collected. Source
collection and review are shared by all four team members.

Approved `.txt` and `.pdf` files belong in the ignored `data/legal/raw/`
directory. Copy `data/legal/metadata/manifest.template.json` to
`data/legal/metadata/manifest.json`, then add only reviewed metadata and one of
the approved provenance labels. Official sources should retain their original
URL where available. Never invent unavailable metadata or citations, and never
classify synthetic fixtures as real legal sources.

Run the metadata summary, duplicate scan, extraction validation, and BM25 /
semantic readiness report:

```powershell
python -m retrieval.ingestion.prepare `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw
```

The preparation command is the dataset summary and validation command. It
reports counts, missing optional metadata, duplicate identifiers/files/
citations/source URLs, expected chunks, and readiness without building indexes
or calling Gemini. Run the lower-level ingestion dry-run with:

```powershell
python -m retrieval.ingestion.run `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw `
  --dry-run
```

See [LEGAL_DATASET_COLLECTION.md](docs/LEGAL_DATASET_COLLECTION.md) for the
four-member collection checklist, manifest rules, provenance guidance, and
evaluation preparation. **CaseLens does NOT claim complete coverage of Sri
Lankan law.**

## Planned architecture

The React frontend will communicate with the FastAPI backend over HTTP using
structured JSON. The backend will validate shared API contracts and, in later
phases, delegate work through a coordinator to specialized agents. Retrieval,
NLP, persistence, storage, and security remain separate modules so team members
can develop them in parallel.

## Grounded summarization contract

The Coordinator can now pass assembled RAG evidence to a provider-neutral
`SummarizationAgent`. Its deterministic prompt permits only supplied context,
requires inline `[CTX-nnn]` references, and prohibits invented authorities,
cases, pages, or citations. Both inline and declared citations are checked
against the Coordinator citation map without silently repairing failures.

Empty context bypasses generation and returns a controlled insufficient-context
answer. Weak-retrieval warnings are propagated into the prompt and must be
acknowledged in output limitations. No external summarization provider is
enabled; automated tests use network-free fakes. See
[SUMMARIZATION_AGENT.md](docs/SUMMARIZATION_AGENT.md) for the complete contract
and future Verification Agent handoff.

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
