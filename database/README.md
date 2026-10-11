# CaseLens persistence foundation

Milestone 6C-1 uses SQLAlchemy 2 with PostgreSQL through psycopg 3. Alembic owns
production schema changes. SQLite is supported only for isolated offline tests;
it is not the production database or an authorization substitute.

## Schema

- `cases` stores the opaque trusted owner subject, case metadata, lifecycle, and
  timestamps.
- `case_documents` belongs to a case and stores only a safe filename, media type,
  relative/opaque private storage key, and source metadata. It never stores bytes
  or trusts an absolute user path.
- `research_tasks` belongs to a case and owner. Its primary key is durable task
  identity; `(owner_subject, request_fingerprint)` provides an idempotency guard.
  Stage states and error categories are sanitized values, not error messages.
- `legal_reports` belongs to exactly one case and one task. The complete validated
  `LegalResearchReport` is stored as PostgreSQL JSONB, preserving CTX IDs,
  document/chunk/page provenance, verifier findings, disclaimers, and optional
  unverified comparison drafts. A task can have only one report.
- `agent_logs` belongs to a task and contains only allowlisted operational scalar
  metadata. Raw evidence, prompts, credentials, and arbitrary diagnostic text are
  outside this contract.

Foreign keys use `RESTRICT`; deleting parent data is deliberately not part of this
milestone. Application checks ensure redundant owner subjects agree across related
records. Database constraints provide a second boundary for identity, lifecycle,
and one-report-per-task rules.

## Ownership boundary

Every service operation requires `TrustedPrincipal(subject=...)`. The subject must
come from a future server-side authentication dependency after token/session
validation. Never construct it from a browser-supplied owner ID. A case, task, or
report ID is an identifier only and is never proof of access.

No HTTP routes are provided. Full reports and document metadata are sensitive and
must eventually be returned only by authenticated, owner-scoped APIs. The current
development token dependency is not a production identity provider.

## Internal interface

Construct an engine and inject its session factory:

```python
from database import PersistenceService, TrustedPrincipal, build_engine, build_session_factory

engine = build_engine(database_url)
persistence = PersistenceService(build_session_factory(engine))
principal = TrustedPrincipal(subject=verified_auth_subject)
```

`PersistenceService` exposes `create_case`, `get_case`, `add_document`,
`create_task`, `get_task`, `update_task`, `save_report`, `get_report`,
`find_reports_by_case`, `find_report_by_task`, `record_agent_event`, and
`list_agent_events`. Each write owns one transaction. `CaseLensRepository` is the
session-scoped form for a future Coordinator adapter that needs several related
writes in one transaction.

The future Coordinator integration should create a durable queued task before
execution, record sanitized stage events/status transitions, save the final
structured report once, then mark the task completed/partial/failed. Restart and
cancellation recovery remain pending.

## Installation and migration

From PowerShell in the repository root, install the declared dependencies into the
project environment:

```powershell
python -m pip install -r requirements.txt
```

Create a PostgreSQL database and user outside this repository, then set the URL for
the current shell. Do not commit credentials:

```powershell
$env:DATABASE_URL = 'postgresql+psycopg://caselens_app:<password>@localhost:5432/caselens'
python -m alembic upgrade head
python -m alembic current
```

Alembic refuses to run without an explicit `DATABASE_URL`. Application startup does
not call `create_all`, drop tables, or run migrations. `downgrade` exists for
operator-controlled migration management and is never automatic.

For offline verification, tests create a unique SQLite file under pytest's temporary
directory and explicitly create the schema there. They never read `DATABASE_URL` or
connect to PostgreSQL:

```powershell
python -m pytest tests/unit/database/test_persistence.py -q
```
