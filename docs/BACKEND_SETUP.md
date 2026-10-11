# CaseLens backend setup

The FastAPI entry point is `backend.main:app`. The backend uses FastAPI,
SQLAlchemy 2, Alembic, psycopg 3, and a verified Supabase access-token boundary.

Configure `DATABASE_URL` for Supabase PostgreSQL, `SUPABASE_JWT_ISSUER`, and,
where required by the project token configuration, `SUPABASE_JWT_AUDIENCE` and
`SUPABASE_JWKS_URL`. Configure `PRIVATE_UPLOAD_DIR` outside static/public
directories before document upload support is enabled. Local private storage is
development-only and is not a production document-storage strategy.

Run migrations after configuring PostgreSQL:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Run locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Swagger is available at `/docs`; health endpoints are `/health` and
`/api/health`. Private endpoints require a valid Supabase bearer token. Tokens
are signature-verified against configured trusted JWKS, and issuer, expiration,
subject, audience (when configured), and RS256/ES256 algorithms are checked.

Run tests on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_cache\caselens-test-temp\full_run
```

PostgreSQL connectivity and Supabase JWKS verification require deliberately
configured external infrastructure and are not claimed by SQLite/unit tests.

## Implemented private routes

`POST /api/cases/{case_id}/documents`, `GET /api/cases/{case_id}/documents`,
`GET /api/cases/{case_id}/documents/{document_id}`, and `DELETE
/api/cases/{case_id}/documents/{document_id}` require a verified bearer token
and case ownership. Uploads accept only UTF-8 TXT, valid PDF, and valid DOCX;
they are streamed in bounded chunks to a server-generated identifier below
`PRIVATE_UPLOAD_DIR`. Original names are metadata only and cannot select a
filesystem path. Failed writes are removed. Local private storage is for
development only and must be replaced by managed encrypted object storage for
production.

Extraction uses the existing local extraction module. A PDF with no readable
text is recorded as `no_text`; OCR is not performed. Retrieval, research-job,
report, and Coordinator HTTP endpoints require a configured prepared corpus
provider and remain unimplemented at the API boundary.
