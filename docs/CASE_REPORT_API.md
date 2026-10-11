# Authenticated case and report API foundation

Milestone 6C-2 provides a localhost-only demonstration session and owner-scoped
case/report reads. It does not provide production authentication, uploads, research
execution, or frontend integration.

## Development authentication

The development login is disabled unless every condition is satisfied:

- `APP_ENV=development`
- `ENABLE_DEV_CASE_API=true`
- `SECRET_KEY` contains at least 32 characters
- `DEV_AUTH_OPERATOR_SUBJECT` is configured server-side
- `DEV_AUTH_PASSWORD` is configured server-side
- `DEV_AUTH_ALLOWED_ORIGIN` is an explicit `http://localhost:<port>` or
  `http://127.0.0.1:<port>` origin

`POST /api/auth/login` accepts only the configured password. It creates an HttpOnly,
SameSite=Strict signed cookie scoped to `/api` and returns a per-session CSRF token.
The password, signing secret, and owner subject are never returned. Login and case
creation require an exact allowed `Origin`; case creation additionally requires the
CSRF token in `X-CSRF-Token`.

The cookie is deliberately not marked `Secure` because this mechanism is limited to
local HTTP demonstration. Production mode always disables it. Deployment requires a
real identity provider, HTTPS-only secure cookies, production session lifecycle,
and authorization policy review.

The old summarization development token is independent and cannot authenticate case
or report requests.

## Endpoints

- `POST /api/auth/login`
- `POST /api/cases`
- `GET /api/cases?offset=0&limit=25`
- `GET /api/cases/{case_id}`
- `GET /api/cases/{case_id}/reports`
- `GET /api/reports/{report_id}`

Every case/report endpoint derives `TrustedPrincipal` from the signed session.
Requests cannot submit an owner. Foreign and missing resources both return 404.
Report endpoints return the persisted structured `LegalResearchReport`, preserving
CTX/provenance, verifier outcomes, comparison review labels, and the human-review
disclaimer. They never return `ResearchWorkflowResult`, task logs, SQL details, or
database connection information.

## Local setup and manual request

Install declared dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set temporary development values in one PowerShell process. Use your own generated
values; never commit them:

```powershell
$env:APP_ENV = 'development'
$env:ENABLE_DEV_CASE_API = 'true'
$env:DEV_AUTH_OPERATOR_SUBJECT = 'local-demo-operator'
$env:DEV_AUTH_PASSWORD = '<choose-a-local-demo-password>'
$env:DEV_AUTH_ALLOWED_ORIGIN = 'http://127.0.0.1:5173'
$env:SECRET_KEY = '<generate-a-random-value-of-at-least-32-characters>'
$env:DATABASE_URL = 'postgresql+psycopg://caselens_app:<password>@localhost:5433/caselens'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell process with matching non-secret variables, log in while
retaining the cookie, then create/list/read one case:

```powershell
$origin = 'http://127.0.0.1:5173'
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$loginBody = @{ password = '<your-local-demo-password>' } | ConvertTo-Json
$login = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/auth/login' -Headers @{ Origin = $origin } -ContentType 'application/json' -Body $loginBody -WebSession $session
$caseBody = @{ title = 'Local demonstration case'; research_question = 'What law applies?' } | ConvertTo-Json
$created = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/cases' -Headers @{ Origin = $origin; 'X-CSRF-Token' = $login.csrf_token } -ContentType 'application/json' -Body $caseBody -WebSession $session
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/api/cases' -WebSession $session
Invoke-RestMethod -Method Get -Uri ("http://127.0.0.1:8000/api/cases/{0}" -f $created.id) -WebSession $session
```

Verify unauthenticated denial:

```powershell
try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/cases' } catch { $_.Exception.Response.StatusCode.value__ }
```

## Tests

Offline SQLite API and persistence tests:

```powershell
python -m pytest tests/unit/test_case_report_api.py tests/unit/database/test_persistence.py tests/unit/coordinator -q
```

The PostgreSQL test is opt-in and refuses database names without `test`. Create and
migrate a separate disposable test database; never point it at `caselens`:

```powershell
$env:CASELENS_TEST_DATABASE_URL = 'postgresql+psycopg://caselens_test_app:<password>@localhost:5433/caselens_test'
$env:DATABASE_URL = $env:CASELENS_TEST_DATABASE_URL
python -m alembic upgrade head
$env:RUN_POSTGRES_PERSISTENCE_TESTS = 'true'
python -m pytest tests/integration/test_postgresql_persistence.py -q
```

The integration test uses a unique identity, verifies an actual insert/read, rolls
back its transaction, and confirms no record remains. It never migrates, truncates,
drops, or otherwise resets the database itself.
