"""FastAPI entry point for the CaseLens backend foundation."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.router import api_router
from backend.config import get_settings
from backend.schemas import HealthResponse, RootResponse

get_settings()

app = FastAPI(
    title="CaseLens Legal Case Analysis API",
    description="Phase 1 API foundation for the CaseLens academic project.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(api_router)


@app.get("/", response_model=RootResponse, tags=["system"])
async def root() -> RootResponse:
    """Return basic service information without implying feature readiness."""

    return RootResponse(
        name="CaseLens",
        phase="Phase 1 Foundation",
        message="CaseLens backend foundation is running.",
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report whether the HTTP application is available."""

    return HealthResponse(status="ok", service="caselens-backend")


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def api_health() -> HealthResponse:
    return await health()
