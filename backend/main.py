"""FastAPI entry point for the CaseLens backend foundation."""

from fastapi import FastAPI

from backend.api.router import api_router
from backend.schemas import HealthResponse, RootResponse

app = FastAPI(
    title="CaseLens Legal Case Analysis API",
    description="Phase 1 API foundation for the CaseLens academic project.",
    version="0.1.0",
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
