"""Environment-backed settings for the Phase 1 backend foundation."""

from dataclasses import dataclass
from functools import lru_cache
from os import getenv

from retrieval.config import load_retrieval_environment

load_retrieval_environment()


@dataclass(frozen=True, slots=True)
class Settings:
    """Small settings surface shared by future backend modules."""

    app_env: str
    backend_host: str
    backend_port: int
    database_url: str
    llm_api_key: str
    secret_key: str
    supabase_jwt_issuer: str
    supabase_jwt_audience: str | None
    supabase_jwks_url: str | None
    private_upload_dir: str
    max_upload_bytes: int


@lru_cache
def get_settings() -> Settings:
    """Read application settings once from the process environment."""

    return Settings(
        app_env=getenv("APP_ENV", "development"),
        backend_host=getenv("BACKEND_HOST", "127.0.0.1"),
        backend_port=int(getenv("BACKEND_PORT", "8000")),
        database_url=getenv("DATABASE_URL", ""),
        llm_api_key=getenv("LLM_API_KEY", ""),
        secret_key=getenv("SECRET_KEY", ""),
        supabase_jwt_issuer=getenv("SUPABASE_JWT_ISSUER", ""),
        supabase_jwt_audience=getenv("SUPABASE_JWT_AUDIENCE") or None,
        supabase_jwks_url=getenv("SUPABASE_JWKS_URL") or None,
        private_upload_dir=getenv("PRIVATE_UPLOAD_DIR", "data/private_uploads"),
        max_upload_bytes=int(getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
    )
