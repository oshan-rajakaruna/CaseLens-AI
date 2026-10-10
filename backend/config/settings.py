"""Environment-backed settings for the Phase 1 backend foundation."""

from dataclasses import dataclass
from functools import lru_cache
from os import getenv

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    """Small settings surface shared by future backend modules."""

    app_env: str
    backend_host: str
    backend_port: int
    database_url: str
    llm_api_key: str
    gemini_model: str
    gemini_max_output_tokens: int
    secret_key: str


@lru_cache
def get_settings() -> Settings:
    """Read application settings once from the process environment."""

    return Settings(
        app_env=getenv("APP_ENV", "development"),
        backend_host=getenv("BACKEND_HOST", "127.0.0.1"),
        backend_port=int(getenv("BACKEND_PORT", "8000")),
        database_url=getenv("DATABASE_URL", ""),
        llm_api_key=getenv("LLM_API_KEY", ""),
        gemini_model=getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_max_output_tokens=int(getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
        secret_key=getenv("SECRET_KEY", ""),
    )
