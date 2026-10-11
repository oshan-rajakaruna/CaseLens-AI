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
    enable_dev_summarization_api: bool
    dev_summarization_api_token: str
    enable_dev_case_api: bool
    dev_auth_operator_subject: str
    dev_auth_password: str
    dev_auth_allowed_origin: str


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
        enable_dev_summarization_api=(
            getenv("ENABLE_DEV_SUMMARIZATION_API", "false") == "true"
        ),
        dev_summarization_api_token=getenv("DEV_SUMMARIZATION_API_TOKEN", ""),
        enable_dev_case_api=(getenv("ENABLE_DEV_CASE_API", "false") == "true"),
        dev_auth_operator_subject=getenv("DEV_AUTH_OPERATOR_SUBJECT", ""),
        dev_auth_password=getenv("DEV_AUTH_PASSWORD", ""),
        dev_auth_allowed_origin=getenv("DEV_AUTH_ALLOWED_ORIGIN", ""),
    )
