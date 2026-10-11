"""Temporary fail-closed access controls for development-only endpoints."""

from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from backend.config import Settings, get_settings

_ACCESS_DENIED = "Development summarization access denied"


def require_dev_summarization_access(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[
        str | None,
        Header(alias="X-CaseLens-Dev-Token"),
    ] = None,
) -> None:
    """Allow the temporary endpoint only with explicit local-dev configuration."""

    allowed = (
        settings.app_env == "development"
        and settings.enable_dev_summarization_api
        and bool(settings.dev_summarization_api_token)
        and token is not None
        and compare_digest(token, settings.dev_summarization_api_token)
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_ACCESS_DENIED,
        )
