"""Explicitly enabled local-development session login."""

from secrets import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.auth.session import (
    COOKIE_NAME,
    COOKIE_PATH,
    SESSION_MAX_AGE_SECONDS,
    development_auth_ready,
    issue_session,
    require_local_origin,
)
from backend.config import Settings, get_settings
from backend.schemas.auth import DevelopmentLoginRequest, DevelopmentLoginResponse

router = APIRouter(prefix="/auth", tags=["development authentication"])


@router.post("/login", response_model=DevelopmentLoginResponse)
def development_login(
    request: DevelopmentLoginRequest,
    response: Response,
    _origin: Annotated[None, Depends(require_local_origin)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DevelopmentLoginResponse:
    if not development_auth_ready(settings):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not compare_digest(request.password, settings.dev_auth_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    session, csrf_token = issue_session(settings)
    response.set_cookie(
        COOKIE_NAME,
        session,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=False,
        samesite="strict",
        path=COOKIE_PATH,
    )
    response.headers["Cache-Control"] = "no-store"
    return DevelopmentLoginResponse(csrf_token=csrf_token)
