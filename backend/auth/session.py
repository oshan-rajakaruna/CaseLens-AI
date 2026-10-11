"""Signed, development-only localhost sessions and CSRF enforcement."""

from hashlib import sha256
from secrets import compare_digest, token_urlsafe
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, Header, HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from backend.config import Settings, get_settings
from database.schemas import TrustedPrincipal

COOKIE_NAME = "caselens_dev_session"
COOKIE_PATH = "/api"
SESSION_MAX_AGE_SECONDS = 3600
_SALT = "caselens-development-case-session-v1"
_UNAUTHENTICATED = "Authentication required"


def development_auth_ready(settings: Settings) -> bool:
    return (
        settings.app_env == "development"
        and settings.enable_dev_case_api
        and len(settings.secret_key) >= 32
        and bool(settings.dev_auth_operator_subject)
        and bool(settings.dev_auth_password)
        and _is_local_origin(settings.dev_auth_allowed_origin)
    )


def issue_session(settings: Settings) -> tuple[str, str]:
    csrf_token = token_urlsafe(32)
    signed = _serializer(settings).dumps(
        {"sub": settings.dev_auth_operator_subject, "csrf": csrf_token}
    )
    return signed, csrf_token


def require_local_origin(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    origin: Annotated[str | None, Header(alias="Origin")] = None,
) -> None:
    del request
    if not development_auth_ready(settings):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if origin is None or not compare_digest(origin, settings.dev_auth_allowed_origin):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origin denied")


def require_authenticated_principal(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TrustedPrincipal:
    session = _read_session(request, settings)
    return TrustedPrincipal(subject=session["sub"])


def require_csrf_principal(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    origin: Annotated[str | None, Header(alias="Origin")] = None,
) -> TrustedPrincipal:
    session = _read_session(request, settings)
    valid = (
        origin is not None
        and compare_digest(origin, settings.dev_auth_allowed_origin)
        and csrf_token is not None
        and compare_digest(csrf_token, session["csrf"])
    )
    if not valid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")
    return TrustedPrincipal(subject=session["sub"])


def _read_session(request: Request, settings: Settings) -> dict[str, str]:
    if not development_auth_ready(settings):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHENTICATED)
    value = request.cookies.get(COOKIE_NAME)
    if not value:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHENTICATED)
    try:
        payload = _serializer(settings).loads(value, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHENTICATED) from None
    if (
        not isinstance(payload, dict)
        or payload.get("sub") != settings.dev_auth_operator_subject
        or not isinstance(payload.get("csrf"), str)
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_UNAUTHENTICATED)
    return {"sub": payload["sub"], "csrf": payload["csrf"]}


def _serializer(settings: Settings) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(
        settings.secret_key,
        salt=_SALT,
        signer_kwargs={"digest_method": sha256},
    )


def _is_local_origin(origin: str) -> bool:
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port is not None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False
