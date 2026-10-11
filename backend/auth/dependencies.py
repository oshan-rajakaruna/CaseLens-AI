"""Verified Supabase bearer-token authentication boundary."""
from dataclasses import dataclass
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from backend.config import get_settings

_bearer = HTTPBearer(auto_error=False)

@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject: str

def get_current_principal(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> AuthenticatedPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication is required")
    settings = get_settings()
    if not settings.supabase_jwt_issuer:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    try:
        jwks_url = settings.supabase_jwks_url or settings.supabase_jwt_issuer.rstrip("/") + "/.well-known/jwks.json"
        key = jwt.PyJWKClient(jwks_url).get_signing_key_from_jwt(credentials.credentials).key
        claims = jwt.decode(
            credentials.credentials, key, algorithms=["RS256", "ES256"],
            issuer=settings.supabase_jwt_issuer,
            audience=settings.supabase_jwt_audience,
            options={"require": ["exp", "sub", "iss"], "verify_aud": settings.supabase_jwt_audience is not None},
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid access token")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise HTTPException(status_code=401, detail="Invalid access token")
    return AuthenticatedPrincipal(subject=subject)
