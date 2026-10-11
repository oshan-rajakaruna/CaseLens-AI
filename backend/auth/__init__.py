"""Verified authentication dependencies."""
from backend.auth.dependencies import AuthenticatedPrincipal, get_current_principal
__all__ = ["AuthenticatedPrincipal", "get_current_principal"]
