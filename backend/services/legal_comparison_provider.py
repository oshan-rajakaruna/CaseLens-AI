"""Provider-neutral boundary for optional legal-comparison drafts."""

from typing import Protocol

from backend.schemas.legal_comparison import (
    LegalComparisonGenerationRequest,
    LegalComparisonProviderOutput,
)


class LegalComparisonProvider(Protocol):
    """Generate one structured draft for an explicitly selected source pair."""

    provider_name: str
    model_name: str | None

    def generate(
        self,
        request: LegalComparisonGenerationRequest,
    ) -> LegalComparisonProviderOutput:
        """Return an unverified comparative explanation for the supplied pair."""
        ...
