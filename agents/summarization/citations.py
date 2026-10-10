"""Validation of prompt-local CTX citations in future generated text."""

from collections.abc import Iterable
import re

from pydantic import BaseModel, Field

_VALID_CITATION = re.compile(r"\[(CTX-\d{3,})\]")
_BRACKETED_CTX_LIKE = re.compile(r"\[[^\]\n]*CTX[^\]\n]*\]", re.IGNORECASE)
_BARE_CTX_LIKE = re.compile(r"(?<!\[)\bCTX-[A-Za-z0-9_-]+\b(?!\])", re.IGNORECASE)


class CitationValidationResult(BaseModel):
    """Structured citation validation without altering generated text."""

    cited_context_ids: list[str] = Field(default_factory=list)
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    malformed_citations: list[str] = Field(default_factory=list)
    has_citations: bool
    all_citations_valid: bool


def validate_context_citations(
    generated_text: str,
    allowed_context_ids: Iterable[str],
) -> CitationValidationResult:
    """Classify CTX references while preserving occurrence order."""

    if not isinstance(generated_text, str):
        raise TypeError("generated_text must be a string")
    allowed = set(allowed_context_ids)
    cited = _VALID_CITATION.findall(generated_text)
    valid = list(dict.fromkeys(item for item in cited if item in allowed))
    invalid = list(dict.fromkeys(item for item in cited if item not in allowed))
    exact_tokens = {match.group(0) for match in _VALID_CITATION.finditer(generated_text)}
    malformed = [
        token
        for token in _BRACKETED_CTX_LIKE.findall(generated_text)
        if token not in exact_tokens
    ]
    malformed.extend(_BARE_CTX_LIKE.findall(generated_text))
    malformed = list(dict.fromkeys(malformed))
    return CitationValidationResult(
        cited_context_ids=cited,
        valid_citations=valid,
        invalid_citations=invalid,
        malformed_citations=malformed,
        has_citations=bool(cited or malformed),
        all_citations_valid=not invalid and not malformed,
    )
