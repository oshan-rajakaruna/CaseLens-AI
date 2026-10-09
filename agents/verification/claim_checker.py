"""Conservative, deterministic checks of claim wording against cited passages."""

from collections.abc import Sequence
import re

from agents.verification.citation_validator import validate_citations
from backend.schemas import (
    CitationValidationResult,
    ClaimVerificationResult,
    LegalClaim,
    RetrievedEvidence,
)


_COPULAS = {"is", "are", "was", "were"}


def check_claim(
    claim: LegalClaim,
    evidence: Sequence[RetrievedEvidence],
    citation_results: Sequence[CitationValidationResult],
) -> ClaimVerificationResult:
    """Compare a claim with passages identified by independently checked citations."""

    checks = list(citation_results)
    expected = validate_citations(claim, evidence)
    if len(checks) != len(expected) or any(
        (given.citation, given.status, given.evidence_id, given.source)
        != (actual.citation, actual.status, actual.evidence_id, actual.source)
        for given, actual in zip(checks, expected)
    ):
        return ClaimVerificationResult(
            claim=claim,
            status="uncertain",
            citation_results=checks,
            issues=["Citation checks do not match the supplied claim and evidence."],
        )

    passages_by_id = {passage.id: passage for passage in evidence}
    if len(passages_by_id) != len(evidence):
        return ClaimVerificationResult(
            claim=claim,
            status="uncertain",
            citation_results=checks,
            issues=["Evidence IDs are not unique."],
        )

    evidence_ids = list(
        dict.fromkeys(check.evidence_id for check in checks if check.status == "valid")
    )
    if any(check.status != "valid" for check in checks):
        return ClaimVerificationResult(
            claim=claim,
            status="uncertain",
            evidence_ids=evidence_ids,
            citation_results=checks,
            issues=["A citation is missing, invalid, or uncertain."],
        )

    passages = [passages_by_id[evidence_id] for evidence_id in evidence_ids]
    statements = _claim_parts(claim.text)
    passage_texts = {_normalize(passage.text) for passage in passages}
    passage_statements = {
        statement for passage in passages for statement in _claim_parts(passage.text)
    }
    full_claim_match = _normalize(claim.text) in passage_texts
    matched = [full_claim_match or statement in passage_statements for statement in statements]
    contradicted = [
        (opposite := _direct_opposite(statement)) is not None and opposite in passage_statements
        for statement in statements
    ]

    if any(support and contradiction for support, contradiction in zip(matched, contradicted)):
        status = "uncertain"
        issues = ["Cited passages contain conflicting statements."]
    elif any(contradicted):
        status = "unsupported"
        issues = ["A cited passage directly negates a claim statement."]
    elif all(matched):
        status = "supported"
        issues = []
    elif any(matched):
        status = "partially_supported"
        issues = ["Only some claim statements match cited passages."]
    else:
        status = "uncertain"
        issues = ["Cited passages do not directly establish or negate the claim."]

    return ClaimVerificationResult(
        claim=claim,
        status=status,
        evidence_ids=evidence_ids,
        citation_results=checks,
        issues=issues,
    )


def _claim_parts(text: str) -> list[str]:
    statement = _normalize(text)
    parts = re.split(r"\s+and\s+", statement)
    if len(parts) > 1 and all(_direct_opposite(part) is not None for part in parts):
        return parts
    return [statement]


def _direct_opposite(statement: str) -> str | None:
    words = statement.split()
    verbs = [index for index, word in enumerate(words) if word.lower() in _COPULAS]
    if len(verbs) != 1 or any(word.lower() in {"and", "or", "that", "if"} for word in words):
        return None
    index = verbs[0]
    if index == 0 or index == len(words) - 1:
        return None
    if any(word.lower() == "not" for word in words[index + 2 :]):
        return None
    if words[index + 1].lower() == "not":
        return " ".join(words[: index + 1] + words[index + 2 :])
    return " ".join(words[: index + 1] + ["not"] + words[index + 1 :])


def _normalize(text: str) -> str:
    return " ".join(text.split()).removesuffix(".")
