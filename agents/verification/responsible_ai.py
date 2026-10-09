"""Deterministic warnings derived from structured verification results."""

from collections.abc import Sequence

from backend.schemas import ClaimVerificationResult, ResponsibleAIWarning


_WARNING_SPECS = (
    (
        "unsupported_claim",
        "critical",
        "Verification marked these claims unsupported. Have a qualified legal professional review them.",
    ),
    (
        "partially_supported_claim",
        "caution",
        "Only part of these claims is supported by the reviewed evidence.",
    ),
    (
        "uncertain_claim",
        "caution",
        "Verification could not determine whether these claims are supported.",
    ),
    (
        "missing_citation",
        "caution",
        "These claims are missing citations.",
    ),
    (
        "invalid_citation",
        "caution",
        "One or more citations for these claims were marked invalid.",
    ),
    (
        "conflicting_evidence",
        "critical",
        "Cited passages for these claims contain conflicting statements. Seek qualified legal review.",
    ),
)


def generate_warnings(
    claim_results: Sequence[ClaimVerificationResult],
) -> list[ResponsibleAIWarning]:
    """Summarize only conditions recorded in claim and citation results."""

    affected: dict[str, list[str]] = {code: [] for code, _, _ in _WARNING_SPECS}
    for result in claim_results:
        claim_id = result.claim.id
        if result.status == "unsupported":
            affected["unsupported_claim"].append(claim_id)
        elif result.status == "partially_supported":
            affected["partially_supported_claim"].append(claim_id)
        elif result.status == "uncertain":
            affected["uncertain_claim"].append(claim_id)

        if result.missing_citations:
            affected["missing_citation"].append(claim_id)
        if any(check.status == "invalid" for check in result.citation_results):
            affected["invalid_citation"].append(claim_id)
        if "conflicting_evidence" in result.issue_codes:
            affected["conflicting_evidence"].append(claim_id)

    warnings = [
        ResponsibleAIWarning(
            code="legal_information_only",
            message=(
                "CaseLens provides legal information support and does not guarantee legal "
                "outcomes. Consult a qualified legal professional before relying on this "
                "analysis for a legal decision."
            ),
            severity="info",
        )
    ]
    warnings.extend(
        ResponsibleAIWarning(
            code=code,
            severity=severity,
            message=message,
            claim_ids=list(dict.fromkeys(affected[code])),
        )
        for code, severity, message in _WARNING_SPECS
        if affected[code]
    )
    return warnings
