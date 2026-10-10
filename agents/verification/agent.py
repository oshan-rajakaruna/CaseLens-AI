"""Deterministic orchestration of CaseLens verification components."""

from agents.verification.citation_validator import validate_citations
from agents.verification.claim_checker import check_claim
from agents.verification.responsible_ai import generate_warnings
from backend.schemas import VerificationRequest, VerificationResponse
from backend.schemas.verification import determine_overall_status


class VerificationAgent:
    """Run citation, claim, and warning checks for one verification request."""

    def verify(self, request: VerificationRequest) -> VerificationResponse:
        claim_results = []
        for claim in request.claims:
            citation_results = validate_citations(claim, request.evidence)
            claim_results.append(check_claim(claim, request.evidence, citation_results))

        warnings = generate_warnings(claim_results)
        overall_status = determine_overall_status(claim_results)

        return VerificationResponse(
            task_id=request.task_id,
            verified=overall_status == "supported",
            overall_status=overall_status,
            claim_results=claim_results,
            citations=[citation for claim in request.claims for citation in claim.citations],
            warnings=warnings,
            notes=[
                f"{result.claim.id}: {issue}"
                for result in claim_results
                for issue in result.issues
            ],
        )
