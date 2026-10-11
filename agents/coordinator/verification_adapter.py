"""Coordinator-owned mapping to the existing deterministic Verification Agent."""

from hashlib import sha256
from typing import Iterable

from agents.summarization.schemas import EvidenceSummary, SourceReference
from agents.verification import VerificationAgent
from backend.schemas import Citation, LegalClaim, RetrievedEvidence, SourceMetadata, VerificationRequest, VerificationResponse


class VerificationMappingError(ValueError):
    """Raised when one request-local evidence identity would be ambiguous."""


class CoordinatorVerificationAdapter:
    """Invoke Verification without creating legal claims or evidence content."""

    def __init__(self, verification_agent: VerificationAgent | None = None) -> None:
        self._verification_agent = verification_agent or VerificationAgent()

    def verify(self, claims: list[LegalClaim], evidence: list[RetrievedEvidence], task_id: str) -> VerificationResponse:
        return self._verification_agent.verify(
            VerificationRequest(task_id=task_id, claims=claims, evidence=evidence)
        )

    def from_evidence_summary(
        self,
        summary: EvidenceSummary,
        *,
        task_id: str,
        source_titles: dict[str, str] | None = None,
    ) -> tuple[list[LegalClaim], list[RetrievedEvidence]]:
        """Use only exact extractive points with traceable source references.

        Points without exactly one source are deliberately omitted: combining
        sources would imply a relationship the deterministic verifier cannot
        establish.
        """

        titles = source_titles or {}
        evidence_by_id: dict[str, RetrievedEvidence] = {}
        claims: list[LegalClaim] = []
        for index, point in enumerate(summary.points, start=1):
            if point.provenance_limited or len(point.source_references) != 1:
                continue
            source = point.source_references[0]
            if source.excerpt is None or not source.excerpt.strip():
                continue
            evidence = self._evidence(source, titles)
            prior = evidence_by_id.get(evidence.id)
            if prior is not None and prior != evidence:
                raise VerificationMappingError("One evidence identifier mapped to different source material.")
            evidence_by_id[evidence.id] = evidence
            claims.append(LegalClaim(
                id=f"summary-claim-{index}",
                text=point.text,
                citations=[Citation(
                    document_id=source.document_id,
                    locator=source.locator,
                    excerpt=source.excerpt,
                )],
            ))
        return claims, list(evidence_by_id.values())

    @staticmethod
    def _evidence(source: SourceReference, titles: dict[str, str]) -> RetrievedEvidence:
        title = titles.get(source.document_id)
        if title is None:
            # The verifier requires a title to call a citation valid. Preserve
            # the source identity as an explicit fallback, not a case title.
            title = source.document_id
        digest = sha256(
            f"{source.document_id}\0{source.locator or ''}\0{source.excerpt}".encode("utf-8")
        ).hexdigest()[:20]
        return RetrievedEvidence(
            id=f"evidence-{digest}",
            source=SourceMetadata(document_id=source.document_id, title=title),
            text=source.excerpt or "",
            locator=source.locator,
        )
