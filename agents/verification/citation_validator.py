"""Deterministic checks of citations against supplied retrieved passages."""

from collections.abc import Sequence

from backend.schemas import Citation, CitationValidationResult, LegalClaim, RetrievedEvidence


def validate_citations(
    claim: LegalClaim, evidence: Sequence[RetrievedEvidence]
) -> list[CitationValidationResult]:
    """Check source references only; claim support requires a separate decision."""

    if not claim.citations:
        return [
            CitationValidationResult(
                citation=None, status="missing", issues=["Claim has no citation."]
            )
        ]

    return [_validate_citation(citation, evidence) for citation in claim.citations]


def _validate_citation(
    citation: Citation, evidence: Sequence[RetrievedEvidence]
) -> CitationValidationResult:
    if not citation.document_id.strip():
        return CitationValidationResult(
            citation=citation, status="invalid", issues=["Citation has no document ID."]
        )
    if citation.locator is not None and not citation.locator.strip():
        return CitationValidationResult(
            citation=citation, status="invalid", issues=["Citation locator is blank."]
        )
    if citation.excerpt is not None and not citation.excerpt.strip():
        return CitationValidationResult(
            citation=citation, status="invalid", issues=["Citation excerpt is blank."]
        )
    if not evidence:
        return CitationValidationResult(
            citation=citation, status="uncertain", issues=["No evidence was supplied."]
        )

    document_matches = [
        passage for passage in evidence if passage.source.document_id == citation.document_id
    ]
    if not document_matches:
        return CitationValidationResult(
            citation=citation,
            status="invalid",
            issues=["Document ID was not found in the supplied evidence."],
        )

    matches = document_matches
    if citation.locator is not None:
        locator = citation.locator.strip()
        matches = [
            passage
            for passage in document_matches
            if passage.locator is not None and passage.locator.strip() == locator
        ]
        if any(passage.locator is None or not passage.locator.strip() for passage in document_matches):
            return CitationValidationResult(
                citation=citation,
                status="uncertain",
                issues=["A matching document has no locator metadata."],
            )
        if not matches:
            return CitationValidationResult(
                citation=citation,
                status="invalid",
                issues=["Citation locator does not match the supplied evidence."],
            )

    if len(matches) != 1:
        return CitationValidationResult(
            citation=citation,
            status="uncertain",
            issues=["More than one passage could match this citation."],
        )

    passage = matches[0]
    if passage.source.title is None:
        return CitationValidationResult(
            citation=citation,
            status="uncertain",
            evidence_id=passage.id,
            source=passage.source,
            issues=["Matched source has no title metadata."],
        )
    if citation.excerpt is not None and _normalize_space(citation.excerpt) not in _normalize_space(
        passage.text
    ):
        return CitationValidationResult(
            citation=citation,
            status="uncertain",
            evidence_id=passage.id,
            source=passage.source,
            issues=["Citation excerpt was not found in the supplied passage."],
        )

    return CitationValidationResult(
        citation=citation,
        status="valid",
        evidence_id=passage.id,
        source=passage.source,
    )


def _normalize_space(text: str) -> str:
    return " ".join(text.split())
