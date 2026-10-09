"""Tests for conservative, text-based claim decisions."""

from agents.verification.citation_validator import validate_citations
from agents.verification.claim_checker import check_claim
from backend.schemas import (
    Citation,
    CitationValidationResult,
    ClaimVerificationResult,
    LegalClaim,
    RetrievedEvidence,
    SourceMetadata,
)


def _passage(
    text: str,
    *,
    evidence_id: str = "passage-1",
    locator: str = "p. 3",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        id=evidence_id,
        source=SourceMetadata(document_id="doc-1", title="Decision"),
        locator=locator,
        text=text,
    )


def _claim(text: str, *citations: Citation) -> LegalClaim:
    return LegalClaim(id="claim-1", text=text, citations=list(citations))


def _check(claim: LegalClaim, evidence: list[RetrievedEvidence]) -> ClaimVerificationResult:
    return check_claim(claim, evidence, validate_citations(claim, evidence))


def test_direct_statement_match_is_supported() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3")
    claim = _claim("The appeal was allowed.", citation)

    result = _check(claim, [_passage("The appeal was allowed.")])

    assert isinstance(result, ClaimVerificationResult)
    assert result.status == "supported"
    assert result.evidence_ids == ["passage-1"]
    assert result.citation_results[0].status == "valid"
    assert result.issue_codes == []


def test_direct_negation_is_unsupported() -> None:
    claim = _claim("The appeal was allowed.", Citation(document_id="doc-1", locator="p. 3"))

    result = _check(claim, [_passage("The appeal was not allowed.")])

    assert result.status == "unsupported"
    assert result.evidence_ids == ["passage-1"]
    assert result.issue_codes == []


def test_one_of_two_simple_statements_is_partially_supported() -> None:
    claim = _claim(
        "The appeal was allowed and costs were awarded.",
        Citation(document_id="doc-1", locator="p. 3"),
    )

    result = _check(claim, [_passage("The appeal was allowed.")])

    assert result.status == "partially_supported"
    assert result.evidence_ids == ["passage-1"]


def test_full_compound_statement_match_is_supported() -> None:
    statement = "The appeal was allowed and costs were awarded."
    claim = _claim(statement, Citation(document_id="doc-1", locator="p. 3"))

    result = _check(claim, [_passage(statement)])

    assert result.status == "supported"


def test_valid_citation_with_unrelated_text_is_uncertain() -> None:
    claim = _claim("The appeal was allowed.", Citation(document_id="doc-1", locator="p. 3"))

    result = _check(claim, [_passage("The court considered the appeal.")])

    assert result.status == "uncertain"
    assert result.citation_results[0].status == "valid"


def test_invalid_citation_cannot_borrow_uncited_evidence() -> None:
    claim = _claim("The appeal was allowed.", Citation(document_id="doc-unknown"))

    result = _check(claim, [_passage("The appeal was allowed.")])

    assert result.status == "uncertain"
    assert result.citation_results[0].status == "invalid"
    assert result.evidence_ids == []


def test_missing_evidence_and_missing_citations_remain_uncertain() -> None:
    cited = _claim("The appeal was allowed.", Citation(document_id="doc-1"))
    uncited = _claim("The appeal was allowed.")

    no_evidence = _check(cited, [])
    no_citation = _check(uncited, [_passage("The appeal was allowed.")])

    assert no_evidence.status == "uncertain"
    assert no_evidence.missing_evidence is True
    assert no_citation.status == "uncertain"
    assert no_citation.missing_citations is True


def test_ambiguous_citation_match_remains_uncertain() -> None:
    claim = _claim("The appeal was allowed.", Citation(document_id="doc-1", locator="p. 3"))
    evidence = [
        _passage("The appeal was allowed."),
        _passage("The appeal was allowed.", evidence_id="passage-2"),
    ]

    result = _check(claim, evidence)

    assert result.status == "uncertain"
    assert result.citation_results[0].status == "uncertain"


def test_conflicting_cited_passages_remain_uncertain() -> None:
    claim = _claim(
        "The appeal was allowed.",
        Citation(document_id="doc-1", locator="p. 3"),
        Citation(document_id="doc-1", locator="p. 4"),
    )
    evidence = [
        _passage("The appeal was allowed."),
        _passage("The appeal was not allowed.", evidence_id="passage-2", locator="p. 4"),
    ]

    result = _check(claim, evidence)

    assert result.status == "uncertain"
    assert result.evidence_ids == ["passage-1", "passage-2"]
    assert result.issues == ["Cited passages contain conflicting statements."]
    assert result.issue_codes == ["conflicting_evidence"]


def test_conflicting_compound_passages_remain_uncertain() -> None:
    claim = _claim(
        "The appeal was allowed and costs were awarded.",
        Citation(document_id="doc-1", locator="p. 3"),
        Citation(document_id="doc-1", locator="p. 4"),
    )
    evidence = [
        _passage("The appeal was allowed and costs were awarded."),
        _passage(
            "The appeal was allowed and costs were not awarded.",
            evidence_id="passage-2",
            locator="p. 4",
        ),
    ]

    result = _check(claim, evidence)

    assert result.status == "uncertain"
    assert result.issue_codes == ["conflicting_evidence"]


def test_inconsistent_citation_result_is_not_trusted() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3")
    claim = _claim("The appeal was allowed.", citation)
    evidence = [_passage("The appeal was allowed.")]
    wrong_record = CitationValidationResult(
        citation=citation,
        status="valid",
        evidence_id="passage-not-supplied",
        source=evidence[0].source,
    )

    result = check_claim(claim, evidence, [wrong_record])

    assert result.status == "uncertain"
    assert result.evidence_ids == []
