"""Tests for deterministic source and locator checks."""

from agents.verification.citation_validator import validate_citations
from backend.schemas import Citation, CitationValidationResult, LegalClaim, RetrievedEvidence, SourceMetadata


def _passage(
    *,
    evidence_id: str = "passage-1",
    document_id: str = "doc-1",
    title: str | None = "Decision",
    locator: str | None = "p. 3",
    text: str = "The court considered the record.",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        id=evidence_id,
        source=SourceMetadata(document_id=document_id, title=title),
        locator=locator,
        text=text,
    )


def _claim(*citations: Citation) -> LegalClaim:
    return LegalClaim(id="claim-1", text="A generated legal assertion.", citations=list(citations))


def test_valid_citation_matches_document_locator_and_excerpt() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3", excerpt="considered the record")

    results = validate_citations(_claim(citation), [_passage()])

    assert len(results) == 1
    assert isinstance(results[0], CitationValidationResult)
    assert results[0].status == "valid"
    assert results[0].evidence_id == "passage-1"
    assert results[0].source.title == "Decision"
    assert results[0].issues == []


def test_unknown_document_id_is_invalid_against_supplied_evidence() -> None:
    citation = Citation(document_id="doc-unknown")

    result = validate_citations(_claim(citation), [_passage()])[0]

    assert result.status == "invalid"
    assert result.evidence_id is None
    assert result.source is None


def test_each_citation_gets_its_own_result_in_order() -> None:
    known = Citation(document_id="doc-1", locator="p. 3")
    unknown = Citation(document_id="doc-unknown")

    results = validate_citations(_claim(known, unknown), [_passage()])

    assert [result.status for result in results] == ["valid", "invalid"]
    assert [result.citation for result in results] == [known, unknown]


def test_mismatched_locator_is_invalid() -> None:
    citation = Citation(document_id="doc-1", locator="p. 9")

    result = validate_citations(_claim(citation), [_passage()])[0]

    assert result.status == "invalid"
    assert "locator" in result.issues[0]


def test_claim_without_citations_returns_missing_result() -> None:
    results = validate_citations(_claim(), [_passage()])

    assert len(results) == 1
    assert results[0].status == "missing"
    assert results[0].citation is None
    assert results[0].evidence_id is None


def test_missing_title_or_locator_metadata_is_uncertain() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3")

    no_title = validate_citations(_claim(citation), [_passage(title=None)])[0]
    no_locator = validate_citations(_claim(citation), [_passage(locator=None)])[0]

    assert no_title.status == "uncertain"
    assert no_title.evidence_id == "passage-1"
    assert no_locator.status == "uncertain"
    assert no_locator.evidence_id is None


def test_ambiguous_passages_and_absent_evidence_are_uncertain() -> None:
    citation = Citation(document_id="doc-1")
    passages = [_passage(), _passage(evidence_id="passage-2", locator="p. 4")]

    ambiguous = validate_citations(_claim(citation), passages)[0]
    no_evidence = validate_citations(_claim(citation), [])[0]

    assert ambiguous.status == "uncertain"
    assert ambiguous.evidence_id is None
    assert no_evidence.status == "uncertain"


def test_duplicate_document_and_locator_matches_are_uncertain() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3")
    passages = [_passage(), _passage(evidence_id="passage-2")]

    result = validate_citations(_claim(citation), passages)[0]

    assert result.status == "uncertain"
    assert result.evidence_id is None


def test_excerpt_not_found_in_passage_is_uncertain() -> None:
    citation = Citation(document_id="doc-1", locator="p. 3", excerpt="Different wording")

    result = validate_citations(_claim(citation), [_passage()])[0]

    assert result.status == "uncertain"
    assert result.evidence_id == "passage-1"
