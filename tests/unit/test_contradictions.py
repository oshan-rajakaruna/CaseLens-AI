"""Tests for narrow, review-only contradiction candidates."""

from agents.analysis.schemas import EvidenceSource, ExtractedFact
from nlp.contradictions import detect_contradictions


def make_fact(text: str, start: int, *, negated: bool = False, date: str = "12 January 2025", amount: str = "LKR 50,000", subject: str = "Nimal Perera", **kwargs) -> ExtractedFact:
    return ExtractedFact(statement=text, category="payment_or_transaction", event_date=date, money_text=amount, entity_texts=[subject], is_negated=negated, source=EvidenceSource(document_id=kwargs.pop("document_id", "doc-a"), page=1, locator="page:1", start_char=start, end_char=start + len(text), excerpt=text), method="test", **kwargs)


def test_detects_same_event_affirmation_vs_denial_with_provenance() -> None:
    yes = make_fact("Nimal Perera paid LKR 50,000 on 12 January 2025.", 0)
    no = make_fact("Nimal Perera did not pay LKR 50,000 on 12 January 2025.", 60, negated=True, document_id="doc-b", epistemic_status="alleged", is_attributed=True)
    result = detect_contradictions([yes, no], case_id="case-1")
    assert len(result) == 1
    assert result[0].category == "affirmation_vs_denial"
    assert result[0].review_status == "needs_review"
    assert [source.document_id for source in result[0].sources] == ["doc-a", "doc-b"]
    assert result[0].evidence_context["right_attributed"] is True


def test_rejects_different_dates_amounts_subjects_conditions_and_missing_context() -> None:
    yes = make_fact("Nimal Perera paid LKR 50,000 on 12 January 2025.", 0)
    different_date = make_fact("Nimal Perera did not pay LKR 50,000 on 15 February 2025.", 60, negated=True, date="15 February 2025")
    different_amount = make_fact("Nimal Perera did not pay LKR 60,000 on 12 January 2025.", 120, negated=True, amount="LKR 60,000")
    different_subject = make_fact("Kamal Perera did not pay LKR 50,000 on 12 January 2025.", 180, negated=True, subject="Kamal Perera")
    conditional = make_fact("If Nimal Perera does not pay LKR 50,000 on 12 January 2025.", 240, negated=True, is_conditional=True)
    assert detect_contradictions([yes, different_date, different_amount, different_subject, conditional], case_id="case-1") == []
    assert detect_contradictions([yes], case_id="case-1") == []
    assert detect_contradictions([yes], case_id=None) == []


def test_repeated_statements_no_reverse_duplicates_and_stable_ids() -> None:
    yes = make_fact("Nimal Perera paid LKR 50,000 on 12 January 2025.", 0)
    no = make_fact("Nimal Perera did not pay LKR 50,000 on 12 January 2025.", 60, negated=True)
    first = detect_contradictions([yes, no], case_id="case-1")
    second = detect_contradictions([yes, no], case_id="case-1")
    assert first[0].candidate_id == second[0].candidate_id
    assert len(first[0].sources) == 2
