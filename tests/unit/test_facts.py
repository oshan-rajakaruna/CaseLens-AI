"""Tests for conservative evidence-grounded fact extraction."""

from agents.analysis.schemas import EvidenceSource, ExtractedEntity
from nlp.facts import extract_facts
from nlp.extraction import ExtractionSegment


def test_execution_payment_and_entity_linking() -> None:
    text = "Nimal signed the agreement on 1 January 2024. Nimal paid LKR 1000."
    entities = [ExtractedEntity(text="Nimal", label="PERSON", source=EvidenceSource(document_id="d", start_char=0, end_char=5), method="test")]
    facts = extract_facts(text, "d", entities=entities)
    assert [fact.category for fact in facts] == ["agreement_execution", "payment_or_transaction"]
    assert facts[0].event_date == "1 January 2024"
    assert facts[1].money_text == "LKR 1000"
    assert facts[0].entity_texts == ["Nimal"]
    assert facts[0].source.excerpt == facts[0].statement


def test_allegation_negation_uncertainty_notice_and_proceeding_are_preserved() -> None:
    text = "ABC alleges that Nimal failed to pay rent. Nimal did not sign the agreement. The tenant may have transferred money. Notice was served. ABC filed a proceeding in court."
    facts = extract_facts(text, "d")
    assert [(fact.category, fact.epistemic_status) for fact in facts] == [
        ("non_payment_or_breach", "alleged"), ("agreement_execution", "stated"),
        ("payment_or_transaction", "uncertain"), ("notice_or_communication", "stated"),
        ("legal_proceeding", "stated"),
    ]
    assert "did not sign" in facts[1].statement
    assert facts[1].is_negated is True
    assert facts[0].is_attributed is True
    assert all(fact.verification_status == "not_independently_verified" for fact in facts)


def test_segments_offsets_empty_and_unsupported_narrative() -> None:
    text = "First page. Payment was made."
    segment = ExtractionSegment(segment_id="d:2", document_id="d", text="Payment was made.", source=EvidenceSource(document_id="d", page=2, locator="page:2"), start_char=12, end_char=29)
    facts = extract_facts(text, "d", [segment])
    assert facts[0].source.page == 2
    assert text[facts[0].source.start_char:facts[0].source.end_char] == facts[0].source.excerpt
    assert extract_facts("", "d") == []
    assert extract_facts("The sky is blue.", "d") == []


def test_denied_conditional_repeated_and_entity_span_semantics() -> None:
    text = "Kamal Perera denies receiving notice. If LKR 100,000 is paid, delivery may occur. Kamal Perera paid LKR 100,000."
    entity = ExtractedEntity(text="Kamal Perera", label="PERSON", source=EvidenceSource(document_id="d", start_char=0, end_char=12), method="test")
    facts = extract_facts(text, "d", entities=[entity])
    assert [(fact.category, fact.epistemic_status, fact.is_negated, fact.is_conditional) for fact in facts] == [
        ("notice_or_communication", "disputed", True, False),
        ("payment_or_transaction", "uncertain", False, True),
        ("payment_or_transaction", "stated", False, False),
    ]
    assert facts[0].entity_texts == ["Kamal Perera"]
    assert facts[2].entity_texts == []
    assert facts[1].statement.startswith("If LKR 100,000")
    assert facts[1].event_date is None
    assert facts[1].money_text == "LKR 100,000"


def test_month_precision_date_is_preserved_for_downstream_timeline() -> None:
    fact = extract_facts("The parties entered into the lease agreement in March 2025.", "d")[0]
    assert fact.event_date == "March 2025"
