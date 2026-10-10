"""Tests for evidence-grounded hybrid NER without accuracy assumptions."""

from dataclasses import dataclass

from agents.analysis import AnalysisAgent, AnalysisRequest
from agents.analysis.schemas import EvidenceSource
from backend.schemas import Case, Document
from nlp.entities import extract_entities
from nlp.extraction import ExtractionSegment


@dataclass
class FakeEntity:
    text: str
    label_: str
    start_char: int
    end_char: int


class FakeModel:
    def __init__(self, entities: list[FakeEntity]) -> None:
        self.entities = entities

    def __call__(self, text: str):  # type: ignore[no-untyped-def]
        return type("FakeDocument", (), {"ents": self.entities})()


def fake_entity(text: str, value: str, label: str) -> FakeEntity:
    start = text.index(value)
    return FakeEntity(value, label, start, start + len(value))


def test_spacy_labels_are_normalized_with_exact_offsets() -> None:
    text = "Nimal Perera of Acme Ltd visited Colombo."
    model = FakeModel(
        [
            fake_entity(text, "Nimal Perera", "PERSON"),
            fake_entity(text, "Acme Ltd", "ORG"),
            fake_entity(text, "Colombo", "GPE"),
        ]
    )

    entities = extract_entities(text, "doc-001", nlp_model=model)

    assert [(entity.text, entity.label) for entity in entities] == [
        ("Nimal Perera", "PERSON"),
        ("Acme Ltd", "ORGANIZATION"),
        ("Colombo", "LOCATION"),
    ]
    for entity in entities:
        assert text[entity.source.start_char : entity.source.end_char] == entity.text
        assert entity.method == "spacy:injected"


def test_dates_money_courts_cases_and_acts_use_precise_rules() -> None:
    text = (
        "On 1 January 2024, the Supreme Court of Sri Lanka considered SC FR 123/2024. "
        "The claim was LKR 1,250.50 under Act No. 10 of 2020."
    )

    entities = extract_entities(text, "doc-rules", model_name="missing_model_for_test")

    assert {(entity.text, entity.label) for entity in entities} == {
        ("1 January 2024", "DATE"),
        ("Supreme Court of Sri Lanka", "COURT"),
        ("SC FR 123/2024", "CASE_REFERENCE"),
        ("LKR 1,250.50", "MONEY"),
        ("Act No. 10 of 2020", "STATUTE_REFERENCE"),
    }
    assert {entity.method for entity in entities} == {"rule-based-fallback"}


def test_rs_money_and_district_court_rule_do_not_absorb_nearby_words() -> None:
    text = "The District Court of Colombo heard a claim for Rs. 5000."

    entities = extract_entities(text, "doc-court", model_name="missing_model_for_test")

    assert [(entity.text, entity.label) for entity in entities] == [
        ("District Court of Colombo", "COURT"),
        ("Rs. 5000", "MONEY"),
    ]


def test_pdf_page_provenance_maps_to_containing_segment() -> None:
    text = "First page\nSC FR 123/2024"
    segments = [
        ExtractionSegment(
            segment_id="doc-pdf:segment:1",
            document_id="doc-pdf",
            text="First page",
            source=EvidenceSource(document_id="doc-pdf", page=1, locator="page:1"),
            start_char=0,
            end_char=10,
        ),
        ExtractionSegment(
            segment_id="doc-pdf:segment:2",
            document_id="doc-pdf",
            text="SC FR 123/2024",
            source=EvidenceSource(document_id="doc-pdf", page=2, locator="page:2"),
            start_char=11,
            end_char=25,
        ),
    ]

    entity = extract_entities(text, "doc-pdf", segments, model_name="missing_model_for_test")[0]

    assert entity.source.page == 2
    assert entity.source.locator == "page:2"
    assert entity.source.excerpt == "SC FR 123/2024"


def test_docx_paragraph_and_table_provenance_maps_correctly() -> None:
    text = "Opening\nAct No. 10 of 2020\nClosing"
    segments = [
        ExtractionSegment(segment_id="docx:1", document_id="docx", text="Opening", source=EvidenceSource(document_id="docx", paragraph=1, locator="paragraph:1"), start_char=0, end_char=7),
        ExtractionSegment(segment_id="docx:2", document_id="docx", text="Act No. 10 of 2020", source=EvidenceSource(document_id="docx", locator="table:1"), start_char=8, end_char=26),
        ExtractionSegment(segment_id="docx:3", document_id="docx", text="Closing", source=EvidenceSource(document_id="docx", paragraph=2, locator="paragraph:2"), start_char=27, end_char=34),
    ]

    entity = extract_entities(text, "docx", segments, model_name="missing_model_for_test")[0]

    assert entity.source.locator == "table:1"
    assert entity.source.paragraph is None


def test_same_entity_at_multiple_offsets_remains_multiple_occurrences() -> None:
    text = "SC FR 123/2024 was cited. Later SC FR 123/2024 was distinguished."

    entities = extract_entities(text, "doc-repeat", model_name="missing_model_for_test")

    assert len(entities) == 2
    assert entities[0].source.start_char != entities[1].source.start_char


def test_rule_entity_wins_over_overlapping_spacy_entity() -> None:
    text = "The Supreme Court of Sri Lanka sat today."
    model = FakeModel([fake_entity(text, "Supreme Court of Sri Lanka", "ORG")])

    entities = extract_entities(text, "doc-overlap", nlp_model=model)

    assert [(entity.text, entity.label, entity.method) for entity in entities] == [
        ("Supreme Court of Sri Lanka", "COURT", "rule-based"),
    ]


def test_output_order_is_deterministic_and_cross_segment_spans_are_explicit() -> None:
    text = "SC FR 123/2024\nAct No. 10 of 2020"
    segments = [
        ExtractionSegment(segment_id="x:1", document_id="x", text="SC FR 123/2024", source=EvidenceSource(document_id="x", page=1), start_char=0, end_char=14),
        ExtractionSegment(segment_id="x:2", document_id="x", text="Act No. 10 of 2020", source=EvidenceSource(document_id="x", page=2), start_char=15, end_char=33),
    ]

    first = extract_entities(text, "x", segments, model_name="missing_model_for_test")
    second = extract_entities(text, "x", segments, model_name="missing_model_for_test")

    assert [entity.model_dump() for entity in first] == [entity.model_dump() for entity in second]
    assert [entity.label for entity in first] == ["CASE_REFERENCE", "STATUTE_REFERENCE"]


def test_empty_whitespace_unknown_language_and_no_match_do_not_fabricate_entities() -> None:
    assert extract_entities("", "doc", model_name="missing_model_for_test") == []
    assert extract_entities(" \t\n", "doc", model_name="missing_model_for_test") == []
    assert extract_entities("Nimal Perera", "doc", nlp_model=FakeModel([fake_entity("Nimal Perera", "Nimal Perera", "PERSON")]), language="si") == []
    assert extract_entities("Ordinary prose without supported references.", "doc", model_name="missing_model_for_test") == []


def test_entities_are_compatible_with_existing_analysis_agent_contract() -> None:
    text = "LKR 1000"
    entities = extract_entities(text, "doc-analysis", model_name="missing_model_for_test")
    request = AnalysisRequest(
        case=Case(id="case-001", title="Example"),
        document=Document(id="doc-analysis", case_id="case-001", name="evidence.txt"),
        text=text,
    )

    response = AnalysisAgent().analyze(request)

    assert entities[0].source.document_id == request.document.id
    assert response.entities[0].label == "MONEY"
    assert response.status == "completed"
