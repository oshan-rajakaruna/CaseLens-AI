"""Tests for conservative source-grounded chronology."""

import pytest

from agents.analysis.schemas import EvidenceSource, ExtractedFact
from nlp.timeline import build_timeline, normalize_date


@pytest.mark.parametrize(("value", "order", "expected"), [("12 January 2025", None, ("2025-01-12", "day")), ("January 12, 2025", None, ("2025-01-12", "day")), ("2025-01-12", None, ("2025-01-12", "day")), ("12/01/2025", "day_first", ("2025-01-12", "day")), ("March 2025", None, ("2025-03", "month")), ("2025", None, ("2025", "year"))])
def test_normalization(value, order, expected) -> None:
    assert normalize_date(value, order) == expected


def test_ambiguous_and_invalid_dates_are_unresolved() -> None:
    assert normalize_date("03/04/2025") is None
    assert normalize_date("31 February 2025") is None
    assert normalize_date("yesterday") is None


def make_fact(statement: str, date: str | None, start: int, **kwargs) -> ExtractedFact:
    return ExtractedFact(statement=statement, category="payment_or_transaction", event_date=date, source=EvidenceSource(document_id="doc", page=2, locator="page:2", start_char=start, end_char=start + len(statement), excerpt=statement), method="test", **kwargs)


def test_orders_events_preserves_semantics_and_provenance() -> None:
    later = make_fact("Nimal paid LKR 50,000 on 12 January 2025.", "12 January 2025", 50)
    early = make_fact("Kamal paid LKR 10,000 on 2025-01-02.", "2025-01-02", 0, epistemic_status="alleged", is_attributed=True)
    events = build_timeline([later, early])
    assert [event.normalized_date for event in events] == ["2025-01-02", "2025-01-12"]
    assert events[0].epistemic_status == "alleged"
    assert events[0].source.page == 2
    assert events[1].source.excerpt == later.statement


def test_statement_date_uses_described_event_date_and_skips_undated_or_deadlines() -> None:
    fact = make_fact("The witness stated on 10 April 2025 that Nimal paid on 5 February 2025.", "10 April 2025", 0, is_attributed=True)
    undated = make_fact("Nimal paid LKR 1000.", None, 100)
    events = build_timeline([fact, undated])
    assert events[0].normalized_date == "2025-02-05"
    assert events[0].is_attributed is True
    assert build_timeline([]) == []
