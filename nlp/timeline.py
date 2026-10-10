"""Conservative timeline generation from already grounded extracted facts."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Iterable, Literal

from agents.analysis.schemas import ExtractedFact, TimelineEvent

_DATES = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}|[A-Za-z]+\s+\d{4}|\d{4})\b")


def normalize_date(expression: str, numeric_date_order: Literal["day_first", "month_first"] | None = None) -> tuple[str, str] | None:
    """Return ISO-style value and explicit precision; never resolve ambiguity."""
    value = expression.strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat(), "day"
        except ValueError:
            pass
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", value):
        if numeric_date_order is None:
            return None
        try:
            fmt = "%d/%m/%Y" if numeric_date_order == "day_first" else "%m/%d/%Y"
            return datetime.strptime(value, fmt).date().isoformat(), "day"
        except ValueError:
            return None
    try:
        return datetime.strptime(value, "%B %Y").strftime("%Y-%m"), "month"
    except ValueError:
        pass
    return (value, "year") if re.fullmatch(r"\d{4}", value) else None


def build_timeline(facts: Iterable[ExtractedFact], clauses=None, date_entities=None, reference_date=None, *, numeric_date_order: Literal["day_first", "month_first"] | None = None) -> list[TimelineEvent]:
    """Build chronological event candidates without treating clauses/deadlines as events."""
    events = []
    for index, fact in enumerate(facts):
        expression = _event_date(fact.statement, fact.event_date)
        if expression is None:
            continue
        normalized = normalize_date(expression, numeric_date_order)
        if normalized is None:
            continue
        value, precision = normalized
        events.append(TimelineEvent(event_id=f"{fact.source.document_id}:{fact.source.start_char}:{fact.source.end_char}", description=fact.statement, category=fact.category, date_text=expression, normalized_date=value, date_precision=precision, temporal_role="event_date", epistemic_status=fact.epistemic_status, is_negated=fact.is_negated, is_conditional=fact.is_conditional, is_attributed=fact.is_attributed, verification_status=fact.verification_status, source=fact.source, method="rule-based-timeline"))
    return sorted(events, key=lambda item: (item.normalized_date or "9999", item.source.document_id, item.source.start_char or -1))


def _event_date(statement: str, supplied: str | None) -> str | None:
    matches = _DATES.findall(statement)
    if re.search(r"\b(?:stated|said|reported) on\b", statement, re.I) and len(matches) > 1:
        return matches[-1]
    return supplied or (matches[0] if matches else None)
