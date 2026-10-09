"""Conservative, evidence-grounded extraction of explicit factual statements."""

from __future__ import annotations

import re
from typing import Sequence

from agents.analysis.schemas import ExtractedEntity, ExtractedFact
from nlp.entities import _source_for_span
from nlp.extraction import ExtractionSegment

_FACT_RULES = (
    ("agreement_execution", re.compile(r"\b(?:agreement|contract)\b.*\b(?:sign(?:ed)?|executed|entered into)\b|\b(?:sign(?:ed)?|executed|entered into)\b.*\b(?:agreement|contract)\b", re.I)),
    ("non_payment_or_breach", re.compile(r"\b(?:failed to pay|did not pay|non-payment|breached|defaulted)\b", re.I)),
    ("payment_or_transaction", re.compile(r"\b(?:paid|payment was made|transferred|received)\b", re.I)),
    ("notice_or_communication", re.compile(r"\b(?:notice was served|notice was sent|notified|communicated|receiving notice)\b", re.I)),
    ("legal_proceeding", re.compile(r"\b(?:filed|commenced|instituted)\b.*\b(?:case|action|proceeding|petition|court)\b", re.I)),
    ("documented_event", re.compile(r"\b(?:delivered|terminated|issued|recorded)\b", re.I)),
)
_DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b")
_MONEY = re.compile(r"\b(?:LKR|Rs\.?)\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?\b", re.I)


def extract_facts(text: str, document_id: str, segments: Sequence[ExtractionSegment] | None = None, entities: Sequence[ExtractedEntity] | None = None) -> list[ExtractedFact]:
    """Return only explicit supported propositions; no truth verification occurs."""
    if not text or not text.strip() or not document_id or not document_id.strip():
        return []
    facts: list[ExtractedFact] = []
    for sentence, start, end in _sentences(text):
        category = next((name for name, rule in _FACT_RULES if rule.search(sentence)), None)
        if category is None:
            continue
        status, negated, conditional, attributed = _semantics(sentence)
        linked = [entity.text for entity in entities or () if entity.source.start_char is not None and start <= entity.source.start_char and entity.source.end_char <= end]
        date = _DATE.search(sentence)
        money = _MONEY.search(sentence)
        facts.append(ExtractedFact(statement=sentence, category=category, epistemic_status=status, is_negated=negated, is_conditional=conditional, is_attributed=attributed, entity_texts=linked, event_date=date.group(0) if date else None, money_text=money.group(0) if money else None, source=_source_for_span(text, document_id, start, end, segments), method="rule-based-fact"))
    return facts


def _sentences(text: str):
    for match in re.finditer(r".+?(?:[.!?]+(?=\s+[A-Z]|$)|$)", text):
        raw = match.group(0)
        value = raw.strip()
        if value:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield value, start, start + len(value)


def _semantics(sentence: str) -> tuple[str, bool, bool, bool]:
    lower = sentence.lower()
    conditional = bool(re.search(r"\b(?:if|provided that|subject to)\b", lower))
    attributed = bool(re.search(r"\b(?:alleges?|allegedly|stated|said|claimed|witness)\b", lower))
    negated = bool(re.search(r"\b(?:did not|was not|not|denies|denied)\b", lower))
    if any(word in lower for word in ("alleges", "alleged", "allegedly")):
        return "alleged", negated, conditional, True
    if any(word in lower for word in ("denies", "denied", "disputes", "disputed")):
        return "disputed", negated, conditional, attributed
    if conditional or re.search(r"\b(?:may|might|could)\b", lower) or attributed:
        return "uncertain", negated, conditional, attributed
    return "stated", negated, conditional, attributed
