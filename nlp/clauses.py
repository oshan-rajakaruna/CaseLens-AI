"""Conservative extraction of explicit contractual language."""

from __future__ import annotations

import re
from typing import Sequence

from agents.analysis.schemas import ExtractedClause, ExtractedEntity
from nlp.entities import _source_for_span
from nlp.extraction import ExtractionSegment

_MONEY = re.compile(r"\b(?:LKR|Rs\.?)\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?\b", re.I)
_DEADLINE = re.compile(r"\b(?:within \d+ days|no later than \d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b", re.I)


def extract_clauses(text: str, document_id: str, segments: Sequence[ExtractionSegment] | None = None, entities: Sequence[ExtractedEntity] | None = None) -> list[ExtractedClause]:
    """Extract only explicit clause language, retaining attribution and modality."""
    if not text or not text.strip() or not document_id or not document_id.strip():
        return []
    results: list[ExtractedClause] = []
    for sentence, start, end in _sentences(text):
        category = _category(sentence)
        if category is None:
            continue
        lower = sentence.lower()
        status = "alleged" if any(word in lower for word in ("alleged", "alleges", "witness")) else ("uncertain" if re.search(r"\b(?:may|might)\b", lower) else "stated")
        parties = [entity.text for entity in entities or () if entity.label in {"PERSON", "ORGANIZATION"} and entity.source.start_char is not None and start <= entity.source.start_char and entity.source.end_char <= end]
        money, deadline = _MONEY.search(sentence), _DEADLINE.search(sentence)
        results.append(ExtractedClause(text=sentence, clause_type=category, epistemic_status=status, is_conditional=bool(re.search(r"\b(?:if|provided that|subject to)\b", lower)), party_texts=parties, money_text=money.group(0) if money else None, deadline_text=deadline.group(0) if deadline else None, source=_source_for_span(text, document_id, start, end, segments), method="rule-based-clause"))
    return results


def _category(sentence: str) -> str | None:
    lower = sentence.lower()
    if "may be terminated" in lower or "termination" in lower:
        return "termination"
    if re.search(r"\b(?:shall not|must not|prohibited from)\b", lower):
        return "prohibition"
    if re.search(r"\b(?:shall|must|required(?: to)?)\b", lower):
        return "obligation"
    if re.search(r"\b(?:entitled to|may)\b", lower):
        return "permission_or_right"
    if re.search(r"\b(?:if|provided that|subject to)\b", lower):
        return "condition"
    if _DEADLINE.search(sentence):
        return "deadline"
    if re.search(r"\b(?:monthly payment|rent|payment)\b", lower) and _MONEY.search(sentence):
        return "payment_term"
    return None


def _sentences(text: str):
    for match in re.finditer(r".+?(?:[.!?]+(?=\s+[A-Z]|$)|$)", text):
        raw = match.group(0)
        value = raw.strip()
        if value:
            start = match.start() + len(raw) - len(raw.lstrip())
            yield value, start, start + len(value)
