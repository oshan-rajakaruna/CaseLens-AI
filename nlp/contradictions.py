"""Narrow, review-only contradiction candidates from grounded facts."""

from hashlib import sha256
import re
from typing import Iterable

from agents.analysis.schemas import ContradictionCandidate, ExtractedFact


def detect_contradictions(facts: Iterable[ExtractedFact], *, case_id: str | None = None) -> list[ContradictionCandidate]:
    """Compare explicit same-event affirmation/denial pairs in a supplied case context."""
    if not case_id or not case_id.strip():
        return []
    items = list(facts)
    candidates = []
    for left_index, left in enumerate(items):
        if left.is_conditional:
            continue
        for right in items[left_index + 1 :]:
            if right.is_conditional or left.source == right.source:
                continue
            if left.is_negated == right.is_negated or not _same_event(left, right):
                continue
            ordered = sorted((left, right), key=lambda fact: (fact.source.document_id, fact.source.start_char or -1))
            digest = sha256(f"{case_id}|{ordered[0].source.model_dump_json()}|{ordered[1].source.model_dump_json()}".encode()).hexdigest()[:16]
            context = {"case_id": case_id, "left_status": ordered[0].epistemic_status, "right_status": ordered[1].epistemic_status, "left_attributed": ordered[0].is_attributed, "right_attributed": ordered[1].is_attributed, "date": ordered[0].event_date, "amount": ordered[0].money_text}
            candidates.append(ContradictionCandidate(candidate_id=f"contradiction:{digest}", category="affirmation_vs_denial", description="Potential conflict between explicit affirmative and negated claims about the same identified event; human review required.", sources=[ordered[0].source, ordered[1].source], evidence_context=context))
    return sorted(candidates, key=lambda item: item.candidate_id or "")


def _same_event(left: ExtractedFact, right: ExtractedFact) -> bool:
    payment_categories = {"payment_or_transaction", "non_payment_or_breach"}
    same_category = left.category == right.category or {left.category, right.category} <= payment_categories
    if not same_category or not left.event_date or left.event_date != right.event_date:
        return False
    if not left.money_text or left.money_text != right.money_text:
        return False
    left_subjects, right_subjects = set(left.entity_texts), set(right.entity_texts)
    if not left_subjects or left_subjects != right_subjects:
        return False
    return _core(left.statement) == _core(right.statement)


def _core(statement: str) -> str:
    value = statement.lower()
    value = re.sub(r"\bpaid\b", "pay", value)
    value = re.sub(r"\b(?:did not|didn't|not|denies|denied)\b", "", value)
    value = re.sub(r"\b(?:on\s+)?\d{1,2}\s+[a-z]+\s+\d{4}\b", "", value)
    value = re.sub(r"\b(?:lkr|rs\.?)\s*[\d,.]+\b", "", value)
    return re.sub(r"\W+", " ", value).strip()
