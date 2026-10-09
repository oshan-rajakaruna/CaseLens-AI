"""Evidence-grounded hybrid named-entity recognition for English legal text."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Callable, Iterable, Protocol, Sequence

from agents.analysis.schemas import EvidenceSource, ExtractedEntity
from nlp.extraction import ExtractionSegment

DEFAULT_SPACY_MODEL = "en_core_web_sm"


class _Span(Protocol):
    text: str
    label_: str
    start_char: int
    end_char: int


class _NlpModel(Protocol):
    def __call__(self, text: str) -> object: ...


@dataclass(frozen=True, slots=True)
class _Candidate:
    text: str
    label: str
    start: int
    end: int
    method: str
    priority: int


_SPACY_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "MONEY": "MONEY",
}

_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "COURT",
        re.compile(
            r"\b(?:Supreme Court of Sri Lanka|Court of Appeal(?: of Sri Lanka)?|"
            r"District Court of [A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}|"
            r"Magistrate's Court(?: of [A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})?)\b"
        ),
    ),
    (
        "CASE_REFERENCE",
        re.compile(r"\b(?:SC\s+(?:FR\s+)?\d{1,5}/\d{4}|CA\s+\d{1,5}/\d{4})\b"),
    ),
    (
        "STATUTE_REFERENCE",
        re.compile(r"\b(?:[A-Z][A-Za-z]*(?:[ \t]+[A-Z][A-Za-z]*){0,5}[ \t]+)?Act[ \t]+No\.[ \t]*\d{1,4}[ \t]+of[ \t]+\d{4}\b"),
    ),
    (
        "MONEY",
        re.compile(r"\b(?:LKR|Rs\.?|Rupees?)\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?\b", re.IGNORECASE),
    ),
    (
        "DATE",
        re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}|"
            r"\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b"
        ),
    ),
)


@lru_cache(maxsize=4)
def load_spacy_model(model_name: str = DEFAULT_SPACY_MODEL) -> _NlpModel | None:
    """Load a configured local spaCy model once; never download at inference time."""

    try:
        import spacy

        return spacy.load(model_name)
    except (ImportError, OSError):
        return None


def extract_entities(
    text: str,
    document_id: str,
    segments: Sequence[ExtractionSegment] | None = None,
    *,
    model_name: str = DEFAULT_SPACY_MODEL,
    nlp_model: _NlpModel | None = None,
    language: str = "en",
) -> list[ExtractedEntity]:
    """Extract deduplicated entities with exact, source-grounded provenance.

    For non-English or unavailable-model calls, only deterministic legal rules
    run. Rules can still safely recognise their explicit Latin-script patterns.
    """

    if not text or not text.strip() or not document_id or not document_id.strip():
        return []

    use_model = language.lower().startswith("en")
    model = nlp_model if nlp_model is not None and use_model else (load_spacy_model(model_name) if use_model else None)
    fallback = model is None
    candidates = _rule_candidates(text, "rule-based-fallback" if fallback else "rule-based")
    if model is not None:
        candidates.extend(_spacy_candidates(text, model, model_name if nlp_model is None else "injected"))

    selected = _resolve_overlaps(candidates)
    return [
        ExtractedEntity(
            text=candidate.text,
            label=candidate.label,
            source=_source_for_span(text, document_id, candidate.start, candidate.end, segments),
            method=candidate.method,
        )
        for candidate in selected
    ]


def _rule_candidates(text: str, method: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for label, pattern in _RULES:
        for match in pattern.finditer(text):
            candidates.append(
                _Candidate(match.group(0), label, match.start(), match.end(), method, priority=100)
            )
    return candidates


def _spacy_candidates(text: str, model: _NlpModel, model_name: str) -> list[_Candidate]:
    document = model(text)
    entities: Iterable[_Span] = getattr(document, "ents", ())
    candidates: list[_Candidate] = []
    for entity in entities:
        label = _SPACY_LABELS.get(entity.label_)
        if label is None or entity.start_char >= entity.end_char:
            continue
        # Use source slicing rather than the model's display value as evidence.
        candidates.append(
            _Candidate(
                text=text[entity.start_char : entity.end_char],
                label=label,
                start=entity.start_char,
                end=entity.end_char,
                method=f"spacy:{model_name}",
                priority=50,
            )
        )
    return candidates


def _resolve_overlaps(candidates: list[_Candidate]) -> list[_Candidate]:
    """Prefer precise rules, then return all non-overlapping occurrences in order."""

    selected: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.priority, item.start, -(item.end - item.start), item.label)):
        if any(candidate.start < existing.end and existing.start < candidate.end for existing in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.start, item.end, item.label))


def _source_for_span(
    text: str,
    document_id: str,
    start: int,
    end: int,
    segments: Sequence[ExtractionSegment] | None,
) -> EvidenceSource:
    """Map one contained span to a segment; cross-segment spans remain explicit."""

    excerpt = text[start:end]
    if segments:
        containing = next(
            (segment for segment in segments if segment.start_char <= start and end <= segment.end_char),
            None,
        )
        if containing is not None:
            return containing.source.model_copy(
                update={"start_char": start, "end_char": end, "excerpt": excerpt}
            )
        return EvidenceSource(
            document_id=document_id,
            locator="multiple_segments",
            start_char=start,
            end_char=end,
            excerpt=excerpt,
        )
    return EvidenceSource(
        document_id=document_id,
        locator=f"text:{start}-{end}",
        start_char=start,
        end_char=end,
        excerpt=excerpt,
    )
