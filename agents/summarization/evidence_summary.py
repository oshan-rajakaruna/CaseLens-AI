"""Deterministic, extractive evidence selection for legal summarization."""

from __future__ import annotations

from dataclasses import dataclass
import re

from agents.summarization.schemas import (
    AnalysisFinding,
    EvidenceItem,
    EvidenceSummary,
    EvidenceSummaryPoint,
    RetrievedPassage,
    SourceReference,
    SummarizationRequest,
)

MAX_INPUT_CHARACTERS = 20_000
MAX_CANDIDATES = 100
MAX_SELECTED_POINTS = 6
MAX_SUMMARY_CHARACTERS = 1_800

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")
_NON_TERMINAL_ABBREVIATIONS = ("rs.", "mr.", "mrs.", "ms.", "dr.", "ltd.", "no.", "v.")
_SIGNIFICANCE_PATTERNS = (
    re.compile(r"\b(?:lkr|rs\.?|usd|eur|gbp|\$)\s?[\d,]+", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b(?:agreement|contract|lease|purchase|payment|pay|paid|notice|terminate|termination|obligation|shall|must|due|breach|proceedings?|court|hearing|invoice)\b", re.IGNORECASE),
    re.compile(r"\b(?:alleged|alleges|claimed|stated|denied|denies|not|never|may|might|if|unless|conditional|subject to|does not establish)\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class _Candidate:
    text: str
    source: SourceReference | None
    is_negated: bool
    is_conditional: bool
    is_attributed: bool
    epistemic_status: str
    verification_status: str
    order: int

    @property
    def normalized_key(self) -> str:
        return _normalize(self.text).casefold()


class EvidenceSummarizer:
    """Select concise, exact source statements with transparent rules.

    This component intentionally performs no paraphrasing. Whitespace is
    normalized only for output readability and candidate comparison; legal
    qualifiers such as ``not``, ``alleged``, ``may``, and ``if`` remain in the
    selected sentence.
    """

    def summarize(self, request: SummarizationRequest) -> EvidenceSummary:
        candidates, input_truncated = self._candidates(request)
        selected = self._select(candidates)
        points = self._points(selected)
        flags = self._uncertainty_flags(points, input_truncated, bool(candidates))
        return EvidenceSummary(
            summary_text=" ".join(point.text for point in points),
            key_points=[point.text for point in points],
            points=points,
            source_references=_unique_sources(
                source
                for point in points
                for source in point.source_references
            ),
            uncertainty_flags=flags,
        )

    def _candidates(self, request: SummarizationRequest) -> tuple[list[_Candidate], bool]:
        records: list[tuple[str, SourceReference | None, bool, bool, bool, str, str]] = []
        for item in request.evidence_items:
            records.append(_item_record(item))
        for finding in request.analysis_findings:
            records.extend(_finding_records(finding))
        for passage in request.retrieved_passages:
            records.append(_passage_record(passage))
        if request.evidence_text is not None:
            # Raw text has no invented locator, page, or citation.
            records.append((
                request.evidence_text,
                None,
                False,
                False,
                False,
                "stated",
                "not_independently_verified",
            ))

        candidates: list[_Candidate] = []
        consumed = 0
        truncated = False
        for text, source, negated, conditional, attributed, epistemic, verification in records:
            for sentence in _sentences(text):
                if len(candidates) >= MAX_CANDIDATES or consumed + len(sentence) > MAX_INPUT_CHARACTERS:
                    truncated = True
                    break
                candidates.append(_Candidate(
                    text=sentence,
                    source=source,
                    is_negated=negated,
                    is_conditional=conditional,
                    is_attributed=attributed,
                    epistemic_status=epistemic,
                    verification_status=verification,
                    order=len(candidates),
                ))
                consumed += len(sentence)
            if truncated:
                break
        return candidates, truncated

    def _select(self, candidates: list[_Candidate]) -> list[_Candidate]:
        significant = [candidate for candidate in candidates if _score(candidate) > 0]
        ranked = sorted(significant, key=lambda candidate: (-_score(candidate), candidate.order))
        selected: list[_Candidate] = []
        selected_characters = 0
        seen: set[str] = set()
        for candidate in ranked:
            if candidate.normalized_key in seen:
                # Keep a duplicate only to retain another real supporting source.
                if candidate.source is not None and all(
                    existing.normalized_key != candidate.normalized_key
                    or existing.source != candidate.source
                    for existing in selected
                ):
                    selected.append(candidate)
                continue
            if len(selected) >= MAX_SELECTED_POINTS:
                break
            if selected_characters + len(candidate.text) > MAX_SUMMARY_CHARACTERS:
                continue
            selected.append(candidate)
            selected_characters += len(candidate.text)
            seen.add(candidate.normalized_key)
        # Output is stable in original evidence order rather than relevance order.
        return sorted(selected, key=lambda candidate: candidate.order)

    @staticmethod
    def _points(candidates: list[_Candidate]) -> list[EvidenceSummaryPoint]:
        grouped: dict[str, list[_Candidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.normalized_key, []).append(candidate)
        points: list[EvidenceSummaryPoint] = []
        for grouped_candidates in grouped.values():
            first = grouped_candidates[0]
            sources = _unique_sources(
                candidate.source
                for candidate in grouped_candidates
                if candidate.source is not None
            )
            points.append(EvidenceSummaryPoint(
                text=first.text,
                source_references=sources,
                citations=_unique_citations(sources),
                is_negated=any(candidate.is_negated for candidate in grouped_candidates),
                is_conditional=any(candidate.is_conditional for candidate in grouped_candidates),
                is_attributed=any(candidate.is_attributed for candidate in grouped_candidates),
                epistemic_status=first.epistemic_status,
                verification_status=first.verification_status,
                provenance_limited=not sources,
            ))
        return points

    @staticmethod
    def _uncertainty_flags(
        points: list[EvidenceSummaryPoint],
        input_truncated: bool,
        had_candidates: bool,
    ) -> list[str]:
        flags: list[str] = []
        if not points and had_candidates:
            flags.append("no_significant_evidence_selected")
        if input_truncated:
            flags.append("input_truncated_at_safe_sentence_boundary")
        if any(point.provenance_limited for point in points):
            flags.append("limited_provenance")
        if any(point.is_attributed for point in points):
            flags.append("attributed_or_alleged_statement")
        if any(point.is_negated for point in points):
            flags.append("negated_statement")
        if any(point.is_conditional for point in points):
            flags.append("conditional_statement")
        if any(point.verification_status != "independently_verified" for point in points):
            flags.append("not_independently_verified")
        return flags


def _item_record(item: EvidenceItem) -> tuple[str, SourceReference | None, bool, bool, bool, str, str]:
    return (
        item.text,
        item.source,
        item.is_negated,
        item.is_conditional,
        item.is_attributed,
        item.epistemic_status,
        item.verification_status,
    )


def _finding_records(finding: AnalysisFinding) -> list[tuple[str, SourceReference | None, bool, bool, bool, str, str]]:
    sources = finding.sources or [None]
    return [(
        finding.statement,
        source,
        finding.is_negated,
        finding.is_conditional,
        finding.is_attributed,
        finding.epistemic_status,
        finding.verification_status,
    ) for source in sources]


def _passage_record(passage: RetrievedPassage) -> tuple[str, SourceReference | None, bool, bool, bool, str, str]:
    return (
        passage.text,
        passage.source,
        False,
        False,
        False,
        "stated",
        "not_independently_verified",
    )


def _sentences(text: str) -> list[str]:
    normalized = _normalize(text)
    fragments = [sentence for sentence in _SENTENCE_BOUNDARY.split(normalized) if sentence]
    sentences: list[str] = []
    for fragment in fragments:
        if sentences and sentences[-1].casefold().endswith(_NON_TERMINAL_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {fragment}"
        else:
            sentences.append(fragment)
    return sentences


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _score(candidate: _Candidate) -> int:
    text = candidate.text
    score = sum(1 for pattern in _SIGNIFICANCE_PATTERNS if pattern.search(text))
    if candidate.is_negated or candidate.is_conditional or candidate.is_attributed:
        score += 1
    if candidate.epistemic_status != "stated":
        score += 1
    return score


def _unique_sources(sources: object) -> list[SourceReference]:
    unique: list[SourceReference] = []
    for source in sources:  # type: ignore[union-attr]
        if source is not None and source not in unique:
            unique.append(source)
    return unique


def _unique_citations(sources: list[SourceReference]) -> list[str]:
    return list(dict.fromkeys(
        source.citation_id for source in sources if source.citation_id is not None
    ))
