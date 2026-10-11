"""Deterministic extraction of explicitly stated judgment information."""

from __future__ import annotations

from dataclasses import dataclass
import re

from agents.summarization.evidence_summary import MAX_INPUT_CHARACTERS, _sentences
from agents.summarization.schemas import (
    EvidenceItem,
    JudgmentMetadata,
    PrecedentSummary,
    PrecedentSummaryPoint,
    RetrievedPassage,
    SourceReference,
    SummarizationRequest,
)

MAX_JUDGMENT_CANDIDATES = 120
MAX_POINTS_PER_CATEGORY = 3

_BACKGROUND = re.compile(
    r"\b(?:background|claimant|respondent|appellant|defendant|plaintiff|alleged|claim|defen[cs]e|contract|agreement|proceedings?|hearing)\b",
    re.IGNORECASE,
)
_ISSUE = re.compile(
    r"\b(?:issue before (?:the )?court|question for determination|court considered whether|legal issue concerned|question is whether)\b",
    re.IGNORECASE,
)
_REASONING = re.compile(
    r"\b(?:court reasoned|judge found|court found|court concluded|in reaching (?:its|this) decision|court considered whether|we find that|we conclude that)\b",
    re.IGNORECASE,
)
_SUBMISSION = re.compile(
    r"\b(?:claimant|respondent|appellant|defendant|plaintiff|counsel)\b.{0,80}\b(?:argued|submitted|contended|requested|sought)\b",
    re.IGNORECASE,
)
_PRIOR_OR_QUOTED = re.compile(
    r"\b(?:previously|earlier|below|lower court|high court|quoted|quotation|in\s+.+?\s+v\.)\b",
    re.IGNORECASE,
)
_HOLDING = re.compile(
    r"\b(?:(?:the court|this court|we|accordingly|therefore)\b.{0,100})?(?:appeal (?:is )?(?:allowed|dismissed)|claim (?:is )?dismissed|application (?:is )?(?:granted|refused)|order (?:is )?set aside|judgment (?:is )?(?:affirmed|reversed)|matter (?:is )?remitted|allows? the appeal|dismisses? the appeal|grants? the application|refuses? the application)\b",
    re.IGNORECASE,
)
_STATUTE = re.compile(r"\b(?:section|article)\s+\d+[A-Za-z0-9()\-]*\b", re.IGNORECASE)
_AUTHORITY = re.compile(r"\b[A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,3}\s+v\.\s+[A-Z][A-Za-z'&.-]+(?:\s+[A-Z][A-Za-z'&.-]+){0,3}\b")


@dataclass(frozen=True)
class _JudgmentCandidate:
    text: str
    source: SourceReference | None
    order: int


class PrecedentSummarizer:
    """Extract only explicit judgment material from a precedent request.

    The service classifies direct language, not legal implications. In
    particular, requests for relief, party submissions, prior-court outcomes,
    and quoted authorities are excluded from current-court holding extraction.
    """

    def summarize(self, request: SummarizationRequest) -> PrecedentSummary:
        candidates, truncated = self._candidates(request)
        categories = self._categorize(candidates)
        metadata = request.judgment_metadata or JudgmentMetadata()
        points = [point for values in categories.values() for point in values]
        sources = _unique_sources(
            source for point in points for source in point.source_references
        )
        flags = self._flags(categories, sources, truncated, bool(candidates))
        return PrecedentSummary(
            case_identifier=_case_identifier(metadata),
            case_number=metadata.case_number,
            court=metadata.court,
            jurisdiction=metadata.jurisdiction,
            judgment_date=metadata.judgment_date,
            legal_issues=[point.text for point in categories["legal_issue"]],
            background=_join_points(categories["background"]),
            reasoning=_join_points(categories["reasoning"]),
            holding=_join_points(categories["holding"]),
            statutory_provisions=_matched_values(categories["statute"], _STATUTE),
            cited_authorities=_matched_values(categories["authority"], _AUTHORITY),
            points=points,
            source_references=sources,
            citations=_unique_citations(sources),
            uncertainty_flags=flags,
        )

    @staticmethod
    def _candidates(request: SummarizationRequest) -> tuple[list[_JudgmentCandidate], bool]:
        records: list[tuple[str, SourceReference | None]] = []
        records.extend((passage.text, passage.source) for passage in request.retrieved_passages)
        records.extend((item.text, item.source) for item in request.evidence_items)
        if request.evidence_text is not None:
            # A precedent_summary selector explicitly identifies this raw text as judgment material.
            records.append((request.evidence_text, None))

        candidates: list[_JudgmentCandidate] = []
        consumed = 0
        truncated = False
        for text, source in records:
            for sentence in _sentences(text):
                if (
                    len(candidates) >= MAX_JUDGMENT_CANDIDATES
                    or consumed + len(sentence) > MAX_INPUT_CHARACTERS
                ):
                    truncated = True
                    break
                candidates.append(_JudgmentCandidate(sentence, source, len(candidates)))
                consumed += len(sentence)
            if truncated:
                break
        return candidates, truncated

    @staticmethod
    def _categorize(
        candidates: list[_JudgmentCandidate],
    ) -> dict[str, list[PrecedentSummaryPoint]]:
        categories = {
            "background": [], "legal_issue": [], "reasoning": [], "holding": [],
            "statute": [], "authority": [],
        }
        seen: dict[str, set[str]] = {category: set() for category in categories}
        for candidate in candidates:
            for category in _categories_for(candidate.text):
                key = candidate.text.casefold()
                if key in seen[category] or len(categories[category]) >= MAX_POINTS_PER_CATEGORY:
                    continue
                categories[category].append(_point(candidate, category))
                seen[category].add(key)
        return categories

    @staticmethod
    def _flags(
        categories: dict[str, list[PrecedentSummaryPoint]],
        sources: list[SourceReference],
        truncated: bool,
        had_candidates: bool,
    ) -> list[str]:
        flags: list[str] = []
        if not had_candidates:
            flags.append("no_judgment_material")
        if had_candidates and not any(categories.values()):
            flags.append("no_safely_identifiable_judgment_content")
        if not categories["legal_issue"]:
            flags.append("legal_issue_not_explicitly_identified")
        if not categories["reasoning"]:
            flags.append("judicial_reasoning_not_explicitly_identified")
        if not categories["holding"]:
            flags.append("current_court_holding_not_explicitly_identified")
        if truncated:
            flags.append("judgment_input_truncated_at_safe_sentence_boundary")
        if any(point.provenance_limited for values in categories.values() for point in values):
            flags.append("limited_provenance")
        if not sources and any(categories.values()):
            flags.append("no_source_locator_or_citation_supplied")
        return flags


def _categories_for(text: str) -> list[str]:
    categories: list[str] = []
    if _BACKGROUND.search(text):
        categories.append("background")
    if _ISSUE.search(text):
        categories.append("legal_issue")
    if _REASONING.search(text) and not _SUBMISSION.search(text) and not _AUTHORITY.search(text):
        categories.append("reasoning")
    if _HOLDING.search(text) and not _SUBMISSION.search(text) and not _PRIOR_OR_QUOTED.search(text):
        categories.append("holding")
    if _STATUTE.search(text):
        categories.append("statute")
    if _AUTHORITY.search(text):
        categories.append("authority")
    return categories


def _point(candidate: _JudgmentCandidate, category: str) -> PrecedentSummaryPoint:
    sources = [candidate.source] if candidate.source is not None else []
    text = candidate.text.casefold()
    return PrecedentSummaryPoint(
        text=candidate.text,
        category=category,  # type: ignore[arg-type]
        source_references=sources,
        citations=_unique_citations(sources),
        provenance_limited=not sources,
        is_negated=bool(re.search(r"\b(?:not|no|never|denied|dismissed|refused)\b", text)),
        is_conditional=bool(re.search(r"\b(?:if|unless|subject to)\b", text)),
        is_attributed=bool(re.search(r"\b(?:alleged|claimed|stated|argued|submitted)\b", text)),
    )


def _case_identifier(metadata: JudgmentMetadata) -> str:
    if metadata.case_title and metadata.case_number:
        return f"{metadata.case_title} ({metadata.case_number})"
    return metadata.case_title or metadata.case_number or "unknown"


def _join_points(points: list[PrecedentSummaryPoint]) -> str | None:
    return " ".join(point.text for point in points) or None


def _matched_values(points: list[PrecedentSummaryPoint], pattern: re.Pattern[str]) -> list[str]:
    return list(dict.fromkeys(
        match.group(0) for point in points for match in pattern.finditer(point.text)
    ))


def _unique_sources(sources: object) -> list[SourceReference]:
    result: list[SourceReference] = []
    for source in sources:  # type: ignore[union-attr]
        if source is not None and source not in result:
            result.append(source)
    return result


def _unique_citations(sources: list[SourceReference]) -> list[str]:
    return list(dict.fromkeys(
        source.citation_id for source in sources if source.citation_id is not None
    ))
