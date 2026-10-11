"""Conservative deterministic comparison of supplied cases and precedents."""

from __future__ import annotations

import re

from agents.summarization.schemas import (
    CaseComparison,
    CaseComparisonRequest,
    ComparisonPoint,
    EvidenceSummaryPoint,
    PrecedentSummary,
    PrecedentSummaryPoint,
    SourceReference,
)

_TOPICS: dict[str, re.Pattern[str]] = {
    "payment_obligation": re.compile(r"\b(?:pay(?:ment)?|amount|lkr|rs\.?|usd|eur|due)\b", re.I),
    "notice_requirement": re.compile(r"\b(?:notice|served|service)\b", re.I),
    "delivery_obligation": re.compile(r"\b(?:deliver(?:y|ed)?|goods)\b", re.I),
    "termination_condition": re.compile(r"\b(?:terminat(?:e|ion|ed)|cancel(?:led|lation)?|unless|subject to)\b", re.I),
    "contract_execution": re.compile(r"\b(?:agreement|contract|lease|execut(?:e|ed|ion))\b", re.I),
    "procedural_issue": re.compile(r"\b(?:appeal|application|proceedings?|hearing|court)\b", re.I),
}
_AMOUNT = re.compile(r"\b(?:(LKR|Rs\.?|USD|EUR|GBP)\s?)([\d,]+(?:\.\d{1,2})?)\b", re.I)
_DATE = re.compile(
    r"\b(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
    re.I,
)
_PAYMENT_CONTEXT = re.compile(
    r"\b(?:must|shall|required|require[sd]?|agreed|due|invoice|obligation|not made|failed to pay|payment was not made)\b",
    re.I,
)


class CasePrecedentComparator:
    """Match aligned, supplied statements without legal applicability claims.

    A similarity means only that explicit statements share a controlled factual
    topic. It never means that the foreign decision applies to the current case
    or that either party should win.
    """

    def compare(self, request: CaseComparisonRequest) -> list[CaseComparison]:
        current_points = _current_points(request)
        return [
            self._compare_one(request, current_points, precedent)
            for precedent in request.precedents
        ]

    def _compare_one(
        self,
        request: CaseComparisonRequest,
        current_points: list[EvidenceSummaryPoint],
        precedent: PrecedentSummary,
    ) -> CaseComparison:
        precedent_points = _precedent_points(precedent)
        comparisons: list[ComparisonPoint] = []
        current_by_topic = _by_topic(current_points)
        precedent_by_topic = _by_topic(precedent_points)

        for topic in _TOPICS:
            current = current_by_topic.get(topic, [])
            prior = precedent_by_topic.get(topic, [])
            if current and prior:
                comparisons.extend(self._aligned_points(topic, current, prior))
            elif current:
                comparisons.append(_missing_point(topic, current[0], None))
            elif prior:
                comparisons.append(_missing_point(topic, None, prior[0]))

        if request.current_case_jurisdiction and precedent.jurisdiction:
            if request.current_case_jurisdiction.casefold() != precedent.jurisdiction.casefold():
                comparisons.append(ComparisonPoint(
                    comparison_type="jurisdiction",
                    topic="jurisdiction",
                    label=(
                        "The supplied current-case and precedent jurisdictions differ; "
                        "authority is not determined."
                    ),
                    limitations=["human_legal_review_required", "authority_not_determined"],
                ))

        sources = _unique_sources(
            source
            for point in comparisons
            for source in [*point.current_case_sources, *point.precedent_sources]
        )
        similarities = [point.label for point in comparisons if point.comparison_type == "similarity"]
        differences = [point.label for point in comparisons if point.comparison_type == "difference"]
        comparable_issues = list(dict.fromkeys(
            point.label for point in comparisons
            if point.comparison_type == "similarity" and point.topic in {"notice_requirement", "termination_condition", "contract_execution", "procedural_issue"}
        ))
        limitations = _limitations(current_points, precedent_points, comparisons)
        return CaseComparison(
            current_case_context=request.case_id,
            precedent_context=precedent.case_identifier,
            supported_similarities=similarities,
            supported_differences=differences,
            limitations=limitations,
            supporting_sources=sources,
            current_case_id=request.case_id,
            precedent_identifier=precedent.case_identifier,
            precedent_court=precedent.court,
            precedent_jurisdiction=precedent.jurisdiction,
            current_case_jurisdiction=request.current_case_jurisdiction,
            comparable_legal_issues=comparable_issues,
            points=comparisons,
            status="completed" if similarities or differences else "partial",
        )

    @staticmethod
    def _aligned_points(
        topic: str,
        current: list[EvidenceSummaryPoint],
        prior: list[PrecedentSummaryPoint],
    ) -> list[ComparisonPoint]:
        # One best-ordered pair per topic keeps comparison output reviewable.
        current_point, precedent_point = current[0], prior[0]
        differences = _explicit_differences(current_point.text, precedent_point.text)
        semantic_limits = _semantic_limitations(current_point, precedent_point)
        provenance_limited = current_point.provenance_limited or precedent_point.provenance_limited
        if differences:
            return [ComparisonPoint(
                comparison_type="difference",
                topic=topic,
                current_case_statement=current_point.text,
                precedent_statement=precedent_point.text,
                label=f"Explicit {topic.replace('_', ' ')} difference: {', '.join(differences)}.",
                current_case_sources=current_point.source_references,
                precedent_sources=precedent_point.source_references,
                limitations=semantic_limits,
                provenance_limited=provenance_limited,
            )]
        return [ComparisonPoint(
            comparison_type="similarity",
            topic=topic,
            current_case_statement=current_point.text,
            precedent_statement=precedent_point.text,
            label=f"Both supplied materials explicitly address {topic.replace('_', ' ')}.",
            current_case_sources=current_point.source_references,
            precedent_sources=precedent_point.source_references,
            limitations=semantic_limits,
            provenance_limited=provenance_limited,
        )]


def _current_points(request: CaseComparisonRequest) -> list[EvidenceSummaryPoint]:
    if request.current_case_summary is not None:
        return request.current_case_summary.points
    return request.current_case_points


def _precedent_points(precedent: PrecedentSummary) -> list[PrecedentSummaryPoint]:
    # A judgment field without source-grounded point is deliberately ignored.
    return precedent.points


def _by_topic(points: list[EvidenceSummaryPoint] | list[PrecedentSummaryPoint]) -> dict[str, list[EvidenceSummaryPoint | PrecedentSummaryPoint]]:
    result: dict[str, list[EvidenceSummaryPoint | PrecedentSummaryPoint]] = {}
    for point in points:
        for topic, pattern in _TOPICS.items():
            if pattern.search(point.text) and _is_meaningful_topic_match(topic, point.text):
                result.setdefault(topic, []).append(point)
    return result


def _is_meaningful_topic_match(topic: str, text: str) -> bool:
    """Avoid calling a bare shared word (for example, social payment) a legal match."""

    if topic == "payment_obligation":
        return bool(_PAYMENT_CONTEXT.search(text) or _AMOUNT.search(text) or _DATE.search(text))
    return True


def _explicit_differences(current: str, precedent: str) -> list[str]:
    differences: list[str] = []
    current_amounts = _AMOUNT.findall(current)
    precedent_amounts = _AMOUNT.findall(precedent)
    if current_amounts and precedent_amounts and current_amounts[0] != precedent_amounts[0]:
        differences.append("payment amount or currency")
    current_dates = _DATE.findall(current)
    precedent_dates = _DATE.findall(precedent)
    if current_dates and precedent_dates and current_dates[0] != precedent_dates[0]:
        differences.append("stated date or deadline")
    return differences


def _semantic_limitations(
    current: EvidenceSummaryPoint,
    precedent: PrecedentSummaryPoint,
) -> list[str]:
    limits: list[str] = []
    if current.is_attributed != precedent.is_attributed:
        limits.append("different_attribution_or_allegation_status")
    if current.is_negated != precedent.is_negated:
        limits.append("different_negation_status")
    if current.is_conditional != precedent.is_conditional:
        limits.append("different_conditional_status")
    if current.provenance_limited or precedent.provenance_limited:
        limits.append("limited_provenance")
    return limits


def _missing_point(
    topic: str,
    current: EvidenceSummaryPoint | None,
    precedent: PrecedentSummaryPoint | None,
) -> ComparisonPoint:
    if current is not None:
        label = (
            f"Current-case {topic.replace('_', ' ')} information was supplied; "
            "precedent information for that topic was not supplied."
        )
    else:
        label = (
            f"Precedent {topic.replace('_', ' ')} information was supplied; "
            "current-case information for that topic was not supplied."
        )
    return ComparisonPoint(
        comparison_type="missing_information",
        topic=topic,
        current_case_statement=current.text if current else None,
        precedent_statement=precedent.text if precedent else None,
        label=label,
        current_case_sources=current.source_references if current else [],
        precedent_sources=precedent.source_references if precedent else [],
        limitations=["missing_information_is_not_a_contradiction"],
        provenance_limited=(current.provenance_limited if current else False) or (precedent.provenance_limited if precedent else False),
    )


def _limitations(
    current: list[EvidenceSummaryPoint],
    precedent: list[PrecedentSummaryPoint],
    comparisons: list[ComparisonPoint],
) -> list[str]:
    limits = ["authority_not_determined", "human_legal_review_required"]
    if not current:
        limits.append("no_current_case_source_grounded_points")
    if not precedent:
        limits.append("no_precedent_source_grounded_points")
    if not any(point.comparison_type in {"similarity", "difference"} for point in comparisons):
        limits.append("no_supported_topic_alignment")
    if any(point.provenance_limited for point in comparisons):
        limits.append("limited_provenance")
    return limits


def _unique_sources(sources: object) -> list[SourceReference]:
    unique: list[SourceReference] = []
    for source in sources:  # type: ignore[union-attr]
        if source is not None and source not in unique:
            unique.append(source)
    return unique
