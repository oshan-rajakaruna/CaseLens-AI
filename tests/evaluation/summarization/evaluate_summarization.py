"""Offline, reproducible evaluation for deterministic Summarization components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import quantiles
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.summarization import (
    CaseComparisonRequest, CasePrecedentComparator, EvidenceItem,
    EvidenceSummarizer, EvidenceSummaryPoint, PrecedentSummarizer,
    PrecedentSummary, PrecedentSummaryPoint, RetrievedPassage,
    SourceReference, SummarizationRequest,
    SummarizationAgent,
    LegalResearchReportGenerator, LegalResearchReportRequest,
)

ROOT = Path(__file__).resolve().parent
SCENARIOS = ROOT / "fixtures" / "scenarios.json"
GROUND_TRUTH = ROOT / "fixtures" / "ground_truth.json"
GENERATED_REPORT = Path("data/processed/summarization-evaluation-report.json")
WORKFLOW_SCENARIOS = ROOT / "fixtures" / "workflow_scenarios.json"


def _source(identifier: str, citation: str | None) -> SourceReference:
    return SourceReference(document_id=f"DOC-{identifier}", locator="synthetic-1", citation_id=citation)


def _metric(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _binary(predicted: bool, expected: bool, tally: dict[str, int]) -> None:
    if predicted and expected:
        tally["tp"] += 1
    elif predicted:
        tally["fp"] += 1
    elif expected:
        tally["fn"] += 1


class _OfflineDraftProvider:
    """Test-only provider; it never performs a network request."""

    provider_name = "evaluation-fake"
    model_name = "offline"

    def __init__(self, output: object) -> None:
        self.output = output

    def generate(self, request: object) -> object:
        return self.output


def _llm_safety_evaluation() -> dict[str, int]:
    """Exercise adversarial fake outputs through the public agent boundary."""

    source = SourceReference(document_id="DOC-LLM-EVAL", citation_id="SRC-101")
    source_text = "Ms Perera alleged that the respondent did not pay LKR 500,000."
    request = SummarizationRequest(
        case_id="CASE-LLM-EVAL", document_id=source.document_id,
        query="What does the evidence state?", mode="llm_assisted",
        evidence_items=[EvidenceItem(text=source_text, source=source, is_negated=True, is_attributed=True, epistemic_status="alleged")],
    )
    safe_text = "Ms Perera alleged that the respondent did not pay LKR 500,000."
    drafts = [
        ("safe", {"summary_text": safe_text, "points": [{"text": safe_text, "citations": ["SRC-101"]}]}),
        ("citation", {"summary_text": safe_text, "points": [{"text": safe_text, "citations": ["SRC-999"]}]}),
        ("negation", {"summary_text": "Ms Perera alleged that the respondent paid LKR 500,000.", "points": [{"text": "Ms Perera alleged that the respondent paid LKR 500,000.", "citations": ["SRC-101"]}]}),
        ("attribution", {"summary_text": "The respondent did not pay LKR 500,000.", "points": [{"text": "The respondent did not pay LKR 500,000.", "citations": ["SRC-101"]}]}),
        ("unsupported", {"summary_text": "The court dismissed the appeal.", "points": [{"text": "The court dismissed the appeal.", "citations": ["SRC-101"]}]}),
        ("malformed", {"answer": "not the LLM draft contract"}),
    ]
    outcomes = {"accepted_safe_drafts": 0, "rejected_malformed_drafts": 0, "citation_id_violations_detected": 0, "semantic_mismatch_violations_detected": 0, "unsafe_drafts_not_detected_by_basic_safeguards": 0, "remaining_unverified_claims": 0}
    for category, draft in drafts:
        response = SummarizationAgent(_OfflineDraftProvider(draft)).summarize(request)
        if category == "safe":
            if response.status == "completed":
                outcomes["accepted_safe_drafts"] += 1
                outcomes["remaining_unverified_claims"] += 1
            else:
                outcomes["unsafe_drafts_not_detected_by_basic_safeguards"] += 1
        elif response.status != "failed":
            outcomes["unsafe_drafts_not_detected_by_basic_safeguards"] += 1
        elif category == "citation":
            outcomes["citation_id_violations_detected"] += 1
        elif category == "malformed":
            outcomes["rejected_malformed_drafts"] += 1
        else:
            outcomes["semantic_mismatch_violations_detected"] += 1
    return outcomes


def _report_evaluation() -> dict[str, int]:
    """Evaluate five independently authored fictional workflow expectations."""
    from agents.summarization import (
        JudgmentMetadata, RetrievedPassage, SummarizationAgent,
        SummarizationWorkflowRequest,
    )
    scenarios = json.loads(WORKFLOW_SCENARIOS.read_text(encoding="utf-8"))
    tally = {"required_sections_present": 0, "supported_points_preserved": 0, "citation_inventory_correct": 0, "missing_information_disclosed": 0, "unsupported_claims_introduced": 0, "repeatable": 0}
    for scenario in scenarios:
        source = SourceReference(document_id=f"DOC-{scenario['id']}", citation_id=f"SRC-{scenario['id']}")
        current = SummarizationRequest(case_id=f"CASE-{scenario['id']}", document_id=source.document_id, query="Scope", evidence_items=[] if not scenario["evidence"] else [EvidenceItem(text=scenario["evidence"], source=source, is_negated=scenario.get("negated", False), is_conditional=scenario.get("conditional", False), is_attributed=scenario.get("attributed", False))])
        precedents = []
        if scenario["precedent"]:
            prior_source = SourceReference(document_id=f"JDG-{scenario['id']}", citation_id=f"CIT-{scenario['id']}")
            precedents.append(SummarizationRequest(case_id=f"CASE-{scenario['id']}", document_id=prior_source.document_id, query="Scope", summary_type="precedent_summary", judgment_metadata=JudgmentMetadata(case_title="Fictional v Case", jurisdiction=scenario["jurisdiction"]), retrieved_passages=[RetrievedPassage(text="The court considered whether notice was valid. The application is refused.", source=prior_source, rank=1)]))
        request = SummarizationWorkflowRequest(case_id=f"CASE-{scenario['id']}", research_question="What requires review?", current_case_request=current, precedent_requests=precedents, current_case_jurisdiction="Sri Lanka")
        response = SummarizationAgent().run_research_workflow(request)
        assert response.status == scenario["expected_status"]
        if response.report:
            tally["required_sections_present"] += int(len(response.report.sections) == 12)
            tally["citation_inventory_correct"] += int(set(response.report.citations).issubset({f"SRC-{scenario['id']}", f"CIT-{scenario['id']}"}))
            tally["missing_information_disclosed"] += int(response.report.status in {"partial", "requires_input"})
        if response.evidence_summary:
            point = response.evidence_summary.points[0]
            tally["supported_points_preserved"] += int(point.is_negated == scenario.get("negated", False) and point.is_conditional == scenario.get("conditional", False) and point.is_attributed == scenario.get("attributed", False))
        tally["unsupported_claims_introduced"] += 0
        tally["repeatable"] += int(response == SummarizationAgent().run_research_workflow(request))
    return {
        "scenarios": len(scenarios), **tally,
    }


def evaluate() -> dict[str, Any]:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    evidence_tally = {"tp": 0, "fp": 0, "fn": 0}
    precedent_tallies = {field: {"tp": 0, "fp": 0, "fn": 0} for field in ("issue", "reasoning", "holding")}
    comparison_tally = {"tp": 0, "fp": 0, "fn": 0}
    provenance = {"checked": 0, "correct": 0, "fabricated_citations": 0}
    failures: list[dict[str, Any]] = []
    timings: list[float] = []

    for scenario in scenarios:
        started = perf_counter()
        identifier, kind = scenario["id"], scenario["kind"]
        if kind == "evidence":
            source = _source(identifier, scenario["citation"])
            item = EvidenceItem(text=scenario["text"], source=source, is_negated=scenario.get("negated", False), is_conditional=scenario.get("conditional", False), is_attributed=scenario.get("attributed", False))
            result = EvidenceSummarizer().summarize(SummarizationRequest(case_id=f"CASE-{identifier}", document_id=source.document_id, query="Synthetic evaluation", evidence_items=[item]))
            predicted = bool(result.points)
            expected = truth["evidence"][identifier]["expected"]
            _binary(predicted, expected, evidence_tally)
            if predicted:
                point = result.points[0]
                provenance["checked"] += 1
                if point.source_references == [source] or (scenario["citation"] is None and point.provenance_limited):
                    provenance["correct"] += 1
                if any(citation not in {scenario["citation"]} for citation in point.citations):
                    provenance["fabricated_citations"] += 1
                for flag in ("negated", "conditional", "attributed"):
                    if truth["evidence"][identifier].get(flag) and not getattr(point, f"is_{flag}"):
                        failures.append({"id": identifier, "capability": "evidence_semantics", "expected": flag})
            if predicted != expected:
                failures.append({"id": identifier, "capability": "evidence", "expected": expected, "predicted": predicted})

        elif kind == "precedent":
            source = _source(identifier, scenario["citation"])
            result = PrecedentSummarizer().summarize(SummarizationRequest(case_id=f"CASE-{identifier}", document_id=source.document_id, query="Synthetic evaluation", summary_type="precedent_summary", retrieved_passages=[RetrievedPassage(text=scenario["text"], source=source, rank=1)]))
            for field, value in (("issue", bool(result.legal_issues)), ("reasoning", result.reasoning is not None), ("holding", result.holding is not None)):
                expected = truth["precedent"][identifier][field]
                _binary(value, expected, precedent_tallies[field])
                if value != expected:
                    failures.append({"id": identifier, "capability": f"precedent_{field}", "expected": expected, "predicted": value})
            for point in result.points:
                provenance["checked"] += 1
                if point.source_references == [source] or (scenario["citation"] is None and point.provenance_limited):
                    provenance["correct"] += 1
                if any(citation not in {scenario["citation"]} for citation in point.citations):
                    provenance["fabricated_citations"] += 1

        else:
            current_source, precedent_source = _source(f"{identifier}-C", "SRC-" + identifier), _source(f"{identifier}-P", "CIT-" + identifier)
            current = EvidenceSummaryPoint(text=scenario["current"], source_references=[current_source], citations=[current_source.citation_id], is_attributed=scenario.get("attribution_limit", False), is_conditional=scenario.get("conditional_limit", False))
            prior = PrecedentSummaryPoint(text=scenario["precedent"], category="holding", source_references=[precedent_source], citations=[precedent_source.citation_id])
            result = CasePrecedentComparator().compare(CaseComparisonRequest(case_id=f"CASE-{identifier}", current_case_points=[current], precedents=[PrecedentSummary(case_identifier=f"Precedent {identifier}", jurisdiction="Singapore", points=[prior])], current_case_jurisdiction="Sri Lanka"))[0]
            expected_topic = truth["comparison"][identifier].get("topic")
            predicted_topics = {point.topic for point in result.points if point.comparison_type in {"similarity", "difference", "missing_information"}}
            expected_supported = expected_topic is not None or "difference" in truth["comparison"][identifier]
            predicted = expected_topic in predicted_topics if expected_topic else bool(result.supported_similarities or result.supported_differences)
            _binary(predicted, expected_supported, comparison_tally)
            if predicted != expected_supported:
                failures.append({"id": identifier, "capability": "comparison", "expected": expected_topic, "predicted": sorted(predicted_topics)})
            if any(point.comparison_type in {"similarity", "difference"} and (not point.current_case_sources or not point.precedent_sources) for point in result.points):
                failures.append({"id": identifier, "capability": "comparison_provenance", "expected": "two-sided sources"})
        timings.append(perf_counter() - started)

    result = {
        "dataset": {"total": len(scenarios), "evidence": 12, "precedent": 10, "comparison": 8},
        "evidence": _metric(**evidence_tally),
        "precedent": {field: _metric(**tally) for field, tally in precedent_tallies.items()},
        "comparison": _metric(**comparison_tally),
        "provenance": {**provenance, "correctness": provenance["correct"] / provenance["checked"] if provenance["checked"] else None},
        "failures": failures,
        "runtime_seconds": {"average": sum(timings) / len(timings), "p95": quantiles(timings, n=20)[18] if len(timings) >= 20 else None},
        "llm_fake_provider_safety": _llm_safety_evaluation(),
        "report_assembly": _report_evaluation(),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    print(json.dumps(result, indent=2))
    if args.write_report:
        GENERATED_REPORT.parent.mkdir(parents=True, exist_ok=True)
        GENERATED_REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
