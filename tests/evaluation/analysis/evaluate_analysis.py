"""Reproducible quantitative evaluation for the synthetic Analysis Agent corpus.

Run: python tests/evaluation/analysis/evaluate_analysis.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from typing import Iterable

import spacy

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agents.analysis import AnalysisAgent, AnalysisRequest
from backend.schemas import Case, Document

ROOT = Path(__file__).parent


def _scores(predicted: set[tuple], expected: set[tuple]) -> dict[str, float | int]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": len(expected)}


def _fact_key(item) -> tuple:
    return (item.category, item.source.excerpt, item.epistemic_status, item.is_negated, item.is_conditional, item.is_attributed)


def _clause_key(item) -> tuple:
    return (item.clause_type, item.source.excerpt, item.is_conditional)


def _sources(response) -> Iterable:
    for collection in (response.entities, response.facts, response.clauses, response.timeline):
        yield from (item.source for item in collection)
    for issue in response.legal_issues:
        yield from issue.sources
    for candidate in response.contradiction_candidates:
        yield from candidate.sources


def run_evaluation() -> dict:
    scenarios = json.loads((ROOT / "fixtures" / "scenarios.json").read_text(encoding="utf-8"))
    truth = json.loads((ROOT / "ground_truth.json").read_text(encoding="utf-8"))["expected"]
    agent = AnalysisAgent()
    predicted = {"entities": set(), "facts": set(), "clauses": set(), "issues": set(), "timeline": set(), "contradictions": set()}
    expected = {key: set() for key in predicted}
    provenance = {"checked": 0, "failures": 0}
    statuses: dict[str, int] = {}
    lengths = []
    for scenario in scenarios:
        scenario_id, text = scenario["id"], scenario["text"]
        lengths.append(len(text))
        response = agent.analyze(AnalysisRequest(case=Case(id=f"case-{scenario_id}", title="Synthetic evaluation"), document=Document(id=f"doc-{scenario_id}", case_id=f"case-{scenario_id}", name=f"{scenario_id}.txt"), text=text))
        statuses[response.status] = statuses.get(response.status, 0) + 1
        annotation = truth.get(scenario_id, {})
        if "entities" in annotation:
            for entity_text, label in annotation["entities"]:
                start = text.index(entity_text)
                expected["entities"].add((scenario_id, label, start, start + len(entity_text)))
            predicted["entities"].update((scenario_id, item.label, item.source.start_char, item.source.end_char) for item in response.entities)
        for item in annotation.get("facts", []):
            expected["facts"].add((scenario_id, *item))
        predicted["facts"].update((scenario_id, *_fact_key(item)) for item in response.facts)
        for item in annotation.get("clauses", []):
            expected["clauses"].add((scenario_id, *item))
        predicted["clauses"].update((scenario_id, *_clause_key(item)) for item in response.clauses)
        expected["issues"].update((scenario_id, item) for item in annotation.get("issues", []))
        predicted["issues"].update((scenario_id, item.category_id) for item in response.legal_issues)
        expected["timeline"].update((scenario_id, *item) for item in annotation.get("timeline", []))
        predicted["timeline"].update((scenario_id, item.normalized_date, item.date_precision) for item in response.timeline)
        expected["contradictions"].update((scenario_id, item) for item in annotation.get("contradictions", []))
        predicted["contradictions"].update((scenario_id, item.category) for item in response.contradiction_candidates)
        for source in _sources(response):
            if source.start_char is None or source.end_char is None or source.excerpt is None:
                continue
            provenance["checked"] += 1
            if text[source.start_char:source.end_char] != source.excerpt or source.document_id != f"doc-{scenario_id}":
                provenance["failures"] += 1
    return {"documents": len(scenarios), "document_lengths": {"min": min(lengths), "max": max(lengths), "median": median(lengths)}, "statuses": statuses, "metrics": {key: _scores(predicted[key], expected[key]) for key in predicted}, "provenance": provenance, "environment": {"python": platform.python_version(), "spacy": spacy.__version__, "model": "en_core_web_sm"}}


def benchmark(iterations: int = 7) -> dict:
    text = "Nimal Perera paid LKR 50,000 on 12 January 2025. The tenant shall pay monthly rent of Rs. 50,000."
    request = AnalysisRequest(case=Case(id="case-benchmark", title="Synthetic benchmark"), document=Document(id="doc-benchmark", case_id="case-benchmark", name="benchmark.txt"), text=text)
    cold_start = perf_counter(); AnalysisAgent().analyze(request); cold = perf_counter() - cold_start
    agent = AnalysisAgent(); timings = []
    for _ in range(iterations):
        start = perf_counter(); agent.analyze(request); timings.append(perf_counter() - start)
    ordered = sorted(timings)
    return {"iterations": iterations, "cold_seconds": cold, "warm_median_seconds": median(timings), "warm_p95_seconds": ordered[round((len(ordered) - 1) * 0.95)]}


if __name__ == "__main__":
    print(json.dumps({"evaluation": run_evaluation(), "benchmark": benchmark()}, indent=2))
