"""Regression validation for the synthetic analysis evaluation runner."""

from tests.evaluation.analysis.evaluate_analysis import run_evaluation


def test_evaluation_is_reproducible_and_audits_provenance() -> None:
    report = run_evaluation()
    assert report["documents"] == 20
    assert report["statuses"] == {"completed": 20}
    assert report["provenance"]["checked"] > 0
    assert report["provenance"]["failures"] == 0
    assert set(report["metrics"]) == {"entities", "facts", "clauses", "issues", "timeline", "contradictions"}
