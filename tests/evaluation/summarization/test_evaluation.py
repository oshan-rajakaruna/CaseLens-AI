"""Regression checks for the offline fictional summarization evaluation."""

from tests.evaluation.summarization.evaluate_summarization import evaluate


def test_evaluation_dataset_and_metrics_are_reproducible() -> None:
    first = evaluate()
    second = evaluate()

    assert first["dataset"] == {"total": 30, "evidence": 12, "precedent": 10, "comparison": 8}
    assert first["evidence"]["tp"] + first["evidence"]["fn"] == 11
    assert first["provenance"]["fabricated_citations"] == 0
    assert first["evidence"] == second["evidence"]
    assert first["precedent"] == second["precedent"]
    assert first["comparison"] == second["comparison"]
    assert first["llm_fake_provider_safety"] == {
        "accepted_safe_drafts": 1,
        "rejected_malformed_drafts": 1,
        "citation_id_violations_detected": 1,
        "semantic_mismatch_violations_detected": 3,
        "unsafe_drafts_not_detected_by_basic_safeguards": 0,
        "remaining_unverified_claims": 1,
    }
    assert first["report_assembly"] == {
        "scenarios": 5,
        "required_sections_present": 5,
        "supported_points_preserved": 4,
        "citation_inventory_correct": 5,
        "missing_information_disclosed": 5,
        "unsupported_claims_introduced": 0,
        "repeatable": 5,
    }
