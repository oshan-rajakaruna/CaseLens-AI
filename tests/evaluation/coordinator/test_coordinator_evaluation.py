"""Compact synthetic functional checks for Coordinator provenance and safety."""

from agents.coordinator import Coordinator, CoordinatorRequest
from agents.coordinator.summarization_adapter import CoordinatorSummarizationAdapter
from tests.unit.coordinator.test_summarization_adapter import _result


def test_synthetic_unsupported_material_is_repeatable_and_not_a_precedent() -> None:
    request = CoordinatorRequest(case_id="CASE-EVAL-001", user_query="fictional query", requested_task="summarize_precedents")
    first = CoordinatorSummarizationAdapter().summarize_precedents(request, [_result(document_type="contract")])
    second = CoordinatorSummarizationAdapter().summarize_precedents(request, [_result(document_type="contract")])

    assert first == second
    assert first[0] == []
    assert first[1] == {}
    assert first[2][0].code == "unsupported_precedent_document_type"
