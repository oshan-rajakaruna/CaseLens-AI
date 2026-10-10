"""Tests for future summarizer citation and output contracts."""

from pydantic import ValidationError
import pytest

from agents.summarization import SummarizationOutput, validate_context_citations


def test_validator_handles_no_citations() -> None:
    result = validate_context_citations("No source reference.", ["CTX-001"])

    assert result.has_citations is False
    assert result.cited_context_ids == []
    assert result.all_citations_valid is True


def test_validator_accepts_one_multiple_and_repeated_citations() -> None:
    result = validate_context_citations(
        "First [CTX-001], again [CTX-001], then [CTX-002].",
        ["CTX-001", "CTX-002"],
    )

    assert result.cited_context_ids == ["CTX-001", "CTX-001", "CTX-002"]
    assert result.valid_citations == ["CTX-001", "CTX-002"]
    assert result.invalid_citations == []
    assert result.all_citations_valid is True


def test_validator_reports_unknown_and_malformed_context_references() -> None:
    result = validate_context_citations(
        "Known [CTX-001], unknown [CTX-999], malformed [CTX-2] and CTX-003.",
        ["CTX-001"],
    )

    assert result.valid_citations == ["CTX-001"]
    assert result.invalid_citations == ["CTX-999"]
    assert result.malformed_citations == ["[CTX-2]", "CTX-003"]
    assert result.all_citations_valid is False


def test_validator_does_not_modify_generated_text() -> None:
    text = "Evidence [CTX-001]."

    validate_context_citations(text, ["CTX-001"])

    assert text == "Evidence [CTX-001]."


def test_summarization_output_schema_accepts_verifiable_contract() -> None:
    output = SummarizationOutput(
        answer="A supported answer [CTX-001].",
        citations=["CTX-001"],
        confidence=0.75,
        limitations=["Limited query set"],
    )

    assert output.citations == ["CTX-001"]
    assert output.confidence == 0.75


@pytest.mark.parametrize(
    "values",
    [
        {"answer": "   "},
        {"answer": "Answer", "citations": ["DOC-001"]},
        {"answer": "Answer", "confidence": 1.5},
    ],
)
def test_summarization_output_schema_rejects_invalid_values(
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SummarizationOutput.model_validate(values)
