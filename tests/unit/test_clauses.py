"""Tests for conservative contractual clause extraction."""

from nlp.clauses import extract_clauses


def test_obligation_prohibition_permission_and_payment_term() -> None:
    text = "The tenant shall pay monthly payment of Rs. 5000. The tenant must not sublet. The landlord may inspect the premises."
    clauses = extract_clauses(text, "d")
    assert [clause.clause_type for clause in clauses] == ["obligation", "prohibition", "permission_or_right"]
    assert clauses[0].money_text == "Rs. 5000"


def test_condition_deadline_termination_and_attribution() -> None:
    text = "If payment is not made within 14 days, the agreement may be terminated. The witness alleged that the agreement required payment."
    clauses = extract_clauses(text, "d")
    assert clauses[0].clause_type == "termination"
    assert clauses[0].deadline_text == "within 14 days"
    assert clauses[0].epistemic_status == "uncertain"
    assert clauses[0].is_conditional is True
    assert clauses[1].epistemic_status == "alleged"


def test_clause_offsets_empty_and_unsupported_text() -> None:
    text = "The party is entitled to inspect records."
    clause = extract_clauses(text, "d")[0]
    assert text[clause.source.start_char:clause.source.end_char] == clause.text
    assert extract_clauses(" \t", "d") == []
    assert extract_clauses("A witness arrived.", "d") == []
