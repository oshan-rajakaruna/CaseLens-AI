"""Tests for conservative, evidence-grounded issue candidates."""

import pytest
from pydantic import ValidationError

from agents.analysis.schemas import EvidenceSource, ExtractedClause, ExtractedFact
from nlp.issues import IssueDefinition, IssueTaxonomy, identify_legal_issues


def fact(category: str, text: str, **kwargs) -> ExtractedFact:
    return ExtractedFact(statement=text, category=category, source=EvidenceSource(document_id="d", page=1, locator="page:1", start_char=0, end_char=len(text), excerpt=text), method="test", **kwargs)


def clause(kind: str, text: str, **kwargs) -> ExtractedClause:
    return ExtractedClause(text=text, clause_type=kind, source=EvidenceSource(document_id="d", paragraph=1, locator="paragraph:1", start_char=0, end_char=len(text), excerpt=text), method="test", **kwargs)


def test_payment_dispute_alleged_breach_and_exact_provenance() -> None:
    source_fact = fact("non_payment_or_breach", "Saman alleges that Nimal failed to pay LKR 50,000.", epistemic_status="alleged", is_attributed=True)
    issues = identify_legal_issues([source_fact], [])
    assert [(item.category_id, item.epistemic_status) for item in issues] == [("alleged_contractual_breach", "alleged"), ("contractual_payment_dispute", "alleged")]
    assert all(item.review_status == "needs_review" for item in issues)
    assert issues[0].sources[0].excerpt == source_fact.statement


def test_no_false_dispute_from_signed_agreement_or_payment_term() -> None:
    issues = identify_legal_issues([fact("agreement_execution", "The parties signed the agreement.")], [clause("payment_term", "The tenant shall pay LKR 50,000 monthly.")])
    assert [item.category_id for item in issues] == ["contractual_obligation_or_condition"]


def test_notice_denial_and_conditional_termination_are_review_candidates() -> None:
    issues = identify_legal_issues([fact("notice_or_communication", "Kamal denies receiving notice.", epistemic_status="disputed", is_negated=True)], [clause("termination", "If payment is not made, the agreement may be terminated.", epistemic_status="uncertain", is_conditional=True)])
    assert [(item.category_id, item.epistemic_status) for item in issues] == [("notice_or_communication_dispute", "disputed"), ("termination_related_issue", "uncertain")]
    assert "occurred" not in issues[1].evidence_summary


def test_deduplication_empty_and_configurable_taxonomy() -> None:
    repeated = fact("non_payment_or_breach", "The tenant did not pay.", epistemic_status="stated")
    taxonomy = IssueTaxonomy(definitions=[IssueDefinition(category_id="contractual_payment_dispute", name="Payment review")])
    issues = identify_legal_issues([repeated, repeated], [], taxonomy)
    assert issues[0].label == "Payment review"
    assert len(issues[0].sources) == 1
    assert identify_legal_issues([], []) == []
    with pytest.raises(ValidationError):
        IssueTaxonomy(definitions=[IssueDefinition(category_id="x", name="X"), IssueDefinition(category_id="x", name="Again")])
