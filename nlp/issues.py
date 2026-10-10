"""Conservative, configurable issue candidates from grounded facts and clauses."""

from collections import defaultdict
from typing import Iterable

from pydantic import BaseModel, Field, model_validator

from agents.analysis.schemas import ExtractedClause, ExtractedFact, LegalIssue


class IssueDefinition(BaseModel):
    """A provisional, team-reviewable legal research category."""

    category_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class IssueTaxonomy(BaseModel):
    """Validated MVP taxonomy; the team must approve replacements."""

    definitions: list[IssueDefinition]

    @model_validator(mode="after")
    def unique_categories(self) -> "IssueTaxonomy":
        if len({item.category_id for item in self.definitions}) != len(self.definitions):
            raise ValueError("taxonomy category IDs must be unique")
        return self


DEFAULT_TAXONOMY = IssueTaxonomy(definitions=[
    IssueDefinition(category_id="contract_execution_dispute", name="Contract execution dispute"),
    IssueDefinition(category_id="contractual_payment_dispute", name="Contractual payment dispute"),
    IssueDefinition(category_id="alleged_contractual_breach", name="Alleged contractual breach"),
    IssueDefinition(category_id="notice_or_communication_dispute", name="Notice or communication dispute"),
    IssueDefinition(category_id="termination_related_issue", name="Termination-related issue"),
    IssueDefinition(category_id="contractual_obligation_or_condition", name="Contractual obligation or condition"),
])


def identify_legal_issues(facts: Iterable[ExtractedFact], clauses: Iterable[ExtractedClause], taxonomy: IssueTaxonomy | None = None) -> list[LegalIssue]:
    """Return neutral, review-required candidates supported by supplied findings."""
    definitions = {item.category_id: item for item in (taxonomy or DEFAULT_TAXONOMY).definitions}
    evidence: dict[str, list] = defaultdict(list)
    statuses: dict[str, list[str]] = defaultdict(list)

    for fact in facts:
        targets: list[str] = []
        if fact.category == "non_payment_or_breach":
            targets.append("contractual_payment_dispute")
            if fact.epistemic_status in {"alleged", "disputed"} or fact.is_attributed:
                targets.append("alleged_contractual_breach")
        elif fact.category == "notice_or_communication" and fact.epistemic_status == "disputed":
            targets.append("notice_or_communication_dispute")
        elif fact.category == "agreement_execution" and (fact.epistemic_status in {"alleged", "disputed"} or fact.is_negated):
            targets.append("contract_execution_dispute")
        for target in targets:
            if target in definitions:
                evidence[target].append(fact.source)
                statuses[target].append(fact.epistemic_status)

    for clause in clauses:
        target = None
        if clause.clause_type == "termination":
            target = "termination_related_issue"
        elif clause.clause_type in {"obligation", "condition", "deadline", "payment_term"}:
            target = "contractual_obligation_or_condition"
        if target and target in definitions:
            evidence[target].append(clause.source)
            statuses[target].append(clause.epistemic_status)

    issues = []
    for category_id in sorted(evidence):
        sources = _unique_sources(evidence[category_id])
        definition = definitions[category_id]
        status = "alleged" if "alleged" in statuses[category_id] else ("disputed" if "disputed" in statuses[category_id] else ("uncertain" if "uncertain" in statuses[category_id] else "stated"))
        issues.append(LegalIssue(label=definition.name, category_id=category_id, review_status="needs_review", epistemic_status=status, evidence_summary="Potential issue identified from explicit extracted evidence; legal review required.", sources=sources, method="rule-based-issue"))
    return issues


def _unique_sources(sources: list):
    unique = []
    seen = set()
    for source in sources:
        key = source.model_dump_json()
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique
