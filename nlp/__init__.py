"""Reusable local NLP and evidence-processing components."""

from nlp.extraction import DocumentExtractionResult, ExtractionSegment, extract_document
from nlp.entities import DEFAULT_SPACY_MODEL, extract_entities, load_spacy_model
from nlp.facts import extract_facts
from nlp.clauses import extract_clauses
from nlp.issues import DEFAULT_TAXONOMY, IssueDefinition, IssueTaxonomy, identify_legal_issues
from nlp.timeline import build_timeline, normalize_date
from nlp.contradictions import detect_contradictions

__all__ = [
    "DEFAULT_SPACY_MODEL",
    "DocumentExtractionResult",
    "ExtractionSegment",
    "extract_document",
    "extract_entities",
    "extract_facts",
    "extract_clauses",
    "DEFAULT_TAXONOMY",
    "IssueDefinition",
    "IssueTaxonomy",
    "identify_legal_issues",
    "build_timeline",
    "normalize_date",
    "detect_contradictions",
    "load_spacy_model",
]
