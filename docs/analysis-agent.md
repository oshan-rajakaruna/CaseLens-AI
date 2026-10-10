# Analysis Agent foundation

## Purpose

The Analysis Agent is Member 3's local processing boundary for evidence and
case-material analysis. It will eventually coordinate reusable NLP components
for entities, facts, clauses, issues, timelines, and candidate contradictions.
It has no FastAPI, database, Coordinator, model-provider, or network dependency.

## Supported input

`AnalysisRequest` accepts the existing shared `Case` and `Document` models,
supplied document text, and a flexible `document_metadata` mapping. The request
requires non-blank case and document identifiers and requires the document's
`case_id` to match the case identifier. Evidence text is retained as supplied.

## Structured output

`AnalysisResponse` contains the case and document identifiers, a status, typed
collections for each future extraction type, and structured warnings/errors.
`EvidenceSource` records a document identifier, optional page/paragraph or
locator, optional character span, and an exact supplied excerpt. Confidence,
where supplied by future processors, must be from 0 through 1.

The current statuses are:

- `not_analyzed`: text is valid but NLP processing is not implemented.
- `requires_input`: text is empty or whitespace-only.
- `failed`: reserved for a future controlled processing failure.

## Local usage

```python
from agents.analysis import AnalysisAgent, AnalysisRequest
from backend.schemas import Case, Document

request = AnalysisRequest(
    case=Case(id="case-001", title="Example case"),
    document=Document(id="doc-001", case_id="case-001", name="evidence.txt"),
    text="The agreement was signed on 1 January 2024.",
    document_metadata={"language": "en"},
)

response = AnalysisAgent().analyze(request)
print(response.model_dump_json())
```

## Current limitations

The core Analysis Agent still validates local input only. It does not yet invoke
the standalone NER module, and it deliberately performs no fact or clause
extraction, legal issue classification, timeline generation, contradiction
detection, OCR, API call, or legal conclusion. Empty result collections mean no
processor has populated them.

## Future integration points

`nlp.extraction.extract_document()` supports local `.txt`, `.pdf`, and `.docx`
evidence extraction. It accepts a path and document ID and returns a
`DocumentExtractionResult`; it does not upload or log evidence. TXT uses strict
UTF-8 decoding (including UTF-8 BOM), PDF uses PyMuPDF, and DOCX uses
python-docx. Input files are size-limited (25 MiB by default), directories and
unsupported types are rejected, and file-type signatures are checked before PDF
or DOCX parsing.

Each result contains ordered `ExtractionSegment` values. Segments retain exact
extracted text and an `EvidenceSource`: PDFs use 1-based `page:N` locators,
DOCX uses `paragraph:N` or `table:N`, and TXT uses `text:1`. The combined
result text joins adjacent source segments with one newline. Offsets are Python
string indices, start-inclusive/end-exclusive, so
`result.text[start_char:end_char] == segment.text` for every segment.

```python
from agents.analysis import AnalysisAgent, AnalysisRequest
from backend.schemas import Case, Document
from nlp.extraction import extract_document

extraction = extract_document("evidence.pdf", "doc-001")
request = AnalysisRequest(
    case=Case(id="case-001", title="Example case"),
    document=Document(id=extraction.document_id, case_id="case-001", name=extraction.filename),
    text=extraction.text,
    document_metadata=extraction.metadata,
)
response = AnalysisAgent().analyze(request)
```

Extraction `status` is `extracted`, `no_text`, or `failed`. Failures contain a
structured error and no fabricated text. Empty files/documents are `no_text`.
PDFs with no extractable text receive an OCR-required warning; OCR is not part
of this project stage. Extraction does not perform legal analysis: a nonempty
extraction still receives the Analysis Agent's `not_analyzed` status.

### Named Entity Recognition

`nlp.entities.extract_entities()` is a separate hybrid English NER utility. It
uses a cached local spaCy model (`en_core_web_sm` by default) for `PERSON`,
`ORGANIZATION`, `LOCATION`, `DATE`, and `MONEY`, alongside precise deterministic
rules for `COURT`, `CASE_REFERENCE`, `STATUTE_REFERENCE`, Sri Lankan rupee
amounts, and common English legal-date formats. Every entity retains exact
original text, start-inclusive/end-exclusive Python offsets, and an
`EvidenceSource` with the extraction segment's page, paragraph, or table
locator where the full entity belongs to one segment.

Install the package and model separately; model download never happens at
runtime:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```python
from nlp.entities import extract_entities

entities = extract_entities(
    "The Supreme Court of Sri Lanka considered SC FR 123/2024.",
    "doc-001",
)
# ExtractedEntity(text="Supreme Court of Sri Lanka", label="COURT", ...)
```

If spaCy or its configured model is unavailable, or the requested language is
not English, spaCy is not used. Explicit legal rules still run and identify
their results with `method="rule-based-fallback"`; the function never claims a
model result occurred. Rule detections outrank overlapping spaCy detections
(for example, a court rule replaces a generic spaCy ORG label). Same-text
occurrences at different offsets remain distinct. An entity that crosses source
segments receives `locator="multiple_segments"` and no misleading single-page
claim.

This is an English baseline, not an evaluated Sri Lankan legal NER model. It
does not guarantee recognition of local names, courts, laws, Sinhala, or Tamil.
Future work should evaluate multilingual/domain-specific models and integrate
NER outputs into the Analysis Agent only after a shared response contract is
agreed.

### Fact and clause extraction

`nlp.facts.extract_facts()` extracts only explicit English sentence-level
statements about agreement execution, payments, non-payment/breach, notices,
documented actions, and described proceedings. It returns the exact supporting
sentence, a category, any directly contained supplied entities, and an explicit
epistemic status: `alleged`, `disputed`, `uncertain`, or `stated`. For example,
“ABC alleges that Nimal failed to pay rent” remains an alleged statement; “did
not sign” and modal “may” wording remain in the source proposition.

`nlp.clauses.extract_clauses()` identifies explicit obligations, prohibitions,
permissions/rights, conditions, payment terms, deadlines, and termination
language. A witness's alleged description of a clause is retained as
`epistemic_status="alleged"`, not treated as a verified contractual term.
Amounts and deadlines are populated only when directly written in the clause.

Both modules reuse extraction-segment provenance and return no result for empty
or unsupported narrative. They are deterministic rule baselines: they do not
verify the truth of evidence, resolve pronouns, infer unstated relationships,
or provide legal advice. Future evaluation needs labelled, approved Sri Lankan
legal evidence and reviewer agreement on categories and epistemic labels.

### Legal issue candidates

`nlp.issues.identify_legal_issues(facts, clauses, taxonomy=None)` groups only
already extracted, source-grounded findings into neutral, `needs_review` issue
candidates. The provisional configurable taxonomy contains contract execution
dispute, payment dispute, alleged breach, notice dispute, termination-related,
and obligation/condition categories. It never decides liability, validity,
guilt, or court outcomes. A signed agreement or ordinary payment term alone is
not a dispute; a conditional termination clause is not a completed termination.
Allegation, denial, and uncertainty statuses are carried into each issue and
all distinct source references are retained. The team and legal supervisors
must approve any taxonomy replacement and evaluate it with labelled samples.

### Timeline

`nlp.timeline.build_timeline()` builds only dated event candidates from supplied
facts; clauses and deadlines are deliberately not historical events. It supports
ISO, English day/month/year, month/year, year-only, and explicitly configured
numeric dates. Ambiguous numeric, invalid, relative, and undated expressions
are omitted rather than guessed. Day/month/year precision is retained, events
sort by normalized earliest position, and source citations plus allegation,
negation, attribution, conditionality, and verification semantics are copied
unchanged. A timeline is evidence assistance, not an independently verified
case history.

### Integrated Analysis Agent

`AnalysisAgent.analyze()` now orchestrates entities, facts, clauses, issues,
timeline, and contradiction candidates over supplied text. It remains local and
does not open files, call APIs, or claim legal verification. It reports
`completed`, `partial`, `failed`, or `requires_input`, with per-stage status and
structured non-sensitive errors. Optional extraction segments preserve PDF/DOCX
locators through downstream findings. Contradictions remain document-only;
cross-document analysis requires a future validated case aggregation layer.

### Evaluation

The fictional, provisional Step 10 benchmark and reproducible metric runner
live in `tests/evaluation/analysis/`; measured results, provenance audit,
performance caveats, and threats to validity are recorded in
`docs/analysis-evaluation.md`. Synthetic completion is never a claim of legal
verification or real-world accuracy.

### Contradiction candidates

`nlp.contradictions.detect_contradictions()` is a narrow O(n²), review-only
comparison over caller-supplied facts in an explicit case context. It currently
supports only `affirmation_vs_denial`: same fact category, identical supplied
subject set, explicit same date and amount, matching normalized statement core,
and opposite negation. Conditional facts, missing identity, different dates or
amounts, obligations, and repeated claims are rejected. Candidates preserve both
sources, attribution/epistemic context, deterministic IDs, and
`review_status="needs_review"`; they establish neither truth nor credibility.

Reusable NLP modules can later populate the typed collections while attaching
an `EvidenceSource` to every result. A separate Coordinator adapter can convert
between this contract and shared `AgentTask`/`AgentResult` once the team agrees
the integration contract.

## Assumptions requiring team confirmation

- The canonical document metadata and source-locator format.
- Stable document/evidence identifiers and document-version semantics.
- Legal issue taxonomy, including Sri Lanka-specific categories.
- Shared status/error vocabulary and the final Coordinator envelope.
- Evaluation data governance, language detection, and future OCR policy.
