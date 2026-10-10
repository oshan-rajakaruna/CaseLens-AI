# Analysis Agent evaluation (Step 10)

## Objective and corpus

This is a reproducible, synthetic-only evaluation of Member 3's local Analysis
Agent. The `tests/evaluation/analysis` corpus contains 20 fictional contract
evidence scenarios (rental, sale, lease/land-related examples, payment and
execution disputes, notices, conditions, witness statements, proceedings,
normal contracts, and positive/negative contradiction examples). Names, amounts
and case material are invented. It is not evidence of real-world Sri Lankan
legal accuracy.

`ground_truth.json` was authored from scenario text before running the system.
It uses strict entity label/offset matching and exact source-span/category/
semantic matching for facts and clauses. The annotations are provisional and
require independent legal-reviewer validation.

## Setup and metrics

Measured with Python 3.11.9, spaCy 3.8.16, and `en_core_web_sm` 3.8.0. The
runner uses `TP/(TP+FP)`, `TP/(TP+FN)`, and harmonic-mean F1; zero denominators
produce zero. Entity support is intentionally small (3), so its 100% result is
not statistically meaningful.

| Capability | TP | FP | FN | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Entities | 3 | 0 | 0 | 100.0% | 100.0% | 100.0% | 3 |
| Facts | 15 | 3 | 0 | 83.3% | 100.0% | 90.9% | 15 |
| Clauses | 4 | 1 | 0 | 80.0% | 100.0% | 88.9% | 4 |
| Issues | 5 | 2 | 0 | 71.4% | 100.0% | 83.3% | 5 |
| Timeline | 7 | 7 | 1 | 50.0% | 87.5% | 63.6% | 8 |
| Contradictions | 1 | 0 | 0 | 100.0% | 100.0% | 100.0% | 1 |

The provenance audit checked 92 returned source spans: 0 failures. All 20 text
pipeline scenarios completed. These results are measured synthetic-benchmark
findings, not production claims.

## Reliability and performance

Existing unit coverage includes empty/whitespace input, invalid encoding,
corrupt/oversized/unsupported documents, scanned PDFs, missing models,
provenance, stage partial failures, and deterministic output. The end-to-end
evaluation additionally exercises real TXT, multi-page PDF, and DOCX table
extraction before analysis.

On a small 97-character synthetic request (7 warm iterations in the final run):
cold analysis was 0.006396 s; warm median was 0.006935 s; warm P95 was
0.007812 s. These are
local synthetic timings, not production capacity measurements.

## Error analysis and fixes

The measured false positives arise principally from broad rule categories and
from timeline candidates for disputed/attributed statements. This report keeps
them visible instead of treating `completed` as correctness. The benchmark also
revealed an integration defect: affirmative payments were categorized as
`payment_or_transaction` while explicit non-payment used
`non_payment_or_breach`, preventing a strict same-event contradiction pair.
The detector now permits only that narrow category pair when subject, date,
amount, normalized statement core, and opposite polarity all match. A regression
test covers the repair.

The evaluator initially failed when executed directly because its package root
was absent from `sys.path`; it now bootstraps the repository root. No model or
rule was tuned to scenario names.

## Threats and future work

The corpus is small, English-only, synthetic, and authored by the development
team. It does not measure Sinhala/Tamil, OCR, real document variability,
cross-document aggregation, or reviewer agreement. Timeline and issue rules
need larger independently annotated samples, per-category support, error
adjudication, and formal human evaluation before code-review approval for
integration.

## Reproduction

```powershell
python -m pytest tests/evaluation/analysis/test_analysis_evaluation.py tests/integration/test_analysis_end_to_end.py -q
python tests/evaluation/analysis/evaluate_analysis.py
python -m pytest -q
```
