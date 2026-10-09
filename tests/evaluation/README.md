# Evaluation data

Evaluation fixtures in this directory are explicitly classified by provenance.
The retrieval dataset is synthetic legal-style data for deterministic framework
validation; it is not real case law and must not be reported as a legal
benchmark.

Automated retrieval metrics use each query's binary `relevant_ids` set. Optional
human labels (`relevant`, `partially_relevant`, and `not_relevant`) are retained
for manual review but are not converted into graded relevance scores.

Run the offline retrieval comparison from the repository root:

```powershell
python -m retrieval.evaluation.run
python -m retrieval.evaluation.run --weights --json
```
