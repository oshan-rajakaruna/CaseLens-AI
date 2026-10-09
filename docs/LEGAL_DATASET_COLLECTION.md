# Legal Dataset Collection

## Scope and ownership

The initial CaseLens academic prototype target is approximately 30–45 manually
curated legal documents:

| Legal category | Target |
| --- | ---: |
| Employment Law | 10–15 documents |
| Contract Law | 10–15 documents |
| Property Law | 10–15 documents |

This is a planning target, not an assertion that the dataset has been collected
and not a measure of legal coverage. **CaseLens does NOT claim complete coverage
of Sri Lankan law.** Source collection and review are shared responsibilities
of all four project members; they are not assigned solely to the Retrieval
owner.

## Collection checklist

For every document, a team member must:

1. Locate a legally accessible source.
2. Verify that the source is suitable for academic use.
3. Download or record the document manually; do not automate scraping.
4. Assign a stable, unique `document_id`.
5. Record only metadata supported by the source.
6. Record provenance and retain the original source URL where available.
7. Add the approved local `.txt` or `.pdf` file to `data/legal/raw/`.
8. Add its entry to `data/legal/metadata/manifest.json`.
9. Run dataset summary and readiness validation.
10. Run the ingestion dry-run before any indexing.

Do not invent a case name, date, court, citation, or other unavailable
metadata. Do not commit confidential, copyrighted, or unapproved source files.

## Manifest workflow

Copy `data/legal/metadata/manifest.template.json` to `manifest.json`. Required
per-document fields are limited to:

- `document_id`: stable and unique within the dataset
- `file_name`: relative path beneath `data/legal/raw/`
- `provenance`: one approved provenance category

Optional fields are `case_name`, `court`, `date`, `citation`,
`legal_category`, `document_type`, `source`, `source_url`, and `notes`. Omit or
set unavailable optional values to `null`. Never synthesize a citation.

Approved provenance categories are:

- `official_court_source`
- `official_legislation_source`
- `approved_public_source`
- `team_curated_sample`

Official and approved public sources require `source` or `source_url`.
Official sources should retain the original URL whenever it is available.
Synthetic fixtures must use `team_curated_sample` and must never be presented
as real legal sources.

## Validation and readiness

Run the combined metadata summary, duplicate scan, extraction dry-run, and
index-readiness report:

```powershell
python -m retrieval.ingestion.prepare `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw
```

The tool reports counts by category, document type, court, provenance, and file
type; missing optional metadata; invalid entries; and duplicate IDs, files,
citations, and source URLs. It never deletes duplicates and never prints source
document text.

Readiness meanings:

- `READY_FOR_BM25_INDEXING`: all enabled files validate and produce chunks;
  manifest identifiers and file references have no blocking duplicates.
- `READY_FOR_SEMANTIC_INDEXING`: BM25 readiness passes and a backend
  `GEMINI_API_KEY` configuration is available.
- `NOT_READY`: blocking validation issues remain.

Checking semantic readiness does not call Gemini or print the key. Duplicate
citations and source URLs are review warnings; duplicate IDs, duplicate file
references, invalid entries, missing files, unsupported files, and extraction
failures block readiness.

For the lower-level ingestion report, run:

```powershell
python -m retrieval.ingestion.run `
  --manifest data/legal/metadata/manifest.json `
  --input-dir data/legal/raw `
  --dry-run
```

## Evaluation preparation

After the curated dataset exists, all four members should help create reviewed
evaluation queries and relevance judgments. The next evaluation stage will
compare BM25, semantic, and hybrid retrieval using Precision@K, Recall@K, and
MRR. No final score should be reported until the curated dataset and human
relevance judgments exist.
