# Retrieval Evaluation Report Template

> Complete this template only with measured results from a documented dataset.
> Do not treat the bundled synthetic framework fixture as real-world legal
> retrieval evidence.

## Dataset description

- Dataset name and version:
- Provenance and licensing:
- Synthetic, curated prototype, or real:
- Number of documents/chunks:
- Number of evaluation queries:
- Known coverage limitations:

## Evaluation queries

Describe how queries were selected, their legal categories, the relevance unit
(document or chunk), and any exclusions.

## Relevance judgement method

Document who assigned binary relevant IDs and any optional human labels. Explain
how disagreements, partial relevance, and missing judgements were handled.

## BM25 results

Record Precision@K, Recall@K, MRR, and per-query observations with the exact K
values and configuration used.

## Semantic results

Record the embedding model, output dimension, index version, Precision@K,
Recall@K, MRR, and per-query observations.

## Hybrid results

Record candidate-pool settings, normalization, weights, Precision@K, Recall@K,
MRR, and per-query observations.

## Weight comparison

Compare only configurations actually measured. Do not declare an optimum
without a documented selection rule and validation dataset.

## Observations

Summarize measured behavior, including queries where lexical, semantic, or
hybrid retrieval performed differently.

## Limitations

Document dataset size, relevance-judgement uncertainty, domain coverage,
embedding/API constraints, prototype index limitations, and threats to
generalization.
