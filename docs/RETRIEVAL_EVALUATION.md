# Retrieval Evaluation

## Current real-corpus decision

CaseLens completed a human-reviewed evaluation over 36 international-law
documents and 8,179 production chunks. Seven evaluation queries produced 38
reviewed query-document judgments. The binary-qrels policy for this evaluation
counted both `relevant` and `partially_relevant` as relevant; `not_relevant`
was counted as not relevant.

Across BM25, semantic, and five hybrid configurations, BM25 `0.8` and semantic
`0.2` was the best observed hybrid compromise. It tied the best observed P@3,
P@5, and R@3 among tested hybrid configurations while retaining a semantic
component. Pure BM25 still tied or exceeded hybrid on some metrics.

This setting is not universally optimal. The evaluation has only seven queries,
uses no held-out validation set, and relies on judgments pooled from retrieved
candidates. Observed retrieval limitations include chunk concentration,
limited document diversity, semantic drift, and lexical false positives. A
larger query set, deeper judgments, document-level aggregation, graded metrics,
and held-out validation are required before making a strong academic claim.

## Production result diversification

Production API and Retrieval Agent searches apply a post-ranking cap of two
chunks per metadata `document_id` by default. The cap is controlled by
`RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT`, can be overridden by an explicit caller,
and can be intentionally disabled. It preserves final score order and fills
the requested result count from lower-ranked documents when the candidate pool
allows. It does not change BM25 scoring, semantic similarity, or hybrid fusion.

The completed seven-query weight evaluation remains explicitly uncapped. This
preserves its historical results and avoids presenting a diversity-control
change as an improvement in relevance accuracy. The cap should receive its own
human-reviewed context-quality evaluation before RAG conclusions are drawn.

## Report template

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
