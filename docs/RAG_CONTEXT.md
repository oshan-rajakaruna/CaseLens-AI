# RAG Context Assembly

## Purpose

Retrieval returns ranked, scored chunks. The RAG context assembler turns those
chunks into a compact evidence package suitable for later Coordinator and
Summarization Agent integration. It does not invoke an LLM and does not change
BM25, semantic, hybrid, or diversification scoring.

## Selection pipeline

1. Preserve retrieval rank order.
2. Apply `RAG_MIN_RETRIEVAL_SCORE` when configured.
3. Conservatively deduplicate text within the same document.
4. Merge selected chunks only when their stable `document_id` matches and
   sequential chunk IDs or direct text overlap are compatible with page
   metadata.
5. Select higher-ranked passages first until the token or passage limit is
   reached.
6. If a passage only partly fits, retain its leading text within the remaining
   budget and mark both the passage and context as truncated.

The assembler never adds weak results to fill unused budget. Results removed
by retrieval diversification or a configured score threshold stay removed.

## Configuration

```dotenv
RAG_MAX_CONTEXT_TOKENS=4000
RAG_MAX_CONTEXT_PASSAGES=8
RAG_MIN_RETRIEVAL_SCORE=
```

Token and passage limits must be positive integers. A score threshold must be
a finite number. An empty threshold disables score filtering; CaseLens does not
invent a threshold without evaluation evidence.

## Token estimation

CaseLens does not add a model-specific tokenizer solely for this layer. The
deterministic estimator counts Unicode word units and punctuation marks in the
fully formatted context, including provenance headers. This is an approximation
and should be replaced or wrapped by the selected generation model's tokenizer
when an LLM integration is chosen.

## Provenance and citations

Every passage receives a deterministic prompt-local ID (`CTX-001`,
`CTX-002`, ...). It also retains:

- stable document ID and title;
- court and source;
- every contributing chunk ID and retrieval rank;
- page numbers where available;
- official citation, source URL, and provenance;
- per-chunk raw/normalized retrieval scores and the final ranking score.

The context ID never replaces original source provenance. The plain-text
formatter includes these IDs so a future response can cite the evidence used.

## Limitations

Diversification and context assembly control redundancy; they do not prove
relevance or factual accuracy. A document cap can surface lower-ranked weak
material, as observed in the sanitary and phytosanitary measures smoke query.
Near-duplicate detection is intentionally restricted to very similar passages
within one document so distinct authorities quoting the same rule remain
separate. Before production RAG use, context quality requires human-reviewed
evaluation with a held-out query set, score-threshold analysis, and the actual
generation model's tokenizer.
