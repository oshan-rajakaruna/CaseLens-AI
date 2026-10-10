# Evaluation data

This location contains reviewed retrieval judgments tied to the curated legal
dataset. The deterministic framework fixture remains under `tests/evaluation/`
because it is explicitly synthetic test data.

`caselens_real_corpus_review_template.csv` records 38 human-reviewed
query-document judgments across seven queries. For the completed binary metric
comparison, `relevant` and `partially_relevant` were counted as relevant, while
`not_relevant` was counted as not relevant. This is a project evaluation policy
and does not erase the original three-level human labels.

The best observed hybrid compromise was BM25 `0.8` and semantic `0.2`. This is
limited evidence from 36 documents and seven queries without a held-out
validation set, not a universally optimal retrieval configuration. Known
issues include chunk concentration, limited document diversity, semantic
drift, and lexical false positives.

The completed comparison used uncapped ranked chunks and remains reproducible
in that mode. Production retrieval now defaults to a post-ranking cap of two
chunks per document; this diversification control has not been assigned an
accuracy improvement and should be evaluated separately for RAG context
quality.
