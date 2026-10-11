# CaseLens draft API

The Milestone 4 endpoints are deliberately unauthenticated and disabled by
default. Enable them only for local development:

```text
APP_ENV=development
ENABLE_UNAUTHENTICATED_DEV_API=true
LLM_API_KEY=<local Gemini API key>
```

Both `APP_ENV=development` and the explicit opt-in must be present. Do not
deploy these endpoints publicly until authentication, authorization, and case
ownership checks are implemented. Request bodies and source evidence are not
logged by these routes.

## Draft summarization

`POST /api/summarization/draft` accepts a `SummarizationRequest` and makes one
Gemini generation call through `SummarizationAgent`.

```json
{
  "case": {"case_id": "case-demo", "title": "Synthetic matter"},
  "extracted_facts": [
    {
      "fact_id": "fact-1",
      "text": "Synthetic fact for local testing.",
      "source_references": [
        {"document_id": "document-1", "locator": "page 1"}
      ]
    }
  ],
  "retrieved_passages": [],
  "instruction": "Prepare a draft evidence summary."
}
```

The response is the existing `SummarizationResponse`, containing structured
summaries, comparisons, draft claims, citations, and warnings. Unsupported
references are rejected rather than returned.

## Draft report

`POST /api/reports/draft` accepts the original case, source request, and an
existing summarization response:

```json
{
  "case": {"case_id": "case-demo", "title": "Synthetic matter"},
  "source_request": {
    "case": {"case_id": "case-demo", "title": "Synthetic matter"},
    "extracted_facts": [
      {
        "fact_id": "fact-1",
        "text": "Synthetic fact for local testing.",
        "source_references": [
          {"document_id": "document-1", "locator": "page 1"}
        ]
      }
    ],
    "retrieved_passages": []
  },
  "summarization": {
    "evidence_summaries": [
      {
        "text": "Synthetic draft summary.",
        "source_references": [
          {"document_id": "document-1", "locator": "page 1"}
        ]
      }
    ],
    "precedent_summaries": [],
    "comparisons": [],
    "draft_claims": [],
    "citations": [],
    "warnings": ["Requires verification."]
  }
}
```

The response contains `report`, a structured `LegalResearchReport` whose status
is always `pending_verification`, and `markdown`, a deterministic escaped
preview. Report assembly makes no Gemini call and performs no persistence.
