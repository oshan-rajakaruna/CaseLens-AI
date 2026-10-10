"""Deterministic grounding prompt for legal evidence summarization."""

from agents.coordinator.schemas import SummarizationReadyPayload
from agents.summarization.schemas import SummarizationProviderRequest

SYSTEM_INSTRUCTIONS = """SYSTEM / INSTRUCTIONS
You are the CaseLens evidence-grounded legal summarization component.
Answer only from the supplied retrieved context.
Do not invent legal authorities, case names, page numbers, or citations.
Cite factual and legal claims inline using bracketed IDs such as [CTX-001].
Use only CTX IDs listed under ALLOWED CITATIONS.
If the context is insufficient, say so explicitly.
Do not claim comprehensive legal coverage or present the response as legal advice.
Clearly distinguish retrieved evidence from interpretation.
Keep the answer concise and evidence-grounded.
Return a structured answer with answer, citations, optional confidence, and limitations.
Every declared citation must appear inline, and every inline citation must be declared."""


def build_summarization_provider_request(
    payload: SummarizationReadyPayload,
) -> SummarizationProviderRequest:
    """Build a stable provider request from Coordinator-supplied context."""

    legal_issue = payload.legal_issue or "Not separately specified"
    allowed = "\n".join(
        _format_allowed_citation(context_id, entry)
        for context_id, entry in payload.citation_map.items()
    ) or "None"
    weak_context = "\n".join(payload.diagnostics.warnings) or "None"
    prompt = "\n\n".join(
        (
            SYSTEM_INSTRUCTIONS,
            f"USER QUERY\n{payload.query}",
            f"LEGAL ISSUE\n{legal_issue}",
            f"RETRIEVED CONTEXT\n{payload.context_text}",
            f"ALLOWED CITATIONS\n{allowed}",
            "WEAK CONTEXT / RETRIEVAL LIMITATIONS\n"
            f"{weak_context}\n"
            "When warnings are present, include an uncertainty or weak-evidence "
            "statement in limitations.",
        )
    )
    return SummarizationProviderRequest(
        query=payload.query,
        legal_issue=payload.legal_issue,
        formatted_context=payload.context_text,
        citation_map=payload.citation_map,
        generation_instructions=SYSTEM_INSTRUCTIONS,
        prompt=prompt,
    )


def _format_allowed_citation(context_id: str, entry: object) -> str:
    title = getattr(entry, "title", None)
    document_id = getattr(entry, "document_id", None)
    pages = getattr(entry, "pages", [])
    page_text = ", ".join(str(page) for page in pages) or "not available"
    return (
        f"{context_id}: document={document_id}; "
        f"title={title or 'not available'}; pages={page_text}"
    )
