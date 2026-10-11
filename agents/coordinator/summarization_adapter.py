"""Deterministic, source-preserving Retrieval-to-Precedent-Summarization mapping."""

from collections import OrderedDict
from hashlib import sha256
from typing import Iterable

from agents.coordinator.precedent_contracts import CoordinatorCitationProvenance
from agents.coordinator.schemas import CoordinatorMessage, CoordinatorRequest, CoordinatorRetrievalResult
from agents.summarization import SummarizationAgent
from agents.summarization.schemas import (
    JudgmentMetadata,
    RetrievedPassage,
    SourceReference,
    SummarizationRequest,
    SummarizationResponse,
)

_JUDGMENT_TYPES = frozenset({"judgment", "precedent"})


class CitationCollisionError(ValueError):
    """Raised if one generated request-local ID could identify different material."""


class CoordinatorSummarizationAdapter:
    """Invoke deterministic precedent extraction over explicitly classified results."""

    def __init__(self, summarization_agent: SummarizationAgent | None = None) -> None:
        self._summarization_agent = summarization_agent or SummarizationAgent()

    def summarize_precedents(
        self,
        request: CoordinatorRequest,
        results: Iterable[CoordinatorRetrievalResult],
    ) -> tuple[
        list[SummarizationResponse],
        dict[str, CoordinatorCitationProvenance],
        list[CoordinatorMessage],
    ]:
        """Group explicit judgment material and return actual deterministic outputs."""

        grouped: "OrderedDict[str, list[CoordinatorRetrievalResult]]" = OrderedDict()
        warnings: list[CoordinatorMessage] = []
        for result in results:
            if not _is_judgment(result):
                warnings.append(
                    CoordinatorMessage(
                        code="unsupported_precedent_document_type",
                        message="Retrieved material lacks an explicit judgment or precedent document type.",
                        stage="summarization",
                    )
                )
                continue
            grouped.setdefault(result.document_id, []).append(result)

        provenance: dict[str, CoordinatorCitationProvenance] = {}
        responses: list[SummarizationResponse] = []
        for document_id, passages in grouped.items():
            summary_request = self._request_for_document(
                request, document_id, passages, provenance
            )
            response = self._summarization_agent.summarize(summary_request)
            assert isinstance(response, SummarizationResponse)
            responses.append(response)
        return responses, provenance, warnings

    @staticmethod
    def _request_for_document(
        coordinator_request: CoordinatorRequest,
        document_id: str,
        results: list[CoordinatorRetrievalResult],
        provenance: dict[str, CoordinatorCitationProvenance],
    ) -> SummarizationRequest:
        first = results[0]
        passages: list[RetrievedPassage] = []
        seen_chunks: dict[str, str] = {}
        for result in results:
            existing_text = seen_chunks.get(result.chunk_id)
            if existing_text is not None:
                if existing_text == result.chunk_text:
                    continue
                raise CitationCollisionError(
                    "One chunk identifier referred to conflicting retrieved passage text."
                )
            seen_chunks[result.chunk_id] = result.chunk_text
            citation_id = _citation_id(result.document_id, result.chunk_id)
            source = _source_for_result(result, citation_id)
            _record_provenance(provenance, citation_id, result, source)
            if any(item.source.citation_id == citation_id for item in passages):
                continue
            passages.append(
                RetrievedPassage(
                    text=result.chunk_text,
                    source=source,
                    rank=result.rank,
                    relevance_score=_score_for_result(result),
                )
            )
        metadata = first.metadata
        return SummarizationRequest(
            case_id=coordinator_request.case_id,
            document_id=document_id,
            document_ids=[document_id],
            query=coordinator_request.user_query,
            summary_type="precedent_summary",
            mode="deterministic",
            retrieved_passages=passages,
            judgment_metadata=JudgmentMetadata(
                case_title=first.case_name,
                court=first.court or _metadata_attr(metadata, "court"),
                jurisdiction=_metadata_value(metadata, "jurisdiction"),
                judgment_date=first.date or _metadata_attr(metadata, "date"),
                official_citation=first.citation or _metadata_attr(metadata, "citation"),
                source_url=_metadata_attr(metadata, "source_url"),
            ),
            metadata={"retrieval_mode": coordinator_request.retrieval_mode},
        )


def _is_judgment(result: CoordinatorRetrievalResult) -> bool:
    document_type = result.document_type or _metadata_attr(result.metadata, "document_type")
    return document_type is not None and document_type.strip().casefold() in _JUDGMENT_TYPES


def _citation_id(document_id: str, chunk_id: str) -> str:
    digest = sha256(f"{document_id}\0{chunk_id}".encode("utf-8")).hexdigest()[:16]
    return f"SRC-{digest}"


def _source_for_result(
    result: CoordinatorRetrievalResult, citation_id: str
) -> SourceReference:
    metadata = result.metadata
    page = _metadata_attr(metadata, "page_number")
    locator_parts = [f"chunk:{result.chunk_id}"]
    page_start = _metadata_attr(metadata, "page_start")
    page_end = _metadata_attr(metadata, "page_end")
    if page_start is not None:
        page_range = str(page_start)
        if page_end is not None and page_end != page_start:
            page_range = f"{page_range}-{page_end}"
        locator_parts.append(f"pages:{page_range}")
    return SourceReference(
        document_id=result.document_id,
        locator="; ".join(locator_parts),
        excerpt=result.chunk_text,
        page=page,
        citation_id=citation_id,
    )


def _record_provenance(
    provenance: dict[str, CoordinatorCitationProvenance],
    citation_id: str,
    result: CoordinatorRetrievalResult,
    source: SourceReference,
) -> None:
    metadata = result.metadata
    entry = CoordinatorCitationProvenance(
        citation_id=citation_id,
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        locator=source.locator,
        page=source.page,
        page_start=_metadata_attr(metadata, "page_start"),
        page_end=_metadata_attr(metadata, "page_end"),
        title=result.case_name,
        court=result.court or _metadata_attr(metadata, "court"),
        jurisdiction=_metadata_value(metadata, "jurisdiction"),
        official_citation=result.citation or _metadata_attr(metadata, "citation"),
        source_url=_metadata_attr(metadata, "source_url"),
    )
    existing = provenance.get(citation_id)
    if existing is not None and existing != entry:
        raise CitationCollisionError(
            "A request-local citation ID would identify different source material."
        )
    provenance[citation_id] = entry


def _metadata_value(metadata: object, field: str) -> str | None:
    extra = getattr(metadata, "model_extra", None) or {}
    value = extra.get(field)
    return value if isinstance(value, str) and value.strip() else None


def _metadata_attr(metadata: object | None, field: str):
    return getattr(metadata, field, None)


def _score_for_result(result: CoordinatorRetrievalResult) -> float:
    if hasattr(result, "score"):
        return result.score  # type: ignore[return-value]
    if hasattr(result, "similarity_score"):
        return result.similarity_score  # type: ignore[return-value]
    return result.hybrid_score  # type: ignore[return-value]
