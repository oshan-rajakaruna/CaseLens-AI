"""Ranked, budgeted, citation-preserving RAG context assembly."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import os
import re
from typing import TypeAlias

from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    LegalDocumentMetadata,
    SemanticSearchResult,
)
from retrieval.rag.schemas import (
    ChunkRetrievalScore,
    RAGCitation,
    RAGContext,
    RAGContextPassage,
)

DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_MAX_CONTEXT_PASSAGES = 8
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CHUNK_NUMBER_PATTERN = re.compile(r"-chunk-(\d+)$")
_NEAR_DUPLICATE_MIN_CHARS = 120
_NEAR_DUPLICATE_RATIO = 0.985
_MIN_WORD_OVERLAP = 8

RetrievalResult: TypeAlias = (
    BM25SearchResult | SemanticSearchResult | HybridSearchResult
)


@dataclass(frozen=True, slots=True)
class RAGContextSettings:
    """Validated context limits and optional relevance threshold."""

    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_passages: int = DEFAULT_MAX_CONTEXT_PASSAGES
    min_retrieval_score: float | None = None

    def __post_init__(self) -> None:
        _validate_positive_integer(self.max_tokens, "max_tokens")
        _validate_positive_integer(self.max_passages, "max_passages")
        if self.min_retrieval_score is not None:
            _validate_finite_score(
                self.min_retrieval_score,
                "min_retrieval_score",
            )

    @classmethod
    def from_environment(cls) -> "RAGContextSettings":
        """Load configuration without changing normal OS environment precedence."""

        return cls(
            max_tokens=_environment_integer(
                "RAG_MAX_CONTEXT_TOKENS",
                DEFAULT_MAX_CONTEXT_TOKENS,
            ),
            max_passages=_environment_integer(
                "RAG_MAX_CONTEXT_PASSAGES",
                DEFAULT_MAX_CONTEXT_PASSAGES,
            ),
            min_retrieval_score=_environment_optional_score(
                "RAG_MIN_RETRIEVAL_SCORE"
            ),
        )


@dataclass(slots=True)
class _PassageDraft:
    text: str
    document_id: str
    metadata: LegalDocumentMetadata
    results: list[RetrievalResult] = field(default_factory=list)
    truncated: bool = False

    @property
    def first_rank(self) -> int:
        return min(result.rank for result in self.results)


class RAGContextAssembler:
    """Convert ranked retrieval results into compact structured context."""

    def __init__(self, settings: RAGContextSettings | None = None) -> None:
        self.settings = settings or RAGContextSettings.from_environment()

    def assemble(self, results: Iterable[RetrievalResult]) -> RAGContext:
        """Threshold, deduplicate, merge, and budget ranked retrieval results."""

        ranked = [
            result
            for _, result in sorted(
                enumerate(results),
                key=lambda item: (item[1].rank, item[0]),
            )
        ]
        eligible = [
            result
            for result in ranked
            if self.settings.min_retrieval_score is None
            or _final_score(result) >= self.settings.min_retrieval_score
        ]
        deduplicated = _deduplicate_results(eligible)
        drafts = _merge_adjacent_drafts(deduplicated)
        passages, truncated = self._select_with_budget(drafts)
        formatted = format_context_passages(passages)
        return RAGContext(
            passages=passages,
            estimated_tokens=estimate_tokens(formatted),
            source_documents=list(
                dict.fromkeys(passage.document_id for passage in passages)
            ),
            input_result_count=len(ranked),
            eligible_result_count=len(eligible),
            deduplicated_result_count=len(deduplicated),
            truncated=truncated,
        )

    def _select_with_budget(
        self,
        drafts: Sequence[_PassageDraft],
    ) -> tuple[list[RAGContextPassage], bool]:
        selected: list[RAGContextPassage] = []
        truncated = False
        for draft in drafts:
            if len(selected) >= self.settings.max_passages:
                truncated = True
                break
            passage = _to_passage(draft, len(selected) + 1)
            candidate = [*selected, passage]
            if (
                estimate_tokens(format_context_passages(candidate))
                <= self.settings.max_tokens
            ):
                selected.append(passage)
                continue

            empty_text_passage = passage.model_copy(update={"text": "x"})
            overhead = estimate_tokens(
                format_context_passages([*selected, empty_text_passage])
            ) - 1
            available_text_tokens = self.settings.max_tokens - overhead
            if available_text_tokens > 0:
                shortened = truncate_to_tokens(passage.text, available_text_tokens)
                if shortened:
                    selected.append(
                        passage.model_copy(
                            update={"text": shortened, "truncated": True}
                        )
                    )
            truncated = True
            break
        return selected, truncated


def estimate_tokens(text: str) -> int:
    """Estimate tokens as Unicode word-or-punctuation units.

    This dependency-free estimate is deterministic. It deliberately counts
    punctuation separately, which is conservative for citation-heavy legal text.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(_TOKEN_PATTERN.findall(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return the longest leading substring within the approximate token limit."""

    _validate_positive_integer(max_tokens, "max_tokens")
    matches = list(_TOKEN_PATTERN.finditer(text))
    if len(matches) <= max_tokens:
        return text.strip()
    return text[: matches[max_tokens - 1].end()].strip()


def format_context(context: RAGContext) -> str:
    """Format assembled evidence for a future LLM prompt."""

    return format_context_passages(context.passages)


def format_context_passages(passages: Sequence[RAGContextPassage]) -> str:
    """Format passages while preserving their stable context identifiers."""

    return "\n\n".join(_format_passage(passage) for passage in passages)


def _format_passage(passage: RAGContextPassage) -> str:
    pages = ", ".join(str(page) for page in passage.pages) or "Not available"
    chunks = ", ".join(passage.chunk_ids)
    lines = [
        f"[{passage.context_id}]",
        f"Case: {passage.title or 'Not available'}",
        f"Court: {passage.court or 'Not available'}",
        f"Document: {passage.document_id}",
        f"Chunks: {chunks}",
        f"Pages: {pages}",
    ]
    if passage.citation.citation:
        lines.append(f"Citation: {passage.citation.citation}")
    if passage.source:
        lines.append(f"Source: {passage.source}")
    lines.extend(("Text:", passage.text))
    return "\n".join(lines)


def _deduplicate_results(results: Sequence[RetrievalResult]) -> list[_PassageDraft]:
    drafts: list[_PassageDraft] = []
    for result in results:
        metadata = _result_metadata(result)
        duplicate = next(
            (
                draft
                for draft in drafts
                if draft.document_id == result.document_id
                and _texts_are_duplicates(draft.text, result.chunk_text)
            ),
            None,
        )
        if duplicate is not None:
            duplicate.results.append(result)
            continue
        drafts.append(
            _PassageDraft(
                text=result.chunk_text.strip(),
                document_id=result.document_id,
                metadata=metadata,
                results=[result],
            )
        )
    return drafts


def _merge_adjacent_drafts(drafts: Sequence[_PassageDraft]) -> list[_PassageDraft]:
    merged = list(drafts)
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(merged):
            partner_index = next(
                (
                    right_index
                    for right_index in range(left_index + 1, len(merged))
                    if _drafts_are_adjacent(left, merged[right_index])
                ),
                None,
            )
            if partner_index is None:
                continue
            right = merged.pop(partner_index)
            merged[left_index] = _combine_drafts(left, right)
            changed = True
            break
    return sorted(merged, key=lambda draft: draft.first_rank)


def _drafts_are_adjacent(left: _PassageDraft, right: _PassageDraft) -> bool:
    if left.document_id != right.document_id:
        return False
    left_numbers = [_chunk_number(result.chunk_id) for result in left.results]
    right_numbers = [_chunk_number(result.chunk_id) for result in right.results]
    sequential = any(
        left_number is not None
        and right_number is not None
        and abs(left_number - right_number) == 1
        for left_number in left_numbers
        for right_number in right_numbers
    )
    overlap = _word_overlap(left.text, right.text) >= _MIN_WORD_OVERLAP or (
        _word_overlap(right.text, left.text) >= _MIN_WORD_OVERLAP
    )
    if not sequential and not overlap:
        return False
    return _page_ranges_are_compatible(left.results, right.results)


def _page_ranges_are_compatible(
    left_results: Sequence[RetrievalResult],
    right_results: Sequence[RetrievalResult],
) -> bool:
    left_pages = _pages_for_results(left_results)
    right_pages = _pages_for_results(right_results)
    if not left_pages or not right_pages:
        return True
    return any(
        abs(left_page - right_page) <= 1
        for left_page in left_pages
        for right_page in right_pages
    )


def _combine_drafts(left: _PassageDraft, right: _PassageDraft) -> _PassageDraft:
    combined_results = sorted(
        [*left.results, *right.results],
        key=lambda result: (
            _chunk_number(result.chunk_id)
            if _chunk_number(result.chunk_id) is not None
            else math.inf,
            result.rank,
        ),
    )
    ordered_drafts = sorted(
        (left, right),
        key=lambda draft: (
            min(
                (
                    number
                    for result in draft.results
                    if (number := _chunk_number(result.chunk_id)) is not None
                ),
                default=math.inf,
            ),
            draft.first_rank,
        ),
    )
    text = ordered_drafts[0].text
    next_text = ordered_drafts[1].text
    overlap = _word_overlap(text, next_text)
    if overlap:
        next_words = next_text.split()
        text = f"{text.rstrip()} {' '.join(next_words[overlap:])}".strip()
    else:
        text = f"{text.rstrip()}\n\n{next_text.lstrip()}"
    primary = min(combined_results, key=lambda result: result.rank)
    return _PassageDraft(
        text=text,
        document_id=left.document_id,
        metadata=_result_metadata(primary),
        results=combined_results,
    )


def _to_passage(draft: _PassageDraft, position: int) -> RAGContextPassage:
    results = sorted(draft.results, key=lambda result: result.rank)
    primary = results[0]
    metadata = draft.metadata
    chunk_ids = list(dict.fromkeys(result.chunk_id for result in results))
    pages = _pages_for_results(results)
    scores = [_score_record(result) for result in results]
    return RAGContextPassage(
        context_id=f"CTX-{position:03d}",
        text=draft.text,
        document_id=draft.document_id,
        title=metadata.case_name or primary.case_name,
        court=metadata.court or primary.court,
        source=metadata.source or primary.source,
        citation=RAGCitation(
            document_id=draft.document_id,
            citation=metadata.citation or primary.citation,
            source=metadata.source or primary.source,
            source_url=metadata.source_url,
            provenance=metadata.provenance,
            chunk_ids=chunk_ids,
            pages=pages,
        ),
        chunk_ids=chunk_ids,
        pages=pages,
        ranks=[result.rank for result in results],
        score=_final_score(primary),
        retrieval_scores=scores,
        truncated=draft.truncated,
    )


def _score_record(result: RetrievalResult) -> ChunkRetrievalScore:
    if isinstance(result, HybridSearchResult):
        return ChunkRetrievalScore(
            chunk_id=result.chunk_id,
            rank=result.rank,
            final_score=result.hybrid_score,
            bm25_score=result.raw_bm25_score,
            normalized_bm25_score=result.normalized_bm25_score,
            semantic_score=result.raw_semantic_score,
            normalized_semantic_score=result.normalized_semantic_score,
            hybrid_score=result.hybrid_score,
        )
    if isinstance(result, SemanticSearchResult):
        return ChunkRetrievalScore(
            chunk_id=result.chunk_id,
            rank=result.rank,
            final_score=result.similarity_score,
            semantic_score=result.similarity_score,
        )
    return ChunkRetrievalScore(
        chunk_id=result.chunk_id,
        rank=result.rank,
        final_score=result.score,
        bm25_score=result.score,
    )


def _final_score(result: RetrievalResult) -> float:
    if isinstance(result, HybridSearchResult):
        return result.hybrid_score
    if isinstance(result, SemanticSearchResult):
        return result.similarity_score
    return result.score


def _result_metadata(result: RetrievalResult) -> LegalDocumentMetadata:
    if result.metadata is not None:
        return result.metadata
    return LegalDocumentMetadata(
        document_id=result.document_id,
        case_name=result.case_name,
        court=result.court,
        date=result.date,
        citation=result.citation,
        legal_category=result.legal_category,
        document_type=result.document_type,
        source=result.source,
    )


def _pages_for_results(results: Sequence[RetrievalResult]) -> list[int]:
    pages: set[int] = set()
    for result in results:
        metadata = _result_metadata(result)
        if metadata.page_number is not None:
            pages.add(metadata.page_number)
        if metadata.page_start is not None:
            page_end = metadata.page_end or metadata.page_start
            pages.update(range(metadata.page_start, page_end + 1))
    return sorted(pages)


def _chunk_number(chunk_id: str) -> int | None:
    match = _CHUNK_NUMBER_PATTERN.search(chunk_id)
    return int(match.group(1)) if match else None


def _word_overlap(left: str, right: str) -> int:
    left_words = left.split()
    right_words = right.split()
    maximum = min(len(left_words), len(right_words), 100)
    for size in range(maximum, _MIN_WORD_OVERLAP - 1, -1):
        if [word.casefold() for word in left_words[-size:]] == [
            word.casefold() for word in right_words[:size]
        ]:
            return size
    return 0


def _texts_are_duplicates(left: str, right: str) -> bool:
    normalized_left = " ".join(left.casefold().split())
    normalized_right = " ".join(right.casefold().split())
    if normalized_left == normalized_right:
        return True
    if min(len(normalized_left), len(normalized_right)) < _NEAR_DUPLICATE_MIN_CHARS:
        return False
    length_ratio = min(len(normalized_left), len(normalized_right)) / max(
        len(normalized_left), len(normalized_right)
    )
    return length_ratio >= _NEAR_DUPLICATE_RATIO and SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
        autojunk=False,
    ).ratio() >= _NEAR_DUPLICATE_RATIO


def _environment_integer(name: str, default: int) -> int:
    configured = os.getenv(name)
    if configured is None or not configured.strip():
        return default
    try:
        value = int(configured)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer greater than or equal to 1"
        ) from exc
    try:
        return _validate_positive_integer(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an integer greater than or equal to 1"
        ) from exc


def _environment_optional_score(name: str) -> float | None:
    configured = os.getenv(name)
    if configured is None or not configured.strip():
        return None
    try:
        value = float(configured)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number or empty") from exc
    try:
        return _validate_finite_score(value, name)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number or empty") from exc


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _validate_finite_score(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric
