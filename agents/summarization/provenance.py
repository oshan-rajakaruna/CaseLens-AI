"""Rendering helpers for validated CTX citation provenance."""

from collections.abc import Mapping

from agents.coordinator.schemas import ContextCitationMapEntry


class UnknownContextCitationError(ValueError):
    """Raised when provenance is requested for an unavailable context ID."""


def get_citation_provenance(
    context_id: str,
    citation_map: Mapping[str, ContextCitationMapEntry],
) -> ContextCitationMapEntry:
    """Return stored provenance for an allowed prompt-local citation."""

    try:
        return citation_map[context_id]
    except KeyError as exc:
        raise UnknownContextCitationError(
            f"Unknown context citation: {context_id}"
        ) from exc


def render_citation_provenance(
    context_id: str,
    citation_map: Mapping[str, ContextCitationMapEntry],
) -> str:
    """Render only provenance values actually present in the citation map."""

    entry = get_citation_provenance(context_id, citation_map)
    fields = [
        f"[{entry.context_id}]",
        f"Document: {entry.document_id}",
    ]
    optional = (
        ("Title", entry.title),
        ("Court", entry.court),
        ("Source", entry.source),
        ("Pages", ", ".join(str(page) for page in entry.pages) or None),
        ("Citation", entry.official_citation),
        ("URL", entry.source_url),
    )
    fields.extend(f"{label}: {value}" for label, value in optional if value)
    fields.append(f"Chunks: {', '.join(entry.chunk_ids)}")
    return "\n".join(fields)
