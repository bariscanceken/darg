"""Context formatting helpers for DARG RAG generation."""

from __future__ import annotations

import re
from typing import Any

SOURCE_REFERENCE_RE = re.compile(r"\[Kaynak\s+(\d+)\]", re.IGNORECASE)


def page_label(chunk: dict[str, Any]) -> str:
    """Return a human-readable page label from existing chunk metadata."""
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")
    if page_start in (None, "") and page_end in (None, ""):
        return ""
    if page_start == page_end or page_end in (None, ""):
        return f"Sayfa {page_start}"
    return f"Sayfa {page_start}-{page_end}"


def source_metadata(source_number: int, chunk: dict[str, Any]) -> dict[str, Any]:
    """Create display metadata from retrieval metadata without mutating the chunk."""
    metadata = {
        "source_label": f"Kaynak {source_number}",
        "chunk_id": chunk.get("chunk_id"),
        "source_file": chunk.get("source_file"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "page_label": page_label(chunk),
    }

    optional_fields = [
        "relative_path",
        "organization",
        "source_relevances",
        "original_rank",
        "original_similarity_score",
        "reranked_rank",
        "rerank_score",
    ]
    for field in optional_fields:
        if field in chunk:
            metadata[field] = chunk[field]
    return metadata


def format_context(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Format Top-5 retrieval chunks for the LLM and return source metadata."""
    context_blocks: list[str] = []
    sources: list[dict[str, Any]] = []

    for source_number, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        context_blocks.append(f"[Kaynak {source_number}]\nMetin: {text}")
        sources.append(source_metadata(source_number, chunk))

    return {
        "context": "\n\n".join(context_blocks),
        "sources": sources,
    }


def format_sources_for_cli(sources: list[dict[str, Any]]) -> str:
    """Render retrieval metadata as a concise CLI source list."""
    lines: list[str] = []
    for source in sources:
        label = source.get("source_label", "Kaynak")
        source_file = source.get("source_file") or "unknown"
        page = source.get("page_label") or "Sayfa bilgisi yok"
        chunk_id = source.get("chunk_id") or "unknown"
        lines.append(f"[{label}] AFAD - {source_file} - {page} - chunk_id: {chunk_id}")
    return "\n".join(lines)


def referenced_source_numbers(answer: str) -> list[int]:
    """Extract source numbers explicitly cited as [Kaynak N], preserving first-use order."""
    numbers: list[int] = []
    seen: set[int] = set()
    for match in SOURCE_REFERENCE_RE.finditer(answer):
        number = int(match.group(1))
        if number not in seen:
            numbers.append(number)
            seen.add(number)
    return numbers


def filter_sources_by_answer_references(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only retrieval metadata sources explicitly cited in the answer."""
    source_by_number = {
        index: source
        for index, source in enumerate(sources, start=1)
    }
    return [
        source_by_number[number]
        for number in referenced_source_numbers(answer)
        if number in source_by_number
    ]
