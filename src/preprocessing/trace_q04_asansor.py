"""Trace Q04 elevator mentions from extracted pages through relevance and chunks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TEXT_ROOT = Path("data/processed/text")
DEFAULT_RELEVANCE_ROOT = Path("data/processed/earthquake_relevance")
DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/q04_asansor_trace.txt")
SEPARATOR = "=" * 80

ASANSOR_TERMS = ["asansör", "asansörü", "asansörleri", "asansöre", "asansörden", "lift"]
DIRECT_GUIDANCE_PHRASES = [
    "asansör kullanmayın",
    "asansörü kullanmayın",
    "asansörleri kullanmayın",
    "asansöre binmeyin",
    "asansöre binilmemeli",
    "asansör kullanılmamalı",
    "asansör kullanılmaz",
    "asansörden uzak",
    "merdiven ve asansörleri kullanmayın",
]
RELATED_TERMS = [
    "deprem",
    "sarsıntı",
    "yangın",
    "tahliye",
    "acil",
    "güvenli",
    "güvenlik",
    "merdiven",
    "engelli",
    "risk",
    "tehlike",
    "kullan",
    "bin",
]
MENTION_ONLY_TERMS = ["adet", "kat", "bulunmaktadır", "bulunur", "mevcut", "kapasiteli", "tesis"]


@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: str
    contains_asansor: bool


@dataclass(frozen=True)
class TraceRecord:
    source_file: str
    relative_path: str
    page: int
    relevance_status: str
    previous_sentences: list[str]
    matched_sentence: str
    next_sentences: list[str]
    chunk_infos: list[ChunkInfo]
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Q04 asansör mentions through preprocessing outputs.")
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--relevance-root", type=Path, default=DEFAULT_RELEVANCE_ROOT)
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize(term)).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


ASANSOR_PATTERNS = [compile_term(term) for term in ASANSOR_TERMS]
DIRECT_PATTERNS = [compile_term(phrase) for phrase in DIRECT_GUIDANCE_PHRASES]


def read_json_list(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list of page records: {path}")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object records in: {path}")
    return records


def contains_asansor(text: str) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in ASANSOR_PATTERNS)


def build_relevance_lookup(relevance_root: Path) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for path in sorted(relevance_root.rglob("*.json")):
        for record in read_json_list(path):
            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            relevance = str(record.get("relevance", "")).strip()
            if relative_path and isinstance(page, int):
                lookup[(relative_path, page)] = relevance or "not_found"
    return lookup


def read_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with chunks_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {chunks_file}:{line_number}")
            chunks.append(record)
    return chunks


def build_chunk_lookup(chunks_file: Path) -> dict[tuple[str, int], list[ChunkInfo]]:
    lookup: dict[tuple[str, int], list[ChunkInfo]] = {}
    for chunk in read_chunks(chunks_file):
        relative_path = str(chunk.get("relative_path", "")).strip()
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not relative_path or not isinstance(page_start, int) or not isinstance(page_end, int) or not chunk_id:
            continue
        info = ChunkInfo(chunk_id=chunk_id, contains_asansor=contains_asansor(str(chunk.get("text", ""))))
        for page in range(page_start, page_end + 1):
            lookup.setdefault((relative_path, page), []).append(info)
    return lookup


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9•])", text)
    return [part.strip() for part in parts if part.strip()]


def context_for_asansor(text: str) -> tuple[list[str], str, list[str]]:
    sentences = split_sentences(text)
    for index, sentence in enumerate(sentences):
        if contains_asansor(sentence):
            return (
                sentences[max(0, index - 2) : index],
                sentence,
                sentences[index + 1 : index + 3],
            )
    return [], "", []


def has_direct_guidance(text: str) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in DIRECT_PATTERNS)


def classify_context(previous_sentences: list[str], matched_sentence: str, next_sentences: list[str]) -> str:
    context = " ".join([*previous_sentences, matched_sentence, *next_sentences])
    normalized_context = normalize(context)
    normalized_sentence = normalize(matched_sentence)

    if has_direct_guidance(context):
        return "DIRECT_GUIDANCE"
    if contains_asansor(matched_sentence) and any(term in normalized_sentence for term in ["kullanmay", "binmey", "kullanılmamal", "kullanilmamal", "kullanılmaz", "kullanilmaz"]):
        return "DIRECT_GUIDANCE"
    if contains_asansor(context) and any(term in normalized_context for term in RELATED_TERMS):
        return "RELATED_GUIDANCE"
    if contains_asansor(matched_sentence) and any(term in normalized_sentence for term in MENTION_ONLY_TERMS):
        return "MENTION_ONLY"
    return "UNKNOWN"


def collect_traces(
    text_root: Path,
    relevance_lookup: dict[tuple[str, int], str],
    chunk_lookup: dict[tuple[str, int], list[ChunkInfo]],
) -> list[TraceRecord]:
    traces: list[TraceRecord] = []

    for path in sorted(text_root.rglob("*.json")):
        for record in read_json_list(path):
            text = str(record.get("text", ""))
            if not contains_asansor(text):
                continue

            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            if not relative_path or not isinstance(page, int):
                continue

            previous_sentences, matched_sentence, next_sentences = context_for_asansor(text)
            traces.append(
                TraceRecord(
                    source_file=str(record.get("source_file", "")),
                    relative_path=relative_path,
                    page=page,
                    relevance_status=relevance_lookup.get((relative_path, page), "not_found"),
                    previous_sentences=previous_sentences,
                    matched_sentence=matched_sentence,
                    next_sentences=next_sentences,
                    chunk_infos=chunk_lookup.get((relative_path, page), []),
                    category=classify_context(previous_sentences, matched_sentence, next_sentences),
                )
            )

    return traces


def format_chunk_infos(chunk_infos: list[ChunkInfo]) -> str:
    if not chunk_infos:
        return "NONE"
    return ", ".join(info.chunk_id for info in chunk_infos)


def render_trace(index: int, trace: TraceRecord) -> str:
    return "\n".join(
        [
            f"Match {index}",
            f"source_file: {trace.source_file}",
            f"relative_path: {trace.relative_path}",
            f"page: {trace.page}",
            f"relevance_status: {trace.relevance_status}",
            f"category: {trace.category}",
            f"previous_2_sentences: {' | '.join(trace.previous_sentences) if trace.previous_sentences else 'NONE'}",
            f"matched_sentence: {trace.matched_sentence or 'NONE'}",
            f"next_2_sentences: {' | '.join(trace.next_sentences) if trace.next_sentences else 'NONE'}",
            f"covering_chunk_ids: {format_chunk_infos(trace.chunk_infos)}",
            f"chunk_contains_asansor: {str(any(info.contains_asansor for info in trace.chunk_infos)).lower()}",
        ]
    )


def render_summary(traces: list[TraceRecord]) -> str:
    relevance_counts = Counter(trace.relevance_status for trace in traces)
    category_counts = Counter(trace.category for trace in traces)
    pages_represented = sum(1 for trace in traces if trace.chunk_infos)
    pages_not_represented = len(traces) - pages_represented
    chunk_ids_with_asansor = {
        info.chunk_id
        for trace in traces
        for info in trace.chunk_infos
        if info.contains_asansor
    }

    return "\n".join(
        [
            SEPARATOR,
            "Summary",
            f"Total extracted matches: {len(traces)}",
            f"Relevant matches: {relevance_counts['relevant']}",
            f"Uncertain matches: {relevance_counts['uncertain']}",
            f"Excluded matches: {relevance_counts['excluded']}",
            "",
            f"Pages represented in chunks: {pages_represented}",
            f"Pages NOT represented in chunks: {pages_not_represented}",
            "",
            f"Chunks containing \"asansör\": {len(chunk_ids_with_asansor)}",
            "",
            f"DIRECT_GUIDANCE: {category_counts['DIRECT_GUIDANCE']}",
            f"RELATED_GUIDANCE: {category_counts['RELATED_GUIDANCE']}",
            f"MENTION_ONLY: {category_counts['MENTION_ONLY']}",
            f"UNKNOWN: {category_counts['UNKNOWN']}",
        ]
    )


def render_report(traces: list[TraceRecord]) -> str:
    lines = [
        SEPARATOR,
        "Q04 - Deprem sırasında asansör kullanmalı mıyım?",
        "",
    ]
    for index, trace in enumerate(traces, start=1):
        lines.append(render_trace(index, trace))
        lines.append("")
    lines.append(render_summary(traces))
    return "\n".join(lines).rstrip() + "\n"


def run_trace(
    text_root: Path,
    relevance_root: Path,
    chunks_file: Path,
    output_file: Path,
) -> list[TraceRecord]:
    relevance_lookup = build_relevance_lookup(relevance_root)
    chunk_lookup = build_chunk_lookup(chunks_file)
    traces = collect_traces(text_root, relevance_lookup, chunk_lookup)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(traces), encoding="utf-8")
    return traces


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    traces = run_trace(args.text_root, args.relevance_root, args.chunks_file, args.output_file)
    relevance_counts = Counter(trace.relevance_status for trace in traces)
    print(f"Total extracted matches: {len(traces)}")
    print(f"Relevant matches: {relevance_counts['relevant']}")
    print(f"Uncertain matches: {relevance_counts['uncertain']}")
    print(f"Excluded matches: {relevance_counts['excluded']}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
