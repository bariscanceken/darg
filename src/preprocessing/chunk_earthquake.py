"""Build embedding-ready chunks from earthquake relevance page JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("data/processed/earthquake_relevance")
DEFAULT_OUTPUT_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")

INCLUDED_RELEVANCES = {"relevant", "uncertain"}
RELEVANCE_ORDER = ("relevant", "uncertain")
MIN_TOKENS = 700
TARGET_TOKENS = 850
MAX_TOKENS = 1000
OVERLAP_TOKENS = 85
LONG_UNIT_TOKENS = 220
MIN_EMIT_TOKENS = 80

TOKEN_RE = re.compile(r"\S+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9])")


@dataclass(frozen=True)
class TextUnit:
    text: str
    page: int
    relevance: str
    token_count: int


@dataclass
class ChunkStats:
    used_documents: int = 0
    used_pages: int = 0
    relevant_pages: int = 0
    uncertain_pages: int = 0
    total_chunks: int = 0
    total_chunk_tokens: int = 0
    min_chunk_tokens: int | None = None
    max_chunk_tokens: int = 0
    relevant_only_chunks: int = 0
    chunks_with_uncertain: int = 0

    @property
    def average_chunk_tokens(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.total_chunk_tokens / self.total_chunks

    def add_chunk(self, token_count: int, source_relevances: list[str]) -> None:
        self.total_chunks += 1
        self.total_chunk_tokens += token_count
        self.min_chunk_tokens = token_count if self.min_chunk_tokens is None else min(self.min_chunk_tokens, token_count)
        self.max_chunk_tokens = max(self.max_chunk_tokens, token_count)

        if source_relevances == ["relevant"]:
            self.relevant_only_chunks += 1
        if "uncertain" in source_relevances:
            self.chunks_with_uncertain += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create earthquake embedding chunks from page relevance JSON files.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def split_long_text(text: str, max_tokens: int = LONG_UNIT_TOKENS) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= max_tokens:
        return [text.strip()] if text.strip() else []
    return [" ".join(tokens[index : index + max_tokens]) for index in range(0, len(tokens), max_tokens)]


def split_into_units(text: str, page: int, relevance: str) -> list[TextUnit]:
    units: list[TextUnit] = []
    for sentence in SENTENCE_BOUNDARY_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        for piece in split_long_text(sentence):
            token_count = estimate_token_count(piece)
            if token_count:
                units.append(TextUnit(piece, page, relevance, token_count))
    return units


def source_relevances_for(units: list[TextUnit]) -> list[str]:
    present = {unit.relevance for unit in units}
    return [relevance for relevance in RELEVANCE_ORDER if relevance in present]


def make_chunk_id(relative_path: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(f"{relative_path}|{chunk_index}|{text}".encode("utf-8")).hexdigest()[:20]
    return f"afad_earthquake_{digest}"


def build_chunk(
    units: list[TextUnit],
    source_file: str,
    relative_path: str,
    chunk_index: int,
) -> dict[str, Any]:
    text = " ".join(unit.text for unit in units).strip()
    pages = [unit.page for unit in units]
    source_relevances = source_relevances_for(units)
    return {
        "chunk_id": make_chunk_id(relative_path, chunk_index, text),
        "text": text,
        "source_file": source_file,
        "relative_path": relative_path,
        "organization": "AFAD",
        "page_start": min(pages),
        "page_end": max(pages),
        "chunk_index": chunk_index,
        "source_relevances": source_relevances,
    }


def tail_overlap(units: list[TextUnit]) -> list[TextUnit]:
    overlap: list[TextUnit] = []
    token_total = 0
    for unit in reversed(units):
        if overlap and token_total + unit.token_count > OVERLAP_TOKENS:
            break
        overlap.insert(0, unit)
        token_total += unit.token_count
        if token_total >= OVERLAP_TOKENS:
            break
    return overlap


def trim_to_fit(current_units: list[TextUnit], next_unit: TextUnit) -> list[TextUnit]:
    while current_units and sum(unit.token_count for unit in current_units) + next_unit.token_count > MAX_TOKENS:
        current_units.pop(0)
    return current_units


def chunk_units(
    units: list[TextUnit],
    source_file: str,
    relative_path: str,
    starting_chunk_index: int,
) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    current_units: list[TextUnit] = []
    current_tokens = 0
    chunk_index = starting_chunk_index
    has_new_content = False

    def emit() -> None:
        nonlocal chunk_index, current_units, current_tokens, has_new_content
        if not current_units or not has_new_content:
            return
        if current_tokens < MIN_EMIT_TOKENS:
            current_units = []
            current_tokens = 0
            has_new_content = False
            return
        chunks.append(build_chunk(current_units, source_file, relative_path, chunk_index))
        chunk_index += 1
        current_units = tail_overlap(current_units)
        current_tokens = sum(unit.token_count for unit in current_units)
        has_new_content = False

    for unit in units:
        if current_units and current_tokens + unit.token_count > MAX_TOKENS:
            emit()
            current_units = trim_to_fit(current_units, unit)
            current_tokens = sum(existing.token_count for existing in current_units)

        current_units.append(unit)
        current_tokens += unit.token_count
        has_new_content = True

        if current_tokens >= TARGET_TOKENS:
            emit()

    emit()
    return chunks, chunk_index


def included_page_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included = []
    for record in records:
        relevance = record.get("relevance")
        text = str(record.get("text", "")).strip()
        page = record.get("page")
        if relevance in INCLUDED_RELEVANCES and text and isinstance(page, int):
            included.append(record)
    return sorted(included, key=lambda record: int(record["page"]))


def split_consecutive_page_runs(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current_run: list[dict[str, Any]] = []
    previous_page: int | None = None

    for record in records:
        page = int(record["page"])
        if previous_page is not None and page != previous_page + 1:
            if current_run:
                runs.append(current_run)
            current_run = []
        current_run.append(record)
        previous_page = page

    if current_run:
        runs.append(current_run)
    return runs


def process_document(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of page records: {input_path}")

    pages = included_page_records(records)
    if not pages:
        return [], {"used_pages": 0, "relevant_pages": 0, "uncertain_pages": 0}

    relative_path = str(pages[0].get("relative_path", input_path.name))
    source_file = str(pages[0].get("source_file", input_path.with_suffix(".pdf").name))
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page_run in split_consecutive_page_runs(pages):
        units: list[TextUnit] = []
        for page_record in page_run:
            units.extend(
                split_into_units(
                    str(page_record["text"]),
                    int(page_record["page"]),
                    str(page_record["relevance"]),
                )
            )
        run_chunks, chunk_index = chunk_units(units, source_file, relative_path, chunk_index)
        chunks.extend(run_chunks)

    page_relevance_by_number = {int(page["page"]): str(page["relevance"]) for page in pages}
    used_page_numbers: set[int] = set()
    for chunk in chunks:
        used_page_numbers.update(range(int(chunk["page_start"]), int(chunk["page_end"]) + 1))

    page_counts = {
        "used_pages": len(used_page_numbers),
        "relevant_pages": sum(1 for page in used_page_numbers if page_relevance_by_number.get(page) == "relevant"),
        "uncertain_pages": sum(1 for page in used_page_numbers if page_relevance_by_number.get(page) == "uncertain"),
    }
    return chunks, page_counts


def run_chunking(input_root: Path, output_file: Path) -> ChunkStats:
    input_root = input_root.resolve()
    output_file = output_file.resolve()
    stats = ChunkStats()
    all_chunks: list[dict[str, Any]] = []

    for input_path in sorted(input_root.rglob("*.json")):
        document_chunks, page_counts = process_document(input_path)
        if page_counts["used_pages"] > 0:
            stats.used_documents += 1
            stats.used_pages += page_counts["used_pages"]
            stats.relevant_pages += page_counts["relevant_pages"]
            stats.uncertain_pages += page_counts["uncertain_pages"]

        for chunk in document_chunks:
            token_count = estimate_token_count(str(chunk["text"]))
            stats.add_chunk(token_count, list(chunk["source_relevances"]))
            all_chunks.append(chunk)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return stats


def main() -> None:
    args = parse_args()
    stats = run_chunking(args.input_root, args.output_file)
    print(f"Used documents: {stats.used_documents}")
    print(f"Used pages: {stats.used_pages}")
    print(f"Relevant pages: {stats.relevant_pages}")
    print(f"Uncertain pages: {stats.uncertain_pages}")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Average tokens: {stats.average_chunk_tokens:.2f}")
    print(f"Minimum tokens: {stats.min_chunk_tokens or 0}")
    print(f"Maximum tokens: {stats.max_chunk_tokens}")
    print(f"Relevant-only chunks: {stats.relevant_only_chunks}")
    print(f"Chunks with uncertain: {stats.chunks_with_uncertain}")


if __name__ == "__main__":
    main()
