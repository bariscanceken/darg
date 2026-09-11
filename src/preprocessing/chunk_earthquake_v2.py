"""Build smaller focused earthquake chunks without changing the v1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("data/processed/earthquake_relevance")
DEFAULT_OLD_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/processed/chunks/afad_earthquake_chunks_v2.jsonl")

INCLUDED_RELEVANCES = {"relevant", "uncertain"}
RELEVANCE_ORDER = ("relevant", "uncertain")
TARGET_TOKENS = 400
PREFERRED_MIN_TOKENS = 300
PREFERRED_MAX_TOKENS = 500
MAX_TOKENS = 550
OVERLAP_TOKENS = 40
MIN_EMIT_TOKENS = 120
LONG_UNIT_TOKENS = 180

# Keep the same approximate token-counting method used by chunk_earthquake.py.
TOKEN_RE = re.compile(r"\S+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9])")
REQUIRED_FIELDS = {
    "chunk_id",
    "text",
    "source_file",
    "relative_path",
    "organization",
    "page_start",
    "page_end",
    "chunk_index",
    "source_relevances",
}


@dataclass(frozen=True)
class TextUnit:
    text: str
    page: int
    relevance: str
    token_count: int


@dataclass
class CorpusStats:
    total_chunks: int = 0
    total_tokens: int = 0
    min_tokens: int | None = None
    max_tokens: int = 0
    relevant_only_chunks: int = 0
    chunks_with_uncertain: int = 0
    unique_chunk_ids: int = 0
    duplicate_chunk_ids: int = 0
    excluded_page_references: int = 0
    output_jsonl_valid: bool = True
    all_required_fields_present: bool = True

    @property
    def average_tokens(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.total_tokens / self.total_chunks

    @property
    def minimum_tokens(self) -> int:
        return self.min_tokens or 0

    def add_chunk(self, token_count: int, source_relevances: list[str]) -> None:
        self.total_chunks += 1
        self.total_tokens += token_count
        self.min_tokens = token_count if self.min_tokens is None else min(self.min_tokens, token_count)
        self.max_tokens = max(self.max_tokens, token_count)
        if source_relevances == ["relevant"]:
            self.relevant_only_chunks += 1
        if "uncertain" in source_relevances:
            self.chunks_with_uncertain += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create smaller v2 earthquake chunks from relevance-filtered pages.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--old-chunks-file", type=Path, default=DEFAULT_OLD_CHUNKS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def token_slice(text: str, max_tokens: int) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    return [" ".join(tokens[index : index + max_tokens]) for index in range(0, len(tokens), max_tokens)]


def split_long_sentence(sentence: str) -> list[str]:
    if estimate_token_count(sentence) <= LONG_UNIT_TOKENS:
        return [sentence.strip()] if sentence.strip() else []
    return token_slice(sentence, LONG_UNIT_TOKENS)


def split_paragraph_to_units(paragraph: str, page: int, relevance: str) -> list[TextUnit]:
    units: list[TextUnit] = []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(paragraph.strip()) if part.strip()]
    if not sentences:
        sentences = [paragraph.strip()] if paragraph.strip() else []

    for sentence in sentences:
        for piece in split_long_sentence(sentence):
            token_count = estimate_token_count(piece)
            if token_count:
                units.append(TextUnit(piece, page, relevance, token_count))
    return units


def split_into_units(text: str, page: int, relevance: str) -> list[TextUnit]:
    raw_paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part.strip()]
    if len(raw_paragraphs) <= 1:
        raw_paragraphs = [text.strip()] if text.strip() else []

    units: list[TextUnit] = []
    for paragraph in raw_paragraphs:
        token_count = estimate_token_count(paragraph)
        if 0 < token_count <= LONG_UNIT_TOKENS:
            units.append(TextUnit(paragraph, page, relevance, token_count))
        else:
            units.extend(split_paragraph_to_units(paragraph, page, relevance))
    return units


def source_relevances_for(units: list[TextUnit]) -> list[str]:
    present = {unit.relevance for unit in units}
    return [relevance for relevance in RELEVANCE_ORDER if relevance in present]


def make_chunk_id(relative_path: str, chunk_index: int, text: str, page_start: int, page_end: int) -> str:
    payload = f"{relative_path}|{chunk_index}|{page_start}|{page_end}|{text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"afad_earthquake_v2_{digest}"


def build_chunk(
    units: list[TextUnit],
    source_file: str,
    relative_path: str,
    organization: str,
    chunk_index: int,
) -> dict[str, Any]:
    text = " ".join(unit.text for unit in units).strip()
    pages = [unit.page for unit in units]
    page_start = min(pages)
    page_end = max(pages)
    return {
        "chunk_id": make_chunk_id(relative_path, chunk_index, text, page_start, page_end),
        "text": text,
        "source_file": source_file,
        "relative_path": relative_path,
        "organization": organization,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_index": chunk_index,
        "source_relevances": source_relevances_for(units),
    }


def merge_chunks(left: dict[str, Any], right: dict[str, Any], relative_path: str, organization: str) -> dict[str, Any]:
    text = f"{str(left['text']).strip()} {str(right['text']).strip()}".strip()
    source_relevances = [
        relevance
        for relevance in RELEVANCE_ORDER
        if relevance in set(left.get("source_relevances", [])) | set(right.get("source_relevances", []))
    ]
    return {
        "chunk_id": "",
        "text": text,
        "source_file": left["source_file"],
        "relative_path": relative_path,
        "organization": organization,
        "page_start": min(int(left["page_start"]), int(right["page_start"])),
        "page_end": max(int(left["page_end"]), int(right["page_end"])),
        "chunk_index": 0,
        "source_relevances": source_relevances,
    }


def finalize_chunk_ids(chunks: list[dict[str, Any]], relative_path: str) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        updated = dict(chunk)
        updated["chunk_index"] = chunk_index
        updated["chunk_id"] = make_chunk_id(
            relative_path,
            chunk_index,
            str(updated["text"]),
            int(updated["page_start"]),
            int(updated["page_end"]),
        )
        finalized.append(updated)
    return finalized


def merge_short_chunks(
    chunks: list[dict[str, Any]],
    relative_path: str,
    organization: str,
) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return finalize_chunk_ids(chunks, relative_path)

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        current_tokens = estimate_token_count(str(current["text"]))

        if current_tokens >= MIN_EMIT_TOKENS:
            merged.append(current)
            index += 1
            continue

        if merged:
            previous = merged[-1]
            combined = merge_chunks(previous, current, relative_path, organization)
            if estimate_token_count(str(combined["text"])) <= MAX_TOKENS:
                merged[-1] = combined
                index += 1
                continue

        if index + 1 < len(chunks):
            next_chunk = chunks[index + 1]
            combined = merge_chunks(current, next_chunk, relative_path, organization)
            if estimate_token_count(str(combined["text"])) <= MAX_TOKENS:
                merged.append(combined)
                index += 2
                continue

        merged.append(current)
        index += 1

    return finalize_chunk_ids(merged, relative_path)


def tail_overlap(units: list[TextUnit]) -> list[TextUnit]:
    overlap: list[TextUnit] = []
    token_total = 0
    for unit in reversed(units):
        if overlap and token_total + unit.token_count > OVERLAP_TOKENS:
            break
        if unit.token_count > OVERLAP_TOKENS and overlap:
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
    organization: str,
    starting_chunk_index: int,
) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    current_units: list[TextUnit] = []
    current_tokens = 0
    chunk_index = starting_chunk_index
    has_new_content = False

    def emit(force: bool = False) -> None:
        nonlocal chunk_index, current_units, current_tokens, has_new_content
        if not current_units or not has_new_content:
            return
        if current_tokens < MIN_EMIT_TOKENS and not force:
            return
        chunks.append(build_chunk(current_units, source_file, relative_path, organization, chunk_index))
        chunk_index += 1
        current_units = tail_overlap(current_units)
        current_tokens = sum(unit.token_count for unit in current_units)
        has_new_content = False

    for unit in units:
        if unit.token_count > MAX_TOKENS:
            for piece in token_slice(unit.text, MAX_TOKENS):
                piece_unit = TextUnit(piece, unit.page, unit.relevance, estimate_token_count(piece))
                if current_units and current_tokens + piece_unit.token_count > MAX_TOKENS:
                    emit(force=current_tokens >= MIN_EMIT_TOKENS)
                    current_units = trim_to_fit(current_units, piece_unit)
                    current_tokens = sum(existing.token_count for existing in current_units)
                current_units.append(piece_unit)
                current_tokens += piece_unit.token_count
                has_new_content = True
                if current_tokens >= TARGET_TOKENS:
                    emit(force=True)
            continue

        if current_units and current_tokens + unit.token_count > MAX_TOKENS:
            emit(force=current_tokens >= MIN_EMIT_TOKENS)
            current_units = trim_to_fit(current_units, unit)
            current_tokens = sum(existing.token_count for existing in current_units)

        current_units.append(unit)
        current_tokens += unit.token_count
        has_new_content = True

        if current_tokens >= TARGET_TOKENS:
            emit(force=True)

    if has_new_content:
        chunks.append(build_chunk(current_units, source_file, relative_path, organization, chunk_index))
        chunk_index += 1

    return chunks, chunk_index


def included_page_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    included: list[dict[str, Any]] = []
    for record in records:
        relevance = str(record.get("relevance", "")).strip()
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


def process_document(input_path: Path) -> tuple[list[dict[str, Any]], set[int], Counter[str]]:
    records = read_json_list(input_path)
    pages = included_page_records(records)
    if not pages:
        return [], set(), Counter()

    relative_path = str(pages[0].get("relative_path", input_path.name))
    source_file = str(pages[0].get("source_file", input_path.with_suffix(".pdf").name))
    organization = str(pages[0].get("organization", "AFAD")) or "AFAD"
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
        run_chunks, chunk_index = chunk_units(units, source_file, relative_path, organization, chunk_index)
        chunks.extend(merge_short_chunks(run_chunks, relative_path, organization))

    chunks = finalize_chunk_ids(chunks, relative_path)

    included_page_relevance = {int(page["page"]): str(page["relevance"]) for page in pages}
    used_pages: set[int] = set()
    used_relevances: Counter[str] = Counter()
    for chunk in chunks:
        for page in range(int(chunk["page_start"]), int(chunk["page_end"]) + 1):
            if page in included_page_relevance:
                used_pages.add(page)
                used_relevances[included_page_relevance[page]] += 1
    return chunks, used_pages, used_relevances


def read_json_list(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list of page records: {path}")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object records in: {path}")
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(record)
    return records


def stats_for_records(records: list[dict[str, Any]], include_v2_fields: bool = False) -> CorpusStats:
    stats = CorpusStats()
    ids: list[str] = []
    for record in records:
        token_count = estimate_token_count(str(record.get("text", "")))
        source_relevances = [str(value) for value in record.get("source_relevances", [])]
        stats.add_chunk(token_count, source_relevances)
        chunk_id = str(record.get("chunk_id", ""))
        ids.append(chunk_id)
        if any(relevance == "excluded" for relevance in source_relevances):
            stats.excluded_page_references += 1
        if include_v2_fields and not REQUIRED_FIELDS.issubset(record.keys()):
            stats.all_required_fields_present = False
    stats.unique_chunk_ids = len(set(ids))
    stats.duplicate_chunk_ids = len(ids) - stats.unique_chunk_ids
    return stats


def build_relevance_lookup(input_root: Path) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for input_path in sorted(input_root.rglob("*.json")):
        for record in read_json_list(input_path):
            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            relevance = str(record.get("relevance", "")).strip()
            if relative_path and isinstance(page, int):
                lookup[(relative_path, page)] = relevance
    return lookup


def validate_output(output_file: Path, input_root: Path) -> CorpusStats:
    try:
        records = read_jsonl(output_file)
    except Exception:
        stats = CorpusStats(output_jsonl_valid=False)
        return stats

    stats = stats_for_records(records, include_v2_fields=True)
    relevance_lookup = build_relevance_lookup(input_root)
    for record in records:
        if estimate_token_count(str(record.get("text", ""))) > MAX_TOKENS:
            stats.output_jsonl_valid = False
        relative_path = str(record.get("relative_path", ""))
        page_start = int(record.get("page_start", 0))
        page_end = int(record.get("page_end", 0))
        for page in range(page_start, page_end + 1):
            if relevance_lookup.get((relative_path, page)) == "excluded":
                stats.excluded_page_references += 1
    return stats


def run_chunking(input_root: Path, output_file: Path) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for input_path in sorted(input_root.rglob("*.json")):
        document_chunks, _used_pages, _used_relevances = process_document(input_path)
        all_chunks.extend(document_chunks)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return all_chunks


def print_stats(old_stats: CorpusStats, v2_stats: CorpusStats) -> None:
    print("OLD CHUNKS")
    print(f"total chunks: {old_stats.total_chunks}")
    print(f"average tokens: {old_stats.average_tokens:.2f}")
    print(f"min tokens: {old_stats.minimum_tokens}")
    print(f"max tokens: {old_stats.max_tokens}")
    print("")
    print("V2 CHUNKS")
    print(f"total chunks: {v2_stats.total_chunks}")
    print(f"average tokens: {v2_stats.average_tokens:.2f}")
    print(f"min tokens: {v2_stats.minimum_tokens}")
    print(f"max tokens: {v2_stats.max_tokens}")
    print(f"relevant-only chunks: {v2_stats.relevant_only_chunks}")
    print(f"chunks containing uncertain pages: {v2_stats.chunks_with_uncertain}")
    print(f"unique chunk ids: {v2_stats.unique_chunk_ids}")
    print("")
    print(f"duplicate chunk_id = {v2_stats.duplicate_chunk_ids}")
    print(f"excluded page references = {v2_stats.excluded_page_references}")
    print(f"max token <= {MAX_TOKENS}: {str(v2_stats.max_tokens <= MAX_TOKENS).lower()}")
    print(f"output JSONL valid: {str(v2_stats.output_jsonl_valid).lower()}")
    print(f"all required fields present: {str(v2_stats.all_required_fields_present).lower()}")
    print("")
    print(f"Old chunk count: {old_stats.total_chunks}")
    print(f"Old average tokens: {old_stats.average_tokens:.2f}")
    print("")
    print(f"V2 chunk count: {v2_stats.total_chunks}")
    print(f"V2 average tokens: {v2_stats.average_tokens:.2f}")
    print(f"V2 min tokens: {v2_stats.minimum_tokens}")
    print(f"V2 max tokens: {v2_stats.max_tokens}")
    print("")
    print(f"Relevant-only chunks: {v2_stats.relevant_only_chunks}")
    print(f"Chunks with uncertain: {v2_stats.chunks_with_uncertain}")
    print("")
    print(f"Duplicate IDs: {v2_stats.duplicate_chunk_ids}")
    print(f"Excluded refs: {v2_stats.excluded_page_references}")
    print("")
    print(
        "Existing Test 1/Test 2/Test 3 files, old chunks, FAISS indexes, metadata, "
        "embedding model and reranker were not modified."
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    old_records = read_jsonl(args.old_chunks_file)
    old_stats = stats_for_records(old_records)
    run_chunking(args.input_root, args.output_file)
    v2_stats = validate_output(args.output_file, args.input_root)
    print_stats(old_stats, v2_stats)


if __name__ == "__main__":
    main()
