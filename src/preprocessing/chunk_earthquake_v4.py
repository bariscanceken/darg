"""Build model-window-sized earthquake chunks measured in E5 subword tokens.

v2 chunks were sized in whitespace words (target 400, max 550). Turkish text
averages ~2 E5 subword tokens per word, so 91% of v2 chunks exceeded the 512-token
window of both the embedding model and the reranker, and about a third of each
chunk was never seen. v4 measures size with the E5 tokenizer itself and keeps
every chunk small enough that a query plus the whole chunk fits the reranker
window, which also keeps each chunk focused on fewer topics.

Reuses the page selection / page-run helpers of v2; v2 outputs are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.preprocessing.chunk_earthquake_v2 import (
    SENTENCE_BOUNDARY_RE,
    included_page_records,
    read_json_list,
    source_relevances_for,
    split_consecutive_page_runs,
)


DEFAULT_INPUT_ROOT = Path("data/processed/earthquake_relevance_v3")
DEFAULT_OUTPUT_FILE = Path("data/processed/chunks/afad_earthquake_chunks_v4.jsonl")
TOKENIZER_MODEL = "intfloat/multilingual-e5-small"

# Sizes in E5 subword tokens. Reranker input = query (~20-60) + passage, window 512.
TARGET_TOKENS = 280
MAX_TOKENS = 350
MIN_EMIT_TOKENS = 100
OVERLAP_TOKENS = 50
WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Unit:
    text: str
    page: int
    relevance: str
    tokens: int


@lru_cache(maxsize=1)
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, local_files_only=True)


def count_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def split_to_fit(sentence: str) -> list[str]:
    """Split an over-long sentence into word runs that each fit MAX_TOKENS."""
    if count_tokens(sentence) <= MAX_TOKENS:
        return [sentence]
    pieces: list[str] = []
    current: list[str] = []
    for word in WORD_RE.findall(sentence):
        if current and count_tokens(" ".join(current + [word])) > TARGET_TOKENS:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def page_units(text: str, page: int, relevance: str) -> list[Unit]:
    units: list[Unit] = []
    for sentence in (part.strip() for part in SENTENCE_BOUNDARY_RE.split(text.strip())):
        for piece in split_to_fit(sentence) if sentence else []:
            units.append(Unit(piece, page, relevance, count_tokens(piece)))
    return units


def make_chunk_id(relative_path: str, chunk_index: int, text: str, page_start: int, page_end: int) -> str:
    payload = f"{relative_path}|{chunk_index}|{page_start}|{page_end}|{text}"
    return "afad_earthquake_v4_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def tail_overlap(units: list[Unit]) -> list[Unit]:
    """Carry the last sentence(s), up to OVERLAP_TOKENS, into the next chunk."""
    overlap: list[Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.tokens > OVERLAP_TOKENS:
            break
        overlap.insert(0, unit)
        total += unit.tokens
    return overlap


def total_tokens(units: list[Unit]) -> int:
    return sum(unit.tokens for unit in units)


def chunk_run(units: list[Unit]) -> list[list[Unit]]:
    """Greedy sentence packing to TARGET_TOKENS, never above MAX_TOKENS, with overlap."""
    chunks: list[list[Unit]] = []
    current: list[Unit] = []
    fresh = 0  # units in `current` not yet emitted in any chunk
    for unit in units:
        if current and total_tokens(current) + unit.tokens > MAX_TOKENS:
            if fresh:
                chunks.append(current)
                current = tail_overlap(current)
            while current and total_tokens(current) + unit.tokens > MAX_TOKENS:
                current.pop(0)
            fresh = 0
        current.append(unit)
        fresh += 1
        if total_tokens(current) >= TARGET_TOKENS:
            chunks.append(current)
            current = tail_overlap(current)
            fresh = 0
    if fresh:
        new_units = current[len(current) - fresh:]
        if chunks and total_tokens(current) < MIN_EMIT_TOKENS and total_tokens(chunks[-1] + new_units) <= MAX_TOKENS:
            chunks[-1] = chunks[-1] + new_units  # fold a short tail into the previous chunk
        else:
            chunks.append(current)
    return chunks


def process_document(input_path: Path) -> list[dict[str, Any]]:
    pages = included_page_records(read_json_list(input_path))
    if not pages:
        return []
    relative_path = str(pages[0].get("relative_path", input_path.name))
    source_file = str(pages[0].get("source_file", input_path.with_suffix(".pdf").name))
    organization = str(pages[0].get("organization", "AFAD")) or "AFAD"

    chunks: list[dict[str, Any]] = []
    for run in split_consecutive_page_runs(pages):
        units = [unit for page in run for unit in page_units(str(page["text"]), int(page["page"]), str(page["relevance"]))]
        for group in chunk_run(units):
            text = " ".join(unit.text for unit in group).strip()
            page_numbers = [unit.page for unit in group]
            chunk_index = len(chunks)
            chunks.append({
                "chunk_id": make_chunk_id(relative_path, chunk_index, text, min(page_numbers), max(page_numbers)),
                "text": text,
                "source_file": source_file,
                "relative_path": relative_path,
                "organization": organization,
                "page_start": min(page_numbers),
                "page_end": max(page_numbers),
                "chunk_index": chunk_index,
                "source_relevances": source_relevances_for(group),
                "token_count": count_tokens(text),
            })
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build E5-token-sized v4 earthquake chunks.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    chunks: list[dict[str, Any]] = []
    for input_path in sorted(args.input_root.rglob("*.json")):
        chunks.extend(process_document(input_path))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = sorted(chunk["token_count"] for chunk in chunks)
    ids = [chunk["chunk_id"] for chunk in chunks]
    print(f"v4 chunks: {len(chunks)}")
    print(f"E5 tokens: median {sizes[len(sizes) // 2]}  p90 {sizes[int(len(sizes) * 0.9)]}  max {sizes[-1]}")
    print(f"chunks above MAX_TOKENS ({MAX_TOKENS}): {sum(size > MAX_TOKENS for size in sizes)}")
    print(f"duplicate chunk ids: {len(ids) - len(set(ids))}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
