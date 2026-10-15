"""Section-aware chunks with a "document › section › subsection" header (index v5).

v4 chunks were cut from flattened page text: a chunk could start mid-section, and nothing in
it said which section it came from, so advice written for "Afet Sonrası › Açık alanda" looked
like generic advice and a question about being outdoors could not match it by meaning.
v5 chunks are built from extract_structured.py blocks:
  * chunks prefer to start where a section starts (heading level 1-2, or 3 once a chunk is half full);
  * every chunk text begins with its section path, so the embedding, BM25, the reranker and the
    answer step all see where the passage sits;
  * tables are kept as "header: value" lines and are never split mid-row;
  * pages whose text is only about other disasters are dropped, as in v2/v4
    (src/preprocessing/filter_earthquake.py), but OCR'd and table pages are kept.
Size is measured with the E5 tokenizer (as in v4): at most 350 tokens including the header.

  python -m src.preprocessing.chunk_structured_v5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.preprocessing.chunk_earthquake_v2 import SENTENCE_BOUNDARY_RE
from src.preprocessing.chunk_earthquake_v4 import MAX_TOKENS, count_tokens
from src.preprocessing.filter_earthquake import classify_page


DEFAULT_INPUT_ROOT = Path("data/processed/structured_v5")
DEFAULT_OUTPUT_FILE = Path("data/processed/chunks/afad_earthquake_chunks_v5.jsonl")
TITLES_FILE = Path("data/geo/document_titles.json")  # src/preprocessing/build_document_titles.py
TARGET_TOKENS = 280
MIN_SECTION_FLUSH_TOKENS = 60   # a new level-1/2 section starts a new chunk once this much is buffered
OVERLAP_UNITS = 1               # sentences repeated from the previous chunk of the same section
MAX_PATH_DEPTH = 3
WORD_RE = re.compile(r"\S+")
TITLE_SEPARATOR = " › "


@dataclass
class Unit:
    text: str
    page: int
    tokens: int
    kind: str          # "text" | "table" | "heading"
    path: tuple[str, ...]
    ocr: bool = False


@dataclass
class Buffer:
    units: list[Unit] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(unit.tokens for unit in self.units)


@lru_cache(maxsize=1)
def known_titles() -> dict[str, str]:
    if not TITLES_FILE.exists():
        return {}
    return {unicodedata.normalize("NFC", name): title
            for name, title in json.loads(TITLES_FILE.read_text(encoding="utf-8")).items()}


def document_title(record: dict[str, Any]) -> str:
    """Curated title, else the first large heading on the first pages, else a readable file name."""
    title = known_titles().get(unicodedata.normalize("NFC", record["source_file"]))
    if title:
        return title
    for page in record["pages"][:3]:
        for block in page["blocks"]:
            if block["kind"] == "heading" and block["level"] <= 2 and 6 <= len(block["text"]) <= 80:
                return block["text"]
    return re.sub(r"[_\-]+", " ", Path(record["source_file"]).stem).strip()


def split_sentence(sentence: str, budget: int) -> list[str]:
    if count_tokens(sentence) <= budget:
        return [sentence]
    pieces, current = [], []
    for word in WORD_RE.findall(sentence):
        if current and count_tokens(" ".join(current + [word])) > budget:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def block_units(block: dict[str, Any], page: int, path: tuple[str, ...], budget: int) -> list[Unit]:
    ocr = bool(block.get("ocr"))
    if block["kind"] == "table":
        parts = [line for line in block["text"].splitlines() if line.strip()]
    else:
        parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(block["text"].strip()) if part.strip()]
    units = []
    for part in parts:
        for piece in split_sentence(part, budget):
            units.append(Unit(piece, page, count_tokens(piece), block["kind"], path, ocr))
    return units


def page_is_included(page: dict[str, Any], relative_path: str) -> bool:
    text = " ".join(block["text"] for block in page["blocks"])
    if not text.strip():
        return False
    relevance, _, _, _ = classify_page(text, relative_path, False)
    return relevance != "excluded"


def header_for(title: str, path: tuple[str, ...]) -> str:
    return TITLE_SEPARATOR.join([title, *path])


def document_chunks(record: dict[str, Any]) -> list[dict[str, Any]]:
    title = document_title(record)
    relative_path = record["relative_path"].removeprefix("afad/")  # same form as v4 ("plans/irap/...")
    stack: list[str] = []
    chunks: list[dict[str, Any]] = []
    buffer = Buffer()

    def budget(path: tuple[str, ...]) -> int:
        return MAX_TOKENS - count_tokens(header_for(title, path)) - 2

    def flush(keep_overlap: bool) -> None:
        body_units = [unit for unit in buffer.units if unit.kind != "heading"]
        if body_units:
            path = buffer.units[0].path
            body = " ".join(unit.text if unit.kind != "table" else unit.text + "\n" for unit in buffer.units).strip()
            body = re.sub(r"\s*\n\s*", "\n", body)
            text = header_for(title, path) + "\n" + body
            pages = [unit.page for unit in buffer.units]
            payload = f"{relative_path}|{len(chunks)}|{text}"
            chunks.append({
                "chunk_id": "afad_earthquake_v5_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20],
                "text": text,
                "title": title,
                "section_path": list(path),
                "source_file": record["source_file"],
                "relative_path": relative_path,
                "organization": record.get("organization", "AFAD"),
                "page_start": min(pages),
                "page_end": max(pages),
                "chunk_index": len(chunks),
                "has_table": any(unit.kind == "table" for unit in buffer.units),
                "ocr": any(unit.ocr for unit in buffer.units),
                "token_count": count_tokens(text),
            })
        tail = [unit for unit in body_units[-OVERLAP_UNITS:]] if keep_overlap and body_units else []
        tail = [unit for unit in tail if unit.kind == "text" and unit.tokens < TARGET_TOKENS // 3]
        buffer.units = tail

    for page in record["pages"]:
        if not page_is_included(page, relative_path):
            continue
        for block in page["blocks"]:
            if block["kind"] == "heading":
                level = min(block["level"], MAX_PATH_DEPTH)
                stack = stack[:level - 1] + [block["text"]]
                path = tuple(stack)
                starts_section = level <= 2 and buffer.tokens >= MIN_SECTION_FLUSH_TOKENS
                starts_subsection = level == 3 and buffer.tokens >= TARGET_TOKENS // 2
                if starts_section or starts_subsection:
                    flush(keep_overlap=False)
                heading = Unit(block["text"] + ":", page["page"], count_tokens(block["text"]) + 1, "heading", path,
                               bool(block.get("ocr")))
                if any(unit.kind != "heading" for unit in buffer.units):
                    buffer.units.append(heading)  # sub-heading inside a running chunk stays visible inline
                else:
                    buffer.units = [Unit("", page["page"], 0, "heading", path)]
                continue
            path = tuple(stack)
            for unit in block_units(block, page["page"], path, budget(path)):
                current_path = buffer.units[0].path if buffer.units else path
                if buffer.units and buffer.tokens + unit.tokens > budget(current_path):
                    flush(keep_overlap=unit.path == current_path)
                    if buffer.units and buffer.tokens + unit.tokens > budget(path):
                        buffer.units = []
                if not buffer.units:
                    buffer.units = [Unit("", unit.page, 0, "heading", path)]
                buffer.units.append(unit)
                if buffer.tokens >= TARGET_TOKENS and unit.kind != "table":
                    flush(keep_overlap=True)
    flush(keep_overlap=False)
    return chunks


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Section-aware v5 chunks from structured extraction.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    total = 0
    over = 0
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for path in sorted(args.input_root.rglob("*.json")):
            chunks = document_chunks(json.loads(path.read_text(encoding="utf-8")))
            for chunk in chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total += len(chunks)
            over += sum(chunk["token_count"] > MAX_TOKENS for chunk in chunks)
    print(f"{total} chunks written to {args.output_file} ({over} over {MAX_TOKENS} tokens)")


if __name__ == "__main__":
    main()
