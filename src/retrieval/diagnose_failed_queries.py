"""Lexical diagnostics for failed retrieval evaluation queries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/failed_query_diagnostics.txt")
MAX_MATCHES_PER_QUERY = 10
PREVIEW_CHARS = 1000
SEPARATOR = "=" * 80

QUERY_TERMS: dict[str, dict[str, Any]] = {
    "Q04": {
        "question": "Deprem sırasında asansör kullanmalı mıyım?",
        "terms": ["asansör", "lift"],
    },
    "Q06": {
        "question": "Depremden sonra doğalgaz kokusu alırsam ne yapmalıyım?",
        "terms": ["doğalgaz", "doğal gaz", "gaz kokusu", "gaz kaçağı", "vana"],
    },
    "Q09": {
        "question": "Deprem sonrası artçı sarsıntılarda ne yapmalıyım?",
        "terms": ["artçı", "artçı sarsıntı", "artçı deprem"],
    },
    "Q13": {
        "question": "Deprem sırasında dışarıdaysam ne yapmalıyım?",
        "terms": ["dışarıda", "dış alanda", "açık alanda", "bina dışında"],
    },
    "Q14": {
        "question": "Deprem sırasında araç kullanıyorsam ne yapmalıyım?",
        "terms": ["araç", "otomobil", "araba", "sürücü", "trafikte"],
    },
    "Q19": {
        "question": "Depremden sonra yaralı birine nasıl yaklaşmalıyım?",
        "terms": ["yaralı", "ilk yardım", "ilkyardım", "hasta/yaralı"],
    },
    "Q20": {
        "question": "Deprem sonrasında resmi makamların tahliye talimatına uymalı mıyım?",
        "terms": ["tahliye", "yetkili", "resmi makam", "talimat", "uyarı"],
    },
}


@dataclass(frozen=True)
class Candidate:
    chunk: dict[str, Any]
    matched_terms: list[str]
    match_count: int
    first_position: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose failed retrieval queries with lexical chunk search.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-matches", type=int, default=MAX_MATCHES_PER_QUERY)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return text.casefold()


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_text(term))
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            chunks.append(record)
    return chunks


def find_candidates(chunks: list[dict[str, Any]], terms: list[str]) -> list[Candidate]:
    patterns = [(term, term_pattern(term)) for term in terms]
    candidates: list[Candidate] = []

    for chunk in chunks:
        text = normalize_text(str(chunk.get("text", "")))
        matched_terms: list[str] = []
        first_positions: list[int] = []
        match_count = 0

        for term, pattern in patterns:
            matches = list(pattern.finditer(text))
            if matches:
                matched_terms.append(term)
                first_positions.append(matches[0].start())
                match_count += len(matches)

        if matched_terms:
            candidates.append(
                Candidate(
                    chunk=chunk,
                    matched_terms=matched_terms,
                    match_count=match_count,
                    first_position=min(first_positions),
                )
            )

    return sorted(
        candidates,
        key=lambda candidate: (
            -len(candidate.matched_terms),
            -candidate.match_count,
            candidate.first_position,
            str(candidate.chunk.get("source_file", "")),
            int(candidate.chunk.get("chunk_index", 0)),
        ),
    )


def preview_text(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_relevances(chunk: dict[str, Any]) -> str:
    relevances = chunk.get("source_relevances", [])
    if isinstance(relevances, list):
        return ", ".join(str(item) for item in relevances)
    return str(relevances)


def render_candidate(index: int, candidate: Candidate) -> str:
    chunk = candidate.chunk
    return "\n".join(
        [
            f"Match {index}",
            f"chunk_id: {chunk.get('chunk_id')}",
            f"source_file: {chunk.get('source_file')}",
            f"pages: {chunk.get('page_start')}-{chunk.get('page_end')}",
            f"source_relevances: {format_relevances(chunk)}",
            f"matched_terms: {', '.join(candidate.matched_terms)}",
            f"text: {preview_text(str(chunk.get('text', '')))}",
        ]
    )


def render_report(
    diagnostics: dict[str, dict[str, Any]],
    max_matches: int,
) -> str:
    blocks: list[str] = []
    for question_id, config in QUERY_TERMS.items():
        candidates: list[Candidate] = diagnostics[question_id]["candidates"]
        lines = [
            SEPARATOR,
            question_id,
            f"Soru: {config['question']}",
            f"Keywords: {', '.join(config['terms'])}",
            f"Candidate chunks: {len(candidates)}",
            "",
        ]
        if not candidates:
            lines.append("No lexical matches found.")
        else:
            for index, candidate in enumerate(candidates[:max_matches], start=1):
                lines.append(render_candidate(index, candidate))
                lines.append("")
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def run_diagnostics(chunks_file: Path, output_file: Path, max_matches: int) -> dict[str, int]:
    chunks = read_chunks(chunks_file)
    diagnostics: dict[str, dict[str, Any]] = {}
    summary: dict[str, int] = {}

    for question_id, config in QUERY_TERMS.items():
        candidates = find_candidates(chunks, list(config["terms"]))
        diagnostics[question_id] = {"candidates": candidates}
        summary[question_id] = len(candidates)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(diagnostics, max_matches), encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    summary = run_diagnostics(args.chunks_file, args.output_file, args.max_matches)
    for question_id in QUERY_TERMS:
        print(f"{question_id}: {summary[question_id]} candidate chunks")


if __name__ == "__main__":
    main()
