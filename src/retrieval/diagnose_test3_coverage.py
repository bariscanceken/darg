"""Detailed corpus coverage diagnostics before Retrieval Test 3."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/test3_coverage_diagnostics.txt")
MAX_RESULTS_PER_QUERY = 20
PREVIEW_CHARS = 1200
SEPARATOR = "=" * 80


@dataclass(frozen=True)
class AnswerPattern:
    label: str
    term_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CompiledPattern:
    label: str
    term_groups: tuple[tuple[re.Pattern[str], ...], ...]


@dataclass(frozen=True)
class Candidate:
    chunk: dict[str, Any]
    matched_patterns: list[str]
    occurrence_count: int
    first_position: int


def variants(*terms: str) -> tuple[str, ...]:
    return terms


QUERY_PATTERNS: dict[str, dict[str, Any]] = {
    "Q04": {
        "question": "Deprem sırasında asansör kullanmalı mıyım?",
        "patterns": [
            AnswerPattern("asansör kullanmayın", (variants("asansör kullanmayın", "asansor kullanmayin"),)),
            AnswerPattern("asansöre binmeyin", (variants("asansöre binmeyin", "asansore binmeyin"),)),
            AnswerPattern("asansörleri kullanmayın", (variants("asansörleri kullanmayın", "asansorleri kullanmayin"),)),
            AnswerPattern("merdiven veya asansör", (variants("merdiven"), variants("asansör", "asansor"))),
            AnswerPattern("asansörden uzak", (variants("asansörden uzak", "asansorden uzak"),)),
        ],
    },
    "Q09": {
        "question": "Deprem sonrası artçı sarsıntılarda ne yapmalıyım?",
        "patterns": [
            AnswerPattern("artçı sarsıntı", (variants("artçı sarsıntı", "artçÈ sarsÈntÈ", "artci sarsinti"),)),
            AnswerPattern("artçı deprem", (variants("artçı deprem", "artçÈ deprem", "artci deprem"),)),
            AnswerPattern("artçılara hazırlıklı", (variants("artçılara", "artçÈlara", "artci"), variants("hazırlıklı", "hazÈrlÈklÈ", "hazirlikli"))),
            AnswerPattern("artçı sırasında", (variants("artçı", "artçÈ", "artci"), variants("sırasında", "sÈrasÈnda", "sirasinda"))),
            AnswerPattern("artçı sarsıntılarda", (variants("artçı sarsıntılarda", "artçÈ sarsÈntÈlarda", "artci sarsintilarda"),)),
        ],
    },
    "Q13": {
        "question": "Deprem sırasında dışarıdaysam ne yapmalıyım?",
        "patterns": [
            AnswerPattern("dışarıdaysanız", (variants("dışarıdaysanız", "dÈşarÈdaysanÈz", "disaridaysaniz"),)),
            AnswerPattern("açık alandaysanız", (variants("açık alandaysanız", "açÈk alandaysanÈz", "acik alandaysaniz"),)),
            AnswerPattern("bina dışındaysanız", (variants("bina dışındaysanız", "bina dÈşÈndaysanÈz", "bina disindaysaniz"),)),
            AnswerPattern("binalardan uzak durun", (variants("binalardan uzak durun"),)),
            AnswerPattern("enerji hatlarından uzak", (variants("enerji hatlarından uzak", "enerji hatlarÈndan uzak"),)),
        ],
    },
    "Q14": {
        "question": "Deprem sırasında araç kullanıyorsam ne yapmalıyım?",
        "patterns": [
            AnswerPattern("araç kullanıyorsanız", (variants("araç kullanıyorsanız", "araç kullanÈyorsanÈz", "arac kullaniyorsaniz"),)),
            AnswerPattern("araç içindeyseniz", (variants("araç içindeyseniz", "araç içindeyseniz", "arac icindeyseniz"),)),
            AnswerPattern("otomobil", (variants("otomobil"), variants("deprem", "sarsıntı", "sarsÈntÈ", "güvenli", "güvenlİ", "durun", "durdur"))),
            AnswerPattern("güvenli bir yerde durun", (variants("güvenli bir yerde durun", "güvenli bir yerde", "güvenlİ bir yerde"), variants("durun", "durdur"))),
            AnswerPattern("köprü", (variants("köprü", "kopru"), variants("araç", "arac", "deprem", "sarsıntı", "sarsÈntÈ", "güvenli", "durun"))),
            AnswerPattern("viyadük", (variants("viyadük", "viyaduk"), variants("araç", "arac", "deprem", "sarsıntı", "sarsÈntÈ", "güvenli", "durun"))),
            AnswerPattern("tünel", (variants("tünel", "tunel"), variants("araç", "arac", "deprem", "sarsıntı", "sarsÈntÈ", "güvenli", "durun"))),
        ],
    },
    "Q20": {
        "question": "Deprem sonrasında resmi makamların tahliye talimatına uymalı mıyım?",
        "patterns": [
            AnswerPattern("yetkililerin talimatlarına uyun", (variants("yetkililerin talimatlarına", "yetkililerin talimatlarÈna"), variants("uyun"))),
            AnswerPattern("resmi makam", (variants("resmi makam", "resmî makam"), variants("tahliye", "talimat", "uyarı", "uyarÈ", "uyun"))),
            AnswerPattern("tahliye talimatı", (variants("tahliye talimatı", "tahliye talimatÈ"),)),
            AnswerPattern("tahliye emri", (variants("tahliye emri"),)),
            AnswerPattern("uyarılara uyun", (variants("uyarılara uyun", "uyarÈlara uyun"),)),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose direct answer coverage for failed retrieval queries.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS_PER_QUERY)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return text.casefold()


def compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_text(term)).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def compile_patterns(patterns: list[AnswerPattern]) -> list[CompiledPattern]:
    return [
        CompiledPattern(
            pattern.label,
            tuple(tuple(compile_term(term) for term in group) for group in pattern.term_groups),
        )
        for pattern in patterns
    ]


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


def find_pattern_hit(text: str, pattern: CompiledPattern) -> tuple[int, int] | None:
    first_positions: list[int] = []
    occurrence_count = 0

    for group in pattern.term_groups:
        group_matches: list[re.Match[str]] = []
        for term_pattern in group:
            group_matches.extend(term_pattern.finditer(text))
        if not group_matches:
            return None
        group_matches.sort(key=lambda match: match.start())
        first_positions.append(group_matches[0].start())
        occurrence_count += len(group_matches)

    return occurrence_count, min(first_positions)


def find_candidates(chunks: list[dict[str, Any]], patterns: list[AnswerPattern]) -> list[Candidate]:
    compiled_patterns = compile_patterns(patterns)
    candidates: list[Candidate] = []

    for chunk in chunks:
        text = normalize_text(str(chunk.get("text", "")))
        matched_patterns: list[str] = []
        first_positions: list[int] = []
        occurrence_count = 0

        for pattern in compiled_patterns:
            hit = find_pattern_hit(text, pattern)
            if hit is None:
                continue
            pattern_occurrences, first_position = hit
            matched_patterns.append(pattern.label)
            occurrence_count += pattern_occurrences
            first_positions.append(first_position)

        if matched_patterns:
            candidates.append(
                Candidate(
                    chunk=chunk,
                    matched_patterns=matched_patterns,
                    occurrence_count=occurrence_count,
                    first_position=min(first_positions),
                )
            )

    return sorted(
        candidates,
        key=lambda candidate: (
            -len(candidate.matched_patterns),
            -candidate.occurrence_count,
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


def render_candidate(question_id: str, candidate: Candidate) -> str:
    chunk = candidate.chunk
    return "\n".join(
        [
            f"question_id: {question_id}",
            f"source_file: {chunk.get('source_file')}",
            f"pages: {chunk.get('page_start')}-{chunk.get('page_end')}",
            f"chunk_id: {chunk.get('chunk_id')}",
            f"matched_pattern: {'; '.join(candidate.matched_patterns)}",
            f"text: {preview_text(str(chunk.get('text', '')))}",
        ]
    )


def render_report(diagnostics: dict[str, list[Candidate]], max_results: int) -> str:
    blocks: list[str] = []

    for question_id, config in QUERY_PATTERNS.items():
        candidates = diagnostics[question_id]
        lines = [
            SEPARATOR,
            question_id,
            f"Question: {config['question']}",
            f"Total candidate matches: {len(candidates)}",
            "",
        ]

        if not candidates:
            lines.append("No strong answer-pattern candidates found.")
        else:
            for candidate in candidates[:max_results]:
                lines.append(render_candidate(question_id, candidate))
                lines.append("")

        blocks.append("\n".join(lines).rstrip())

    blocks.append(
        "\n".join(
            [
                SEPARATOR,
                "Manual Coverage Review",
                "Q04 Coverage: UNKNOWN",
                "Q09 Coverage: UNKNOWN",
                "Q13 Coverage: UNKNOWN",
                "Q14 Coverage: UNKNOWN",
                "Q20 Coverage: UNKNOWN",
            ]
        )
    )
    return "\n\n".join(blocks) + "\n"


def run_diagnostics(chunks_file: Path, output_file: Path, max_results: int) -> tuple[dict[str, int], int]:
    chunks = read_chunks(chunks_file)
    diagnostics: dict[str, list[Candidate]] = {}
    summary: dict[str, int] = {}

    for question_id, config in QUERY_PATTERNS.items():
        candidates = find_candidates(chunks, list(config["patterns"]))
        diagnostics[question_id] = candidates
        summary[question_id] = len(candidates)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(diagnostics, max_results), encoding="utf-8")
    return summary, len(chunks)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    _, chunk_count = run_diagnostics(args.chunks_file, args.output_file, args.max_results)
    print(f"Questions analyzed: {len(QUERY_PATTERNS)}")
    print(f"Corpus chunks scanned: {chunk_count}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
