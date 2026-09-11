"""Targeted corpus coverage diagnostics for failed retrieval questions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/answer_coverage_diagnostics.txt")
MAX_RESULTS_PER_QUERY = 15
PREVIEW_CHARS = 1200
SEPARATOR = "=" * 80


@dataclass(frozen=True)
class MatchRule:
    label: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class CompiledRule:
    label: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class Candidate:
    chunk: dict[str, Any]
    matched_rules: list[str]
    total_occurrences: int
    first_position: int


QUERY_RULES: dict[str, dict[str, Any]] = {
    "Q04": {
        "title": "Asansör",
        "question": "Deprem sırasında asansör kullanmalı mıyım?",
        "rules": [
            MatchRule('"asansör kullanmayın"', ("asansör kullanmayın",)),
            MatchRule('"asansörü kullanmayın"', ("asansörü kullanmayın",)),
            MatchRule('"asansörleri kullanmayın"', ("asansörleri kullanmayın",)),
            MatchRule('"asansöre binmeyin"', ("asansöre binmeyin",)),
            MatchRule("asansör + merdiven", ("asansör", "merdiven")),
        ],
    },
    "Q06": {
        "title": "Doğalgaz",
        "question": "Depremden sonra doğalgaz kokusu alırsam ne yapmalıyım?",
        "rules": [
            MatchRule('"gaz kokusu"', ("gaz kokusu",)),
            MatchRule('"doğalgaz kokusu"', ("doğalgaz kokusu",)),
            MatchRule('"doğal gaz kokusu"', ("doğal gaz kokusu",)),
            MatchRule('"gaz kaçağı"', ("gaz kaçağı",)),
            MatchRule('"gaz vanasını"', ("gaz vanasını",)),
            MatchRule('"doğalgaz vanasını"', ("doğalgaz vanasını",)),
            MatchRule("gaz + kibrit", ("gaz", "kibrit")),
            MatchRule("gaz + elektrik", ("gaz", "elektrik")),
        ],
    },
    "Q09": {
        "title": "Artçı",
        "question": "Deprem sonrası artçı sarsıntılarda ne yapmalıyım?",
        "rules": [
            MatchRule('"artçı sarsıntı"', ("artçı sarsıntı",)),
            MatchRule('"artçı deprem"', ("artçı deprem",)),
            MatchRule('artçı + "hasarlı bina"', ("artçı", "hasarlı bina")),
            MatchRule("artçı + uzak", ("artçı", "uzak")),
            MatchRule("artçı + girme", ("artçı", "girme")),
        ],
    },
    "Q13": {
        "title": "Dışarıda",
        "question": "Deprem sırasında dışarıdaysam ne yapmalıyım?",
        "rules": [
            MatchRule('"açık alanda"', ("açık alanda",)),
            MatchRule('"bina dışında"', ("bina dışında",)),
            MatchRule('"dışarıdaysanız"', ("dışarıdaysanız",)),
            MatchRule('"dışarıda iseniz"', ("dışarıda iseniz",)),
            MatchRule('"binalardan uzak"', ("binalardan uzak",)),
            MatchRule('"enerji hatlarından uzak"', ("enerji hatlarından uzak",)),
            MatchRule('"elektrik hatlarından uzak"', ("elektrik hatlarından uzak",)),
        ],
    },
    "Q14": {
        "title": "Araç",
        "question": "Deprem sırasında araç kullanıyorsam ne yapmalıyım?",
        "rules": [
            MatchRule('"araç kullanıyorsanız"', ("araç kullanıyorsanız",)),
            MatchRule('"araç kullanırken"', ("araç kullanırken",)),
            MatchRule('"aracı durdur"', ("aracı durdur",)),
            MatchRule('"aracınızı durdur"', ("aracınızı durdur",)),
            MatchRule("araç + köprü", ("araç", "köprü")),
            MatchRule("araç + üst geçit", ("araç", "üst geçit")),
            MatchRule("araç + güvenli", ("araç", "güvenli")),
        ],
    },
    "Q19": {
        "title": "Yaralı",
        "question": "Depremden sonra yaralı birine nasıl yaklaşmalıyım?",
        "rules": [
            MatchRule('"yaralıyı hareket ettirmeyin"', ("yaralıyı hareket ettirmeyin",)),
            MatchRule('"yaralıya"', ("yaralıya",)),
            MatchRule('"ilk yardım"', ("ilk yardım",)),
            MatchRule('"ilkyardım"', ("ilkyardım",)),
            MatchRule("yaralı + 112", ("yaralı", "112")),
            MatchRule("yaralı + solunum", ("yaralı", "solunum")),
            MatchRule("yaralı + bilinç", ("yaralı", "bilinç")),
        ],
    },
    "Q20": {
        "title": "Tahliye",
        "question": "Deprem sonrasında resmi makamların tahliye talimatına uymalı mıyım?",
        "rules": [
            MatchRule('"yetkililerin uyarılarına"', ("yetkililerin uyarılarına",)),
            MatchRule('"yetkililerin talimatlarına"', ("yetkililerin talimatlarına",)),
            MatchRule('"resmi makam"', ("resmi makam",)),
            MatchRule('"resmî makam"', ("resmî makam",)),
            MatchRule('"tahliye talimatı"', ("tahliye talimatı",)),
            MatchRule("tahliye + uyun", ("tahliye", "uyun")),
            MatchRule("tahliye + uymalı", ("tahliye", "uymalı")),
            MatchRule("tahliye + yetkili", ("tahliye", "yetkili")),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create targeted answer coverage diagnostics from chunk text.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-results", type=int, default=MAX_RESULTS_PER_QUERY)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return text.casefold()


def compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_text(term)).replace(r"\ ", r"\s+")
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


def compile_rules(rules: list[MatchRule]) -> list[CompiledRule]:
    return [CompiledRule(rule.label, tuple(compile_term(term) for term in rule.terms)) for rule in rules]


def find_rule_hits(text: str, rule: CompiledRule) -> tuple[int, int] | None:
    positions: list[int] = []
    occurrence_total = 0

    for pattern in rule.patterns:
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        positions.append(matches[0].start())
        occurrence_total += len(matches)

    return occurrence_total, min(positions)


def find_candidates(chunks: list[dict[str, Any]], rules: list[MatchRule]) -> list[Candidate]:
    candidates: list[Candidate] = []
    compiled_rules = compile_rules(rules)

    for chunk in chunks:
        text = normalize_text(str(chunk.get("text", "")))
        matched_rules: list[str] = []
        total_occurrences = 0
        first_positions: list[int] = []

        for rule in compiled_rules:
            hit = find_rule_hits(text, rule)
            if hit is None:
                continue
            occurrence_count, first_position = hit
            matched_rules.append(rule.label)
            total_occurrences += occurrence_count
            first_positions.append(first_position)

        if matched_rules:
            candidates.append(
                Candidate(
                    chunk=chunk,
                    matched_rules=matched_rules,
                    total_occurrences=total_occurrences,
                    first_position=min(first_positions),
                )
            )

    return sorted(
        candidates,
        key=lambda candidate: (
            -len(candidate.matched_rules),
            -candidate.total_occurrences,
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


def render_candidate(index: int, candidate: Candidate) -> str:
    chunk = candidate.chunk
    return "\n".join(
        [
            f"Match {index}",
            f"chunk_id: {chunk.get('chunk_id')}",
            f"source_file: {chunk.get('source_file')}",
            f"page_start: {chunk.get('page_start')}",
            f"page_end: {chunk.get('page_end')}",
            f"matched phrase: {'; '.join(candidate.matched_rules)}",
            f"text: {preview_text(str(chunk.get('text', '')))}",
        ]
    )


def render_report(diagnostics: dict[str, list[Candidate]], max_results: int) -> str:
    blocks: list[str] = []
    for question_id, config in QUERY_RULES.items():
        candidates = diagnostics[question_id]
        lines = [
            SEPARATOR,
            f"{question_id} - {config['title']}",
            f"Soru: {config['question']}",
            f"Total matches: {len(candidates)}",
            "",
        ]

        if not candidates:
            lines.append("No matching chunks found.")
        else:
            for index, candidate in enumerate(candidates[:max_results], start=1):
                lines.append(render_candidate(index, candidate))
                lines.append("")

        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def run_diagnostics(chunks_file: Path, output_file: Path, max_results: int) -> dict[str, int]:
    chunks = read_chunks(chunks_file)
    diagnostics: dict[str, list[Candidate]] = {}
    summary: dict[str, int] = {}

    for question_id, config in QUERY_RULES.items():
        candidates = find_candidates(chunks, list(config["rules"]))
        diagnostics[question_id] = candidates
        summary[question_id] = len(candidates)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(diagnostics, max_results), encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    summary = run_diagnostics(args.chunks_file, args.output_file, args.max_results)
    for question_id in QUERY_RULES:
        print(f"{question_id}: {summary[question_id]} matches")


if __name__ == "__main__":
    main()
