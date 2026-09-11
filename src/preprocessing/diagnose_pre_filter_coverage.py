"""Diagnose pre-filter page coverage for specific failed retrieval questions."""

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
DEFAULT_OUTPUT_FILE = Path("data/evaluation/pre_filter_coverage_diagnostics.txt")
MAX_CANDIDATES_PER_QUESTION = 30
SEPARATOR = "=" * 80
STATUSES = ("relevant", "uncertain", "excluded", "not_found")


QUERY_TERMS: dict[str, dict[str, Any]] = {
    "Q04": {
        "question": "Deprem sırasında asansör kullanmalı mıyım?",
        "terms": ["asansör", "asansörü", "asansörleri", "asansöre", "asansörden", "lift"],
    },
    "Q14": {
        "question": "Deprem sırasında araç kullanıyorsam ne yapmalıyım?",
        "terms": [
            "araç kullanıyorsanız",
            "araç kullanırken",
            "araç içindeyseniz",
            "araç içerisindeyseniz",
            "otomobil",
            "sürücü",
            "trafikte",
            "güvenli bir yerde durun",
            "güvenli yerde durun",
            "köprü",
            "viyadük",
            "tünel",
        ],
    },
}


@dataclass(frozen=True)
class CandidatePage:
    question_id: str
    source_file: str
    relative_path: str
    page: int
    matched_terms: list[str]
    relevance_status: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find pre-filter page candidates for failed retrieval questions.")
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--relevance-root", type=Path, default=DEFAULT_RELEVANCE_ROOT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_PER_QUESTION)
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize(term)).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of page records: {path}")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object records in: {path}")
    return records


def build_relevance_lookup(relevance_root: Path) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for path in sorted(relevance_root.rglob("*.json")):
        for record in read_json_list(path):
            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            relevance = str(record.get("relevance", "")).strip()
            if relative_path and isinstance(page, int) and relevance:
                lookup[(relative_path, page)] = relevance
    return lookup


def find_matched_terms(text: str, compiled_terms: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    normalized_text = normalize(text)
    return [term for term, pattern in compiled_terms if pattern.search(normalized_text)]


def scan_pages(
    text_root: Path,
    relevance_lookup: dict[tuple[str, int], str],
) -> tuple[dict[str, list[CandidatePage]], int]:
    compiled_by_question = {
        question_id: [(term, compile_term(term)) for term in config["terms"]]
        for question_id, config in QUERY_TERMS.items()
    }
    candidates: dict[str, list[CandidatePage]] = {question_id: [] for question_id in QUERY_TERMS}
    scanned_pages = 0

    for path in sorted(text_root.rglob("*.json")):
        for record in read_json_list(path):
            scanned_pages += 1
            text = str(record.get("text", ""))
            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            if not relative_path or not isinstance(page, int):
                continue

            relevance_status = relevance_lookup.get((relative_path, page), "not_found")
            for question_id, compiled_terms in compiled_by_question.items():
                matched_terms = find_matched_terms(text, compiled_terms)
                if matched_terms:
                    candidates[question_id].append(
                        CandidatePage(
                            question_id=question_id,
                            source_file=str(record.get("source_file", "")),
                            relative_path=relative_path,
                            page=page,
                            matched_terms=matched_terms,
                            relevance_status=relevance_status,
                            text=text,
                        )
                    )

    return candidates, scanned_pages


def status_summary(candidates: list[CandidatePage]) -> Counter[str]:
    counts: Counter[str] = Counter(candidate.relevance_status for candidate in candidates)
    for status in STATUSES:
        counts.setdefault(status, 0)
    return counts


def render_candidate(candidate: CandidatePage) -> str:
    return "\n".join(
        [
            f"question_id: {candidate.question_id}",
            f"source_file: {candidate.source_file}",
            f"relative_path: {candidate.relative_path}",
            f"page: {candidate.page}",
            f"relevance_status: {candidate.relevance_status}",
            f"matched_terms: {', '.join(candidate.matched_terms)}",
            f"text: {candidate.text}",
        ]
    )


def render_report(candidates: dict[str, list[CandidatePage]], max_candidates: int) -> str:
    blocks: list[str] = []

    for question_id, config in QUERY_TERMS.items():
        question_candidates = candidates[question_id]
        lines = [
            SEPARATOR,
            question_id,
            f"Question: {config['question']}",
            f"Total pre-filter matches: {len(question_candidates)}",
            "",
        ]
        if not question_candidates:
            lines.append("No pre-filter candidate pages found.")
        else:
            for candidate in question_candidates[:max_candidates]:
                lines.append(render_candidate(candidate))
                lines.append("")
        blocks.append("\n".join(lines).rstrip())

    summary_lines = [SEPARATOR, "Summary"]
    for question_id in QUERY_TERMS:
        counts = status_summary(candidates[question_id])
        summary_lines.extend(
            [
                "",
                f"{question_id}:",
                f"Total pre-filter matches: {len(candidates[question_id])}",
                f"Relevant: {counts['relevant']}",
                f"Uncertain: {counts['uncertain']}",
                f"Excluded: {counts['excluded']}",
                f"Not found in relevance output: {counts['not_found']}",
            ]
        )
    blocks.append("\n".join(summary_lines))
    return "\n\n".join(blocks) + "\n"


def run_diagnostics(
    text_root: Path,
    relevance_root: Path,
    output_file: Path,
    max_candidates: int,
) -> tuple[dict[str, list[CandidatePage]], int]:
    relevance_lookup = build_relevance_lookup(relevance_root)
    candidates, scanned_pages = scan_pages(text_root, relevance_lookup)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(candidates, max_candidates), encoding="utf-8")
    return candidates, scanned_pages


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    candidates, scanned_pages = run_diagnostics(
        args.text_root,
        args.relevance_root,
        args.output_file,
        args.max_candidates,
    )

    print(f"Questions analyzed: {len(QUERY_TERMS)}")
    print(f"Extracted pages scanned: {scanned_pages}")
    print(f"Q04 pre-filter matches: {len(candidates['Q04'])}")
    print(f"Q14 pre-filter matches: {len(candidates['Q14'])}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
