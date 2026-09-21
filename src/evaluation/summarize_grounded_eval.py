"""Summarize manually verified grounded-answer evaluation labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path("data/evaluation/rag_grounded_eval_v1.jsonl")
GROUNDING_LABELS = {"PASS", "MINOR", "FAIL"}
CITATION_LABELS = {"PASS", "PARTIAL", "FAIL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize DARG grounded answer evaluation labels.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row, path, line_number)
            rows.append(row)
    return rows


def validate_row(row: dict[str, Any], path: Path, line_number: int) -> None:
    required_fields = [
        "id",
        "question",
        "generated_answer",
        "sources",
        "grounding_label",
        "citation_label",
        "evaluation_note",
        "manually_verified",
    ]
    for field in required_fields:
        if field not in row:
            raise ValueError(f"{path}:{line_number} missing field: {field}")

    if row["grounding_label"] is not None and row["grounding_label"] not in GROUNDING_LABELS:
        raise ValueError(f"{path}:{line_number} invalid grounding_label: {row['grounding_label']}")
    if row["citation_label"] is not None and row["citation_label"] not in CITATION_LABELS:
        raise ValueError(f"{path}:{line_number} invalid citation_label: {row['citation_label']}")


def verified_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("manually_verified") is True]


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified = verified_rows(rows)
    grounding_counts = Counter(row.get("grounding_label") for row in verified)
    citation_counts = Counter(row.get("citation_label") for row in verified)
    evaluated = len(verified)

    pass_count = grounding_counts.get("PASS", 0)
    minor_count = grounding_counts.get("MINOR", 0)
    fail_count = grounding_counts.get("FAIL", 0)

    return {
        "evaluated_answers": evaluated,
        "pass_count": pass_count,
        "minor_count": minor_count,
        "fail_count": fail_count,
        "strict_grounded_answer_accuracy": pass_count / evaluated if evaluated else 0.0,
        "acceptable_grounded_answer_accuracy": (pass_count + minor_count) / evaluated if evaluated else 0.0,
        "citation_pass_count": citation_counts.get("PASS", 0),
        "citation_partial_count": citation_counts.get("PARTIAL", 0),
        "citation_fail_count": citation_counts.get("FAIL", 0),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("Grounded Answer Evaluation v1 Summary")
    print(f"evaluated answers: {summary['evaluated_answers']}")
    print(f"PASS count: {summary['pass_count']}")
    print(f"MINOR count: {summary['minor_count']}")
    print(f"FAIL count: {summary['fail_count']}")
    print(f"Strict Grounded Answer Accuracy: {summary['strict_grounded_answer_accuracy']:.4f}")
    print(f"Acceptable Grounded Answer Accuracy: {summary['acceptable_grounded_answer_accuracy']:.4f}")
    print(f"Citation PASS count: {summary['citation_pass_count']}")
    print(f"Citation PARTIAL count: {summary['citation_partial_count']}")
    print(f"Citation FAIL count: {summary['citation_fail_count']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    rows = read_jsonl(args.input_file)
    summary = compute_summary(rows)
    print_summary(summary)


if __name__ == "__main__":
    main()
