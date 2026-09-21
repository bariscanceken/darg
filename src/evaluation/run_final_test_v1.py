"""Run the locked independent Final Test v1 against the existing DARG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.rag.answerability import ANSWERABLE, NOT_ANSWERABLE
from src.rag.generate_answer import generate_answer

from .inspect_final_test_v1 import read_jsonl


DEFAULT_FINAL_TEST_FILE = Path("data/evaluation/final_test_v1.jsonl")
DEFAULT_RESULTS_FILE = Path("data/evaluation/final_test_v1_results.jsonl")
DEFAULT_SUMMARY_FILE = Path("data/evaluation/final_test_v1_summary.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DARG independent Final Test v1.")
    parser.add_argument("--final-test-file", type=Path, default=DEFAULT_FINAL_TEST_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if local retrieval models are not in cache.",
    )
    return parser.parse_args()


def verified_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("ground_truth_verified") is True]


def predicted_answerability(result: dict[str, Any]) -> str:
    label = str(result.get("answerability", {}).get("label", "")).strip()
    if label == ANSWERABLE:
        return ANSWERABLE
    return NOT_ANSWERABLE


def result_row(record: dict[str, Any], result: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    expected = str(record["top5_answerability"])
    if result is None:
        predicted = ""
        answer = ""
        sources: list[dict[str, Any]] = []
    else:
        predicted = predicted_answerability(result)
        answer = str(result.get("answer", ""))
        sources = list(result.get("sources", []))

    return {
        "id": record["id"],
        "question": record["question"],
        "expected_answerability": expected,
        "predicted_answerability": predicted,
        "answerability_correct": predicted == expected if predicted else False,
        "generated_answer": answer,
        "sources": sources,
        "error": error,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("answerability_correct") is True)
    wrong = total - correct
    tp = sum(
        1
        for row in rows
        if row.get("expected_answerability") == ANSWERABLE and row.get("predicted_answerability") == ANSWERABLE
    )
    fn = sum(
        1
        for row in rows
        if row.get("expected_answerability") == ANSWERABLE and row.get("predicted_answerability") == NOT_ANSWERABLE
    )
    fp = sum(
        1
        for row in rows
        if row.get("expected_answerability") == NOT_ANSWERABLE and row.get("predicted_answerability") == ANSWERABLE
    )
    tn = sum(
        1
        for row in rows
        if row.get("expected_answerability") == NOT_ANSWERABLE and row.get("predicted_answerability") == NOT_ANSWERABLE
    )
    return {
        "evaluated_questions": total,
        "correct": correct,
        "wrong": wrong,
        "answerability_accuracy": correct / total if total else 0.0,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def render_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Final Test v1 Summary",
            "",
            f"evaluated questions: {summary['evaluated_questions']}",
            f"correct: {summary['correct']}",
            f"wrong: {summary['wrong']}",
            f"answerability accuracy: {summary['answerability_accuracy']:.4f}",
            f"TP: {summary['tp']}",
            f"FN: {summary['fn']}",
            f"FP: {summary['fp']}",
            f"TN: {summary['tn']}",
        ]
    ) + "\n"


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_summary(summary), encoding="utf-8")


def run_final_test(final_test_file: Path, results_file: Path, summary_file: Path, allow_model_download: bool) -> dict[str, Any]:
    records = verified_records(read_jsonl(final_test_file))
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("", encoding="utf-8")

    for record in records:
        try:
            result = generate_answer(str(record["question"]), allow_model_download=allow_model_download)
            row = result_row(record, result)
        except Exception as exc:  # noqa: BLE001 - record error without retries.
            row = result_row(record, None, error=f"{type(exc).__name__}: {exc}")
        append_jsonl(results_file, row)
        print(f"{record['id']}: {row['predicted_answerability'] or 'ERROR'}")

    rows = load_results(results_file)
    summary = compute_summary(rows)
    write_summary(summary_file, summary)
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    summary = run_final_test(args.final_test_file, args.results_file, args.summary_file, args.allow_model_download)
    print("")
    print("Final Test v1")
    print(f"Evaluated questions: {summary['evaluated_questions']}")
    print(f"Correct: {summary['correct']}")
    print(f"Wrong: {summary['wrong']}")
    print(f"Answerability accuracy: {summary['answerability_accuracy']:.4f}")
    print(f"Results: {args.results_file}")
    print(f"Summary: {args.summary_file}")


if __name__ == "__main__":
    main()
