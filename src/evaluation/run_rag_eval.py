"""Run RAG Evaluation v1 against the existing DARG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.rag.answerability import ANSWERABLE, NOT_ANSWERABLE
from src.rag.generate_answer import generate_answer

from .inspect_rag_eval import read_eval_questions


DEFAULT_EVAL_FILE = Path("data/evaluation/rag_eval_v1.jsonl")
DEFAULT_RESULTS_FILE = Path("data/evaluation/rag_eval_v1_results.jsonl")
DEFAULT_SUMMARY_FILE = Path("data/evaluation/rag_eval_v1_summary.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DARG RAG Evaluation v1.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if local retrieval models are not in cache.",
    )
    return parser.parse_args()


def verified_questions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only records with verified ground-truth labels."""
    return [record for record in records if record.get("ground_truth_verified") is True]


def predicted_answerability(result: dict[str, Any]) -> str:
    """Extract the normal pipeline's answerability decision."""
    label = str(result.get("answerability", {}).get("label", "")).strip()
    if label == ANSWERABLE:
        return ANSWERABLE
    return NOT_ANSWERABLE


def evaluation_row(record: dict[str, Any], result: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    """Build one JSONL evaluation row."""
    expected = str(record["expected_answerability"])
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
        "retrieval_miss": bool(record.get("retrieval_miss", False)),
        "generated_answer": answer,
        "sources": sources,
        "error": error,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row so completed results survive interrupted runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("answerability_correct") is True)
    wrong = total - correct
    actual_answerable = sum(1 for row in rows if row.get("expected_answerability") == ANSWERABLE)
    actual_not_answerable = sum(1 for row in rows if row.get("expected_answerability") == NOT_ANSWERABLE)

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

    wrong_rows = [row for row in rows if row.get("answerability_correct") is not True]
    retrieval_miss_rows = [row for row in rows if row.get("retrieval_miss") is True]

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
        "actual_answerable": actual_answerable,
        "actual_not_answerable": actual_not_answerable,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "wrong_rows": wrong_rows,
        "retrieval_miss_rows": retrieval_miss_rows,
    }


def render_summary(summary: dict[str, Any], title: str = "RAG Evaluation v1 Summary") -> str:
    lines = [
        title,
        "",
        f"Toplam değerlendirilen soru: {summary['total']}",
        f"Doğru answerability kararı sayısı: {summary['correct']}",
        f"Yanlış answerability kararı sayısı: {summary['wrong']}",
        f"Answerability Accuracy: {summary['accuracy']:.4f}",
        f"ANSWERABLE ground truth sayısı: {summary['actual_answerable']}",
        f"NOT_ANSWERABLE ground truth sayısı: {summary['actual_not_answerable']}",
        "",
        "Confusion matrix:",
        "              Predicted",
        "            A        N",
        f"Actual A    {summary['tp']}        {summary['fn']}",
        f"Actual N    {summary['fp']}        {summary['tn']}",
        "",
        "Yanlış karar verilen sorular:",
    ]

    if summary["wrong_rows"]:
        for row in summary["wrong_rows"]:
            lines.extend(
                [
                    f"ID: {row['id']}",
                    f"question: {row['question']}",
                    f"expected: {row['expected_answerability']}",
                    f"predicted: {row['predicted_answerability'] or 'ERROR'}",
                    f"retrieval_miss: {str(row['retrieval_miss']).lower()}",
                    "",
                ]
            )
    else:
        lines.append("NONE")
        lines.append("")

    lines.append("Retrieval miss soruları:")
    if summary["retrieval_miss_rows"]:
        for row in summary["retrieval_miss_rows"]:
            lines.extend(
                [
                    f"ID: {row['id']}",
                    f"question: {row['question']}",
                    f"expected: {row['expected_answerability']}",
                    f"predicted: {row['predicted_answerability'] or 'ERROR'}",
                    "",
                ]
            )
    else:
        lines.append("NONE")

    return "\n".join(lines).rstrip() + "\n"


def write_summary(path: Path, summary: dict[str, Any], title: str = "RAG Evaluation v1 Summary") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_summary(summary, title), encoding="utf-8")


def run_evaluation(
    eval_file: Path,
    results_file: Path,
    summary_file: Path,
    allow_model_download: bool,
    summary_title: str = "RAG Evaluation v1 Summary",
) -> dict[str, Any]:
    records = verified_questions(read_eval_questions(eval_file))
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("", encoding="utf-8")

    for record in records:
        try:
            result = generate_answer(str(record["question"]), allow_model_download=allow_model_download)
            row = evaluation_row(record, result)
        except Exception as exc:  # noqa: BLE001 - keep eval moving without retries.
            row = evaluation_row(record, None, error=f"{type(exc).__name__}: {exc}")
        append_jsonl(results_file, row)
        print(f"{record['id']}: {row['predicted_answerability'] or 'ERROR'}")

    rows = load_results(results_file)
    summary = compute_summary(rows)
    write_summary(summary_file, summary, summary_title)
    return summary


def print_summary(summary: dict[str, Any], results_file: Path, summary_file: Path, label: str = "RAG Evaluation v1") -> None:
    print("")
    print(label)
    print(f"Evaluated questions: {summary['total']}")
    print(f"Correct answerability decisions: {summary['correct']}")
    print(f"Wrong answerability decisions: {summary['wrong']}")
    print(f"Answerability Accuracy: {summary['accuracy']:.4f}")
    print(f"Results: {results_file}")
    print(f"Summary: {summary_file}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    summary = run_evaluation(args.eval_file, args.results_file, args.summary_file, args.allow_model_download)
    print_summary(summary, args.results_file, args.summary_file)


if __name__ == "__main__":
    main()
