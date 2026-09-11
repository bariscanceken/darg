"""Compute Retrieval Test 3 metrics from human judgments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_top50_reranked_judgments.csv")
DEFAULT_METRICS_FILE = Path("data/evaluation/retrieval_top50_reranked_metrics.json")
DEFAULT_BASELINE_METRICS_FILE = Path("data/evaluation/retrieval_metrics.json")
DEFAULT_TEST2_METRICS_FILE = Path("data/evaluation/retrieval_reranked_metrics.json")
TOP_K = 5


@dataclass(frozen=True)
class Metrics:
    precision_at_5: float
    hit_rate_at_5: float
    mrr: float
    total_questions: int
    evaluated_results: int
    questions_with_no_relevant: int
    failed_question_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_at_5": self.precision_at_5,
            "hit_rate_at_5": self.hit_rate_at_5,
            "mrr": self.mrr,
            "total_questions": self.total_questions,
            "evaluated_results": self.evaluated_results,
            "questions_with_no_relevant": self.questions_with_no_relevant,
            "failed_question_ids": self.failed_question_ids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for Retrieval Test 3 Top-50 reranked judgments.")
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--baseline-metrics-file", type=Path, default=DEFAULT_BASELINE_METRICS_FILE)
    parser.add_argument("--test2-metrics-file", type=Path, default=DEFAULT_TEST2_METRICS_FILE)
    return parser.parse_args()


def read_judgments(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Judgments file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required_fields = {
        "question_id",
        "question",
        "reranked_rank",
        "original_faiss_rank",
        "chunk_id",
        "source_file",
        "page_start",
        "page_end",
        "relevant",
    }
    fieldnames = set(rows[0].keys()) if rows else set()
    missing = required_fields - fieldnames
    if missing:
        raise ValueError(f"Judgments CSV is missing columns: {', '.join(sorted(missing))}")
    return rows


def count_blank_labels(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not str(row.get("relevant", "")).strip())


def validate_labels(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        value = str(row.get("relevant", "")).strip()
        if value not in {"0", "1"}:
            raise ValueError(f"Invalid relevant value at CSV line {row_number}: {value!r}")


def group_by_question(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["question_id"]), []).append(row)

    for question_id, question_rows in grouped.items():
        question_rows.sort(key=lambda item: int(item["reranked_rank"]))
        if len(question_rows) != TOP_K:
            raise ValueError(f"{question_id} has {len(question_rows)} rows, expected {TOP_K}")
        expected_ranks = list(range(1, TOP_K + 1))
        observed_ranks = [int(row["reranked_rank"]) for row in question_rows]
        if observed_ranks != expected_ranks:
            raise ValueError(f"{question_id} has ranks {observed_ranks}, expected {expected_ranks}")
    return grouped


def compute_metrics(rows: list[dict[str, str]]) -> Metrics:
    validate_labels(rows)
    grouped = group_by_question(rows)

    precision_total = 0.0
    hit_total = 0.0
    reciprocal_rank_total = 0.0
    failed_question_ids: list[str] = []

    for question_id in sorted(grouped):
        labels = [int(row["relevant"]) for row in grouped[question_id]]
        relevant_ranks = [index + 1 for index, label in enumerate(labels) if label == 1]
        precision_total += len(relevant_ranks) / TOP_K
        if relevant_ranks:
            hit_total += 1.0
            reciprocal_rank_total += 1.0 / min(relevant_ranks)
        else:
            failed_question_ids.append(question_id)

    total_questions = len(grouped)
    if total_questions == 0:
        raise ValueError("No question rows found in judgments CSV.")

    return Metrics(
        precision_at_5=precision_total / total_questions,
        hit_rate_at_5=hit_total / total_questions,
        mrr=reciprocal_rank_total / total_questions,
        total_questions=total_questions,
        evaluated_results=len(rows),
        questions_with_no_relevant=len(failed_question_ids),
        failed_question_ids=failed_question_ids,
    )


def write_metrics(path: Path, metrics: Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_metrics_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def failed_display(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "N/A"
    failed = metrics.get("questions_with_no_relevant")
    total = metrics.get("total_questions")
    if failed is None or total is None:
        return "N/A"
    return f"{failed}/{total}"


def print_comparison_table(
    baseline_metrics: dict[str, Any] | None,
    test2_metrics: dict[str, Any] | None,
    test3_metrics: Metrics,
) -> None:
    rows = [
        ("Test 1", baseline_metrics),
        ("Test 2", test2_metrics),
        ("Test 3", test3_metrics.to_dict()),
    ]
    print("")
    print("Test | Precision@5 | Hit Rate@5 | MRR | Failed")
    for name, metrics in rows:
        print(
            f"{name} | "
            f"{format_number(metrics.get('precision_at_5') if metrics else None)} | "
            f"{format_number(metrics.get('hit_rate_at_5') if metrics else None)} | "
            f"{format_number(metrics.get('mrr') if metrics else None)} | "
            f"{failed_display(metrics)}"
        )


def print_test3_summary(metrics: Metrics) -> None:
    print("Test 3 - FAISS Top-50 + Cross-Encoder Top-5")
    print("")
    print(f"Precision@5: {metrics.precision_at_5:.4f}")
    print(f"Hit Rate@5: {metrics.hit_rate_at_5:.4f}")
    print(f"MRR: {metrics.mrr:.4f}")
    print(f"Failed questions: {metrics.questions_with_no_relevant}/{metrics.total_questions}")
    print(f"Failed IDs: {', '.join(metrics.failed_question_ids) if metrics.failed_question_ids else 'NONE'}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    rows = read_judgments(args.judgments_file)
    blank_labels = count_blank_labels(rows)
    if blank_labels:
        print("Judgments required: yes")
        print(f"Blank labels: {blank_labels}")
        return

    metrics = compute_metrics(rows)
    write_metrics(args.metrics_file, metrics)
    print_test3_summary(metrics)

    baseline_metrics = read_metrics_if_exists(args.baseline_metrics_file)
    test2_metrics = read_metrics_if_exists(args.test2_metrics_file)
    print_comparison_table(baseline_metrics, test2_metrics, metrics)


if __name__ == "__main__":
    main()
