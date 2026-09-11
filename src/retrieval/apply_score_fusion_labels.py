"""Apply deduplicated Test 6 manual labels and compute score-fusion metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_score_fusion_judgments.csv")
DEFAULT_MAPPING_FILE = Path("data/evaluation/retrieval_score_fusion_manual_mapping.json")
DEFAULT_METRICS_FILE = Path("data/evaluation/retrieval_score_fusion_metrics.json")
TOP_K = 5

MANUAL_LABELS = {
    "M001": "0",
    "M002": "1",
    "M003": "1",
    "M004": "1",
    "M005": "1",
    "M006": "0",
    "M007": "1",
    "M008": "0",
    "M009": "0",
    "M010": "0",
    "M011": "1",
    "M012": "1",
    "M013": "0",
    "M014": "0",
    "M015": "1",
    "M016": "1",
    "M017": "0",
    "M018": "0",
    "M019": "0",
    "M020": "1",
    "M021": "0",
    "M022": "1",
    "M023": "0",
    "M024": "0",
    "M025": "0",
    "M026": "1",
    "M027": "0",
    "M028": "0",
    "M029": "1",
    "M030": "1",
    "M031": "1",
    "M032": "1",
    "M033": "0",
    "M034": "0",
    "M035": "0",
    "M036": "1",
    "M037": "0",
    "M038": "0",
    "M039": "0",
    "M040": "1",
    "M041": "0",
    "M042": "1",
}


@dataclass(frozen=True)
class Metrics:
    alpha: str
    relevant_results: int
    precision_at_5: float
    hit_rate_at_5: float
    mrr: float
    total_questions: int
    evaluated_results: int
    questions_with_no_relevant: int
    failed_question_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "relevant_results": self.relevant_results,
            "precision_at_5": self.precision_at_5,
            "hit_rate_at_5": self.hit_rate_at_5,
            "mrr": self.mrr,
            "total_questions": self.total_questions,
            "evaluated_results": self.evaluated_results,
            "questions_with_no_relevant": self.questions_with_no_relevant,
            "failed_question_ids": self.failed_question_ids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual Test 6 score-fusion labels and compute metrics.")
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"No judgment rows found: {path}")
    required = {"alpha", "question_id", "final_rank", "chunk_id", "relevant"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Judgments CSV is missing columns: {', '.join(sorted(missing))}")
    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_mapping(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    expected_ids = {f"M{index:03d}" for index in range(1, len(MANUAL_LABELS) + 1)}
    observed_ids = {str(record["manual_review_id"]) for record in records}
    if observed_ids != expected_ids:
        missing = expected_ids - observed_ids
        extra = observed_ids - expected_ids
        raise ValueError(f"Manual mapping ID mismatch. Missing={sorted(missing)}, extra={sorted(extra)}")
    return records


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        f"{float(row['alpha']):.2f}",
        str(row["question_id"]),
        str(row["final_rank"]),
        str(row["chunk_id"]),
    )


def mapping_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        f"{float(row['alpha']):.2f}",
        str(row["question_id"]),
        str(row["final_rank"]),
        str(row["chunk_id"]),
    )


def apply_labels(rows: list[dict[str, str]], mapping_records: list[dict[str, Any]]) -> int:
    rows_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = row_key(row)
        if key in rows_by_key:
            raise ValueError(f"Duplicate judgment row key: {key}")
        rows_by_key[key] = row

    updated = 0
    for record in mapping_records:
        manual_id = str(record["manual_review_id"])
        label = MANUAL_LABELS[manual_id]
        for csv_row in record.get("csv_rows", []):
            key = mapping_key(csv_row)
            row = rows_by_key.get(key)
            if row is None:
                raise ValueError(f"Could not find judgment row for {manual_id}: {key}")
            current = str(row.get("relevant", "")).strip()
            if current and current != label:
                raise ValueError(f"{manual_id} would overwrite an existing different label at {key}: {current} -> {label}")
            if not current:
                updated += 1
            row["relevant"] = label
    return updated


def blank_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not str(row.get("relevant", "")).strip())


def validate_labels(rows: list[dict[str, str]]) -> None:
    for row_number, row in enumerate(rows, start=2):
        value = str(row.get("relevant", "")).strip()
        if value not in {"0", "1"}:
            raise ValueError(f"Invalid relevant value at CSV line {row_number}: {value!r}")


def compute_metrics_for_alpha(alpha: str, rows: list[dict[str, str]]) -> Metrics:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if f"{float(row['alpha']):.2f}" == alpha:
            grouped.setdefault(str(row["question_id"]), []).append(row)

    if not grouped:
        raise ValueError(f"No rows found for alpha={alpha}")

    precision_total = 0.0
    hit_total = 0.0
    reciprocal_rank_total = 0.0
    relevant_results = 0
    failed_question_ids: list[str] = []

    for question_id in sorted(grouped):
        question_rows = sorted(grouped[question_id], key=lambda row: int(row["final_rank"]))
        if len(question_rows) != TOP_K:
            raise ValueError(f"alpha={alpha} {question_id}: expected {TOP_K} rows, got {len(question_rows)}")
        ranks = [int(row["final_rank"]) for row in question_rows]
        if ranks != list(range(1, TOP_K + 1)):
            raise ValueError(f"alpha={alpha} {question_id}: bad rank sequence {ranks}")

        labels = [int(row["relevant"]) for row in question_rows]
        relevant_ranks = [index + 1 for index, label in enumerate(labels) if label == 1]
        relevant_results += len(relevant_ranks)
        precision_total += len(relevant_ranks) / TOP_K
        if relevant_ranks:
            hit_total += 1.0
            reciprocal_rank_total += 1.0 / min(relevant_ranks)
        else:
            failed_question_ids.append(question_id)

    total_questions = len(grouped)
    evaluated_results = sum(len(items) for items in grouped.values())
    return Metrics(
        alpha=alpha,
        relevant_results=relevant_results,
        precision_at_5=precision_total / total_questions,
        hit_rate_at_5=hit_total / total_questions,
        mrr=reciprocal_rank_total / total_questions,
        total_questions=total_questions,
        evaluated_results=evaluated_results,
        questions_with_no_relevant=len(failed_question_ids),
        failed_question_ids=failed_question_ids,
    )


def compute_all_metrics(rows: list[dict[str, str]]) -> list[Metrics]:
    validate_labels(rows)
    alphas = sorted({f"{float(row['alpha']):.2f}" for row in rows}, key=float)
    return [compute_metrics_for_alpha(alpha, rows) for alpha in alphas]


def write_metrics(path: Path, rows: list[dict[str, str]], metrics: list[Metrics], updated_rows: int) -> None:
    relevant_results_total = sum(1 for row in rows if str(row["relevant"]).strip() == "1")
    payload = {
        "experiment": "Test 6 - V2 FAISS Top-50 + Cross-Encoder + dense/reranker score fusion",
        "judgments_file": str(DEFAULT_JUDGMENTS_FILE),
        "manual_mapping_file": str(DEFAULT_MAPPING_FILE),
        "total_judgment_rows": len(rows),
        "updated_rows_from_manual_mapping": updated_rows,
        "blank_relevant_rows": blank_count(rows),
        "relevant_results_total": relevant_results_total,
        "metrics_by_alpha": [metric.to_dict() for metric in metrics],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(rows: list[dict[str, str]], metrics: list[Metrics], updated_rows: int, metrics_file: Path) -> None:
    total_rows = len(rows)
    alpha_counts = Counter(f"{float(row['alpha']):.2f}" for row in rows)
    print(f"Total Test 6 judgment rows: {total_rows}")
    print(f"Updated rows from manual mapping: {updated_rows}")
    print(f"Blank relevant rows: {blank_count(rows)}")
    print(f"Relevant results / {total_rows}: {sum(1 for row in rows if row['relevant'] == '1')}")
    print("")
    print("Alpha | Precision@5 | Hit Rate@5 | MRR | Failed")
    for metric in metrics:
        print(
            f"alpha={metric.alpha} | "
            f"{metric.precision_at_5:.4f} | "
            f"{metric.hit_rate_at_5:.4f} | "
            f"{metric.mrr:.4f} | "
            f"{metric.questions_with_no_relevant}/{metric.total_questions} "
            f"({', '.join(metric.failed_question_ids) if metric.failed_question_ids else 'NONE'})"
        )
    print("")
    print(f"Alpha row counts: {dict(alpha_counts)}")
    print(f"Metrics JSON path: {metrics_file}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    fieldnames, rows = read_csv(args.judgments_file)
    mapping_records = read_mapping(args.mapping_file)
    updated_rows = apply_labels(rows, mapping_records)
    write_csv(args.judgments_file, fieldnames, rows)
    metrics = compute_all_metrics(rows)
    write_metrics(args.metrics_file, rows, metrics, updated_rows)
    print_summary(rows, metrics, updated_rows, args.metrics_file)


if __name__ == "__main__":
    main()
