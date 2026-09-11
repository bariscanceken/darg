"""Apply deduplicated holdout manual labels and compute per-system metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/holdout_retrieval_judgments.csv")
DEFAULT_MAPPING_FILE = Path("data/evaluation/holdout_retrieval_manual_mapping.json")
DEFAULT_METRICS_FILE = Path("data/evaluation/holdout_retrieval_metrics.json")
TOP_K = 5
LABEL_RE = re.compile(r"^(M\d{3})\s*=\s*([01])\s*$")


@dataclass(frozen=True)
class Metrics:
    system: str
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
            "system": self.system,
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
    parser = argparse.ArgumentParser(description="Apply holdout manual labels and compute Test 4/Test 6 metrics.")
    parser.add_argument("labels_file", type=Path, help="Text file containing M001 = 0/1 style manual labels.")
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    return parser.parse_args()


def read_manual_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = LABEL_RE.match(stripped)
        if not match:
            raise ValueError(f"Invalid label line {line_number}: {line!r}")
        manual_id, label = match.groups()
        if manual_id in labels:
            raise ValueError(f"Duplicate manual_review_id in labels file: {manual_id}")
        labels[manual_id] = label
    return labels


def read_mapping(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    if not records:
        raise ValueError(f"No mapping records found: {path}")
    return records


def read_judgments(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"question_id", "system", "final_rank", "chunk_id", "relevant"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"Judgment CSV is missing columns: {', '.join(sorted(missing))}")
    return fieldnames, rows


def write_judgments(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def judgment_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["question_id"]),
        str(row["system"]),
        str(row["final_rank"]),
        str(row["chunk_id"]),
    )


def validate_label_coverage(labels: dict[str, str], mapping_records: list[dict[str, Any]]) -> None:
    expected_ids = {str(record["manual_review_id"]) for record in mapping_records}
    observed_ids = set(labels)
    if expected_ids != observed_ids:
        missing = expected_ids - observed_ids
        extra = observed_ids - expected_ids
        raise ValueError(f"Label/mapping mismatch. Missing={sorted(missing)}, extra={sorted(extra)}")


def apply_labels(rows: list[dict[str, str]], mapping_records: list[dict[str, Any]], labels: dict[str, str]) -> int:
    rows_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = judgment_key(row)
        if key in rows_by_key:
            raise ValueError(f"Duplicate judgment row key: {key}")
        rows_by_key[key] = row

    updated = 0
    for record in mapping_records:
        label = labels[str(record["manual_review_id"])]
        for mapped_row in record.get("judgment_rows", []):
            key = judgment_key(mapped_row)
            row = rows_by_key.get(key)
            if row is None:
                raise ValueError(f"Mapped judgment row not found: {key}")
            current = str(row.get("relevant", "")).strip()
            if current and current != label:
                raise ValueError(f"Refusing to overwrite different label at {key}: {current} -> {label}")
            if not current:
                updated += 1
            row["relevant"] = label
    return updated


def blank_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if not str(row.get("relevant", "")).strip())


def validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 300:
        raise ValueError(f"Expected 300 holdout judgment rows, got {len(rows)}")
    for row_number, row in enumerate(rows, start=2):
        value = str(row.get("relevant", "")).strip()
        if value not in {"0", "1"}:
            raise ValueError(f"Invalid relevant value at CSV line {row_number}: {value!r}")

    counts = Counter((row["question_id"], row["system"]) for row in rows)
    bad_counts = {key: value for key, value in counts.items() if value != TOP_K}
    if bad_counts:
        raise ValueError(f"Each question/system must have 5 rows. Bad counts: {bad_counts}")

    duplicates = [
        key
        for key, value in Counter((row["question_id"], row["system"], row["chunk_id"]) for row in rows).items()
        if value > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate chunk_id inside same question/system: {duplicates[:5]}")


def compute_metrics_for_system(system: str, rows: list[dict[str, str]]) -> Metrics:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["system"] == system:
            grouped.setdefault(row["question_id"], []).append(row)

    precision_total = 0.0
    hit_total = 0.0
    reciprocal_rank_total = 0.0
    relevant_results = 0
    failed_question_ids: list[str] = []

    for question_id in sorted(grouped):
        question_rows = sorted(grouped[question_id], key=lambda row: int(row["final_rank"]))
        if len(question_rows) != TOP_K:
            raise ValueError(f"{system} {question_id}: expected {TOP_K} rows, got {len(question_rows)}")
        ranks = [int(row["final_rank"]) for row in question_rows]
        if ranks != list(range(1, TOP_K + 1)):
            raise ValueError(f"{system} {question_id}: bad rank sequence {ranks}")

        labels = [int(row["relevant"]) for row in question_rows]
        relevant_ranks = [rank for rank, label in enumerate(labels, start=1) if label == 1]
        relevant_results += len(relevant_ranks)
        precision_total += len(relevant_ranks) / TOP_K
        if relevant_ranks:
            hit_total += 1.0
            reciprocal_rank_total += 1.0 / min(relevant_ranks)
        else:
            failed_question_ids.append(question_id)

    total_questions = len(grouped)
    return Metrics(
        system=system,
        relevant_results=relevant_results,
        precision_at_5=precision_total / total_questions,
        hit_rate_at_5=hit_total / total_questions,
        mrr=reciprocal_rank_total / total_questions,
        total_questions=total_questions,
        evaluated_results=sum(len(question_rows) for question_rows in grouped.values()),
        questions_with_no_relevant=len(failed_question_ids),
        failed_question_ids=failed_question_ids,
    )


def compute_all_metrics(rows: list[dict[str, str]]) -> list[Metrics]:
    systems = sorted({row["system"] for row in rows})
    return [compute_metrics_for_system(system, rows) for system in systems]


def write_metrics(
    path: Path,
    labels_file: Path,
    judgments_file: Path,
    mapping_file: Path,
    rows: list[dict[str, str]],
    metrics: list[Metrics],
    updated_rows: int,
) -> None:
    metric_by_system = {metric.system: metric for metric in metrics}
    test4 = metric_by_system.get("test4_baseline")
    test6 = metric_by_system.get("test6_score_fusion_alpha_0.50")
    delta = None
    if test4 and test6:
        delta = {
            "precision_at_5": test6.precision_at_5 - test4.precision_at_5,
            "hit_rate_at_5": test6.hit_rate_at_5 - test4.hit_rate_at_5,
            "mrr": test6.mrr - test4.mrr,
            "relevant_results": test6.relevant_results - test4.relevant_results,
            "questions_with_no_relevant": test6.questions_with_no_relevant - test4.questions_with_no_relevant,
        }
    payload = {
        "experiment": "Unseen holdout retrieval evaluation",
        "labels_file": str(labels_file),
        "judgments_file": str(judgments_file),
        "manual_mapping_file": str(mapping_file),
        "total_judgment_rows": len(rows),
        "updated_rows_from_manual_mapping": updated_rows,
        "blank_relevant_rows": blank_count(rows),
        "metrics_by_system": [metric.to_dict() for metric in metrics],
        "test6_minus_test4": delta,
        "note": "This holdout set was used only for final retrieval evaluation. No retrieval parameters were tuned using holdout results.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(rows: list[dict[str, str]], metrics: list[Metrics], updated_rows: int, metrics_file: Path) -> None:
    print(f"Total holdout judgment rows: {len(rows)}")
    print(f"Updated rows from manual mapping: {updated_rows}")
    print(f"Blank relevant rows: {blank_count(rows)}")
    print("")
    print("System | Relevant | Precision@5 | Hit Rate@5 | MRR | Failed")
    for metric in metrics:
        print(
            f"{metric.system} | "
            f"{metric.relevant_results}/{metric.evaluated_results} | "
            f"{metric.precision_at_5:.4f} | "
            f"{metric.hit_rate_at_5:.4f} | "
            f"{metric.mrr:.4f} | "
            f"{metric.questions_with_no_relevant}/{metric.total_questions} "
            f"({', '.join(metric.failed_question_ids) if metric.failed_question_ids else 'NONE'})"
        )
    print("")
    print(f"Metrics JSON path: {metrics_file}")
    print("This holdout set was used only for final retrieval evaluation.")
    print("No retrieval parameters were tuned using holdout results.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    labels = read_manual_labels(args.labels_file)
    mapping_records = read_mapping(args.mapping_file)
    validate_label_coverage(labels, mapping_records)
    fieldnames, rows = read_judgments(args.judgments_file)
    updated_rows = apply_labels(rows, mapping_records, labels)
    validate_rows(rows)
    write_judgments(args.judgments_file, fieldnames, rows)
    metrics = compute_all_metrics(rows)
    write_metrics(args.metrics_file, args.labels_file, args.judgments_file, args.mapping_file, rows, metrics, updated_rows)
    print_summary(rows, metrics, updated_rows, args.metrics_file)


if __name__ == "__main__":
    main()
