"""Prepare a deduplicated manual review file for Test 6 score fusion labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_score_fusion_judgments.csv")
DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_score_fusion_results.json")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_score_fusion_manual_review.txt")
DEFAULT_MAPPING_FILE = Path("data/evaluation/retrieval_score_fusion_manual_mapping.json")
SEPARATOR = "=" * 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate blank Test 6 score-fusion labels for manual review.")
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    return parser.parse_args()


def alpha_key(value: Any) -> str:
    return f"{float(value):.2f}"


def read_judgment_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            row["_csv_line_number"] = str(line_number)
            rows.append(row)
    return rows


def build_result_lookup(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for experiment in payload.get("experiments", []):
        alpha = alpha_key(experiment["alpha"])
        for question in experiment.get("questions", []):
            question_id = str(question["question_id"])
            for result in question.get("final_top5", []):
                lookup[(alpha, question_id, str(result["chunk_id"]))] = result
    return lookup


def first_existing_result(
    rows: list[dict[str, str]],
    result_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    for row in rows:
        result = result_lookup.get((alpha_key(row["alpha"]), row["question_id"], row["chunk_id"]))
        if result is not None:
            return result
    return {}


def group_blank_rows(rows: list[dict[str, str]]) -> OrderedDict[tuple[str, str], list[dict[str, str]]]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for row in rows:
        if str(row.get("relevant", "")).strip():
            continue
        key = (str(row["question_id"]), str(row["chunk_id"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def sorted_group_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (float(row["alpha"]), int(row["final_rank"])))


def render_review_record(
    manual_review_id: str,
    rows: list[dict[str, str]],
    result_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> str:
    sorted_rows = sorted_group_rows(rows)
    first_row = sorted_rows[0]
    first_result = first_existing_result(sorted_rows, result_lookup)
    appeared_in_alphas = ", ".join(row["alpha"] for row in sorted_rows)
    final_ranks = "\n".join(f"alpha={row['alpha']} -> rank {row['final_rank']}" for row in sorted_rows)
    text = str(first_result.get("text", "")).strip()

    faiss_rank = first_row.get("faiss_rank", "")
    faiss_score = first_row.get("faiss_score", "")
    reranker_score = first_row.get("reranker_score", "")

    return "\n".join(
        [
            SEPARATOR,
            f"manual_review_id: {manual_review_id}",
            f"question_id: {first_row['question_id']}",
            f"question: {first_row['question']}",
            f"chunk_id: {first_row['chunk_id']}",
            f"source_file: {first_row['source_file']}",
            f"page_start/page_end: {first_row['page_start']}-{first_row['page_end']}",
            f"appeared_in_alphas: {appeared_in_alphas}",
            "final_ranks:",
            final_ranks,
            f"FAISS rank: {faiss_rank}",
            f"FAISS score: {faiss_score}",
            f"reranker score: {reranker_score}",
            "",
            "text:",
            text,
        ]
    )


def mapping_record(
    manual_review_id: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    sorted_rows = sorted_group_rows(rows)
    first_row = sorted_rows[0]
    return {
        "manual_review_id": manual_review_id,
        "question_id": first_row["question_id"],
        "question": first_row["question"],
        "chunk_id": first_row["chunk_id"],
        "source_file": first_row["source_file"],
        "page_start": int(first_row["page_start"]),
        "page_end": int(first_row["page_end"]),
        "csv_rows": [
            {
                "csv_line_number": int(row["_csv_line_number"]),
                "alpha": row["alpha"],
                "question_id": row["question_id"],
                "final_rank": int(row["final_rank"]),
                "chunk_id": row["chunk_id"],
            }
            for row in sorted_rows
        ],
    }


def write_review(
    path: Path,
    grouped_rows: OrderedDict[tuple[str, str], list[dict[str, str]]],
    result_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    blocks: list[str] = []
    for index, rows in enumerate(grouped_rows.values(), start=1):
        manual_review_id = f"M{index:03d}"
        blocks.append(render_review_record(manual_review_id, rows, result_lookup))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def write_mapping(
    path: Path,
    source_judgments_file: Path,
    grouped_rows: OrderedDict[tuple[str, str], list[dict[str, str]]],
) -> None:
    records = [
        mapping_record(f"M{index:03d}", rows)
        for index, rows in enumerate(grouped_rows.values(), start=1)
    ]
    payload = {
        "source_judgments_file": str(source_judgments_file),
        "unique_key": "question_id + chunk_id",
        "manual_review_count": len(records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    rows = read_judgment_rows(args.judgments_file)
    result_lookup = build_result_lookup(args.results_file)
    grouped_rows = group_blank_rows(rows)

    total_rows = len(rows)
    blank_rows = sum(1 for row in rows if not str(row.get("relevant", "")).strip())
    historical_labels = total_rows - blank_rows
    unique_manual = len(grouped_rows)
    duplicates_eliminated = blank_rows - unique_manual

    write_review(args.review_file, grouped_rows, result_lookup)
    write_mapping(args.mapping_file, args.judgments_file, grouped_rows)

    print(f"Total Test 6 judgment rows: {total_rows}")
    print(f"Historical labels already available: {historical_labels}")
    print(f"Blank rows before deduplication: {blank_rows}")
    print(f"Unique manual judgments required: {unique_manual}")
    print(f"Duplicates eliminated: {duplicates_eliminated}")
    print(f"Review output: {args.review_file}")
    print(f"Mapping output: {args.mapping_file}")


if __name__ == "__main__":
    main()
