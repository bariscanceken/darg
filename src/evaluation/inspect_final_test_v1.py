"""Inspect local retrieval results for the independent Final Test v1 set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.rag.generate_answer import retrieve_top_chunks


DEFAULT_FINAL_TEST_FILE = Path("data/evaluation/final_test_v1.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/final_test_v1_retrieval_inspection.txt")
DEV_EVAL_FILE = Path("data/evaluation/rag_eval_v1.jsonl")
SEPARATOR = "=" * 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local Top-5 retrieval for Final Test v1.")
    parser.add_argument("--final-test-file", type=Path, default=DEFAULT_FINAL_TEST_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if local retrieval models are not in cache.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, path, line_number)
            rows.append(record)
    return rows


def validate_record(record: dict[str, Any], path: Path, line_number: int) -> None:
    required_fields = [
        "id",
        "question",
        "category",
        "candidate_answerability",
        "corpus_answerability",
        "top5_answerability",
        "retrieval_miss",
        "ground_truth_verified",
    ]
    for field in required_fields:
        if field not in record:
            raise ValueError(f"{path}:{line_number} missing field: {field}")


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def assert_not_exact_dev_questions(final_rows: list[dict[str, Any]], dev_file: Path = DEV_EVAL_FILE) -> None:
    if not dev_file.exists():
        return
    dev_rows: list[dict[str, Any]] = []
    with dev_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                dev_rows.append(json.loads(line))
    dev_questions = {normalize_question(str(row["question"])) for row in dev_rows if str(row.get("id", "")).startswith("Q")}
    duplicates = [
        row["id"]
        for row in final_rows
        if normalize_question(str(row["question"])) in dev_questions
    ]
    if duplicates:
        raise RuntimeError(f"Final questions duplicate development questions: {', '.join(duplicates)}")


def render_result(rank: int, chunk: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Rank: {rank}",
            f"source_file: {chunk.get('source_file')}",
            f"page_start/page_end: {chunk.get('page_start')}-{chunk.get('page_end')}",
            f"chunk_id: {chunk.get('chunk_id')}",
            "text:",
            str(chunk.get("text", "")).strip(),
            "",
        ]
    )


def render_inspection(final_rows: list[dict[str, Any]], allow_model_download: bool) -> str:
    blocks: list[str] = []
    for row in final_rows:
        chunks = retrieve_top_chunks(str(row["question"]), allow_model_download=allow_model_download)
        lines = [
            SEPARATOR,
            f"id: {row['id']}",
            f"question: {row['question']}",
            f"category: {row['category']}",
            f"candidate_answerability: {row['candidate_answerability']}",
            f"corpus_answerability: {row['corpus_answerability']}",
            f"top5_answerability: {row['top5_answerability']}",
            f"retrieval_miss: {row['retrieval_miss']}",
            f"ground_truth_verified: {str(row['ground_truth_verified']).lower()}",
            "",
            "Top-5 retrieval results:",
            "",
        ]
        for rank, chunk in enumerate(chunks, start=1):
            lines.append(render_result(rank, chunk))
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    final_rows = read_jsonl(args.final_test_file)
    assert_not_exact_dev_questions(final_rows)
    report = render_inspection(final_rows, args.allow_model_download)
    write_text(args.output_file, report)

    print(f"Final test questions loaded: {len(final_rows)}")
    print("Exact duplicate development questions: 0")
    print("OpenAI API calls: 0")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
