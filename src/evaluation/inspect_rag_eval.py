"""Inspect local retrieval results for RAG evaluation v1 questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.rag.generate_answer import retrieve_top_chunks


DEFAULT_EVAL_FILE = Path("data/evaluation/rag_eval_v1.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/rag_eval_v1_retrieval_inspection.txt")
SEPARATOR = "=" * 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local Top-5 retrieval for RAG eval v1 Q01-Q20.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if local retrieval models are not in cache.",
    )
    return parser.parse_args()


def read_eval_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("id", "question", "category", "expected_answerability", "ground_truth_verified"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number} missing field: {field}")
            questions.append(record)
    return questions


def is_q01_to_q20(question_id: str) -> bool:
    if not question_id.startswith("Q"):
        return False
    try:
        number = int(question_id[1:])
    except ValueError:
        return False
    return 1 <= number <= 20


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


def render_inspection(eval_questions: list[dict[str, Any]], allow_model_download: bool) -> str:
    blocks: list[str] = []
    target_questions = [question for question in eval_questions if is_q01_to_q20(str(question["id"]))]

    for question in target_questions:
        chunks = retrieve_top_chunks(str(question["question"]), allow_model_download=allow_model_download)
        lines = [
            SEPARATOR,
            f"id: {question['id']}",
            f"question: {question['question']}",
            f"category: {question['category']}",
            f"expected_answerability: {question['expected_answerability']}",
            f"ground_truth_verified: {str(question['ground_truth_verified']).lower()}",
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
    questions = read_eval_questions(args.eval_file)
    target_count = sum(1 for question in questions if is_q01_to_q20(str(question["id"])))
    report = render_inspection(questions, args.allow_model_download)
    write_text(args.output_file, report)

    print(f"Evaluation questions loaded: {len(questions)}")
    print(f"Questions inspected: {target_count}")
    print("OpenAI API calls: 0")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
