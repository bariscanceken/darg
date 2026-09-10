"""Print retrieval evaluation results in a human-readable review format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_results.json")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_review.txt")
SEPARATOR = "=" * 50
PREVIEW_CHARS = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show retrieval evaluation results for manual judgment.")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--no-write", action="store_true", help="Only print to terminal; do not write a review file.")
    return parser.parse_args()


def preview_text(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def render_result(question: dict[str, Any]) -> str:
    lines = [
        SEPARATOR,
        str(question["id"]),
        f"Soru: {question['question']}",
        "",
    ]

    for result in question["results"]:
        lines.extend(
            [
                f"Rank {result['rank']}",
                f"Score: {float(result['similarity_score']):.4f}",
                f"Source: {result['source_file']}",
                f"Pages: {result['page_start']}-{result['page_end']}",
                f"Text: {preview_text(str(result.get('text_preview', '')))}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def render_results(results_file: Path) -> str:
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError(f"Expected questions list in {results_file}")
    return "\n\n".join(render_result(question) for question in questions) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    output = render_results(args.results_file)
    print(output, end="")

    if not args.no_write:
        args.review_file.parent.mkdir(parents=True, exist_ok=True)
        args.review_file.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
