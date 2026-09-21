"""Run RAG Evaluation regression v2 without overwriting v1 outputs."""

from __future__ import annotations

import sys
from pathlib import Path

from .run_rag_eval import DEFAULT_EVAL_FILE, parse_args, print_summary, run_evaluation


DEFAULT_V2_RESULTS_FILE = Path("data/evaluation/rag_eval_v2_results.jsonl")
DEFAULT_V2_SUMMARY_FILE = Path("data/evaluation/rag_eval_v2_summary.txt")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    eval_file = args.eval_file or DEFAULT_EVAL_FILE
    summary = run_evaluation(
        eval_file=eval_file,
        results_file=DEFAULT_V2_RESULTS_FILE,
        summary_file=DEFAULT_V2_SUMMARY_FILE,
        allow_model_download=args.allow_model_download,
        summary_title="RAG Evaluation regression v2 Summary",
    )
    print_summary(summary, DEFAULT_V2_RESULTS_FILE, DEFAULT_V2_SUMMARY_FILE, label="RAG Evaluation regression v2")


if __name__ == "__main__":
    main()
