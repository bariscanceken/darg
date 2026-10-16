"""Pipeline v3 check against the form's criteria (accuracy >= 85%, speed), without the paid gpt-4o judge.

Runs a fixed random sample of the 74 answerable questions plus all 13 unanswerable controls through
pipeline v3, timing each answer, and writes the answers for manual grading (expert / user check).

  python -m src.evaluation.run_v3_check --sample 25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from src.evaluation.evaluate_faithfulness import negative_questions, question_list
from src.rag.generate_answer import generate_answer_v3

OUTPUT = Path("data/evaluation/v3_check_results.jsonl")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=25)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    answerable = question_list()
    sample = random.Random(42).sample(answerable, args.sample)
    questions = sample + negative_questions()
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (set_name, question) in enumerate(questions, start=1):
            start = time.perf_counter()
            result = generate_answer_v3(question)
            seconds = time.perf_counter() - start
            row = {"set": set_name, "question": question, "answer": result["answer"], "seconds": round(seconds, 1),
                   "sentences": result.get("sentences", []),
                   "sources": [f"{s.get('source_file')} s.{s.get('page_label')}" for s in result["sources"]]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(questions)}] {seconds:.1f}s | {set_name} | {question}", flush=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
