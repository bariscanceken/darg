"""Final Test v2: run the user's independent questions through the CLI system ONCE.

The questions in data/evaluation/final_test_v2.jsonl are written by the user and are not read by
Claude before this run (project rule). Each line: {"id": ..., "soru": ..., "beklenen": ...}.
The run refuses to start if results already exist, so the test cannot be repeated or tuned against.
Answers are written for grading; grading happens after the run and is recorded in the report.

  python -m src.evaluation.run_final_test_v2
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.rag.generate_answer import generate_answer_v3

QUESTIONS = Path("data/evaluation/final_test_v2.jsonl")
RESULTS = Path("data/evaluation/final_test_v2_results.jsonl")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS} already exists: Final Test v2 runs only once.")
    if not QUESTIONS.exists():
        raise SystemExit(f"{QUESTIONS} not found: the user writes the questions first.")
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    with RESULTS.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            question = row.get("soru") or row.get("question") or ""
            start = time.perf_counter()
            result = generate_answer_v3(question)
            seconds = round(time.perf_counter() - start, 1)
            handle.write(json.dumps({
                **row,
                "answer": result["answer"],
                "seconds": seconds,
                "used_web": result.get("used_web", False),
                "sentences": result.get("sentences", []),
                "sources": [f"{s.get('source_file')} {s.get('page_label') or ''}".strip() for s in result["sources"]],
            }, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {seconds}s", flush=True)  # questions are not printed
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
