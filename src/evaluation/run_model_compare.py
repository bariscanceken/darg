"""Cost test: does a cheaper model write answers as well as gpt-4o?

Re-asks the questions of the v3 regression check (data/evaluation/v3_check_results_after_fixes.jsonl, answered by
gpt-4o) with another answer model, web search off. Only the questions where gpt-4o did not use the web are compared
(web search needs gpt-4o anyway). Decision rule, fixed before the run (17 Oct 2026 conversation, 25 Oct): the cheap
model may be worse on at most 1 question, with 0 dangerous answers and all control questions right.

    python -m src.evaluation.run_model_compare --model gpt-4o-mini
Costs OpenAI money (~0.05-0.10 $ per model for 33 questions): run only with the project owner's approval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.rag.generate_answer import generate_answer_v3

BASELINE = Path("data/evaluation/v3_check_results_after_fixes.jsonl")


def used_web(row: dict) -> bool:
    return "Web bilgileri" in row["answer"] or any("http" in str(s) for s in row["sources"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in BASELINE.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if not used_web(row)]
    output = Path(f"data/evaluation/model_compare_{args.model}.jsonl")
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            start = time.perf_counter()
            result = generate_answer_v3(row["question"], web_search=False, answer_model=args.model)
            seconds = round(time.perf_counter() - start, 1)
            handle.write(json.dumps({"set": row["set"], "question": row["question"], "model": args.model,
                                     "answer": result["answer"], "baseline_answer": row["answer"],
                                     "seconds": seconds, "sentences": result.get("sentences", [])},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {seconds}s | {row['set']} | {row['question']}", flush=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
