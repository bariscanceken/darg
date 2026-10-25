"""Independent one-shot check of a cheaper answer model (model_check_v1, 32 questions written before any run).

Runs the full v3 system as deployed (web search on), only the answer model differs. One shot: refuses to run when the
result file exists, so the questions cannot be re-run until a result looks good. Decision rule fixed before the run
(26 Oct 2026): accuracy >= 85% (correct / all, partial = not correct), wrong answers < 10%, dangerous answers 0, every
control question right.

    python -m src.evaluation.run_model_check --model gpt-4.1-mini
Costs OpenAI money (~0.10-0.40 $ depending on how often the web is searched): only with the project owner's approval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.rag.generate_answer import generate_answer_v3

QUESTIONS = Path("data/evaluation/model_check_v1.jsonl")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    output = Path(f"data/evaluation/model_check_v1_results_{args.model}.jsonl")
    if output.exists():
        raise SystemExit(f"{output} already exists: this check runs once.")
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            start = time.perf_counter()
            result = generate_answer_v3(row["soru"], answer_model=args.model)
            seconds = round(time.perf_counter() - start, 1)
            handle.write(json.dumps({**row, "model": args.model, "answer": result["answer"],
                                     "used_web": result.get("used_web", False), "seconds": seconds,
                                     "sentences": result.get("sentences", [])}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {seconds}s web={result.get('used_web')} | {row['soru']}", flush=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
