"""Run a small pilot question set through the CLI pipeline and save answers for manual review.

  python -m src.evaluation.run_pilot --questions data/evaluation/hatay_pilot_v1.jsonl --tag baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.rag.generate_answer import answer_for_cli, format_local_info, parse_args as parse_cli_args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args, cli_rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *cli_rest]
    cli_args = parse_cli_args()  # CLI defaults, plus any pipeline flags passed through

    items = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = args.questions.with_name(f"{args.questions.stem}_results_{args.tag}.jsonl")
    review = args.questions.with_name(f"{args.questions.stem}_review_{args.tag}.txt")
    with output.open("w", encoding="utf-8") as results, review.open("w", encoding="utf-8") as text:
        for item in items:
            result = answer_for_cli(item["question"], cli_args)
            local = format_local_info(result["local_info"]) if result.get("local_info") else ""
            record = {**item, "answer": result["answer"], "gate": result["answerability"]["label"],
                      "question_type_pred": result.get("question_type"),
                      "normalized_question": result.get("normalized_question"),
                      "sources": [f"{s.get('source_file')} s.{s.get('page')}" for s in result["sources"]],
                      "retrieved": [f"{s.get('source_file')} s.{s.get('page')}" for s in result["retrieved_sources"]],
                      "local_info": local}
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            block = (f"=== {item['id']} [{item['type']} → {record['question_type_pred']}] {item['question']}\n"
                     f"(anlaşılan: {record['normalized_question']}) gate={record['gate']}\n"
                     f"Beklenen: {item['expect']}\n--- Cevap:\n{record['answer']}\n"
                     + (f"--- Yerel:\n{local}\n" if local else "")
                     + f"--- Kaynaklar: {record['sources']}\n--- Getirilen: {record['retrieved']}\n\n")
            text.write(block)
            text.flush()
            print(f"{item['id']} done ({record['gate']})", flush=True)
    print(f"Wrote {output} and {review}")


if __name__ == "__main__":
    main()
