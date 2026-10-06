"""Evaluate local normalization + LLM spelling repair on query_typo_eval_v2 (independent set).

Groups: should_fix (expected exact spelling), should_keep (must stay unchanged),
should_not_guess (garbled tokens must survive), observe (reported, unlabeled).
Final Test v1 files are never read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.rag.generate_answer import DEFAULT_OPENAI_MODEL, load_environment, load_openai_client
from src.rag.llm_query_repair import repair_query
from src.rag.query_normalization import TOKEN_RE, normalize_query, turkish_lower

from .evaluate_gate_robustness import assert_not_final_test, read_jsonl


DEFAULT_INPUT_FILE = Path("data/evaluation/query_typo_eval_v2.jsonl")
DEFAULT_RESULTS_FILE = Path("data/evaluation/query_repair_v2_results.jsonl")
DEFAULT_SUMMARY_FILE = Path("data/evaluation/query_repair_v2_summary.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM spelling repair on query_typo_eval_v2.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--openai-model", default=None)
    return parser.parse_args()


def words(text: str) -> list[str]:
    return [turkish_lower(token) for token in TOKEN_RE.findall(text)]


def outcome(row: dict[str, Any], final: str) -> str:
    group = row["group"]
    if group == "should_fix":
        if words(final) == words(row["expected_query"]):
            return "exact"
        return "unchanged" if final == row["query"] else "different"
    if group == "should_keep":
        return "kept" if final == row["query"] else "changed"
    if group == "should_not_guess":
        kept = all(turkish_lower(token) in words(final) for token in row["must_keep_tokens"])
        return "kept_unknown" if kept else "guessed"
    return "observed"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    assert_not_final_test(args.input_file, DEFAULT_RESULTS_FILE, DEFAULT_SUMMARY_FILE)
    rows = read_jsonl(args.input_file)

    load_environment()
    client = load_openai_client()
    model = args.openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    results = []
    for row in rows:
        local = normalize_query(row["query"])["normalized"]
        repair = repair_query(local, client, model)
        final = repair["repaired"]
        results.append({
            **row,
            "local_only": local,
            "final": final,
            "repair_status": repair["status"],
            "repair_proposed": repair.get("proposed"),
            "local_outcome": outcome(row, local),
            "final_outcome": outcome(row, final),
        })
        print(f"{row['id']}: {results[-1]['final_outcome']} ({repair['status']})", flush=True)

    with DEFAULT_RESULTS_FILE.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    def count(group: str, key: str, value: str) -> str:
        items = [r for r in results if r["group"] == group]
        return f"{sum(r[key] == value for r in items)}/{len(items)}"

    lines = [
        "DARG Query Repair v2 Evaluation (independent set, run once)",
        "=" * 60,
        f"Model: {model}",
        "",
        f"{'':38}{'local only':>12}{'local+LLM':>12}",
        f"{'should_fix exact':38}{count('should_fix', 'local_outcome', 'exact'):>12}{count('should_fix', 'final_outcome', 'exact'):>12}",
        f"{'should_keep unchanged':38}{count('should_keep', 'local_outcome', 'kept'):>12}{count('should_keep', 'final_outcome', 'kept'):>12}",
        f"{'should_not_guess unknown word kept':38}{count('should_not_guess', 'local_outcome', 'kept_unknown'):>12}{count('should_not_guess', 'final_outcome', 'kept_unknown'):>12}",
        "",
        "Details:",
    ]
    for r in results:
        lines.append(f"  [{r['final_outcome']}] {r['id']} ({r['repair_status']}): {r['query']}")
        lines.append(f"      local:  {r['local_only']}")
        lines.append(f"      final:  {r['final']}")
        if r.get("expected_query"):
            lines.append(f"      expect: {r['expected_query']}")
        if r.get("repair_proposed"):
            lines.append(f"      LLM proposed (not used): {r['repair_proposed']}")
    summary = "\n".join(lines) + "\n"
    DEFAULT_SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
