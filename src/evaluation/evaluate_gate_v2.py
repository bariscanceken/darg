"""Compare answerability gate prompt v1 and v2 on identical retrieval contexts.

Tuning sets: gate_robustness_v1 (filtered by the manual review) and rag_eval_v1.
The held-out set (gate_heldout_v2) runs only with --heldout and must be run once,
after the v2 prompt is frozen. Final Test v1 files are never read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.rag.answerability import ANSWERABLE, NOT_ANSWERABLE, check_answerability
from src.rag.generate_answer import DEFAULT_OPENAI_MODEL, load_environment, load_openai_client

from .evaluate_gate_robustness import assert_not_final_test, majority, read_jsonl, retrieve_context


ROBUSTNESS_FILE = Path("data/evaluation/gate_robustness_v1.jsonl")
REVIEW_FILE = Path("data/evaluation/gate_robustness_v1_review.jsonl")
RAG_EVAL_FILE = Path("data/evaluation/rag_eval_v1.jsonl")
HELDOUT_FILE = Path("data/evaluation/gate_heldout_v2.jsonl")
OUTPUT_DIR = Path("data/evaluation")
VERSIONS = ("v1", "v2")
DEFAULT_REPEATS = 3
TARGET_GATE_ERRORS = ("Q01-V1", "Q01-V2", "Q01-V3", "Q01-V4", "Q12-V2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare answerability gate prompt v1 vs v2.")
    parser.add_argument("--heldout", action="store_true", help="Run the held-out set instead of the tuning sets.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def load_tuning_sets() -> dict[str, list[dict[str, Any]]]:
    excluded = {row["id"] for row in read_jsonl(REVIEW_FILE) if not row["include_in_gate_v2_eval"]}
    robustness = [row for row in read_jsonl(ROBUSTNESS_FILE) if row["id"] not in excluded]
    rag_eval = read_jsonl(RAG_EVAL_FILE)
    return {"robustness_v1_reviewed": robustness, "rag_eval_v1": rag_eval}


def run_set(rows: list[dict[str, Any]], client: Any, model: str, repeats: int, allow_model_download: bool) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows, start=1):
        retrieved = retrieve_context(row["question"], allow_model_download)
        labels = {
            version: [
                check_answerability(client, model, retrieved["retrieval_query"], retrieved["context"], version)["label"]
                for _ in range(repeats)
            ]
            for version in VERSIONS
        }
        result = {
            "id": row["id"],
            "question": row["question"],
            "expected_answerability": row["expected_answerability"],
            "chunk_ids": retrieved["chunk_ids"],
            **{f"{version}_labels": labels[version] for version in VERSIONS},
            **{f"{version}_majority": majority(labels[version]) for version in VERSIONS},
        }
        results.append(result)
        print(f"[{index}/{len(rows)}] {row['id']}: v1={result['v1_majority'][:1]} v2={result['v2_majority'][:1]}", flush=True)
    return results


def metrics(results: list[dict[str, Any]], version: str) -> dict[str, Any]:
    key = f"{version}_majority"
    answerable = [r for r in results if r["expected_answerability"] == ANSWERABLE]
    negative = [r for r in results if r["expected_answerability"] == NOT_ANSWERABLE]
    return {
        "accuracy": sum(r[key] == r["expected_answerability"] for r in results),
        "total": len(results),
        "answerable_accepted": sum(r[key] == ANSWERABLE for r in answerable),
        "answerable_total": len(answerable),
        "false_accepts": [r["id"] for r in negative if r[key] == ANSWERABLE],
        "negative_total": len(negative),
        "unstable": sum(len(set(r[f"{version}_labels"])) > 1 for r in results),
    }


def render(set_results: dict[str, list[dict[str, Any]]], model: str, repeats: int, heldout: bool) -> str:
    title = "Held-out evaluation (run once, prompt frozen)" if heldout else "Tuning evaluation"
    lines = [f"DARG Answerability Gate v1 vs v2 — {title}", "=" * 60, f"Model: {model} | repeats: {repeats}", ""]
    for name, results in set_results.items():
        lines.append(f"## {name} ({len(results)} questions)")
        lines.append(f"{'':28}{'v1':>12}{'v2':>12}")
        m = {version: metrics(results, version) for version in VERSIONS}
        rows = [
            ("Accuracy", lambda x: f"{x['accuracy']}/{x['total']}"),
            ("ANSWERABLE accepted", lambda x: f"{x['answerable_accepted']}/{x['answerable_total']}"),
            ("False accepts", lambda x: f"{len(x['false_accepts'])}/{x['negative_total']}"),
            ("Unstable repeats", lambda x: f"{x['unstable']}/{x['total']}"),
        ]
        for label, fmt in rows:
            lines.append(f"{label:28}{fmt(m['v1']):>12}{fmt(m['v2']):>12}")
        for version in VERSIONS:
            if m[version]["false_accepts"]:
                lines.append(f"  {version} false accepts: {', '.join(m[version]['false_accepts'])}")
        changed = [r for r in results if r["v1_majority"] != r["v2_majority"]]
        lines.append(f"Changed decisions v1 -> v2: {len(changed)}")
        for r in changed:
            ok = "OK " if r["v2_majority"] == r["expected_answerability"] else "BAD"
            lines.append(f"  [{ok}] {r['id']}: {r['v1_majority'][:1]} -> {r['v2_majority'][:1]} "
                         f"(expected {r['expected_answerability'][:1]}) {r['question']}")
        targets = [r for r in results if r["id"] in TARGET_GATE_ERRORS]
        if targets:
            fixed = sum(r["v2_majority"] == ANSWERABLE for r in targets)
            lines.append(f"Target gate errors fixed by v2: {fixed}/{len(targets)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    prefix = "gate_v2_heldout" if args.heldout else "gate_v2_tuning"
    results_file = OUTPUT_DIR / f"{prefix}_results.jsonl"
    summary_file = OUTPUT_DIR / f"{prefix}_summary.txt"
    assert_not_final_test(results_file, summary_file, HELDOUT_FILE)

    sets = {"gate_heldout_v2": read_jsonl(HELDOUT_FILE)} if args.heldout else load_tuning_sets()

    load_environment()
    client = load_openai_client()
    model = args.openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    set_results = {name: run_set(rows, client, model, args.repeats, args.allow_model_download) for name, rows in sets.items()}
    with results_file.open("w", encoding="utf-8") as handle:
        for name, results in set_results.items():
            for result in results:
                handle.write(json.dumps({"set": name, **result}, ensure_ascii=False) + "\n")

    summary = render(set_results, model, args.repeats, args.heldout)
    summary_file.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Results: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
