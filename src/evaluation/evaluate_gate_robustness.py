"""Measure Answerability Gate robustness to meaning-preserving question paraphrases.

Read-only with respect to the RAG pipeline: retrieval, the gate prompt and the
generation prompt are used as-is. Final Test v1 files are never read.

For every paraphrase variant the gate is also run on two cross combinations so
a failure can be attributed to phrasing or to the retrieved context:
  variant question + base context   -> phrasing effect
  base question + variant context   -> context effect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.rag.answerability import ANSWERABLE, NOT_ANSWERABLE, check_answerability
from src.rag.context import format_context
from src.rag.generate_answer import DEFAULT_OPENAI_MODEL, load_environment, load_openai_client, retrieve_top_chunks
from src.rag.query_normalization import normalize_query


DEFAULT_INPUT_FILE = Path("data/evaluation/gate_robustness_v1.jsonl")
DEFAULT_RESULTS_FILE = Path("data/evaluation/gate_robustness_v1_results.jsonl")
DEFAULT_SUMMARY_FILE = Path("data/evaluation/gate_robustness_v1_summary.txt")
DEFAULT_REPEATS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Answerability Gate robustness to paraphrases.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def assert_not_final_test(*paths: Path) -> None:
    for path in paths:
        if "final_test" in path.as_posix().lower():
            raise SystemExit(f"Final Test dosyaları bu değerlendirmede kullanılamaz: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def majority(labels: list[str]) -> str:
    counts = Counter(labels)
    return ANSWERABLE if counts[ANSWERABLE] > counts[NOT_ANSWERABLE] else NOT_ANSWERABLE


def retrieve_context(question: str, allow_model_download: bool) -> dict[str, Any]:
    """Mirror the CLI path: normalized query for retrieval only."""
    retrieval_query = normalize_query(question)["normalized"]
    chunks = retrieve_top_chunks(retrieval_query, allow_model_download=allow_model_download)
    return {
        "retrieval_query": retrieval_query,
        "chunk_ids": [str(chunk.get("chunk_id")) for chunk in chunks],
        "context": format_context(chunks)["context"],
    }


def gate_label(client: Any, model: str, question: str, context: str) -> str:
    return check_answerability(client, model, question, context)["label"]


def evaluate(rows: list[dict[str, Any]], client: Any, model: str, repeats: int, allow_model_download: bool,
             results_file: Path) -> list[dict[str, Any]]:
    bases = {row["base_id"]: row for row in rows if row["variant"] == "base"}
    retrieved = {row["id"]: retrieve_context(row["question"], allow_model_download) for row in rows}

    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text("", encoding="utf-8")
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        own = retrieved[row["id"]]
        base = bases[row["base_id"]]
        base_ctx = retrieved[base["id"]]

        own_labels = [gate_label(client, model, row["question"], own["context"]) for _ in range(repeats)]
        result = {
            **row,
            "retrieval_query": own["retrieval_query"],
            "chunk_ids": own["chunk_ids"],
            "own_labels": own_labels,
            "own_majority": majority(own_labels),
        }
        if row["variant"] != "base":
            result["top5_overlap_with_base"] = len(set(own["chunk_ids"]) & set(base_ctx["chunk_ids"]))
            result["base_top1_in_top5"] = base_ctx["chunk_ids"][0] in own["chunk_ids"]
            result["same_set_different_order"] = (
                set(own["chunk_ids"]) == set(base_ctx["chunk_ids"]) and own["chunk_ids"] != base_ctx["chunk_ids"]
            )
            result["variant_question_base_context"] = gate_label(client, model, row["question"], base_ctx["context"])
            result["base_question_variant_context"] = gate_label(client, model, base["question"], own["context"])

        with results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
        print(f"[{index}/{len(rows)}] {row['id']}: {result['own_majority']} {own_labels}", flush=True)
    return results


def attribute_failure(result: dict[str, Any]) -> str:
    """Why did a variant get a different label than expected?"""
    expected = result["expected_answerability"]
    phrasing_ok = result["variant_question_base_context"] == expected
    context_ok = result["base_question_variant_context"] == expected
    if phrasing_ok and not context_ok:
        return "context"
    if context_ok and not phrasing_ok:
        return "phrasing"
    if not phrasing_ok and not context_ok:
        return "both"
    return "interaction"


def render_summary(results: list[dict[str, Any]], model: str, repeats: int) -> str:
    bases = [r for r in results if r["variant"] == "base"]
    variants = [r for r in results if r["variant"] != "base"]
    base_by_id = {r["base_id"]: r for r in bases}

    def accuracy(items: list[dict[str, Any]]) -> str:
        correct = sum(r["own_majority"] == r["expected_answerability"] for r in items)
        return f"{correct}/{len(items)} ({correct / len(items):.0%})" if items else "0/0"

    answerable_variants = [r for r in variants if r["expected_answerability"] == ANSWERABLE]
    control_variants = [r for r in variants if r["expected_answerability"] == NOT_ANSWERABLE]
    flips = [r for r in variants if r["own_majority"] != base_by_id[r["base_id"]]["own_majority"]]
    unstable = [r for r in results if len(set(r["own_labels"])) > 1]
    failures = [r for r in answerable_variants if r["own_majority"] != ANSWERABLE]
    attribution = Counter(attribute_failure(r) for r in failures)
    false_accepts = [r for r in results if r["expected_answerability"] == NOT_ANSWERABLE and r["own_majority"] == ANSWERABLE]

    lines = [
        "DARG Answerability Gate Robustness v1",
        "=" * 40,
        f"Model: {model} | repeats per question: {repeats}",
        f"Questions: {len(bases)} base + {len(variants)} paraphrase variants (dev-derived; Final Test not used)",
        "",
        "1) Accuracy (majority label vs expected)",
        f"   Base questions:                 {accuracy(bases)}",
        f"   ANSWERABLE paraphrase variants: {accuracy(answerable_variants)}",
        f"   NOT_ANSWERABLE control variants:{accuracy(control_variants)}",
        f"   False accepts (N expected, A predicted): {len(false_accepts)}",
        "",
        "2) Stability",
        f"   Variants whose label differs from their base question: {len(flips)}/{len(variants)}",
        f"   Questions with disagreeing repeats (same input): {len(unstable)}/{len(results)}",
        "",
        "3) Attribution of ANSWERABLE-variant failures",
        f"   Failures: {len(failures)}",
        f"   context (variant question passes with base context):  {attribution['context']}",
        f"   phrasing (base question passes with variant context): {attribution['phrasing']}",
        f"   both fail:                                             {attribution['both']}",
        f"   interaction (each passes when swapped):                {attribution['interaction']}",
        "",
        "4) Per base question",
    ]
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_base[r["base_id"]].append(r)
    for base_id, items in by_base.items():
        lines.append(f"   {base_id} (expected {items[0]['expected_answerability']})")
        for r in items:
            mark = "OK " if r["own_majority"] == r["expected_answerability"] else "XX "
            extra = ""
            if r["variant"] != "base":
                extra = (f" | overlap {r['top5_overlap_with_base']}/5"
                         f" | varQ+baseCtx={r['variant_question_base_context'][:1]}"
                         f" baseQ+varCtx={r['base_question_variant_context'][:1]}")
                if r["own_majority"] != r["expected_answerability"]:
                    extra += f" | cause={attribute_failure(r)}"
            labels = "".join(label[:1] for label in r["own_labels"])
            lines.append(f"     {mark}{r['variant']:4} [{labels}] {r['question']}{extra}")
    lines.append("")
    lines.append("Legend: A=ANSWERABLE, N=NOT_ANSWERABLE; [AAN] = repeated gate labels on own context.")
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    assert_not_final_test(args.input_file, args.results_file, args.summary_file)
    rows = read_jsonl(args.input_file)

    load_environment()
    client = load_openai_client()
    model = args.openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    results = evaluate(rows, client, model, args.repeats, args.allow_model_download, args.results_file)
    summary = render_summary(results, model, args.repeats)
    args.summary_file.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Results: {args.results_file}")
    print(f"Summary: {args.summary_file}")


if __name__ == "__main__":
    main()
