"""Evaluate query typo normalization: clean-query identity and an independent typo set.

Calibration sets (rag_eval, dev, holdout) check that clean queries are unchanged.
query_typo_eval_v1.jsonl is an independent evaluation set, never used for tuning.
Final Test v1 files are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.rag.query_normalization import TOKEN_RE, load_query_vocab, normalize_query


RAG_EVAL_FILE = Path("data/evaluation/rag_eval_v1.jsonl")
DEV_QUESTIONS_FILE = Path("data/evaluation/retrieval_questions.json")
DEFAULT_TYPO_FILE = Path("data/evaluation/query_typo_eval_v1.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/query_normalization_v1_summary.txt")
SPEC_EXAMPLES = [
    ("Deprem sırasında asaönser kullanmalı mıyım?", "Deprem sırasında asansör kullanmalı mıyım?"),
    ("Deprem sırasında asansor kullanmalı mıyım?", "Deprem sırasında asansör kullanmalı mıyım?"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DARG query typo normalization.")
    parser.add_argument("--typo-file", type=Path, default=DEFAULT_TYPO_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--retrieval-check", action="store_true", help="Compare frozen Top-5 retrieval for clean/typo/normalized queries.")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def assert_not_final_test(*paths: Path) -> None:
    for path in paths:
        if "final_test" in path.as_posix().lower():
            raise SystemExit(f"Final Test dosyaları bu değerlendirmede kullanılamaz: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_calibration_questions() -> list[tuple[str, str]]:
    from src.retrieval.evaluate_holdout import HOLDOUT_QUESTIONS

    questions = [("rag_eval_v1", str(row["question"])) for row in read_jsonl(RAG_EVAL_FILE)]
    questions += [("dev", str(row["question"])) for row in json.loads(DEV_QUESTIONS_FILE.read_text(encoding="utf-8"))]
    questions += [("holdout", str(row["question"])) for row in HOLDOUT_QUESTIONS]
    return questions


def evaluate_calibration(questions: list[tuple[str, str]]) -> dict[str, Any]:
    changed = []
    for source, question in questions:
        result = normalize_query(question)
        if result["normalized"] != question:
            changed.append({"source": source, "question": question, "normalized": result["normalized"]})
    return {"total": len(questions), "changed": changed}


def compare_tokens(typo: str, normalized: str, expected: str) -> dict[str, int]:
    """Token-level outcome; corrections replace whole tokens so the token lists align."""
    typo_tokens = TOKEN_RE.findall(typo)
    normalized_tokens = TOKEN_RE.findall(normalized)
    expected_tokens = TOKEN_RE.findall(expected)
    counts = {"correct": 0, "wrong": 0, "missed": 0}
    for typed, got, want in zip(typo_tokens, normalized_tokens, expected_tokens):
        if got != typed:
            counts["correct" if got == want else "wrong"] += 1
        elif typed != want:
            counts["missed"] += 1
    return counts


def evaluate_typo_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    totals = {"correct": 0, "wrong": 0, "missed": 0}
    exact = {"should_fix": 0, "should_keep": 0}
    group_sizes = {"should_fix": 0, "should_keep": 0}
    for row in rows:
        result = normalize_query(row["typo_query"])
        counts = compare_tokens(row["typo_query"], result["normalized"], row["expected_query"])
        for key in totals:
            totals[key] += counts[key]
        group_sizes[row["group"]] += 1
        exact[row["group"]] += result["normalized"] == row["expected_query"]
        details.append({**row, "normalized": result["normalized"], "skipped_reason": result["skipped_reason"], **counts})
    return {"details": details, "token_totals": totals, "exact": exact, "group_sizes": group_sizes}


def top5_ids(query: str, allow_model_download: bool) -> list[str]:
    from src.rag.generate_answer import retrieve_top_chunks

    return [str(chunk.get("chunk_id")) for chunk in retrieve_top_chunks(query, allow_model_download=allow_model_download)]


def retrieval_check(pairs: list[tuple[str, str, str]], allow_model_download: bool) -> list[dict[str, Any]]:
    """For (label, typo, expected) pairs: Top-5 overlap with the clean query's Top-5."""
    rows = []
    for label, typo, expected in pairs:
        normalized = normalize_query(typo)["normalized"]
        clean_ids = top5_ids(expected, allow_model_download)
        typo_ids = top5_ids(typo, allow_model_download)
        normalized_ids = top5_ids(normalized, allow_model_download) if normalized != typo else typo_ids
        rows.append({
            "label": label,
            "typo_overlap": len(set(clean_ids) & set(typo_ids)),
            "normalized_overlap": len(set(clean_ids) & set(normalized_ids)),
        })
    return rows


def render_report(vocab_sha: str, calibration: dict[str, Any], typo: dict[str, Any], spec: list[dict[str, str]],
                  retrieval: list[dict[str, Any]] | None) -> str:
    lines = ["DARG Query Normalization v1 Evaluation", "=" * 40, f"Vocabulary source sha256: {vocab_sha}", ""]

    lines.append("1) Calibration sets (clean queries, used for tuning)")
    lines.append(f"Changed: {len(calibration['changed'])}/{calibration['total']}")
    for item in calibration["changed"]:
        lines.append(f"  [{item['source']}] {item['question']} -> {item['normalized']}")
    lines.append("")

    lines.append("2) Spec examples (seen during design, not independent)")
    for item in spec:
        status = "OK" if item["normalized"] == item["expected"] else "FAIL"
        lines.append(f"  [{status}] {item['query']} -> {item['normalized']}")
    lines.append("")

    totals = typo["token_totals"]
    lines.append("3) Independent typo set (query_typo_eval_v1.jsonl, never used for tuning)")
    lines.append(f"should_fix exact match: {typo['exact']['should_fix']}/{typo['group_sizes']['should_fix']}")
    lines.append(f"should_keep unchanged: {typo['exact']['should_keep']}/{typo['group_sizes']['should_keep']}")
    lines.append(f"Token corrections: correct={totals['correct']} wrong={totals['wrong']} missed={totals['missed']}")
    for item in typo["details"]:
        mark = "OK" if item["normalized"] == item["expected_query"] else ("WRONG" if item["wrong"] else "MISS")
        lines.append(f"  [{mark}] {item['id']}: {item['typo_query']}")
        if item["normalized"] != item["typo_query"] or mark != "OK":
            lines.append(f"         got:      {item['normalized']}")
            lines.append(f"         expected: {item['expected_query']}")
        if item["skipped_reason"]:
            lines.append(f"         skipped:  {item['skipped_reason']}")
    lines.append("")

    if retrieval is not None:
        lines.append("4) Retrieval check: Top-5 overlap with the clean query's Top-5 (frozen pipeline)")
        lines.append("   label | typo query | normalized query")
        for row in retrieval:
            lines.append(f"   {row['label']:5} | {row['typo_overlap']}/5 | {row['normalized_overlap']}/5")
        lines.append(f"   Mean typo overlap: {sum(r['typo_overlap'] for r in retrieval) / len(retrieval):.2f}/5")
        lines.append(f"   Mean normalized overlap: {sum(r['normalized_overlap'] for r in retrieval) / len(retrieval):.2f}/5")
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    assert_not_final_test(args.typo_file, args.output_file)

    vocab = load_query_vocab()
    calibration = evaluate_calibration(load_calibration_questions())
    spec = [
        {"query": query, "expected": expected, "normalized": normalize_query(query)["normalized"]}
        for query, expected in SPEC_EXAMPLES
    ]
    typo_rows = read_jsonl(args.typo_file)
    typo = evaluate_typo_set(typo_rows)

    retrieval = None
    if args.retrieval_check:
        pairs = [(f"S{index}", query, expected) for index, (query, expected) in enumerate(SPEC_EXAMPLES, start=1)]
        pairs += [(row["id"], row["typo_query"], row["expected_query"]) for row in typo_rows if row["group"] == "should_fix"]
        retrieval = retrieval_check(pairs, args.allow_model_download)

    report = render_report(vocab.source_sha256, calibration, typo, spec, retrieval)
    args.output_file.write_text(report, encoding="utf-8")
    print(report)
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
