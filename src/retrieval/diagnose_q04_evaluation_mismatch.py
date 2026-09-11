"""Diagnose Q04 mismatch between direct guidance trace and reranked evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_CHUNK_ID = "afad_earthquake_628e4978a3a3a946879b"
QUESTION_ID = "Q04"
QUESTION = "Deprem sırasında asansör kullanmalı mıyım?"

DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_RERANKED_RESULTS_FILE = Path("data/evaluation/retrieval_reranked_results.json")
DEFAULT_RERANKED_JUDGMENTS_FILE = Path("data/evaluation/retrieval_reranked_judgments.csv")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/q04_evaluation_mismatch.txt")
SEPARATOR = "=" * 80

DEPREM_CONTEXT_TERMS = [
    "deprem",
    "sarsıntı",
    "sarsinti",
    "sarsÈntÈ",
]
ASANSOR_TERMS = [
    "asansör",
    "asansor",
    "asansöre",
    "asansore",
    "asansörü",
    "asansoru",
    "asansörleri",
    "asansorleri",
]
NEGATIVE_INSTRUCTION_TERMS = [
    "kullanmay",
    "binmey",
    "binilmemeli",
    "kullanılmamal",
    "kullanilmamal",
    "kullanılmaz",
    "kullanilmaz",
    "uzak",
]


@dataclass(frozen=True)
class RuleDecision:
    has_deprem_context: bool
    has_asansor_no_use_instruction: bool
    same_context: bool
    direct_answer: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Q04 evaluation mismatch for a target direct-guidance chunk.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--reranked-results-file", type=Path, default=DEFAULT_RERANKED_RESULTS_FILE)
    parser.add_argument("--reranked-judgments-file", type=Path, default=DEFAULT_RERANKED_JUDGMENTS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def contains_any(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def read_target_chunk(chunks_file: Path) -> dict[str, Any]:
    with chunks_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("chunk_id") == TARGET_CHUNK_ID:
                return record
    raise ValueError(f"Target chunk not found: {TARGET_CHUNK_ID}")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9•])", text)
    return [part.strip() for part in parts if part.strip()]


def asansor_sentence_contexts(text: str) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    contexts: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        if not contains_any(sentence, ASANSOR_TERMS):
            continue
        previous = sentences[max(0, index - 3) : index]
        following = sentences[index + 1 : index + 4]
        context_text = " ".join([*previous, sentence, *following])
        contexts.append(
            {
                "sentence_index": index,
                "previous_3_sentences": previous,
                "asansor_sentence": sentence,
                "next_3_sentences": following,
                "has_deprem_context": contains_any(context_text, DEPREM_CONTEXT_TERMS),
                "has_asansor_no_use_instruction": contains_any(sentence, ASANSOR_TERMS)
                and contains_any(context_text, NEGATIVE_INSTRUCTION_TERMS),
                "same_context": contains_any(context_text, DEPREM_CONTEXT_TERMS)
                and contains_any(sentence, ASANSOR_TERMS)
                and contains_any(context_text, NEGATIVE_INSTRUCTION_TERMS),
            }
        )
    return contexts


def evaluate_direct_answer(contexts: list[dict[str, Any]]) -> RuleDecision:
    has_deprem_context = any(bool(context["has_deprem_context"]) for context in contexts)
    has_instruction = any(bool(context["has_asansor_no_use_instruction"]) for context in contexts)
    same_context = any(bool(context["same_context"]) for context in contexts)
    return RuleDecision(
        has_deprem_context=has_deprem_context,
        has_asansor_no_use_instruction=has_instruction,
        same_context=same_context,
        direct_answer=same_context,
    )


def read_q04_reranked_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for question in payload.get("questions", []):
        if question.get("id") == QUESTION_ID:
            return list(question.get("results", []))
    raise ValueError(f"{QUESTION_ID} not found in {path}")


def read_q04_judgments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("question_id") == QUESTION_ID]


def classify_mismatch(
    direct_answer: bool,
    target_in_top5: bool,
    target_judgment: str | None,
) -> str:
    if direct_answer and target_in_top5 and target_judgment == "0":
        return "EVALUATION_LABEL_ERROR"
    if (not direct_answer) and target_in_top5:
        return "DIRECT_GUIDANCE_FALSE_POSITIVE"
    if not target_in_top5:
        return "EVALUATION_PIPELINE_MISMATCH"
    return "OTHER"


def render_list(values: list[str]) -> str:
    if not values:
        return "NONE"
    return "\n".join(f"  - {value}" for value in values)


def render_report(
    chunk: dict[str, Any],
    contexts: list[dict[str, Any]],
    rule_decision: RuleDecision,
    q04_results: list[dict[str, Any]],
    q04_judgments: list[dict[str, str]],
) -> str:
    target_result = next((result for result in q04_results if result.get("chunk_id") == TARGET_CHUNK_ID), None)
    target_judgment = next((row.get("relevant", "").strip() for row in q04_judgments if row.get("chunk_id") == TARGET_CHUNK_ID), None)
    if target_judgment == "":
        target_judgment = "BLANK"

    mismatch_class = classify_mismatch(
        direct_answer=rule_decision.direct_answer,
        target_in_top5=target_result is not None,
        target_judgment=target_judgment,
    )

    lines = [
        SEPARATOR,
        "Q04 Evaluation Mismatch Diagnostic",
        f"Question: {QUESTION}",
        f"Target chunk_id: {TARGET_CHUNK_ID}",
        "",
        "Target Chunk",
        f"source_file: {chunk.get('source_file')}",
        f"page_start: {chunk.get('page_start')}",
        f"page_end: {chunk.get('page_end')}",
        f"source_relevances: {chunk.get('source_relevances')}",
        "full text:",
        str(chunk.get("text", "")).strip(),
        "",
        SEPARATOR,
        "Asansör Sentence Contexts",
    ]

    if not contexts:
        lines.append("No asansör sentence found in target chunk.")
    else:
        for context_number, context in enumerate(contexts, start=1):
            lines.extend(
                [
                    "",
                    f"Context {context_number}",
                    f"sentence_index: {context['sentence_index']}",
                    "previous_3_sentences:",
                    render_list(context["previous_3_sentences"]),
                    f"asansör_sentence: {context['asansor_sentence']}",
                    "next_3_sentences:",
                    render_list(context["next_3_sentences"]),
                    f"has_deprem_context: {str(context['has_deprem_context']).lower()}",
                    f"has_asansor_no_use_instruction: {str(context['has_asansor_no_use_instruction']).lower()}",
                    f"same_context: {str(context['same_context']).lower()}",
                ]
            )

    lines.extend(
        [
            "",
            SEPARATOR,
            "Rule-Based Direct Answer Check",
            f"deprem/sarsıntı bağlamı var mı: {str(rule_decision.has_deprem_context).lower()}",
            f"asansör kullanmama/binmeme talimatı var mı: {str(rule_decision.has_asansor_no_use_instruction).lower()}",
            f"aynı bağlam içerisinde mi: {str(rule_decision.same_context).lower()}",
            f"direct_answer: {str(rule_decision.direct_answer).lower()}",
            "",
            SEPARATOR,
            "Test 2 Q04 Reranked Results",
        ]
    )

    for result in q04_results:
        lines.extend(
            [
                "",
                f"rerank_rank: {result.get('reranked_rank')}",
                f"chunk_id: {result.get('chunk_id')}",
                f"source_file: {result.get('source_file')}",
                "full text:",
                str(result.get("text", "")).strip(),
            ]
        )

    lines.extend(
        [
            "",
            SEPARATOR,
            "Test 2 Q04 Judgments CSV Rows",
        ]
    )

    for row in q04_judgments:
        lines.append(json.dumps(row, ensure_ascii=False))

    lines.extend(
        [
            "",
            SEPARATOR,
            "Summary",
            f"target_chunk_in_q04_reranked_top5: {str(target_result is not None).lower()}",
            f"target_chunk_reranked_rank: {target_result.get('reranked_rank') if target_result else 'NOT_FOUND'}",
            f"target_chunk_judgment_relevant: {target_judgment if target_judgment is not None else 'NOT_FOUND'}",
            f"mismatch_classification: {mismatch_class}",
            "",
            "Classification options:",
            "A) EVALUATION_LABEL_ERROR",
            "B) DIRECT_GUIDANCE_FALSE_POSITIVE",
            "C) EVALUATION_PIPELINE_MISMATCH",
            "D) OTHER",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run_diagnostic(
    chunks_file: Path,
    reranked_results_file: Path,
    reranked_judgments_file: Path,
    output_file: Path,
) -> str:
    chunk = read_target_chunk(chunks_file)
    contexts = asansor_sentence_contexts(str(chunk.get("text", "")))
    rule_decision = evaluate_direct_answer(contexts)
    q04_results = read_q04_reranked_results(reranked_results_file)
    q04_judgments = read_q04_judgments(reranked_judgments_file)
    report = render_report(chunk, contexts, rule_decision, q04_results, q04_judgments)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    report = run_diagnostic(
        args.chunks_file,
        args.reranked_results_file,
        args.reranked_judgments_file,
        args.output_file,
    )
    summary = report.split("Summary", maxsplit=1)[-1]
    for line in summary.splitlines():
        if line.startswith("target_chunk_") or line.startswith("mismatch_classification:"):
            print(line)
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
