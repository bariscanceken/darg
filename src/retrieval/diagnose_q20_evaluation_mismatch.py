"""Diagnose Q20 reranked evaluation labels against exact chunk text."""

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


QUESTION_ID = "Q20"
QUESTION = "Deprem sonrasında resmi makamların tahliye talimatına uymalı mıyım?"

DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_RERANKED_RESULTS_FILE = Path("data/evaluation/retrieval_reranked_results.json")
DEFAULT_RERANKED_JUDGMENTS_FILE = Path("data/evaluation/retrieval_reranked_judgments.csv")
DEFAULT_TRACE_FILE = Path("data/evaluation/remaining_failed_queries_trace.txt")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/q20_evaluation_mismatch.txt")
SEPARATOR = "=" * 88

TURKISH_AND_EXTRACTION_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "I": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
        "È": "i",
        "è": "g",
        "Ë": "s",
        "ë": "s",
        "É": "i",
        "é": "i",
    }
)

CONTEXT_TERMS = ["yetkili", "talimat", "tahliye", "uyarı", "uyari"]


@dataclass(frozen=True)
class DirectTrace:
    chunk_id: str
    context_mismatch: bool
    trace_category: str
    trace_matched_sentence: str
    trace_faiss_rank: str
    trace_rerank_rank: str


@dataclass(frozen=True)
class RuleDecision:
    authority_instruction_present: bool
    evacuation_context_present: bool
    earthquake_context_present: bool
    citizen_guidance_present: bool
    direct_answer: bool
    context_mismatch: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Q20 Test 2 evaluation mismatch.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--reranked-results-file", type=Path, default=DEFAULT_RERANKED_RESULTS_FILE)
    parser.add_argument("--reranked-judgments-file", type=Path, default=DEFAULT_RERANKED_JUDGMENTS_FILE)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(TURKISH_AND_EXTRACTION_MAP).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def regex_search(text: str, pattern: str) -> bool:
    return re.search(pattern, normalize(text)) is not None


def read_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError(f"Missing chunk_id at {path}:{line_number}")
            chunks[chunk_id] = record
    return chunks


def field_value(block: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.*)$", block, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_q20_trace_section(trace_text: str) -> str:
    match = re.search(r"(?ms)^Q20\s*$([\s\S]*?)(?=^={20,}\s*$\nSummary Table)", trace_text)
    if not match:
        raise ValueError("Could not find Q20 section in remaining failed queries trace.")
    return match.group(1)


def read_q20_direct_traces(path: Path) -> list[DirectTrace]:
    section = extract_q20_trace_section(path.read_text(encoding="utf-8"))
    traces: list[DirectTrace] = []
    seen: set[str] = set()
    for block in re.split(r"(?m)^Candidate\s+\d+\s*$", section):
        if "category: DIRECT_GUIDANCE" not in block:
            continue
        chunk_id = field_value(block, "chunk_id")
        if not chunk_id or chunk_id == "NO_CHUNK" or chunk_id in seen:
            continue
        seen.add(chunk_id)
        traces.append(
            DirectTrace(
                chunk_id=chunk_id,
                context_mismatch=field_value(block, "context_mismatch").casefold() == "true",
                trace_category=field_value(block, "category"),
                trace_matched_sentence=field_value(block, "matched sentence"),
                trace_faiss_rank=field_value(block, "FAISS rank"),
                trace_rerank_rank=field_value(block, "Rerank rank"),
            )
        )
    return traces


def split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9•])", compact)
    return [part.strip() for part in parts if part.strip()]


def q20_sentence_contexts(text: str) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    contexts: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        if not contains_any(sentence, CONTEXT_TERMS):
            continue
        previous = sentences[max(0, index - 3) : index]
        following = sentences[index + 1 : index + 4]
        matched_terms = [term for term in CONTEXT_TERMS if contains_any(sentence, [term])]
        contexts.append(
            {
                "sentence_index": index,
                "matched_terms": matched_terms,
                "previous_3_sentences": previous,
                "matched_sentence": sentence,
                "next_3_sentences": following,
            }
        )
    return contexts


def evaluate_q20_direct_answer(text: str, trace_context_mismatch: bool = False) -> RuleDecision:
    normalized = normalize(text)
    authority_source_context = any(
        term in normalized
        for term in [
            "yetkili",
            "resmi makam",
            "makamlar",
            "idare",
            "idarenin",
            "devlet kurulus",
            "afad",
            "gorevli",
            "profesyonel mudahaleci",
        ]
    )
    instruction_context = any(
        term in normalized
        for term in [
            "talimat",
            "emir",
            "uyari",
            "direktif",
        ]
    )
    evacuation_context = "tahliye" in normalized or "acil durum" in normalized
    earthquake_context = any(
        term in normalized
        for term in [
            "deprem",
            "sarsinti",
            "artci",
            "yikim",
            "enkaz",
            "hasarli bina",
        ]
    )
    citizen_guidance = any(
        term in normalized
        for term in [
            "tahliye olun",
            "tahliye edin",
            "tahliye edil",
            "uyun",
            "uyunuz",
            "uymali",
            "uymalidir",
            "direktiflerine titizlikle uyun",
            "talimatlarina uy",
            "emri verilirse tahliye",
        ]
    ) or regex_search(text, r"(talimat|uyari|direktif|emir).{0,120}\b(uy|uyun|uyunuz|uymal|tahliye)")
    other_disaster_context = any(term in normalized for term in ["sel", "taskin", "tsunami", "cig", "heyelan", "yangin"])
    context_mismatch = trace_context_mismatch or (other_disaster_context and not earthquake_context)
    authority_instruction = authority_source_context and instruction_context and citizen_guidance
    direct_answer = authority_instruction and evacuation_context and citizen_guidance and earthquake_context and not context_mismatch

    return RuleDecision(
        authority_instruction_present=authority_instruction,
        evacuation_context_present=evacuation_context,
        earthquake_context_present=earthquake_context,
        citizen_guidance_present=citizen_guidance,
        direct_answer=direct_answer,
        context_mismatch=context_mismatch,
    )


def read_q20_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for question in payload.get("questions", []):
        if question.get("id") == QUESTION_ID:
            return list(question.get("results", []))
    raise ValueError(f"{QUESTION_ID} not found in {path}")


def read_q20_judgments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("question_id") == QUESTION_ID]


def judgment_by_chunk_id(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["chunk_id"]: row.get("relevant", "").strip() for row in rows}


def render_list(values: list[str]) -> str:
    if not values:
        return "  - NONE"
    return "\n".join(f"  - {value}" for value in values)


def render_contexts(contexts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if not contexts:
        return ["No sentence containing yetkili/talimat/tahliye/uyarı was found."]
    for context_number, context in enumerate(contexts, start=1):
        lines.extend(
            [
                f"Context {context_number}",
                f"sentence_index: {context['sentence_index']}",
                f"matched_terms: {', '.join(context['matched_terms'])}",
                "previous_3_sentences:",
                render_list(context["previous_3_sentences"]),
                f"matched_sentence: {context['matched_sentence']}",
                "next_3_sentences:",
                render_list(context["next_3_sentences"]),
                "",
            ]
        )
    return lines


def render_rule_decision(decision: RuleDecision) -> list[str]:
    return [
        f"authority_instruction_present: {str(decision.authority_instruction_present).lower()}",
        f"evacuation_context_present: {str(decision.evacuation_context_present).lower()}",
        f"earthquake_context_present: {str(decision.earthquake_context_present).lower()}",
        f"citizen_guidance_present: {str(decision.citizen_guidance_present).lower()}",
        f"direct_answer: {str(decision.direct_answer).lower()}",
        f"context_mismatch: {str(decision.context_mismatch).lower()}",
    ]


def classify_final(
    q20_results: list[dict[str, Any]],
    direct_chunk_ids: set[str],
    direct_decisions: dict[str, RuleDecision],
    top5_decisions: dict[str, RuleDecision],
    judgments: dict[str, str],
) -> str:
    top5_direct_answer_ids = [
        str(result.get("chunk_id"))
        for result in q20_results
        if top5_decisions.get(str(result.get("chunk_id"))) is not None
        and top5_decisions[str(result.get("chunk_id"))].direct_answer
    ]
    if any(judgments.get(chunk_id) == "0" for chunk_id in top5_direct_answer_ids):
        return "EVALUATION_LABEL_ERROR"

    top5_direct_ids = [str(result.get("chunk_id")) for result in q20_results if str(result.get("chunk_id")) in direct_chunk_ids]
    if top5_direct_ids:
        if any(direct_decisions[chunk_id].context_mismatch for chunk_id in top5_direct_ids):
            return "CONTEXT_MISMATCH"
        return "DIRECT_GUIDANCE_FALSE_POSITIVE"

    if any(decision.context_mismatch for decision in top5_decisions.values()) and any(
        decision.authority_instruction_present and decision.evacuation_context_present
        for decision in top5_decisions.values()
    ):
        return "CONTEXT_MISMATCH"

    if any(decision.context_mismatch for decision in direct_decisions.values()) and not any(
        decision.direct_answer for decision in direct_decisions.values()
    ):
        return "CONTEXT_MISMATCH"
    return "OTHER"


def render_report(
    direct_traces: list[DirectTrace],
    chunks_by_id: dict[str, dict[str, Any]],
    q20_results: list[dict[str, Any]],
    q20_judgments: list[dict[str, str]],
) -> tuple[str, str, int, str, str]:
    direct_chunk_ids = {trace.chunk_id for trace in direct_traces}
    trace_by_id = {trace.chunk_id: trace for trace in direct_traces}
    judgments = judgment_by_chunk_id(q20_judgments)
    direct_decisions: dict[str, RuleDecision] = {}
    top5_decisions: dict[str, RuleDecision] = {}

    lines = [
        SEPARATOR,
        "Q20 Evaluation Mismatch Diagnostic",
        f"Question: {QUESTION}",
        "",
        f"Q20 direct guidance chunks: {len(direct_traces)}",
        "",
    ]

    lines.extend([SEPARATOR, "DIRECT_GUIDANCE Chunks"])
    for trace in direct_traces:
        chunk = chunks_by_id.get(trace.chunk_id)
        if chunk is None:
            lines.extend(["", f"chunk_id: {trace.chunk_id}", "ERROR: chunk not found in chunk corpus"])
            continue

        text = str(chunk.get("text", ""))
        decision = evaluate_q20_direct_answer(text, trace.context_mismatch)
        direct_decisions[trace.chunk_id] = decision
        lines.extend(
            [
                "",
                f"chunk_id: {trace.chunk_id}",
                f"source_file: {chunk.get('source_file')}",
                f"page_start: {chunk.get('page_start')}",
                f"page_end: {chunk.get('page_end')}",
                f"context_mismatch: {str(trace.context_mismatch).lower()}",
                f"trace_faiss_rank: {trace.trace_faiss_rank}",
                f"trace_rerank_rank: {trace.trace_rerank_rank}",
                f"trace_matched_sentence: {trace.trace_matched_sentence}",
                "",
                "Rule-based direct answer check:",
                *render_rule_decision(decision),
                "",
                "Sentence contexts:",
                *render_contexts(q20_sentence_contexts(text)),
                "full text:",
                text.strip(),
            ]
        )

    lines.extend(["", SEPARATOR, "Q20 Test 2 Reranked Top-5 Results"])
    top5_direct_answer_chunks = 0
    rank5_chunk_id = ""
    rank5_label = ""
    rank5_in_direct_list = False

    for result in q20_results:
        chunk_id = str(result.get("chunk_id"))
        top5_decision = evaluate_q20_direct_answer(str(result.get("text", "")), False)
        top5_decisions[chunk_id] = top5_decision
        if result.get("reranked_rank") == 5:
            rank5_chunk_id = chunk_id
            rank5_label = judgments.get(chunk_id, "")
            rank5_in_direct_list = chunk_id in direct_chunk_ids

        if top5_decision.direct_answer:
            top5_direct_answer_chunks += 1

        pages = f"{result.get('page_start')}-{result.get('page_end')}"
        lines.extend(
            [
                "",
                f"reranked_rank: {result.get('reranked_rank')}",
                f"original_faiss_rank: {result.get('original_rank')}",
                f"chunk_id: {chunk_id}",
                f"source_file: {result.get('source_file')}",
                f"pages: {pages}",
                f"rerank_score: {float(result.get('rerank_score', 0.0)):.6f}",
                f"current_relevant_label: {judgments.get(chunk_id, '') or 'BLANK'}",
                f"in_DIRECT_GUIDANCE_chunk_list: {str(chunk_id in direct_chunk_ids).lower()}",
                "",
                "Rule-based direct answer check:",
                *render_rule_decision(top5_decision),
                "",
                "full text:",
                str(result.get("text", "")).strip(),
            ]
        )

    lines.extend(["", SEPARATOR, "Q20 Test 2 Judgments CSV Rows"])
    for row in q20_judgments:
        lines.append(json.dumps(row, ensure_ascii=False))

    lines.extend(["", SEPARATOR, "Rank 5 Check"])
    rank5_decision = top5_decisions.get(rank5_chunk_id)
    lines.extend(
        [
            f"Q20 rank5 chunk_id: {rank5_chunk_id}",
            f"Q20 rank5 current label: {rank5_label or 'BLANK'}",
            f"Q20 rank5 in DIRECT_GUIDANCE chunk list: {str(rank5_in_direct_list).lower()}",
        ]
    )
    if rank5_decision is not None:
        lines.extend(render_rule_decision(rank5_decision))
    else:
        lines.append("rank5_rule_based_direct_answer_check: NOT_DIRECT_GUIDANCE_TRACE_CANDIDATE")

    diagnosis = classify_final(q20_results, direct_chunk_ids, direct_decisions, top5_decisions, judgments)
    lines.extend(
        [
            "",
            SEPARATOR,
            "Summary",
            f"Q20 direct guidance chunks: {len(direct_traces)}",
            f"Q20 top5 direct-answer chunks: {top5_direct_answer_chunks}",
            f"Q20 rank5 chunk: {rank5_chunk_id}",
            f"Q20 rank5 current label: {rank5_label or 'BLANK'}",
            f"Diagnosis: {diagnosis}",
            "",
            "Classification options:",
            "A) EVALUATION_LABEL_ERROR",
            "B) DIRECT_GUIDANCE_FALSE_POSITIVE",
            "C) CONTEXT_MISMATCH",
            "D) OTHER",
        ]
    )
    return "\n".join(lines).rstrip() + "\n", diagnosis, top5_direct_answer_chunks, rank5_chunk_id, rank5_label or "BLANK"


def run_diagnostic(
    chunks_file: Path,
    reranked_results_file: Path,
    reranked_judgments_file: Path,
    trace_file: Path,
    output_file: Path,
) -> tuple[str, int, int, str, str]:
    direct_traces = read_q20_direct_traces(trace_file)
    chunks_by_id = read_chunks(chunks_file)
    q20_results = read_q20_results(reranked_results_file)
    q20_judgments = read_q20_judgments(reranked_judgments_file)
    report, diagnosis, top5_direct_answer_count, rank5_chunk_id, rank5_label = render_report(
        direct_traces,
        chunks_by_id,
        q20_results,
        q20_judgments,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    return diagnosis, len(direct_traces), top5_direct_answer_count, rank5_chunk_id, rank5_label


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    diagnosis, direct_count, top5_direct_answer_count, rank5_chunk_id, rank5_label = run_diagnostic(
        args.chunks_file,
        args.reranked_results_file,
        args.reranked_judgments_file,
        args.trace_file,
        args.output_file,
    )

    print(f"Q20 direct guidance chunks: {direct_count}")
    print(f"Q20 top5 direct-answer chunks: {top5_direct_answer_count}")
    print(f"Q20 rank5 chunk: {rank5_chunk_id}")
    print(f"Q20 rank5 current label: {rank5_label}")
    print(f"Diagnosis: {diagnosis}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
