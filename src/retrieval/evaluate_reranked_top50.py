"""Evaluate FAISS Top-50 plus CrossEncoder reranking as Retrieval Test 3."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from .evaluate import DEFAULT_QUESTIONS_FILE, read_questions
from .rerank import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOP_K,
    load_embedding_model,
    load_index_and_metadata,
    load_reranker,
    rerank_candidates,
    retrieve_faiss_candidates,
)
from .search import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME


DEFAULT_FAISS_TOP_K = 50
DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_top50_reranked_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_top50_reranked_judgments.csv")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_top50_reranked_review.txt")
DEFAULT_TRACE_FILE = Path("data/evaluation/remaining_failed_queries_trace.txt")
PREVIEW_CHARS = 800
SEPARATOR = "=" * 80
Q09_DIRECT_CHUNK_ID = "afad_earthquake_9934dddf1886db4d4037"
Q14_ID = "Q14"
Q09_ID = "Q09"

CSV_FIELDS = [
    "question_id",
    "question",
    "reranked_rank",
    "original_faiss_rank",
    "chunk_id",
    "source_file",
    "page_start",
    "page_end",
    "relevant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reranked retrieval with FAISS Top-50 candidates.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def preview_text(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def result_for_json(question_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "reranked_rank": result["reranked_rank"],
        "original_faiss_rank": result["original_rank"],
        "faiss_similarity_score": result["original_similarity_score"],
        "rerank_score": result["rerank_score"],
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "text_preview": preview_text(str(result["text"])),
        "text": result["text"],
    }


def candidate_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(candidate["chunk_id"]): candidate for candidate in candidates}


def extract_q14_direct_chunk_ids(trace_file: Path) -> list[str]:
    if not trace_file.exists():
        return []
    text = trace_file.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^Q14\s*$(.*?)(?=^={20,}\s*$\nQ20\s*$)", text)
    if not match:
        return []

    chunk_ids: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"(?m)^Candidate\s+\d+\s*$", match.group(1)):
        if "category: DIRECT_GUIDANCE" not in block:
            continue
        chunk_match = re.search(r"(?m)^chunk_id:\s*(.+)$", block)
        if not chunk_match:
            continue
        chunk_id = chunk_match.group(1).strip()
        if chunk_id and chunk_id != "NO_CHUNK" and chunk_id not in seen:
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
    return chunk_ids


def run_evaluation(
    questions: list[dict[str, str]],
    index_dir: Path,
    embedding_model_name: str,
    reranker_model_name: str,
    faiss_top_k: int,
    top_k: int,
    allow_model_download: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    index, metadata = load_index_and_metadata(index_dir)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    evaluated_questions: list[dict[str, Any]] = []
    faiss_candidates_by_question: dict[str, list[dict[str, Any]]] = {}
    reranked_all_by_question: dict[str, list[dict[str, Any]]] = {}

    for question in questions:
        faiss_candidates = retrieve_faiss_candidates(
            question["question"],
            index,
            metadata,
            embedding_model,
            faiss_top_k,
        )
        reranked_all = rerank_candidates(question["question"], faiss_candidates, reranker, top_k=faiss_top_k)
        reranked_top = reranked_all[: max(top_k, 1)]
        if len(reranked_top) != top_k:
            raise RuntimeError(f"{question['id']}: expected {top_k} reranked results, got {len(reranked_top)}")

        faiss_candidates_by_question[question["id"]] = faiss_candidates
        reranked_all_by_question[question["id"]] = reranked_all
        evaluated_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "results": [result_for_json(question["id"], question["question"], result) for result in reranked_top],
            }
        )

    payload = {
        "pipeline": "multilingual-e5-small -> FAISS Top-50 -> CrossEncoder reranker -> Top-5",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(index_dir / "afad_e5_small.faiss"),
        "metadata_path": str(index_dir / "metadata.jsonl"),
        "faiss_candidates_per_question": faiss_top_k,
        "reranked_results_per_question": top_k,
        "questions": evaluated_questions,
    }
    return payload, faiss_candidates_by_question, reranked_all_by_question


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_judgments(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for question in payload["questions"]:
            for result in question["results"]:
                writer.writerow(
                    {
                        "question_id": question["id"],
                        "question": question["question"],
                        "reranked_rank": result["reranked_rank"],
                        "original_faiss_rank": result["original_faiss_rank"],
                        "chunk_id": result["chunk_id"],
                        "source_file": result["source_file"],
                        "page_start": result["page_start"],
                        "page_end": result["page_end"],
                        "relevant": "",
                    }
                )


def render_review(
    payload: dict[str, Any],
    q09_direct_status: dict[str, Any],
    q14_direct_statuses: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for question in payload["questions"]:
        lines = [
            SEPARATOR,
            f"Question ID: {question['id']}",
            f"Question: {question['question']}",
            "",
        ]
        for result in question["results"]:
            lines.extend(
                [
                    f"Reranked Rank: {result['reranked_rank']}",
                    f"Original FAISS Rank: {result['original_faiss_rank']}",
                    f"FAISS Similarity Score: {result['faiss_similarity_score']:.4f}",
                    f"Rerank Score: {result['rerank_score']:.4f}",
                    f"Source: {result['source_file']}",
                    f"Pages: {result['page_start']}-{result['page_end']}",
                    f"Chunk ID: {result['chunk_id']}",
                    f"Text Preview: {result['text_preview']}",
                    "",
                ]
            )
        blocks.append("\n".join(lines).rstrip())

    summary_lines = [
        SEPARATOR,
        "Target Diagnostics",
        "Q09 direct guidance chunk:",
        f"chunk_id: {q09_direct_status['chunk_id']}",
        f"in_top5: {str(q09_direct_status['in_top5']).lower()}",
        f"original_faiss_rank: {q09_direct_status['original_faiss_rank']}",
        f"faiss_similarity_score: {q09_direct_status['faiss_similarity_score']}",
        f"rerank_rank: {q09_direct_status['rerank_rank']}",
        f"rerank_score: {q09_direct_status['rerank_score']}",
        "",
        "Q14 direct chunk observation:",
    ]
    if not q14_direct_statuses:
        summary_lines.append("No Q14 direct chunk IDs found in prior trace report.")
    else:
        for status in q14_direct_statuses:
            summary_lines.extend(
                [
                    f"chunk_id: {status['chunk_id']}",
                    f"original_faiss_rank: {status['original_faiss_rank']}",
                    f"faiss_similarity_score: {status['faiss_similarity_score']}",
                    f"rerank_rank: {status['rerank_rank']}",
                    f"rerank_score: {status['rerank_score']}",
                    "",
                ]
            )
    blocks.append("\n".join(summary_lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def write_review(
    path: Path,
    payload: dict[str, Any],
    q09_direct_status: dict[str, Any],
    q14_direct_statuses: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(payload, q09_direct_status, q14_direct_statuses), encoding="utf-8")


def rank_status(
    chunk_id: str,
    top_results: list[dict[str, Any]],
    faiss_candidates: list[dict[str, Any]],
    reranked_all: list[dict[str, Any]],
) -> dict[str, Any]:
    faiss_lookup = candidate_lookup(faiss_candidates)
    rerank_lookup = candidate_lookup(reranked_all)
    faiss_candidate = faiss_lookup.get(chunk_id)
    rerank_candidate = rerank_lookup.get(chunk_id)
    return {
        "chunk_id": chunk_id,
        "in_top5": any(str(result.get("chunk_id")) == chunk_id for result in top_results),
        "original_faiss_rank": faiss_candidate["original_rank"] if faiss_candidate else "NOT_IN_TOP_50",
        "faiss_similarity_score": f"{float(faiss_candidate['original_similarity_score']):.6f}"
        if faiss_candidate
        else "NOT_IN_TOP_50",
        "rerank_rank": rerank_candidate["reranked_rank"] if rerank_candidate else "NOT_IN_TOP_50",
        "rerank_score": f"{float(rerank_candidate['rerank_score']):.6f}" if rerank_candidate else "NOT_IN_TOP_50",
    }


def question_results(payload: dict[str, Any], question_id: str) -> list[dict[str, Any]]:
    for question in payload["questions"]:
        if question["id"] == question_id:
            return list(question["results"])
    raise ValueError(f"Question not found in payload: {question_id}")


def validate_payload(payload: dict[str, Any], question_count: int, top_k: int) -> None:
    questions = payload.get("questions", [])
    if len(questions) != question_count:
        raise RuntimeError(f"Expected {question_count} question blocks, got {len(questions)}")
    result_count = sum(len(question.get("results", [])) for question in questions)
    if result_count != question_count * top_k:
        raise RuntimeError(f"Expected {question_count * top_k} results, got {result_count}")
    for question in questions:
        if len(question.get("results", [])) != top_k:
            raise RuntimeError(f"{question.get('id')}: expected {top_k} results, got {len(question.get('results', []))}")


def print_summary(
    question_count: int,
    faiss_top_k: int,
    top_k: int,
    q09_direct_status: dict[str, Any],
    q14_direct_statuses: list[dict[str, Any]],
    results_file: Path,
    judgments_file: Path,
    review_file: Path,
) -> None:
    print(f"Questions: {question_count}")
    print(f"FAISS candidates per question: {faiss_top_k}")
    print(f"Reranked results per question: {top_k}")
    print(f"Total review results: {question_count * top_k}")
    print("")
    print(f"Q09 direct chunk in Top-5: {'yes' if q09_direct_status['in_top5'] else 'no'}")
    print(f"Q09 direct chunk rerank rank: {q09_direct_status['rerank_rank']}")
    print("")
    print("Q14 direct chunk observation:")
    if not q14_direct_statuses:
        print("Q14 direct chunk: NOT_FOUND_IN_PRIOR_TRACE")
    else:
        for status in q14_direct_statuses:
            print(
                f"{status['chunk_id']}: original FAISS rank={status['original_faiss_rank']}, "
                f"rerank rank={status['rerank_rank']}"
            )
    print("")
    print("Output files:")
    print(results_file)
    print(judgments_file)
    print(review_file)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    questions = read_questions(args.questions_file)
    payload, faiss_candidates_by_question, reranked_all_by_question = run_evaluation(
        questions,
        args.index_dir,
        args.embedding_model,
        args.reranker_model,
        args.faiss_top_k,
        args.top_k,
        args.allow_model_download,
    )
    validate_payload(payload, len(questions), args.top_k)

    q09_direct_status = rank_status(
        Q09_DIRECT_CHUNK_ID,
        question_results(payload, Q09_ID),
        faiss_candidates_by_question[Q09_ID],
        reranked_all_by_question[Q09_ID],
    )
    q14_direct_statuses = [
        rank_status(
            chunk_id,
            question_results(payload, Q14_ID),
            faiss_candidates_by_question[Q14_ID],
            reranked_all_by_question[Q14_ID],
        )
        for chunk_id in extract_q14_direct_chunk_ids(args.trace_file)
    ]

    write_results(args.results_file, payload)
    write_judgments(args.judgments_file, payload)
    write_review(args.review_file, payload, q09_direct_status, q14_direct_statuses)
    print_summary(
        len(questions),
        args.faiss_top_k,
        args.top_k,
        q09_direct_status,
        q14_direct_statuses,
        args.results_file,
        args.judgments_file,
        args.review_file,
    )


if __name__ == "__main__":
    main()
