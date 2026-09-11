"""Evaluate Retrieval Test 5 hybrid dense+BM25 candidate generation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from .evaluate import DEFAULT_QUESTIONS_FILE, read_questions
from .hybrid_retrieval import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_FILE,
    DEFAULT_METADATA_FILE,
    DEFAULT_RERANKER_MODEL,
    hybrid_retrieve,
    load_hybrid_resources,
)
from .search import DEFAULT_MODEL_NAME


DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_hybrid_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_hybrid_judgments.csv")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_hybrid_review.txt")
SEPARATOR = "=" * 80
CSV_FIELDS = [
    "question_id",
    "question",
    "final_rank",
    "rerank_score",
    "chunk_id",
    "source_file",
    "page_start",
    "page_end",
    "retrieved_by",
    "faiss_rank",
    "faiss_score",
    "bm25_rank",
    "bm25_score",
    "relevant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Test 5 hybrid retrieval over the v2 corpus.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def nullable_float(value: Any) -> float | None:
    return None if value is None else float(value)


def nullable_int(value: Any) -> int | None:
    return None if value is None else int(value)


def result_for_json(question_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "final_rank": result["final_rank"],
        "reranker_score": float(result["rerank_score"]),
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "text": result["text"],
        "faiss_rank": nullable_int(result.get("faiss_rank")),
        "faiss_score": nullable_float(result.get("faiss_score")),
        "bm25_rank": nullable_int(result.get("bm25_rank")),
        "bm25_score": nullable_float(result.get("bm25_score")),
        "retrieved_by": result["retrieved_by"],
    }


def run_evaluation(
    questions: list[dict[str, str]],
    index_dir: Path,
    index_file: str,
    metadata_file: str,
    embedding_model_name: str,
    reranker_model_name: str,
    dense_top_k: int,
    bm25_top_k: int,
    top_k: int,
    allow_model_download: bool,
) -> dict[str, Any]:
    resources = load_hybrid_resources(
        index_dir=index_dir,
        index_file=index_file,
        metadata_file=metadata_file,
        embedding_model_name=embedding_model_name,
        reranker_model_name=reranker_model_name,
        allow_model_download=allow_model_download,
    )

    evaluated_questions: list[dict[str, Any]] = []
    for question in questions:
        dense_candidates, bm25_candidates, candidate_union, final_results = hybrid_retrieve(
            question["question"],
            resources,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=top_k,
        )
        if len(dense_candidates) != dense_top_k:
            raise RuntimeError(f"{question['id']}: expected {dense_top_k} dense candidates, got {len(dense_candidates)}")
        if len(bm25_candidates) != bm25_top_k:
            raise RuntimeError(f"{question['id']}: expected {bm25_top_k} BM25 candidates, got {len(bm25_candidates)}")
        if len(candidate_union) != len({str(candidate["chunk_id"]) for candidate in candidate_union}):
            raise RuntimeError(f"{question['id']}: duplicate chunk_id in candidate union")
        if len(final_results) != top_k:
            raise RuntimeError(f"{question['id']}: expected {top_k} final results, got {len(final_results)}")

        evaluated_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "candidate_union_size": len(candidate_union),
                "dense_candidates": len(dense_candidates),
                "bm25_candidates": len(bm25_candidates),
                "results": [result_for_json(question["id"], question["question"], result) for result in final_results],
            }
        )

    return {
        "pipeline": "multilingual-e5-small dense Top-50 + BM25 Top-50 union -> CrossEncoder reranker -> Top-5",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(resources.index_path),
        "metadata_path": str(resources.metadata_path),
        "dense_candidates_per_question": dense_top_k,
        "bm25_candidates_per_question": bm25_top_k,
        "final_results_per_question": top_k,
        "questions": evaluated_questions,
    }


def validate_payload(payload: dict[str, Any], question_count: int, top_k: int) -> None:
    questions = payload.get("questions", [])
    if len(questions) != question_count:
        raise RuntimeError(f"Expected {question_count} question blocks, got {len(questions)}")
    total_results = sum(len(question.get("results", [])) for question in questions)
    if total_results != question_count * top_k:
        raise RuntimeError(f"Expected {question_count * top_k} final results, got {total_results}")
    for question in questions:
        if len(question.get("results", [])) != top_k:
            raise RuntimeError(f"{question.get('id')}: expected {top_k} final results")


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
                        "final_rank": result["final_rank"],
                        "rerank_score": f"{result['reranker_score']:.6f}",
                        "chunk_id": result["chunk_id"],
                        "source_file": result["source_file"],
                        "page_start": result["page_start"],
                        "page_end": result["page_end"],
                        "retrieved_by": result["retrieved_by"],
                        "faiss_rank": result["faiss_rank"] if result["faiss_rank"] is not None else "",
                        "faiss_score": f"{result['faiss_score']:.6f}" if result["faiss_score"] is not None else "",
                        "bm25_rank": result["bm25_rank"] if result["bm25_rank"] is not None else "",
                        "bm25_score": f"{result['bm25_score']:.6f}" if result["bm25_score"] is not None else "",
                        "relevant": "",
                    }
                )


def render_review(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for question in payload["questions"]:
        lines = [
            SEPARATOR,
            f"question_id: {question['id']}",
            f"question: {question['question']}",
            f"candidate union size: {question['candidate_union_size']}",
            "",
        ]
        for result in question["results"]:
            lines.extend(
                [
                    f"final rank: {result['final_rank']}",
                    f"reranker score: {float(result['reranker_score']):.6f}",
                    f"chunk_id: {result['chunk_id']}",
                    f"source_file: {result['source_file']}",
                    f"page_start/page_end: {result['page_start']}-{result['page_end']}",
                    f"retrieved_by: {result['retrieved_by']}",
                    f"FAISS rank: {result['faiss_rank'] if result['faiss_rank'] is not None else 'null'}",
                    f"FAISS similarity: {float(result['faiss_score']):.6f}" if result["faiss_score"] is not None else "FAISS similarity: null",
                    f"BM25 rank: {result['bm25_rank'] if result['bm25_rank'] is not None else 'null'}",
                    f"BM25 score: {float(result['bm25_score']):.6f}" if result["bm25_score"] is not None else "BM25 score: null",
                    "chunk text:",
                    str(result["text"]).strip(),
                    "",
                ]
            )
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def write_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(payload), encoding="utf-8")


def union_sizes(payload: dict[str, Any]) -> list[int]:
    return [int(question["candidate_union_size"]) for question in payload["questions"]]


def print_summary(payload: dict[str, Any], results_file: Path, judgments_file: Path, review_file: Path) -> None:
    sizes = union_sizes(payload)
    total_results = sum(len(question["results"]) for question in payload["questions"])
    print(f"Questions: {len(payload['questions'])}")
    print(f"Dense candidates per question: {payload['dense_candidates_per_question']}")
    print(f"BM25 candidates per question: {payload['bm25_candidates_per_question']}")
    print(f"Average unique candidate union: {statistics.mean(sizes):.2f}")
    print(f"Minimum unique candidate union: {min(sizes)}")
    print(f"Maximum unique candidate union: {max(sizes)}")
    print(f"Final results per question: {payload['final_results_per_question']}")
    print(f"Total final results: {total_results}")
    print("")
    print("Candidate union sizes:")
    for question in payload["questions"]:
        print(f"{question['id']}: {question['candidate_union_size']}")
    print("")
    print("Output files:")
    print(results_file)
    print(judgments_file)
    print(review_file)
    print("")
    print(
        "Test 5 hybrid retrieval completed. Existing Test 1/Test 2/Test 3/Test 4 results, "
        "V1/V2 chunks, FAISS indexes, embedding model and reranker were not modified."
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    questions = read_questions(args.questions_file)
    payload = run_evaluation(
        questions=questions,
        index_dir=args.index_dir,
        index_file=args.index_file,
        metadata_file=args.metadata_file,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        top_k=args.top_k,
        allow_model_download=args.allow_model_download,
    )
    validate_payload(payload, len(questions), args.top_k)
    write_results(args.results_file, payload)
    write_judgments(args.judgments_file, payload)
    write_review(args.review_file, payload)
    print_summary(payload, args.results_file, args.judgments_file, args.review_file)


if __name__ == "__main__":
    main()
