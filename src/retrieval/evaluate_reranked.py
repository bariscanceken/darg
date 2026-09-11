"""Evaluate experimental FAISS Top-20 plus CrossEncoder reranking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .evaluate import DEFAULT_QUESTIONS_FILE, read_questions
from .rerank import (
    DEFAULT_FAISS_TOP_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANK_TOP_K,
    load_embedding_model,
    load_index_and_metadata,
    load_reranker,
    rerank_query,
)
from .search import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME


DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_reranked_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_reranked_judgments.csv")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_reranked_review.txt")
PREVIEW_CHARS = 800
SEPARATOR = "=" * 80
CSV_FIELDS = [
    "question_id",
    "question",
    "reranked_rank",
    "original_rank",
    "original_similarity_score",
    "rerank_score",
    "chunk_id",
    "source_file",
    "page_start",
    "page_end",
    "relevant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reranked retrieval on fixed questions.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
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


def result_for_json(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "reranked_rank": result["reranked_rank"],
        "original_rank": result["original_rank"],
        "original_similarity_score": result["original_similarity_score"],
        "rerank_score": result["rerank_score"],
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "text_preview": preview_text(str(result["text"])),
        "text": result["text"],
    }


def run_evaluation(
    questions: list[dict[str, str]],
    index_dir: Path,
    embedding_model_name: str,
    reranker_model_name: str,
    faiss_top_k: int,
    top_k: int,
    allow_model_download: bool,
) -> dict[str, Any]:
    index, metadata = load_index_and_metadata(index_dir)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    evaluated_questions: list[dict[str, Any]] = []
    for question in questions:
        results = rerank_query(
            question["question"],
            index,
            metadata,
            embedding_model,
            reranker,
            faiss_top_k,
            top_k,
        )
        if len(results) != top_k:
            raise RuntimeError(f"{question['id']}: expected {top_k} reranked results, got {len(results)}")
        evaluated_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "results": [result_for_json(result) for result in results],
            }
        )

    return {
        "pipeline": "multilingual-e5-small -> FAISS Top-20 -> CrossEncoder reranker -> Top-5",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(index_dir / "afad_e5_small.faiss"),
        "metadata_path": str(index_dir / "metadata.jsonl"),
        "faiss_candidates_per_question": faiss_top_k,
        "reranked_results_per_question": top_k,
        "questions": evaluated_questions,
    }


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
                        "original_rank": result["original_rank"],
                        "original_similarity_score": f"{result['original_similarity_score']:.6f}",
                        "rerank_score": f"{result['rerank_score']:.6f}",
                        "chunk_id": result["chunk_id"],
                        "source_file": result["source_file"],
                        "page_start": result["page_start"],
                        "page_end": result["page_end"],
                        "relevant": "",
                    }
                )


def render_review(payload: dict[str, Any]) -> str:
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
                    f"Original FAISS Rank: {result['original_rank']}",
                    f"Original Similarity Score: {result['original_similarity_score']:.4f}",
                    f"Rerank Score: {result['rerank_score']:.4f}",
                    f"Source: {result['source_file']}",
                    f"Pages: {result['page_start']}-{result['page_end']}",
                    f"Chunk ID: {result['chunk_id']}",
                    f"Text Preview: {result['text_preview']}",
                    "",
                ]
            )
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def write_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(payload), encoding="utf-8")


def print_summary(question_count: int, faiss_top_k: int, rerank_top_k: int) -> None:
    print(f"Questions: {question_count}")
    print(f"FAISS candidates per question: {faiss_top_k}")
    print(f"Reranked results per question: {rerank_top_k}")
    print(f"Total review results: {question_count * rerank_top_k}")
    print("Judgments required: yes")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    questions = read_questions(args.questions_file)
    payload = run_evaluation(
        questions,
        args.index_dir,
        args.embedding_model,
        args.reranker_model,
        args.faiss_top_k,
        args.top_k,
        args.allow_model_download,
    )
    write_results(args.results_file, payload)
    write_judgments(args.judgments_file, payload)
    write_review(args.review_file, payload)
    print_summary(len(questions), args.faiss_top_k, args.top_k)


if __name__ == "__main__":
    main()
