"""Evaluate Retrieval Test 4 using the v2 FAISS index and metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import faiss

from .evaluate import DEFAULT_QUESTIONS_FILE, read_questions
from .rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_K, load_embedding_model, load_reranker, rerank_candidates
from .search import DEFAULT_MODEL_NAME, embed_query, read_metadata


DEFAULT_INDEX_DIR = Path("data/indexes/mobile_v2")
DEFAULT_INDEX_FILE = "afad_e5_small_v2.faiss"
DEFAULT_METADATA_FILE = "metadata_v2.jsonl"
DEFAULT_FAISS_TOP_K = 50
DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_v2_top50_reranked_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_v2_top50_reranked_review.txt")
SEPARATOR = "=" * 80

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
    parser = argparse.ArgumentParser(description="Evaluate Test 4 retrieval over v2 chunks and v2 FAISS index.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
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


def load_v2_index_and_metadata(index_dir: Path, index_file: str, metadata_file: str) -> tuple[faiss.Index, list[dict[str, Any]], Path, Path]:
    index_path = index_dir / index_file
    metadata_path = index_dir / metadata_file
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"V2 index files not found: {index_path}, {metadata_path}")

    index = faiss.read_index(str(index_path))
    metadata = read_metadata(metadata_path)
    if index.ntotal != len(metadata):
        raise ValueError(f"V2 index row count ({index.ntotal}) does not match metadata rows ({len(metadata)})")
    return index, metadata, index_path, metadata_path


def retrieve_v2_candidates(
    query: str,
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    faiss_top_k: int,
) -> list[dict[str, Any]]:
    k = min(max(faiss_top_k, 1), len(metadata))
    query_embedding = embed_query(embedding_model, query)
    scores, indices = index.search(query_embedding, k)

    candidates: list[dict[str, Any]] = []
    row_indices: list[int] = []
    for original_rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
        row_index = int(row_index)
        if row_index < 0:
            continue
        record = metadata[row_index]
        row_indices.append(row_index)
        candidates.append(
            {
                "original_rank": original_rank,
                "original_similarity_score": float(score),
                "faiss_row": row_index,
                "chunk_id": record["chunk_id"],
                "source_file": record["source_file"],
                "relative_path": record.get("relative_path", ""),
                "organization": record.get("organization", "AFAD"),
                "page_start": record["page_start"],
                "page_end": record["page_end"],
                "chunk_index": record.get("chunk_index"),
                "source_relevances": record.get("source_relevances", []),
                "text": str(record.get("text", "")),
            }
        )

    if len(candidates) != k:
        raise RuntimeError(f"Expected {k} FAISS candidates, got {len(candidates)}")
    if len(set(row_indices)) != len(row_indices):
        raise RuntimeError(f"Expected distinct FAISS rows, got duplicates in Top-{k}")
    return candidates


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
        "text": result["text"],
    }


def run_evaluation(
    questions: list[dict[str, str]],
    index_dir: Path,
    index_file: str,
    metadata_file: str,
    embedding_model_name: str,
    reranker_model_name: str,
    faiss_top_k: int,
    top_k: int,
    allow_model_download: bool,
) -> dict[str, Any]:
    index, metadata, index_path, metadata_path = load_v2_index_and_metadata(index_dir, index_file, metadata_file)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    evaluated_questions: list[dict[str, Any]] = []
    for question in questions:
        candidates = retrieve_v2_candidates(question["question"], index, metadata, embedding_model, faiss_top_k)
        reranked_all = rerank_candidates(question["question"], candidates, reranker, top_k=faiss_top_k)
        reranked_top = reranked_all[: max(top_k, 1)]
        if len(reranked_top) != top_k:
            raise RuntimeError(f"{question['id']}: expected {top_k} reranked results, got {len(reranked_top)}")
        evaluated_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "results": [result_for_json(question["id"], question["question"], result) for result in reranked_top],
            }
        )

    return {
        "pipeline": "multilingual-e5-small -> V2 FAISS Top-50 -> CrossEncoder reranker -> Top-5",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "faiss_candidates_per_question": faiss_top_k,
        "reranked_results_per_question": top_k,
        "questions": evaluated_questions,
    }


def validate_payload(payload: dict[str, Any], question_count: int, top_k: int) -> None:
    questions = payload.get("questions", [])
    if len(questions) != question_count:
        raise RuntimeError(f"Expected {question_count} question blocks, got {len(questions)}")
    total_results = sum(len(question.get("results", [])) for question in questions)
    if total_results != question_count * top_k:
        raise RuntimeError(f"Expected {question_count * top_k} total results, got {total_results}")
    for question in questions:
        if len(question.get("results", [])) != top_k:
            raise RuntimeError(f"{question.get('id')}: expected {top_k} results, got {len(question.get('results', []))}")


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


def render_review(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for question in payload["questions"]:
        lines = [
            SEPARATOR,
            f"question_id: {question['id']}",
            f"question: {question['question']}",
            "",
        ]
        for result in question["results"]:
            lines.extend(
                [
                    f"final rank: {result['reranked_rank']}",
                    f"reranker score: {float(result['rerank_score']):.6f}",
                    f"original FAISS rank: {result['original_faiss_rank']}",
                    f"FAISS similarity: {float(result['faiss_similarity_score']):.6f}",
                    f"chunk_id: {result['chunk_id']}",
                    f"source_file: {result['source_file']}",
                    f"page_start/page_end: {result['page_start']}-{result['page_end']}",
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


def print_summary(payload: dict[str, Any], results_file: Path, judgments_file: Path, review_file: Path) -> None:
    question_count = len(payload["questions"])
    top_k = int(payload["reranked_results_per_question"])
    total_results = sum(len(question["results"]) for question in payload["questions"])
    print(f"Questions: {question_count}")
    print(f"FAISS candidates per question: {payload['faiss_candidates_per_question']}")
    print(f"Reranked results per question: {top_k}")
    print(f"Total review results: {total_results}")
    print("")
    print(f"Used FAISS index: {payload['index_path']}")
    print(f"Used metadata: {payload['metadata_path']}")
    print(f"Every question has exactly 5 results: {str(all(len(q['results']) == top_k for q in payload['questions'])).lower()}")
    print(f"Total results == 100: {str(total_results == 100).lower()}")
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
    payload = run_evaluation(
        questions,
        args.index_dir,
        args.index_file,
        args.metadata_file,
        args.embedding_model,
        args.reranker_model,
        args.faiss_top_k,
        args.top_k,
        args.allow_model_download,
    )
    validate_payload(payload, len(questions), args.top_k)
    write_results(args.results_file, payload)
    write_judgments(args.judgments_file, payload)
    write_review(args.review_file, payload)
    print_summary(payload, args.results_file, args.judgments_file, args.review_file)


if __name__ == "__main__":
    main()
