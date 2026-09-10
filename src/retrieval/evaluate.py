"""Evaluate the existing local retrieval system on a fixed question set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from .search import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, DEFAULT_TOP_K, embed_query, load_model, read_metadata


DEFAULT_QUESTIONS_FILE = Path("data/evaluation/retrieval_questions.json")
DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_judgments.csv")
DEFAULT_METRICS_FILE = Path("data/evaluation/retrieval_metrics.json")
PREVIEW_CHARS = 800
CSV_FIELDS = [
    "question_id",
    "question",
    "rank",
    "similarity_score",
    "chunk_id",
    "source_file",
    "page_start",
    "page_end",
    "relevant",
]


@dataclass
class Metrics:
    total_questions: int
    evaluated_results: int
    questions_with_no_relevant: int
    precision_at_5: float
    hit_rate_at_5: float
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "evaluated_results": self.evaluated_results,
            "questions_with_no_relevant": self.questions_with_no_relevant,
            "precision_at_5": self.precision_at_5,
            "hit_rate_at_5": self.hit_rate_at_5,
            "mrr": self.mrr,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AFAD local retrieval on fixed questions.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if the model is not in the local cache.",
    )
    return parser.parse_args()


def read_questions(path: Path) -> list[dict[str, str]]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError(f"Expected a list of questions: {path}")

    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for item in questions:
        if not isinstance(item, dict):
            raise ValueError(f"Expected question objects in: {path}")
        question_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        if not question_id or not question:
            raise ValueError(f"Question records need id and question fields: {item}")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)
        normalized.append({"id": question_id, "question": question})
    return normalized


def make_preview(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def retrieve_question(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    model: Any,
    question: dict[str, str],
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = embed_query(model, question["question"])
    k = min(max(top_k, 1), len(metadata))
    scores, indices = index.search(query_embedding, k)

    rows: list[dict[str, Any]] = []
    row_indices: list[int] = []
    for rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
        row_index = int(row_index)
        if row_index < 0:
            continue
        record = metadata[row_index]
        row_indices.append(row_index)
        rows.append(
            {
                "rank": rank,
                "similarity_score": float(score),
                "faiss_row": row_index,
                "chunk_id": record["chunk_id"],
                "source_file": record["source_file"],
                "page_start": record["page_start"],
                "page_end": record["page_end"],
                "text_preview": make_preview(str(record.get("text", ""))),
            }
        )

    if len(rows) != k:
        raise RuntimeError(f"{question['id']}: expected {k} results, got {len(rows)}")
    if len(set(row_indices)) != len(row_indices):
        raise RuntimeError(f"{question['id']}: expected {k} distinct FAISS rows, got duplicates")
    return rows


def run_retrieval(
    questions: list[dict[str, str]],
    index_dir: Path,
    model_name: str,
    top_k: int,
    allow_model_download: bool,
) -> dict[str, Any]:
    index_path = index_dir / "afad_e5_small.faiss"
    metadata_path = index_dir / "metadata.jsonl"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Index files not found under {index_dir}. Build them with: python -m src.retrieval.build_index"
        )

    index = faiss.read_index(str(index_path))
    metadata = read_metadata(metadata_path)
    if index.ntotal != len(metadata):
        raise ValueError(f"Index row count ({index.ntotal}) does not match metadata rows ({len(metadata)})")

    model = load_model(model_name, allow_model_download)
    evaluated_questions: list[dict[str, Any]] = []
    for question in questions:
        evaluated_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "results": retrieve_question(index, metadata, model, question, top_k),
            }
        )

    return {
        "model_name": model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "top_k": top_k,
        "questions": evaluated_questions,
    }


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def judgment_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["question_id"]), str(row["rank"]), str(row["chunk_id"]))


def read_existing_judgments(path: Path) -> dict[tuple[str, str, str], str]:
    if not path.exists():
        return {}

    judgments: dict[tuple[str, str, str], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = str(row.get("relevant", "")).strip()
            if value:
                judgments[judgment_key(row)] = value
    return judgments


def result_rows_for_csv(results_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in results_payload["questions"]:
        for result in question["results"]:
            rows.append(
                {
                    "question_id": question["id"],
                    "question": question["question"],
                    "rank": result["rank"],
                    "similarity_score": f"{result['similarity_score']:.6f}",
                    "chunk_id": result["chunk_id"],
                    "source_file": result["source_file"],
                    "page_start": result["page_start"],
                    "page_end": result["page_end"],
                    "relevant": "",
                }
            )
    return rows


def write_judgments(path: Path, results_payload: dict[str, Any]) -> list[dict[str, Any]]:
    existing = read_existing_judgments(path)
    rows = result_rows_for_csv(results_payload)
    for row in rows:
        row["relevant"] = existing.get(judgment_key(row), "")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_relevance(value: str) -> int | None:
    value = value.strip()
    if value == "1":
        return 1
    if value == "0":
        return 0
    return None


def compute_metrics(rows: list[dict[str, Any]], total_questions: int, top_k: int) -> Metrics | None:
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        label = parse_relevance(str(row.get("relevant", "")))
        if label is None:
            return None
        parsed = dict(row)
        parsed["rank"] = int(parsed["rank"])
        parsed["relevant"] = label
        parsed_rows.append(parsed)

    rows_by_question: dict[str, list[dict[str, Any]]] = {}
    for row in parsed_rows:
        rows_by_question.setdefault(str(row["question_id"]), []).append(row)

    precision_total = 0.0
    hit_total = 0.0
    reciprocal_rank_total = 0.0
    questions_with_no_relevant = 0

    for question_rows in rows_by_question.values():
        top_rows = sorted(question_rows, key=lambda row: int(row["rank"]))[:top_k]
        relevant_ranks = [int(row["rank"]) for row in top_rows if int(row["relevant"]) == 1]
        precision_total += len(relevant_ranks) / top_k
        if relevant_ranks:
            hit_total += 1.0
            reciprocal_rank_total += 1.0 / min(relevant_ranks)
        else:
            questions_with_no_relevant += 1

    return Metrics(
        total_questions=total_questions,
        evaluated_results=len(parsed_rows),
        questions_with_no_relevant=questions_with_no_relevant,
        precision_at_5=precision_total / total_questions,
        hit_rate_at_5=hit_total / total_questions,
        mrr=reciprocal_rank_total / total_questions,
    )


def write_metrics(path: Path, metrics: Metrics | None, total_questions: int, evaluated_results: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if metrics is None:
        payload: dict[str, Any] = {
            "judgments_required": True,
            "total_questions": total_questions,
            "evaluated_results": evaluated_results,
            "precision_at_5": None,
            "hit_rate_at_5": None,
            "mrr": None,
            "questions_with_no_relevant": None,
        }
    else:
        payload = {"judgments_required": False, **metrics.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(total_questions: int, retrieved_results: int, metrics: Metrics | None) -> None:
    print(f"Questions: {total_questions}")
    print(f"Retrieved results: {retrieved_results}")
    print(f"Judgments required: {'yes' if metrics is None else 'no'}")
    if metrics is None:
        print("Manual relevance labeling required in data/evaluation/retrieval_judgments.csv")
        return
    print(f"Precision@5: {metrics.precision_at_5:.4f}")
    print(f"Hit Rate@5: {metrics.hit_rate_at_5:.4f}")
    print(f"MRR: {metrics.mrr:.4f}")
    print(f"Questions with no relevant result: {metrics.questions_with_no_relevant}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    questions = read_questions(args.questions_file)
    results_payload = run_retrieval(
        questions,
        args.index_dir,
        args.model_name,
        args.top_k,
        args.allow_model_download,
    )
    write_results(args.results_file, results_payload)
    judgment_rows = write_judgments(args.judgments_file, results_payload)
    metrics = compute_metrics(judgment_rows, len(questions), args.top_k)
    write_metrics(args.metrics_file, metrics, len(questions), len(judgment_rows))
    print_summary(len(questions), len(judgment_rows), metrics)


if __name__ == "__main__":
    main()
