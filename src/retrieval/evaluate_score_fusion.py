"""Evaluate Retrieval Test 6 with dense + reranker score fusion over V2 chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from .evaluate import DEFAULT_QUESTIONS_FILE, read_questions
from .evaluate_reranked_v2 import (
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_FILE,
    DEFAULT_METADATA_FILE,
    load_v2_index_and_metadata,
    retrieve_v2_candidates,
)
from .rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_K, load_embedding_model, load_reranker, rerank_candidates
from .search import DEFAULT_MODEL_NAME


DEFAULT_FAISS_TOP_K = 50
DEFAULT_ALPHAS = (0.50, 0.60, 0.70, 0.80, 0.90)
DEFAULT_RESULTS_FILE = Path("data/evaluation/retrieval_score_fusion_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/retrieval_score_fusion_judgments.csv")
DEFAULT_REVIEW_FILE = Path("data/evaluation/retrieval_score_fusion_review.txt")
DEFAULT_TEST4_RESULTS_FILE = Path("data/evaluation/retrieval_v2_top50_reranked_results.json")
DEFAULT_TEST4_JUDGMENTS_FILE = Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv")
DEFAULT_TRACE_FILE = Path("data/evaluation/remaining_failed_queries_trace.txt")
PREVIEW_CHARS = 800
SEPARATOR = "=" * 80
Q14_ID = "Q14"

CSV_FIELDS = [
    "alpha",
    "question_id",
    "question",
    "final_rank",
    "final_score",
    "faiss_rank",
    "faiss_score",
    "faiss_norm",
    "reranker_score",
    "reranker_norm",
    "chunk_id",
    "source_file",
    "page_start",
    "page_end",
    "relevant",
]

PROTECTED_FILES = [
    Path("data/processed/chunks/afad_earthquake_chunks.jsonl"),
    Path("data/processed/chunks/afad_earthquake_chunks_v2.jsonl"),
    Path("data/indexes/mobile/afad_e5_small.faiss"),
    Path("data/indexes/mobile/metadata.jsonl"),
    Path("data/indexes/mobile_v2/afad_e5_small_v2.faiss"),
    Path("data/indexes/mobile_v2/metadata_v2.jsonl"),
    Path("data/evaluation/retrieval_results.json"),
    Path("data/evaluation/retrieval_judgments.csv"),
    Path("data/evaluation/retrieval_metrics.json"),
    Path("data/evaluation/retrieval_review.txt"),
    Path("data/evaluation/retrieval_reranked_results.json"),
    Path("data/evaluation/retrieval_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_reranked_metrics.json"),
    Path("data/evaluation/retrieval_reranked_review.txt"),
    Path("data/evaluation/retrieval_top50_reranked_results.json"),
    Path("data/evaluation/retrieval_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_top50_reranked_metrics.json"),
    Path("data/evaluation/retrieval_top50_reranked_review.txt"),
    Path("data/evaluation/retrieval_v2_top50_reranked_results.json"),
    Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_v2_top50_reranked_metrics.json"),
    Path("data/evaluation/retrieval_v2_top50_reranked_review.txt"),
    Path("data/evaluation/retrieval_hybrid_results.json"),
    Path("data/evaluation/retrieval_hybrid_judgments.csv"),
    Path("data/evaluation/retrieval_hybrid_metrics.json"),
    Path("data/evaluation/retrieval_hybrid_review.txt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Test 6 dense + reranker score fusion over V2 retrieval.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--test4-results-file", type=Path, default=DEFAULT_TEST4_RESULTS_FILE)
    parser.add_argument("--test4-judgments-file", type=Path, default=DEFAULT_TEST4_JUDGMENTS_FILE)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str | None]:
    return {str(path): file_hash(path) for path in paths}


def preview_text(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def load_test4_top5(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    top5: dict[str, set[str]] = {}
    for question in payload.get("questions", []):
        top5[str(question["id"])] = {str(result["chunk_id"]) for result in question.get("results", [])}
    return top5


def load_test4_labels(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    labels: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = str(row.get("relevant", "")).strip()
            if label in {"0", "1"}:
                labels[(str(row["question_id"]), str(row["chunk_id"]))] = label
    return labels


def result_for_json(question_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "final_rank": result["final_rank"],
        "final_score": float(result["final_score"]),
        "faiss_rank": result["faiss_rank"],
        "faiss_score": float(result["faiss_score"]),
        "faiss_norm": float(result["faiss_norm"]),
        "reranker_score": float(result["reranker_score"]),
        "reranker_norm": float(result["reranker_norm"]),
        "reranker_only_rank": result["reranker_only_rank"],
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "text": result["text"],
    }


def add_scores_and_norms(candidates: list[dict[str, Any]], reranked_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reranker_by_id = {str(candidate["chunk_id"]): candidate for candidate in reranked_all}
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        chunk_id = str(candidate["chunk_id"])
        reranked = reranker_by_id[chunk_id]
        updated = dict(candidate)
        updated["faiss_rank"] = int(candidate["original_rank"])
        updated["faiss_score"] = float(candidate["original_similarity_score"])
        updated["reranker_score"] = float(reranked["rerank_score"])
        updated["reranker_only_rank"] = int(reranked["reranked_rank"])
        scored.append(updated)

    faiss_norms = min_max_normalize([float(candidate["faiss_score"]) for candidate in scored])
    reranker_norms = min_max_normalize([float(candidate["reranker_score"]) for candidate in scored])
    for candidate, faiss_norm, reranker_norm in zip(scored, faiss_norms, reranker_norms):
        candidate["faiss_norm"] = faiss_norm
        candidate["reranker_norm"] = reranker_norm
    return scored


def apply_fusion(candidates: list[dict[str, Any]], alpha: float, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fused: list[dict[str, Any]] = []
    for candidate in candidates:
        updated = dict(candidate)
        updated["final_score"] = alpha * float(candidate["reranker_norm"]) + (1.0 - alpha) * float(candidate["faiss_norm"])
        fused.append(updated)

    fused.sort(
        key=lambda candidate: (
            -float(candidate["final_score"]),
            int(candidate["reranker_only_rank"]),
            int(candidate["faiss_rank"]),
            str(candidate["chunk_id"]),
        )
    )
    for rank, candidate in enumerate(fused, start=1):
        candidate["fusion_rank"] = rank

    final = [dict(candidate) for candidate in fused[: max(top_k, 1)]]
    for rank, candidate in enumerate(final, start=1):
        candidate["final_rank"] = rank
    return final, fused


def historical_relevant(question_id: str, chunk_id: str, test4_labels: dict[tuple[str, str], str]) -> str:
    return test4_labels.get((question_id, chunk_id), "")


def extract_q14_direct_sources(trace_file: Path) -> list[dict[str, Any]]:
    if not trace_file.exists():
        return []
    text = trace_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?ms)^Q14\s*$(.*?)(?=^={20,}\s*$\nQ20\s*$)", text)
    if not match:
        return []

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for block in re.split(r"(?m)^Candidate\s+\d+\s*$", match.group(1)):
        if "category: DIRECT_GUIDANCE" not in block:
            continue
        source_match = re.search(r"(?m)^source_file:\s*(.+)$", block)
        page_match = re.search(r"(?m)^page:\s*(\d+)$", block)
        chunk_match = re.search(r"(?m)^chunk_id:\s*(.+)$", block)
        if not source_match or not page_match:
            continue
        source_file = source_match.group(1).strip()
        page = int(page_match.group(1))
        prior_chunk_id = chunk_match.group(1).strip() if chunk_match else ""
        key = (source_file, page, prior_chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source_file": source_file, "page": page, "prior_chunk_id": prior_chunk_id})
    return sources


def map_sources_to_v2_chunks(sources: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        for record in metadata:
            if str(record.get("source_file")) != source["source_file"]:
                continue
            page_start = int(record.get("page_start", 0))
            page_end = int(record.get("page_end", 0))
            if page_start <= int(source["page"]) <= page_end:
                chunk_id = str(record["chunk_id"])
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                mapped.append(
                    {
                        "chunk_id": chunk_id,
                        "prior_chunk_id": source["prior_chunk_id"],
                        "source_file": record.get("source_file"),
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                )
    return mapped


def rank_lookup(candidates: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(candidate[key]): candidate for candidate in candidates}


def q14_diagnostics(
    q14_candidates: list[dict[str, Any]],
    q14_reranked_all: list[dict[str, Any]],
    fused_by_alpha: dict[float, list[dict[str, Any]]],
    direct_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    faiss_by_id = rank_lookup(q14_candidates, "chunk_id")
    rerank_by_id = rank_lookup(q14_reranked_all, "chunk_id")
    diagnostics: list[dict[str, Any]] = []
    for direct_chunk in direct_chunks:
        chunk_id = str(direct_chunk["chunk_id"])
        faiss_candidate = faiss_by_id.get(chunk_id)
        rerank_candidate = rerank_by_id.get(chunk_id)
        alpha_ranks: dict[str, Any] = {}
        for alpha, fused in fused_by_alpha.items():
            fused_candidate = rank_lookup(fused, "chunk_id").get(chunk_id)
            alpha_ranks[f"{alpha:.2f}"] = fused_candidate["fusion_rank"] if fused_candidate else "NOT_IN_TOP_50"
        diagnostics.append(
            {
                **direct_chunk,
                "faiss_rank": faiss_candidate["original_rank"] if faiss_candidate else "NOT_IN_TOP_50",
                "faiss_score": float(faiss_candidate["original_similarity_score"]) if faiss_candidate else None,
                "reranker_only_rank": rerank_candidate["reranked_rank"] if rerank_candidate else "NOT_IN_TOP_50",
                "reranker_score": float(rerank_candidate["rerank_score"]) if rerank_candidate else None,
                "fusion_rank_by_alpha": alpha_ranks,
            }
        )
    return diagnostics


def run_evaluation(
    questions: list[dict[str, str]],
    index_dir: Path,
    index_file: str,
    metadata_file: str,
    embedding_model_name: str,
    reranker_model_name: str,
    faiss_top_k: int,
    top_k: int,
    alphas: list[float],
    allow_model_download: bool,
    test4_results_file: Path,
    test4_judgments_file: Path,
    trace_file: Path,
) -> dict[str, Any]:
    index, metadata, index_path, metadata_path = load_v2_index_and_metadata(index_dir, index_file, metadata_file)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)
    test4_top5 = load_test4_top5(test4_results_file)
    test4_labels = load_test4_labels(test4_judgments_file)

    base_by_question: dict[str, dict[str, Any]] = {}
    experiments: list[dict[str, Any]] = []
    per_alpha_questions: dict[float, list[dict[str, Any]]] = {alpha: [] for alpha in alphas}
    q14_fused_by_alpha: dict[float, list[dict[str, Any]]] = {}

    for question in questions:
        candidates = retrieve_v2_candidates(question["question"], index, metadata, embedding_model, faiss_top_k)
        if len({str(candidate["chunk_id"]) for candidate in candidates}) != len(candidates):
            raise RuntimeError(f"{question['id']}: duplicate chunk_id in FAISS Top-{faiss_top_k}")
        reranked_all = rerank_candidates(question["question"], candidates, reranker, top_k=faiss_top_k)
        scored_candidates = add_scores_and_norms(candidates, reranked_all)
        base_by_question[question["id"]] = {
            "candidates": candidates,
            "reranked_all": reranked_all,
            "scored_candidates": scored_candidates,
        }

        for alpha in alphas:
            final_top5, fused_all = apply_fusion(scored_candidates, alpha, top_k)
            if question["id"] == Q14_ID:
                q14_fused_by_alpha[alpha] = fused_all
            per_alpha_questions[alpha].append(
                {
                    "question_id": question["id"],
                    "question": question["question"],
                    "candidate_count": len(scored_candidates),
                    "final_top5": [
                        result_for_json(question["id"], question["question"], result) for result in final_top5
                    ],
                }
            )

    alpha_summaries: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        overlap_ratios: list[float] = []
        new_chunks = 0
        label_count = 0
        blank_label_count = 0
        for question in per_alpha_questions[alpha]:
            question_id = str(question["question_id"])
            test4_ids = test4_top5.get(question_id, set())
            fusion_ids = {str(result["chunk_id"]) for result in question["final_top5"]}
            overlap_ratios.append(len(fusion_ids & test4_ids) / float(top_k))
            new_chunks += len(fusion_ids - test4_ids)
            for result in question["final_top5"]:
                if historical_relevant(question_id, str(result["chunk_id"]), test4_labels):
                    label_count += 1
                else:
                    blank_label_count += 1
        alpha_summaries[alpha] = {
            "average_top5_overlap_with_test4": statistics.mean(overlap_ratios) if overlap_ratios else 0.0,
            "new_chunks_entering_top5": new_chunks,
            "historical_labels_available": label_count,
            "blank_relevance_labels": blank_label_count,
            "judgments_required": blank_label_count > 0,
        }
        experiments.append(
            {
                "alpha": alpha,
                **alpha_summaries[alpha],
                "questions": per_alpha_questions[alpha],
            }
        )

    q14_sources = extract_q14_direct_sources(trace_file)
    q14_direct_chunks = map_sources_to_v2_chunks(q14_sources, metadata)
    q14_statuses = q14_diagnostics(
        base_by_question[Q14_ID]["candidates"],
        base_by_question[Q14_ID]["reranked_all"],
        q14_fused_by_alpha,
        q14_direct_chunks,
    )

    return {
        "pipeline": "V2 chunks -> multilingual-e5-small FAISS Top-50 -> CrossEncoder -> dense/reranker score fusion -> Top-5",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "faiss_candidates_per_question": faiss_top_k,
        "final_results_per_question": top_k,
        "alphas": alphas,
        "test4_results_file": str(test4_results_file),
        "test4_judgments_file": str(test4_judgments_file),
        "experiments": experiments,
        "q14_diagnostic": {
            "question_id": Q14_ID,
            "known_direct_chunks": q14_statuses,
        },
    }


def validate_payload(payload: dict[str, Any], question_count: int, top_k: int, faiss_top_k: int) -> None:
    for experiment in payload["experiments"]:
        questions = experiment.get("questions", [])
        if len(questions) != question_count:
            raise RuntimeError(f"alpha={experiment['alpha']}: expected {question_count} questions, got {len(questions)}")
        total_results = sum(len(question.get("final_top5", [])) for question in questions)
        if total_results != question_count * top_k:
            raise RuntimeError(f"alpha={experiment['alpha']}: expected {question_count * top_k} results, got {total_results}")
        for question in questions:
            if int(question["candidate_count"]) != faiss_top_k:
                raise RuntimeError(f"{question['question_id']}: expected {faiss_top_k} candidates")
            if len(question["final_top5"]) != top_k:
                raise RuntimeError(f"{question['question_id']}: expected {top_k} final results")
            if len({str(result["chunk_id"]) for result in question["final_top5"]}) != len(question["final_top5"]):
                raise RuntimeError(f"{question['question_id']}: duplicate chunk_id in final Top-{top_k}")


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_judgments(path: Path, payload: dict[str, Any]) -> None:
    labels = load_test4_labels(Path(str(payload["test4_judgments_file"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for experiment in payload["experiments"]:
            for question in experiment["questions"]:
                for result in question["final_top5"]:
                    relevant = historical_relevant(str(question["question_id"]), str(result["chunk_id"]), labels)
                    writer.writerow(
                        {
                            "alpha": f"{float(experiment['alpha']):.2f}",
                            "question_id": question["question_id"],
                            "question": question["question"],
                            "final_rank": result["final_rank"],
                            "final_score": f"{float(result['final_score']):.6f}",
                            "faiss_rank": result["faiss_rank"],
                            "faiss_score": f"{float(result['faiss_score']):.6f}",
                            "faiss_norm": f"{float(result['faiss_norm']):.6f}",
                            "reranker_score": f"{float(result['reranker_score']):.6f}",
                            "reranker_norm": f"{float(result['reranker_norm']):.6f}",
                            "chunk_id": result["chunk_id"],
                            "source_file": result["source_file"],
                            "page_start": result["page_start"],
                            "page_end": result["page_end"],
                            "relevant": relevant,
                        }
                    )


def render_review(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for experiment in payload["experiments"]:
        lines = [
            "=" * 30,
            f"ALPHA = {float(experiment['alpha']):.2f}",
            "=" * 30,
            f"average Top-5 overlap with Test 4: {float(experiment['average_top5_overlap_with_test4']):.4f}",
            f"new chunks entering Top-5: {experiment['new_chunks_entering_top5']}",
            f"historical labels available: {experiment['historical_labels_available']}/100",
            f"judgments required: {'yes' if experiment['judgments_required'] else 'no'}",
            "",
        ]
        for question in experiment["questions"]:
            lines.extend([SEPARATOR, str(question["question_id"]), f"Question: {question['question']}", ""])
            for result in question["final_top5"]:
                lines.extend(
                    [
                        f"final rank: {result['final_rank']}",
                        f"final score: {float(result['final_score']):.6f}",
                        f"faiss rank: {result['faiss_rank']}",
                        f"faiss score: {float(result['faiss_score']):.6f}",
                        f"faiss norm: {float(result['faiss_norm']):.6f}",
                        f"reranker score: {float(result['reranker_score']):.6f}",
                        f"reranker norm: {float(result['reranker_norm']):.6f}",
                        f"source: {result['source_file']}",
                        f"pages: {result['page_start']}-{result['page_end']}",
                        f"chunk_id: {result['chunk_id']}",
                        f"text: {preview_text(str(result['text']))}",
                        "",
                    ]
                )
        blocks.append("\n".join(lines).rstrip())

    q14_lines = [SEPARATOR, "Q14 Diagnostic", ""]
    statuses = payload["q14_diagnostic"]["known_direct_chunks"]
    if not statuses:
        q14_lines.append("known direct-answer chunk: NOT_FOUND")
    else:
        for status in statuses:
            q14_lines.extend(
                [
                    f"chunk_id: {status['chunk_id']}",
                    f"prior_chunk_id: {status['prior_chunk_id']}",
                    f"source_file: {status['source_file']}",
                    f"pages: {status['page_start']}-{status['page_end']}",
                    f"FAISS rank: {status['faiss_rank']}",
                    f"FAISS score: {status['faiss_score']}",
                    f"reranker-only rank: {status['reranker_only_rank']}",
                    f"reranker score: {status['reranker_score']}",
                    f"fusion ranks by alpha: {json.dumps(status['fusion_rank_by_alpha'], ensure_ascii=False)}",
                    "",
                ]
            )
    blocks.append("\n".join(q14_lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def write_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(payload), encoding="utf-8")


def print_summary(
    payload: dict[str, Any],
    results_file: Path,
    judgments_file: Path,
    review_file: Path,
    protected_unchanged: bool,
) -> None:
    print(f"Questions = {len(payload['experiments'][0]['questions'])}")
    print(f"FAISS candidates/question = {payload['faiss_candidates_per_question']}")
    print("")
    for experiment in payload["experiments"]:
        total_results = sum(len(question["final_top5"]) for question in experiment["questions"])
        print(f"alpha={float(experiment['alpha']):.2f}")
        print(f"average Top-5 overlap with Test 4: {float(experiment['average_top5_overlap_with_test4']):.4f}")
        print(f"new chunks entering Top-5: {experiment['new_chunks_entering_top5']}")
        print(f"historical labels available: {experiment['historical_labels_available']}/100")
        print(f"judgments required: {'yes' if experiment['judgments_required'] else 'no'}")
        print(f"final results/question = {payload['final_results_per_question']}")
        print(f"total final results = {total_results}")
        print("")

    print("Q14 diagnostic:")
    statuses = payload["q14_diagnostic"]["known_direct_chunks"]
    if not statuses:
        print("known direct-answer chunk: NOT_FOUND")
    else:
        for status in statuses:
            print(f"known direct-answer chunk: {status['chunk_id']}")
            print(f"FAISS rank: {status['faiss_rank']}")
            print(f"reranker-only rank: {status['reranker_only_rank']}")
            for alpha, rank in status["fusion_rank_by_alpha"].items():
                print(f"alpha={alpha} final fusion rank: {rank}")
    print("")

    print(f"V2 FAISS index unchanged: {str(protected_unchanged).lower()}")
    print(f"V2 metadata unchanged: {str(protected_unchanged).lower()}")
    print(f"chunk files unchanged: {str(protected_unchanged).lower()}")
    print(f"Test 1/2/3/4/5 outputs unchanged: {str(protected_unchanged).lower()}")
    print("")
    print("Output files:")
    print(results_file)
    print(judgments_file)
    print(review_file)
    print("")
    print(
        "Test 6 score fusion experiment completed. Existing Test 1/Test 2/Test 3/Test 4/Test 5 results, "
        "V1/V2 chunks, FAISS indexes, embedding model and reranker were not modified."
    )
    print("Alpha exploration was performed on the development set and must be validated later on an unseen holdout set.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    before = snapshot(PROTECTED_FILES)
    questions = read_questions(args.questions_file)
    payload = run_evaluation(
        questions=questions,
        index_dir=args.index_dir,
        index_file=args.index_file,
        metadata_file=args.metadata_file,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        faiss_top_k=args.faiss_top_k,
        top_k=args.top_k,
        alphas=args.alphas,
        allow_model_download=args.allow_model_download,
        test4_results_file=args.test4_results_file,
        test4_judgments_file=args.test4_judgments_file,
        trace_file=args.trace_file,
    )
    validate_payload(payload, len(questions), args.top_k, args.faiss_top_k)
    write_results(args.results_file, payload)
    write_judgments(args.judgments_file, payload)
    write_review(args.review_file, payload)
    after = snapshot(PROTECTED_FILES)
    protected_unchanged = before == after
    print_summary(payload, args.results_file, args.judgments_file, args.review_file, protected_unchanged)


if __name__ == "__main__":
    main()
