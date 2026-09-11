"""Evaluate a source-priority soft boost over the Test 4 V2 reranking pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from .evaluate import DEFAULT_QUESTIONS_FILE, read_questions
from .evaluate_reranked_v2 import DEFAULT_INDEX_DIR, DEFAULT_INDEX_FILE, load_v2_index_and_metadata, retrieve_v2_candidates
from .rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_K, load_embedding_model, load_reranker, rerank_candidates
from .search import DEFAULT_MODEL_NAME


DEFAULT_METADATA_FILE = "metadata_v2_source_enriched.jsonl"
DEFAULT_FAISS_TOP_K = 50
DEFAULT_BOOSTS = (0.02, 0.05, 0.10)
DEFAULT_RESULTS_FILE = Path("data/evaluation/source_priority_results.json")
DEFAULT_TRACE_FILE = Path("data/evaluation/source_priority_trace.txt")
DEFAULT_REVIEW_FILE = Path("data/evaluation/source_priority_review.txt")
DEFAULT_CANONICAL_JUDGMENTS_FILE = Path("data/evaluation/development_canonical_judgments.csv")
DEV_JUDGMENT_FILES = [
    Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_score_fusion_judgments.csv"),
    Path("data/evaluation/retrieval_hybrid_judgments.csv"),
    Path("data/evaluation/retrieval_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_judgments.csv"),
]
SEPARATOR = "=" * 80
PREVIEW_CHARS = 800

PROTECTED_FILES = [
    Path("data/processed/chunks/afad_earthquake_chunks.jsonl"),
    Path("data/processed/chunks/afad_earthquake_chunks_v2.jsonl"),
    Path("data/indexes/mobile/afad_e5_small.faiss"),
    Path("data/indexes/mobile/metadata.jsonl"),
    Path("data/indexes/mobile_v2/afad_e5_small_v2.faiss"),
    Path("data/indexes/mobile_v2/metadata_v2.jsonl"),
    Path("data/indexes/mobile_v2/metadata_v2_source_enriched.jsonl"),
    DEFAULT_CANONICAL_JUDGMENTS_FILE,
    *DEV_JUDGMENT_FILES,
]


@dataclass(frozen=True)
class Metrics:
    precision_at_5: float
    hit_rate_at_5: float
    mrr: float
    total_questions: int
    evaluated_results: int
    questions_with_no_relevant: int
    failed_question_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_at_5": self.precision_at_5,
            "hit_rate_at_5": self.hit_rate_at_5,
            "mrr": self.mrr,
            "total_questions": self.total_questions,
            "evaluated_results": self.evaluated_results,
            "questions_with_no_relevant": self.questions_with_no_relevant,
            "failed_question_ids": self.failed_question_ids,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate source-priority soft boost on development Q01-Q20.")
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--canonical-judgments-file", type=Path, default=DEFAULT_CANONICAL_JUDGMENTS_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--boosts", nargs="+", type=float, default=list(DEFAULT_BOOSTS))
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


def source_bonus(source_priority: int, boost: float) -> float:
    if source_priority == 3:
        return boost
    if source_priority == 2:
        return boost / 2.0
    return 0.0


def load_canonical_labels(path: Path) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            label = str(row.get("relevant", "")).strip()
            question_id = str(row.get("question_id", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            if label not in {"0", "1"} or not question_id.startswith("Q") or not chunk_id:
                raise ValueError(f"Invalid canonical judgment at {path}:{row_number}")
            key = (question_id, chunk_id)
            if key in labels and labels[key] != label:
                raise ValueError(f"Conflicting canonical judgment for {question_id}/{chunk_id}")
            labels[key] = label
    return labels


def load_development_labels(
    paths: list[Path],
    canonical_path: Path = DEFAULT_CANONICAL_JUDGMENTS_FILE,
) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]], dict[str, int]]:
    canonical_labels = load_canonical_labels(canonical_path)
    labels: dict[tuple[str, str], str] = dict(canonical_labels)
    conflicts: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = str(row.get("relevant", "")).strip()
                question_id = str(row.get("question_id", "")).strip()
                chunk_id = str(row.get("chunk_id", "")).strip()
                if label not in {"0", "1"} or not question_id.startswith("Q") or not chunk_id:
                    continue
                key = (question_id, chunk_id)
                if key in canonical_labels:
                    continue
                if key in labels and labels[key] != label:
                    conflicts.append(
                        {
                            "question_id": question_id,
                            "chunk_id": chunk_id,
                            "kept_label": labels[key],
                            "ignored_label": label,
                            "source_file": str(path),
                        }
                    )
                    continue
                labels[key] = label
    stats = {
        "canonical_labels_loaded": len(canonical_labels),
        "remaining_conflicting_labels_affecting_evaluation": len(conflicts),
    }
    return labels, conflicts, stats


def label_for(labels: dict[tuple[str, str], str], question_id: str, chunk_id: str) -> str:
    return labels.get((question_id, chunk_id), "")


def add_reranker_norms(
    candidates: list[dict[str, Any]],
    reranked_all: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reranked_by_id = {str(candidate["chunk_id"]): candidate for candidate in reranked_all}
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        reranked = reranked_by_id[str(candidate["chunk_id"])]
        updated = dict(candidate)
        updated["faiss_rank"] = int(candidate["original_rank"])
        updated["faiss_score"] = float(candidate["original_similarity_score"])
        updated["reranker_rank"] = int(reranked["reranked_rank"])
        updated["reranker_score"] = float(reranked["rerank_score"])
        source_record = metadata[int(candidate["faiss_row"])]
        updated["source_priority"] = int(source_record.get("source_priority", 1))
        updated["document_type"] = source_record.get("document_type", "unknown")
        updated["source_priority_reason"] = source_record.get("source_priority_reason", "")
        scored.append(updated)

    norms = min_max_normalize([float(candidate["reranker_score"]) for candidate in scored])
    for candidate, norm in zip(scored, norms):
        candidate["reranker_norm"] = norm
    return scored


def baseline_top5(scored_candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda candidate: (
            -float(candidate["reranker_score"]),
            int(candidate["faiss_rank"]),
            str(candidate["chunk_id"]),
        ),
    )
    top = [dict(candidate) for candidate in sorted_candidates[:top_k]]
    for rank, candidate in enumerate(top, start=1):
        candidate["final_rank"] = rank
        candidate["source_bonus"] = 0.0
        candidate["final_score"] = float(candidate["reranker_norm"])
    return top


def boosted_top5(scored_candidates: list[dict[str, Any]], boost: float, top_k: int) -> list[dict[str, Any]]:
    boosted: list[dict[str, Any]] = []
    for candidate in scored_candidates:
        updated = dict(candidate)
        updated["source_bonus"] = source_bonus(int(candidate["source_priority"]), boost)
        updated["final_score"] = float(candidate["reranker_norm"]) + float(updated["source_bonus"])
        boosted.append(updated)

    boosted.sort(
        key=lambda candidate: (
            -float(candidate["final_score"]),
            int(candidate["reranker_rank"]),
            int(candidate["faiss_rank"]),
            str(candidate["chunk_id"]),
        )
    )
    top = [dict(candidate) for candidate in boosted[:top_k]]
    for rank, candidate in enumerate(top, start=1):
        candidate["final_rank"] = rank
    return top


def result_for_json(question_id: str, question: str, candidate: dict[str, Any], labels: dict[tuple[str, str], str]) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "final_rank": int(candidate["final_rank"]),
        "chunk_id": candidate["chunk_id"],
        "source_file": candidate["source_file"],
        "page_start": candidate["page_start"],
        "page_end": candidate["page_end"],
        "text": str(candidate["text"]),
        "faiss_rank": int(candidate["faiss_rank"]),
        "faiss_score": float(candidate["faiss_score"]),
        "reranker_rank": int(candidate["reranker_rank"]),
        "reranker_score": float(candidate["reranker_score"]),
        "reranker_norm": float(candidate["reranker_norm"]),
        "source_priority": int(candidate["source_priority"]),
        "document_type": candidate["document_type"],
        "source_bonus": float(candidate["source_bonus"]),
        "final_score": float(candidate["final_score"]),
        "relevant": label_for(labels, question_id, str(candidate["chunk_id"])),
    }


def compute_metrics(question_results: list[dict[str, Any]]) -> Metrics | None:
    if any(str(result.get("relevant", "")).strip() not in {"0", "1"} for result in question_results):
        return None

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in question_results:
        grouped.setdefault(str(result["question_id"]), []).append(result)

    precision_total = 0.0
    hit_total = 0.0
    reciprocal_total = 0.0
    failed: list[str] = []
    for question_id in sorted(grouped):
        rows = sorted(grouped[question_id], key=lambda row: int(row["final_rank"]))
        labels = [int(row["relevant"]) for row in rows]
        relevant_ranks = [rank for rank, label in enumerate(labels, start=1) if label == 1]
        precision_total += len(relevant_ranks) / 5.0
        if relevant_ranks:
            hit_total += 1.0
            reciprocal_total += 1.0 / min(relevant_ranks)
        else:
            failed.append(question_id)

    total_questions = len(grouped)
    return Metrics(
        precision_at_5=precision_total / total_questions,
        hit_rate_at_5=hit_total / total_questions,
        mrr=reciprocal_total / total_questions,
        total_questions=total_questions,
        evaluated_results=len(question_results),
        questions_with_no_relevant=len(failed),
        failed_question_ids=failed,
    )


def priority_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(result["source_priority"]) for result in results)
    return {str(priority): counts.get(str(priority), 0) for priority in (3, 2, 1)}


def changed_count(baseline_results: list[dict[str, Any]], boosted_results: list[dict[str, Any]]) -> int:
    baseline_ids = {str(result["chunk_id"]) for result in baseline_results}
    boosted_ids = {str(result["chunk_id"]) for result in boosted_results}
    return len(boosted_ids - baseline_ids)


def run_experiment(
    questions: list[dict[str, str]],
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    reranker: Any,
    labels: dict[tuple[str, str], str],
    faiss_top_k: int,
    top_k: int,
    boosts: list[float],
) -> dict[str, Any]:
    question_blocks: list[dict[str, Any]] = []
    baseline_all: list[dict[str, Any]] = []
    boosted_by_value: dict[float, list[dict[str, Any]]] = {boost: [] for boost in boosts}

    for question in questions:
        candidates = retrieve_v2_candidates(question["question"], index, metadata, embedding_model, faiss_top_k)
        reranked_all = rerank_candidates(question["question"], candidates, reranker, top_k=faiss_top_k)
        scored_candidates = add_reranker_norms(candidates, reranked_all, metadata)
        baseline = [result_for_json(question["id"], question["question"], item, labels) for item in baseline_top5(scored_candidates, top_k)]
        baseline_all.extend(baseline)

        boosted_blocks: list[dict[str, Any]] = []
        for boost in boosts:
            boosted = [
                result_for_json(question["id"], question["question"], item, labels)
                for item in boosted_top5(scored_candidates, boost, top_k)
            ]
            boosted_by_value[boost].extend(boosted)
            baseline_ids = {str(result["chunk_id"]) for result in baseline}
            boosted_ids = {str(result["chunk_id"]) for result in boosted}
            boosted_blocks.append(
                {
                    "boost": boost,
                    "results": boosted,
                    "entered_top5": sorted(boosted_ids - baseline_ids),
                    "exited_top5": sorted(baseline_ids - boosted_ids),
                }
            )

        question_blocks.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "baseline_top5": baseline,
                "boosted": boosted_blocks,
            }
        )

    baseline_metrics = compute_metrics(baseline_all)
    baseline_missing = sum(1 for result in baseline_all if str(result.get("relevant", "")).strip() not in {"0", "1"})
    baseline_priority_counts = priority_counts(baseline_all)
    boost_summaries: list[dict[str, Any]] = []
    for boost in boosts:
        boosted_results = boosted_by_value[boost]
        metrics = compute_metrics(boosted_results)
        counts = priority_counts(boosted_results)
        missing = sum(1 for result in boosted_results if str(result.get("relevant", "")).strip() not in {"0", "1"})
        boost_summaries.append(
            {
                "boost": boost,
                "metrics": metrics.to_dict() if metrics else None,
                "missing_judgments": missing,
                "changed_results_vs_baseline": sum(
                    changed_count(block["baseline_top5"], next(item["results"] for item in block["boosted"] if item["boost"] == boost))
                    for block in question_blocks
                ),
                "source_priority_counts": counts,
                "source_priority_delta_vs_baseline": {
                    priority: counts[priority] - baseline_priority_counts[priority]
                    for priority in baseline_priority_counts
                },
            }
        )

    return {
        "questions": question_blocks,
        "baseline": {
            "name": "Test4",
            "metrics": baseline_metrics.to_dict() if baseline_metrics else None,
            "missing_judgments": baseline_missing,
            "source_priority_counts": baseline_priority_counts,
        },
        "boosts": boost_summaries,
    }


def select_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = payload["baseline"]["metrics"]
    if baseline_metrics is None:
        return {"selected": "Test4", "reason": "Baseline labels are incomplete."}

    viable: list[dict[str, Any]] = []
    for boost in payload["boosts"]:
        metrics = boost["metrics"]
        if metrics is None:
            continue
        if metrics["hit_rate_at_5"] < baseline_metrics["hit_rate_at_5"]:
            continue
        if metrics["precision_at_5"] < baseline_metrics["precision_at_5"]:
            continue
        if metrics["mrr"] < baseline_metrics["mrr"] - 0.05:
            continue
        improves_any_metric = (
            metrics["hit_rate_at_5"] > baseline_metrics["hit_rate_at_5"]
            or metrics["precision_at_5"] > baseline_metrics["precision_at_5"]
            or metrics["mrr"] > baseline_metrics["mrr"]
        )
        if not improves_any_metric and boost["changed_results_vs_baseline"] == 0:
            continue
        viable.append(boost)

    if not viable:
        return {
            "selected": "Test4",
            "reason": "Source-priority soft boost did not improve the development baseline.",
        }

    viable.sort(
        key=lambda item: (
            item["metrics"]["hit_rate_at_5"],
            item["metrics"]["precision_at_5"],
            item["metrics"]["mrr"],
            -float(item["boost"]),
        ),
        reverse=True,
    )
    chosen = viable[0]
    return {
        "selected": f"boost={chosen['boost']:.2f}",
        "reason": "Selected by development-set rule order: Hit Rate@5, Precision@5, MRR, then smaller boost.",
    }


def validate_payload(payload: dict[str, Any], question_count: int, top_k: int) -> None:
    questions = payload["questions"]
    if len(questions) != question_count:
        raise RuntimeError(f"Expected {question_count} questions, got {len(questions)}")
    for block in questions:
        if len(block["baseline_top5"]) != top_k:
            raise RuntimeError(f"{block['question_id']}: baseline expected {top_k} results")
        if len({str(result["chunk_id"]) for result in block["baseline_top5"]}) != top_k:
            raise RuntimeError(f"{block['question_id']}: duplicate baseline chunk")
        for boosted in block["boosted"]:
            if len(boosted["results"]) != top_k:
                raise RuntimeError(f"{block['question_id']} boost={boosted['boost']}: expected {top_k} results")
            if len({str(result["chunk_id"]) for result in boosted["results"]}) != top_k:
                raise RuntimeError(f"{block['question_id']} boost={boosted['boost']}: duplicate boosted chunk")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def result_trace_lines(result: dict[str, Any], boosted: bool) -> list[str]:
    lines = [
        f"rank: {result['final_rank']}",
        f"source_file: {result['source_file']}",
        f"source_priority: {result['source_priority']}",
        f"reranker_norm: {float(result['reranker_norm']):.6f}",
    ]
    if boosted:
        lines.extend(
            [
                f"source_bonus: {float(result['source_bonus']):.6f}",
                f"final_score: {float(result['final_score']):.6f}",
            ]
        )
    lines.extend(
        [
            f"relevant: {result['relevant'] if result['relevant'] != '' else 'MISSING'}",
            f"chunk_id: {result['chunk_id']}",
            "",
        ]
    )
    return lines


def render_trace(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for block in payload["questions"]:
        lines = [SEPARATOR, f"QUESTION ID: {block['question_id']}", f"Question: {block['question']}", "", "Baseline Top5:"]
        for result in block["baseline_top5"]:
            lines.extend(result_trace_lines(result, boosted=False))

        for boosted in block["boosted"]:
            lines.extend([f"Boosted Top5 boost={float(boosted['boost']):.2f}:", ""])
            for result in boosted["results"]:
                lines.extend(result_trace_lines(result, boosted=True))
            lines.extend(
                [
                    f"Entered Top5: {', '.join(boosted['entered_top5']) if boosted['entered_top5'] else 'NONE'}",
                    f"Exited Top5: {', '.join(boosted['exited_top5']) if boosted['exited_top5'] else 'NONE'}",
                    "",
                ]
            )
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def render_review(payload: dict[str, Any]) -> str:
    lines = [
        "SOURCE PRIORITY EXPERIMENT REVIEW",
        "Only development Q01-Q20 were used.",
        "",
        "Missing judgments:",
        "",
    ]
    missing_count = 0
    for block in payload["questions"]:
        for boosted in block["boosted"]:
            for result in boosted["results"]:
                if str(result.get("relevant", "")).strip() in {"0", "1"}:
                    continue
                missing_count += 1
                lines.extend(
                    [
                        SEPARATOR,
                        f"question_id: {block['question_id']}",
                        f"question: {block['question']}",
                        f"boost: {float(boosted['boost']):.2f}",
                        f"rank: {result['final_rank']}",
                        f"chunk_id: {result['chunk_id']}",
                        f"source_file: {result['source_file']}",
                        f"source_priority: {result['source_priority']}",
                        f"page_start/page_end: {result['page_start']}-{result['page_end']}",
                        "text:",
                        str(result["text"]).strip(),
                        "",
                    ]
                )
    if missing_count == 0:
        lines.append("No missing judgments. Existing development labels covered all boosted Top-5 results.")
    return "\n".join(lines).rstrip() + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metric_display(metrics: dict[str, Any] | None) -> tuple[str, str, str, str]:
    if metrics is None:
        return "INCOMPLETE", "INCOMPLETE", "INCOMPLETE", "INCOMPLETE"
    failed = ", ".join(metrics["failed_question_ids"]) if metrics["failed_question_ids"] else "NONE"
    return (
        f"{float(metrics['precision_at_5']):.4f}",
        f"{float(metrics['hit_rate_at_5']):.4f}",
        f"{float(metrics['mrr']):.4f}",
        failed,
    )


def print_summary(payload: dict[str, Any], protected_unchanged: bool, results_file: Path, trace_file: Path, review_file: Path) -> None:
    print("SOURCE PRIORITY EXPERIMENT")
    print("")
    p_at_5, hit, mrr, _ = metric_display(payload["baseline"]["metrics"])
    print("Baseline Test4")
    print(f"P@5: {p_at_5}")
    print(f"Hit@5: {hit}")
    print(f"MRR: {mrr}")
    print(f"Missing judgments baseline: {payload['baseline']['missing_judgments']}")
    print(f"Source priority counts: {payload['baseline']['source_priority_counts']}")
    print("")

    for boost in payload["boosts"]:
        p_at_5, hit, mrr, failed = metric_display(boost["metrics"])
        print(f"Boost {float(boost['boost']):.2f}")
        print(f"P@5: {p_at_5}")
        print(f"Hit@5: {hit}")
        print(f"MRR: {mrr}")
        print(f"Failed: {failed}")
        print(f"Missing judgments: {boost['missing_judgments']}")
        print(f"Changed results vs baseline: {boost['changed_results_vs_baseline']}")
        print(f"Priority deltas vs baseline: {boost['source_priority_delta_vs_baseline']}")
        print("")

    print("Best development configuration:")
    print(f"{payload['selection']['selected']} - {payload['selection']['reason']}")
    print("")
    print("Validation:")
    print("H01-H30 used: false")
    print(f"Canonical labels loaded: {payload['label_stats']['canonical_labels_loaded']}")
    print(
        "Remaining conflicting labels affecting evaluation: "
        f"{payload['label_stats']['remaining_conflicting_labels_affecting_evaluation']}"
    )
    print(f"canonical file contains exactly 13 rows: {payload['label_stats']['canonical_labels_loaded'] == 13}")
    print(f"old judgment files unchanged: {str(protected_unchanged).lower()}")
    print(f"chunk file unchanged: {str(protected_unchanged).lower()}")
    print(f"FAISS unchanged: {str(protected_unchanged).lower()}")
    print("embedding model unchanged: true")
    print("Cross-Encoder unchanged: true")
    print(f"candidate depth unchanged: {payload['faiss_candidates_per_question'] == 50}")
    print("no BM25: true")
    print("no Test6 score fusion: true")
    print("no score fusion: true")
    print("source priority only affects final reranking: true")
    print("no hard filtering: true")
    print("")
    print("Output files:")
    print(results_file)
    print(trace_file)
    print(review_file)
    print("")
    print("Source-priority soft boost experiment completed.")
    print("Holdout was not used for tuning.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    before = snapshot(PROTECTED_FILES)
    questions = read_questions(args.questions_file)
    labels, label_conflicts, label_stats = load_development_labels(DEV_JUDGMENT_FILES, args.canonical_judgments_file)
    index, metadata, index_path, metadata_path = load_v2_index_and_metadata(args.index_dir, args.index_file, args.metadata_file)
    embedding_model = load_embedding_model(args.embedding_model, args.allow_model_download)
    reranker = load_reranker(args.reranker_model, args.allow_model_download)
    experiment = run_experiment(
        questions=questions,
        index=index,
        metadata=metadata,
        embedding_model=embedding_model,
        reranker=reranker,
        labels=labels,
        faiss_top_k=args.faiss_top_k,
        top_k=args.top_k,
        boosts=args.boosts,
    )
    payload = {
        "experiment": "Source-priority soft boost over Test4 V2 reranking pipeline",
        "questions_file": str(args.questions_file),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "faiss_candidates_per_question": args.faiss_top_k,
        "final_results_per_question": args.top_k,
        "boost_values": args.boosts,
        "canonical_judgments_file": str(args.canonical_judgments_file),
        "development_label_files": [str(path) for path in DEV_JUDGMENT_FILES if path.exists()],
        "label_stats": label_stats,
        "label_conflicts": label_conflicts,
        **experiment,
    }
    payload["selection"] = select_configuration(payload)
    validate_payload(payload, len(questions), args.top_k)

    write_json(args.results_file, payload)
    write_text(args.trace_file, render_trace(payload))
    write_text(args.review_file, render_review(payload))
    after = snapshot(PROTECTED_FILES)
    print_summary(payload, before == after, args.results_file, args.trace_file, args.review_file)


if __name__ == "__main__":
    main()
