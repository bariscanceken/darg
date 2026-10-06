"""Compare frozen retrieval on mobile_v2 vs encoding-repaired mobile_v3.

Same embedding model, reranker and Top-50 -> Top-5 flow; only the chunk text
(and therefore the index) differs. Existing v2 relevance labels are carried over
with the v2->v3 chunk-id map. Unlabeled Top-5 pairs are written to a labeling CSV;
fill its `relevant` column (1/0) and rerun to get final metrics.
Final Test v1 is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.retrieval.evaluate_holdout import HOLDOUT_QUESTIONS
from src.retrieval.evaluate_reranked_v2 import DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, load_v2_index_and_metadata, retrieve_v2_candidates
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL, load_embedding_model, load_reranker, rerank_candidates
from src.retrieval.search import DEFAULT_MODEL_NAME


INDEX_DIRS = {"v2": Path("data/indexes/mobile_v2"), "v3": Path("data/indexes/mobile_v3")}
ID_MAP_FILE = Path("data/processed/chunks/chunk_id_map_v2_to_v3.json")
DEV_QUESTIONS_FILE = Path("data/evaluation/retrieval_questions.json")
JUDGMENT_FILES = [
    Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_hybrid_judgments.csv"),
    Path("data/evaluation/retrieval_score_fusion_judgments.csv"),
    Path("data/evaluation/holdout_retrieval_judgments.csv"),
]
LABELING_FILE = Path("data/evaluation/retrieval_v3_labeling.csv")
SUMMARY_FILE = Path("data/evaluation/retrieval_v2_v3_comparison.txt")
FAISS_TOP_K = 50
TOP_K = 5
LABELING_FIELDS = ["question_set", "question", "system", "rank", "chunk_id", "source_file", "page_start", "suggested", "note", "relevant", "text"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval on mobile_v2 vs mobile_v3.")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--use-suggested", action="store_true",
                        help="Provisional run: use the `suggested` column where `relevant` is still empty.")
    return parser.parse_args()


def load_labels(use_suggested: bool = False) -> dict[tuple[str, str], int]:
    """(question, v3 chunk id) -> relevant, from v2 judgments mapped to v3 plus filled labeling rows."""
    id_map = json.loads(ID_MAP_FILE.read_text(encoding="utf-8"))
    labels: dict[tuple[str, str], int] = {}
    for path in JUDGMENT_FILES:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                chunk_id = row["chunk_id"]
                if chunk_id in id_map and row.get("relevant", "").strip() in {"0", "1"}:
                    labels[(row["question"], id_map[chunk_id])] = int(row["relevant"])
    if LABELING_FILE.exists():
        with LABELING_FILE.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row["relevant"].strip()
                if not value and use_suggested:
                    value = row.get("suggested", "").strip()
                if value in {"0", "1"}:
                    labels[(row["question"], row["chunk_id"])] = int(value)
    return labels


def questions() -> dict[str, list[str]]:
    dev = [row["question"] for row in json.loads(DEV_QUESTIONS_FILE.read_text(encoding="utf-8"))]
    return {"dev": dev, "holdout": [row["question"] for row in HOLDOUT_QUESTIONS]}


def run_system(version: str, embedding_model: Any, reranker: Any, all_questions: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    index, metadata, _, _ = load_v2_index_and_metadata(INDEX_DIRS[version], DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE)
    id_map = json.loads(ID_MAP_FILE.read_text(encoding="utf-8"))
    results: dict[str, list[dict[str, Any]]] = {}
    for question_set, items in all_questions.items():
        for question in items:
            candidates = retrieve_v2_candidates(question, index, metadata, embedding_model, FAISS_TOP_K)
            top = rerank_candidates(question, candidates, reranker, top_k=TOP_K)
            for chunk in top:
                # Express every result in v3 ids so labels are shared across systems.
                chunk["v3_chunk_id"] = id_map.get(chunk["chunk_id"], chunk["chunk_id"]) if version == "v2" else chunk["chunk_id"]
            results[question] = top
        print(f"{version}: {question_set} done", flush=True)
    return results


def metrics(results: dict[str, list[dict[str, Any]]], items: list[str], labels: dict[tuple[str, str], int]) -> dict[str, Any]:
    hits = reciprocal = relevant = unlabeled = 0
    for question in items:
        flags = [labels.get((question, chunk["v3_chunk_id"])) for chunk in results[question]]
        unlabeled += sum(flag is None for flag in flags)
        relevant += sum(flag == 1 for flag in flags)
        first = next((rank for rank, flag in enumerate(flags, start=1) if flag == 1), None)
        hits += first is not None
        reciprocal += 1 / first if first else 0
    n = len(items)
    return {"hit": hits, "n": n, "mrr": reciprocal / n, "p5": relevant / (n * TOP_K), "unlabeled": unlabeled}


def write_labeling_file(all_results: dict[str, dict[str, list[dict[str, Any]]]], all_questions: dict[str, list[str]],
                        labels: dict[tuple[str, str], int]) -> int:
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if LABELING_FILE.exists():
        with LABELING_FILE.open(encoding="utf-8") as handle:
            existing = {(row["question"], row["chunk_id"]): row for row in csv.DictReader(handle)}
    rows: dict[tuple[str, str], dict[str, Any]] = dict(existing)
    for question_set, items in all_questions.items():
        for question in items:
            for version, results in all_results.items():
                for rank, chunk in enumerate(results[question], start=1):
                    key = (question, chunk["v3_chunk_id"])
                    if key in labels or key in rows:
                        continue
                    rows[key] = {
                        "question_set": question_set, "question": question, "system": version, "rank": rank,
                        "chunk_id": chunk["v3_chunk_id"], "source_file": chunk["source_file"],
                        "page_start": chunk["page_start"], "text": " ".join(str(chunk["text"]).split()), "relevant": "",
                    }
    with LABELING_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    return sum(1 for row in rows.values() if str(row["relevant"]).strip() not in {"0", "1"})


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    embedding_model = load_embedding_model(DEFAULT_MODEL_NAME, args.allow_model_download)
    reranker = load_reranker(DEFAULT_RERANKER_MODEL, args.allow_model_download)
    all_questions = questions()
    all_results = {version: run_system(version, embedding_model, reranker, all_questions) for version in INDEX_DIRS}
    labels = load_labels(args.use_suggested)
    pending = write_labeling_file(all_results, all_questions, load_labels())

    lines = ["DARG retrieval: mobile_v2 vs mobile_v3 (encoding-repaired text)", "=" * 64,
             "PROVISIONAL: researcher-unreviewed suggested labels used." if args.use_suggested else "Labels: reviewed only.",
             "Same E5 model, reranker, Top-50 -> Top-5. Unlabeled results count as not relevant.", ""]
    for question_set, items in all_questions.items():
        lines.append(f"## {question_set} ({len(items)} questions)")
        for version in INDEX_DIRS:
            m = metrics(all_results[version], items, labels)
            lines.append(f"  {version}: Hit@5 {m['hit']}/{m['n']}  MRR {m['mrr']:.3f}  P@5 {m['p5']:.3f}  unlabeled in Top-5: {m['unlabeled']}")
        lines.append("")
    lines.append(f"Pairs still waiting for a label in {LABELING_FILE}: {pending}")
    summary = "\n".join(lines) + "\n"
    summary_file = SUMMARY_FILE.with_name(SUMMARY_FILE.stem + "_provisional.txt") if args.use_suggested else SUMMARY_FILE
    summary_file.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
