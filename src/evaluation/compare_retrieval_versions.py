"""Compare retrieval versions with the calibrated LLM relevance judge.

v2: original chunks (word-sized, broken encodings in 5 booklets)
v3: same chunks, encoding-repaired text
v4: encoding-repaired text, re-chunked to fit the 512-token model windows
All use the same E5 model, reranker and Top-50 -> Top-5 flow. Every Top-5 result
is labeled by the same judge (gpt-4o, kappa 0.80 vs human labels), so systems are
compared under one labeling rule. Final Test v1 is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.retrieval.evaluate_holdout import HOLDOUT_QUESTIONS
from src.retrieval.evaluate_reranked_v2 import DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, load_v2_index_and_metadata, retrieve_v2_candidates
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL, load_embedding_model, load_reranker, rerank_candidates
from src.retrieval.search import DEFAULT_MODEL_NAME

from .llm_relevance_judge import RelevanceJudge


INDEX_DIRS = {
    "v2": Path("data/indexes/mobile_v2"),
    "v3": Path("data/indexes/mobile_v3"),
    "v4": Path("data/indexes/mobile_v4"),
}
DEV_QUESTIONS_FILE = Path("data/evaluation/retrieval_questions.json")
RESULTS_FILE = Path("data/evaluation/retrieval_versions_results.jsonl")
SUMMARY_FILE = Path("data/evaluation/retrieval_versions_summary.txt")
FAISS_TOP_K = 50
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval versions with the LLM judge.")
    parser.add_argument("--versions", nargs="+", default=list(INDEX_DIRS))
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def question_sets() -> dict[str, list[str]]:
    dev = [row["question"] for row in json.loads(DEV_QUESTIONS_FILE.read_text(encoding="utf-8"))]
    return {"dev": dev, "holdout": [row["question"] for row in HOLDOUT_QUESTIONS]}


def run_version(version: str, embedding_model: Any, reranker: Any, judge: RelevanceJudge,
                sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    index, metadata, _, _ = load_v2_index_and_metadata(INDEX_DIRS[version], DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE)
    rows = []
    for set_name, questions in sets.items():
        for question in questions:
            candidates = retrieve_v2_candidates(question, index, metadata, embedding_model, FAISS_TOP_K)
            top = rerank_candidates(question, candidates, reranker, top_k=TOP_K)
            labels = [judge.judge(question, str(chunk["text"])) for chunk in top]
            rows.append({
                "version": version, "set": set_name, "question": question, "labels": labels,
                "results": [{"chunk_id": c["chunk_id"], "source_file": c["source_file"], "page_start": c["page_start"]} for c in top],
            })
        print(f"{version}: {set_name} done", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    hits = sum(1 in row["labels"] for row in rows)
    mrr = sum(1 / (row["labels"].index(1) + 1) if 1 in row["labels"] else 0 for row in rows) / n
    p5 = sum(sum(row["labels"]) for row in rows) / (n * TOP_K)
    return {"n": n, "hits": hits, "mrr": mrr, "p5": p5}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    from src.rag.generate_answer import load_environment, load_openai_client

    load_environment()
    judge = RelevanceJudge(load_openai_client())
    embedding_model = load_embedding_model(DEFAULT_MODEL_NAME, args.allow_model_download)
    reranker = load_reranker(DEFAULT_RERANKER_MODEL, args.allow_model_download)
    sets = question_sets()

    all_rows = [row for version in args.versions for row in run_version(version, embedding_model, reranker, judge, sets)]
    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["DARG retrieval versions — LLM judge (gpt-4o, kappa 0.80 vs human)", "=" * 64,
             "Same E5 model, reranker, Top-50 -> Top-5.", ""]
    for set_name in sets:
        lines.append(f"## {set_name}")
        for version in args.versions:
            s = summarize([r for r in all_rows if r["version"] == version and r["set"] == set_name])
            lines.append(f"  {version}: Hit@5 {s['hits']}/{s['n']}  MRR {s['mrr']:.3f}  P@5 {s['p5']:.3f}")
        lines.append("")
    lines.append("Per-question Hit@5 (" + " / ".join(args.versions) + "):")
    for set_name, questions in sets.items():
        for question in questions:
            marks = []
            for version in args.versions:
                row = next(r for r in all_rows if r["version"] == version and r["question"] == question)
                marks.append("✅" if 1 in row["labels"] else "❌")
            lines.append(f"  [{set_name}] {' '.join(marks)}  {question}")
    summary = "\n".join(lines) + "\n"
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
