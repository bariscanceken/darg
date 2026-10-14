"""Ablation of retrieval strategies on the v4 index, labeled by the calibrated LLM judge.

Run 1 (retrieval_strategies_run1_*): A baseline, B +BM25-F5 top-30, C +diversity
cap, D both. Run 2 (this configuration):
A  v4 baseline: dense Top-50 -> rerank -> Top-5
E  + BM25-F5 top-10 pool
F  + BM25-F5 top-10 pool + LLM rerank of the cross-encoder top-20 (gpt-4o-mini)
G  + BM25-F5 top-30 pool + LLM rerank
Every strategy also gets the region-aware pool when the question names a place
(questions without a place are unaffected by it). Sets: dev (20), holdout (30),
regional_dev_v1 (24, ambiguous question excluded). Final Test files are never read.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from src.rag.generate_answer import DEFAULT_OPENAI_MODEL, load_environment, load_openai_client, load_retrieval_components
from src.rag.location import detect_location
from src.rag.regional_retrieval import retrieve_enhanced
from src.retrieval.evaluate_holdout import HOLDOUT_QUESTIONS
from src.retrieval.evaluate_reranked_v2 import DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL
from src.retrieval.search import DEFAULT_MODEL_NAME

from .llm_relevance_judge import RelevanceJudge


INDEX_DIR = "data/indexes/mobile_v4"
STRATEGIES = {
    "A_baseline": {"lexical": False},
    "E_bm25_10": {"lexical": True, "bm25_top_k": 10},
    "F_bm25_10_llm": {"lexical": True, "bm25_top_k": 10, "llm": True},
    "G_bm25_30_llm": {"lexical": True, "bm25_top_k": 30, "llm": True},
}
RESULTS_FILE = Path("data/evaluation/retrieval_strategies_results.jsonl")
SUMMARY_FILE = Path("data/evaluation/retrieval_strategies_summary.txt")


def question_sets() -> dict[str, list[str]]:
    dev = [row["question"] for row in json.loads(Path("data/evaluation/retrieval_questions.json").read_text(encoding="utf-8"))]
    regional = [json.loads(line) for line in Path("data/evaluation/regional_dev_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "dev": dev,
        "holdout": [row["question"] for row in HOLDOUT_QUESTIONS],
        "regional": [row["question"] for row in regional if row["type"] != "ambiguous"],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_environment()
    client = load_openai_client()
    judge = RelevanceJudge(client)
    llm = (client, os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL)
    index, metadata, embedding_model, reranker = load_retrieval_components(
        INDEX_DIR, DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, DEFAULT_MODEL_NAME, DEFAULT_RERANKER_MODEL, False)

    rows: list[dict[str, Any]] = []
    for set_name, questions in question_sets().items():
        for question in questions:
            location = detect_location(question)
            for name, options in STRATEGIES.items():
                opts = dict(options)
                use_llm = opts.pop("llm", False)
                top = retrieve_enhanced(question, location, index, metadata, embedding_model, reranker, 5,
                                        llm_reranker=llm if use_llm else None, **opts)
                labels = [judge.judge(question, str(chunk["text"])) for chunk in top]
                rows.append({"set": set_name, "question": question, "strategy": name, "labels": labels,
                             "sources": [f"{c['source_file']} p{c['page_start']}" for c in top]})
            print(f"{set_name}: {question[:60]}", flush=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["DARG retrieval strategy ablation on mobile_v4 (LLM judge, gpt-4o, kappa 0.80)", "=" * 66, ""]
    for set_name in question_sets():
        lines.append(f"## {set_name}")
        for name in STRATEGIES:
            subset = [r for r in rows if r["set"] == set_name and r["strategy"] == name]
            n = len(subset)
            hits = sum(1 in r["labels"] for r in subset)
            mrr = sum(1 / (r["labels"].index(1) + 1) if 1 in r["labels"] else 0 for r in subset) / n
            p5 = sum(sum(r["labels"]) for r in subset) / (5 * n)
            lines.append(f"  {name:12} Hit@5 {hits}/{n}  MRR {mrr:.3f}  P@5 {p5:.3f}")
        lines.append("")
    lines.append("Per question Hit@5 (" + " / ".join(STRATEGIES) + "):")
    for set_name, questions in question_sets().items():
        for question in questions:
            marks = " ".join("✅" if 1 in next(r for r in rows if r["question"] == question and r["strategy"] == name)["labels"] else "❌"
                             for name in STRATEGIES)
            lines.append(f"  [{set_name}] {marks}  {question}")
    summary = "\n".join(lines) + "\n"
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
