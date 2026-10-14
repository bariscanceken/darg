"""Evaluate region-aware retrieval against the plain v4 pipeline.

For each question in regional_dev_v1.jsonl:
- location detection accuracy (expected province / ambiguity);
- Top-5 of v4 and of v4 + regional retrieval, labeled by the calibrated LLM judge;
- Hit@5 (any relevant chunk), regional Hit@5 (a relevant chunk from the right
  province's IRAP or naming the place), and wrong-province chunks (IRAP of another
  province) in the Top-5.
Final Test files are never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from src.rag.generate_answer import load_environment, load_openai_client, load_retrieval_components
from src.rag.location import detect_location, has_regional_plan
from src.rag.regional_retrieval import irap_province_map, mentions, retrieve_regional
from src.retrieval.evaluate_reranked_v2 import DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, retrieve_v2_candidates
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL, rerank_candidates
from src.retrieval.search import DEFAULT_MODEL_NAME

from .llm_relevance_judge import RelevanceJudge


INPUT_FILE = Path("data/evaluation/regional_dev_v1.jsonl")
INDEX_DIR = "data/indexes/mobile_v4"
RESULTS_FILE = Path("data/evaluation/regional_dev_v1_results.jsonl")
SUMMARY_FILE = Path("data/evaluation/regional_dev_v1_summary.txt")


def chunk_province(chunk: dict[str, Any]) -> str | None:
    return irap_province_map().get(chunk["source_file"])


def score(row: dict[str, Any], chunks: list[dict[str, Any]], labels: list[int]) -> dict[str, Any]:
    target = row["province"]
    location = detect_location(row["question"])
    place = (location.district if location and location.district else None) or target
    regional_hit = any(
        label == 1 and (chunk_province(chunk) == target or (place and mentions(place, chunk["text"])))
        for chunk, label in zip(chunks, labels)
    )
    wrong = sum(1 for chunk in chunks if chunk_province(chunk) not in (None, target))
    return {"hit": 1 in labels, "regional_hit": regional_hit, "wrong_province": wrong}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [json.loads(line) for line in INPUT_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    load_environment()
    judge = RelevanceJudge(load_openai_client())
    index, metadata, embedding_model, reranker = load_retrieval_components(
        INDEX_DIR, DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, DEFAULT_MODEL_NAME, DEFAULT_RERANKER_MODEL, False)

    results = []
    for row in rows:
        location = detect_location(row["question"])
        expected_ambiguous = row["type"] == "ambiguous"
        detected_ok = (location is not None and location.is_ambiguous) if expected_ambiguous else (
            location is not None and location.province == row["province"])
        base = rerank_candidates(row["question"], retrieve_v2_candidates(row["question"], index, metadata, embedding_model, 50), reranker, top_k=5)
        regional = retrieve_regional(row["question"], location, index, metadata, embedding_model, reranker) \
            if location and not location.is_ambiguous else base
        systems = {}
        for name, chunks in (("v4", base), ("v4_regional", regional)):
            labels = [judge.judge(row["question"], str(chunk["text"])) for chunk in chunks]
            systems[name] = {**score(row, chunks, labels), "labels": labels,
                             "sources": [f"{c['source_file']} p{c['page_start']}" for c in chunks]}
        results.append({**row, "detected": location.label() if location else None,
                        "detected_ok": detected_ok, "has_plan": has_regional_plan(row["province"]), **systems})
        print(f"{row['id']}: detect={'OK' if detected_ok else 'XX'} "
              f"v4 hit={systems['v4']['hit']} reg={systems['v4']['regional_hit']} wrong={systems['v4']['wrong_province']} | "
              f"regional hit={systems['v4_regional']['hit']} reg={systems['v4_regional']['regional_hit']} wrong={systems['v4_regional']['wrong_province']}",
              flush=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    def count(items: list[dict[str, Any]], system: str, key: str) -> str:
        return f"{sum(bool(r[system][key]) for r in items)}/{len(items)}"

    with_plan = [r for r in results if r["has_plan"] and r["type"] != "ambiguous"]
    no_plan = [r for r in results if r["type"] == "no_irap"]
    lines = ["DARG regional retrieval — regional_dev_v1 (LLM judge)", "=" * 60,
             f"Location detection correct: {sum(r['detected_ok'] for r in results)}/{len(results)}", "",
             f"Provinces WITH a regional plan ({len(with_plan)} questions):",
             f"{'':28}{'v4':>10}{'v4+regional':>14}"]
    for label, key in (("Hit@5", "hit"), ("Regional Hit@5", "regional_hit")):
        lines.append(f"{label:28}{count(with_plan, 'v4', key):>10}{count(with_plan, 'v4_regional', key):>14}")
    for subset_name, subset in (("with plan", with_plan), ("without plan", no_plan)):
        wrong_base = sum(r["v4"]["wrong_province"] for r in subset)
        wrong_reg = sum(r["v4_regional"]["wrong_province"] for r in subset)
        lines.append(f"{'Wrong-province chunks (' + subset_name + ')':28}{wrong_base:>10}{wrong_reg:>14}")
    lines += ["", "Per question (v4 | v4+regional): H=hit R=regional hit W=wrong-province chunks"]
    for r in results:
        a, b = r["v4"], r["v4_regional"]
        lines.append(f"  {r['id']} [{r['type']}] H{int(a['hit'])}R{int(a['regional_hit'])}W{a['wrong_province']} | "
                     f"H{int(b['hit'])}R{int(b['regional_hit'])}W{b['wrong_province']}  {r['question']}  -> {r['detected']}")
    summary = "\n".join(lines) + "\n"
    SUMMARY_FILE.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
