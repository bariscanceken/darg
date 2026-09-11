"""Top-20 retrieval diagnostics for failed evaluation queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import faiss

from .search import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, embed_query, load_model, read_metadata


DEFAULT_OUTPUT_FILE = Path("data/evaluation/top20_failed_queries.txt")
TOP_K = 20
PREVIEW_CHARS = 1000
SEPARATOR = "=" * 80

QUESTIONS: list[dict[str, str]] = [
    {"id": "Q04", "question": "Deprem sırasında asansör kullanmalı mıyım?"},
    {"id": "Q06", "question": "Depremden sonra doğalgaz kokusu alırsam ne yapmalıyım?"},
    {"id": "Q09", "question": "Deprem sonrası artçı sarsıntılarda ne yapmalıyım?"},
    {"id": "Q13", "question": "Deprem sırasında dışarıdaysam ne yapmalıyım?"},
    {"id": "Q14", "question": "Deprem sırasında araç kullanıyorsam ne yapmalıyım?"},
    {"id": "Q19", "question": "Depremden sonra yaralı birine nasıl yaklaşmalıyım?"},
    {"id": "Q20", "question": "Deprem sonrasında resmi makamların tahliye talimatına uymalı mıyım?"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Top-20 diagnostics for failed retrieval queries.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if the model is not in the local cache.",
    )
    return parser.parse_args()


def preview_text(text: str, max_chars: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def retrieve_top20(index: faiss.Index, metadata: list[dict[str, Any]], model: Any, query: str) -> list[dict[str, Any]]:
    query_embedding = embed_query(model, query)
    scores, indices = index.search(query_embedding, TOP_K)

    results: list[dict[str, Any]] = []
    for rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
        row_index = int(row_index)
        if row_index < 0:
            continue
        record = metadata[row_index]
        results.append(
            {
                "rank": rank,
                "similarity_score": float(score),
                "chunk_id": record.get("chunk_id"),
                "source_file": record.get("source_file"),
                "page_start": record.get("page_start"),
                "page_end": record.get("page_end"),
                "text": preview_text(str(record.get("text", ""))),
            }
        )

    if len(results) != TOP_K:
        raise RuntimeError(f"Expected {TOP_K} results, got {len(results)}")
    return results


def render_question(question_id: str, question: str, results: list[dict[str, Any]]) -> str:
    lines = [
        SEPARATOR,
        question_id,
        f"Soru: {question}",
        f"Results: {len(results)}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"Rank {result['rank']}",
                f"Score: {result['similarity_score']:.4f}",
                f"chunk_id: {result['chunk_id']}",
                f"source_file: {result['source_file']}",
                f"page_start: {result['page_start']}",
                f"page_end: {result['page_end']}",
                f"text: {result['text']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def run_diagnostics(index_dir: Path, output_file: Path, model_name: str, allow_model_download: bool) -> dict[str, int]:
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
    if index.ntotal < TOP_K:
        raise ValueError(f"Index has only {index.ntotal} rows; cannot retrieve Top-{TOP_K}")

    model = load_model(model_name, allow_model_download)
    blocks: list[str] = []
    summary: dict[str, int] = {}

    for item in QUESTIONS:
        results = retrieve_top20(index, metadata, model, item["question"])
        blocks.append(render_question(item["id"], item["question"], results))
        summary[item["id"]] = len(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    summary = run_diagnostics(args.index_dir, args.output_file, args.model_name, args.allow_model_download)
    for item in QUESTIONS:
        print(f"{item['id']}: {summary[item['id']]} results")


if __name__ == "__main__":
    main()
