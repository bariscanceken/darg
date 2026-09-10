"""Search the local AFAD earthquake FAISS index from the terminal."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_DIR = Path("data/indexes/mobile")
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_TOP_K = 5
RESULT_SEPARATOR = "=" * 88


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search AFAD earthquake chunks.")
    parser.add_argument("query", help="User query to search for.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if the model is not in the local cache.",
    )
    return parser.parse_args()


def read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return metadata


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    embedding = model.encode(
        [f"query: {query.strip()}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype="float32")


def format_relevance(record: dict[str, Any]) -> str:
    values = record.get("source_relevances", [])
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)
    return str(values)


def print_result(rank: int, score: float, record: dict[str, Any]) -> None:
    page_start = record.get("page_start")
    page_end = record.get("page_end")
    print(f"\n{RESULT_SEPARATOR}")
    print(f"rank: {rank}")
    print(f"similarity score: {score:.4f}")
    print(f"source_file: {record.get('source_file')}")
    print(f"page_start-page_end: {page_start}-{page_end}")
    print(f"source_relevances: {format_relevance(record)}")
    print("chunk text:")
    print(str(record.get("text", "")).strip())


def load_model(model_name: str, allow_model_download: bool) -> SentenceTransformer:
    # Keep CLI search output focused on retrieval results; traceback still appears if loading fails.
    with contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(model_name, local_files_only=not allow_model_download)


def search(index_dir: Path, model_name: str, query: str, top_k: int, allow_model_download: bool) -> None:
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
    query_embedding = embed_query(model, query)
    k = min(max(top_k, 1), len(metadata))
    scores, indices = index.search(query_embedding, k)
    result_pairs = [
        (float(score), int(row_index))
        for score, row_index in zip(scores[0], indices[0])
        if int(row_index) >= 0
    ]
    row_indices = [row_index for _, row_index in result_pairs]
    distinct_row_count = len(set(row_indices))

    if len(result_pairs) != k:
        raise RuntimeError(f"Expected {k} search results, got {len(result_pairs)}")
    if distinct_row_count != len(row_indices):
        raise RuntimeError(
            f"Expected distinct FAISS rows, got {distinct_row_count} unique rows for {len(row_indices)} results"
        )

    print(f"requested_top_k: {top_k}")
    print(f"returned_results: {len(result_pairs)}")
    print(f"distinct_faiss_rows: {distinct_row_count}")

    for rank, (score, row_index) in enumerate(result_pairs, start=1):
        print_result(rank, float(score), metadata[int(row_index)])

    print(f"\n{RESULT_SEPARATOR}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    search(args.index_dir, args.model_name, args.query, args.top_k, args.allow_model_download)


if __name__ == "__main__":
    main()
