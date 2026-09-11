"""Build a separate FAISS index for the v2 AFAD earthquake chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from .build_index import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    embed_passages,
    metadata_for_chunk,
    prefixed_passages,
    read_chunks,
)


DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks_v2.jsonl")
DEFAULT_INDEX_DIR = Path("data/indexes/mobile_v2")
DEFAULT_FAISS_FILENAME = "afad_e5_small_v2.faiss"
DEFAULT_METADATA_FILENAME = "metadata_v2.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS IndexFlatIP over AFAD v2 earthquake chunks.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Load the model only from the local HF cache. Enabled by default for Test 4.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if the model is not already cached.",
    )
    return parser.parse_args()


def write_metadata(metadata_path: Path, chunks: list[dict[str, Any]]) -> None:
    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(metadata_for_chunk(chunk), ensure_ascii=False) + "\n")


def validate_metadata(metadata_path: Path) -> int:
    rows = 0
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected metadata JSON object at {metadata_path}:{line_number}")
            rows += 1
    return rows


def duplicate_chunk_ids(chunks: list[dict[str, Any]]) -> int:
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    return len(chunk_ids) - len(set(chunk_ids))


def build_index_v2(
    chunks_file: Path,
    index_dir: Path,
    model_name: str,
    batch_size: int,
    local_files_only: bool,
) -> dict[str, Any]:
    chunks = read_chunks(chunks_file)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_file}")

    duplicate_ids = duplicate_chunk_ids(chunks)
    if duplicate_ids:
        raise ValueError(f"Duplicate chunk IDs found in v2 chunks: {duplicate_ids}")

    print(f"V2 chunks loaded: {len(chunks)}", flush=True)
    print(f"Loading model: {model_name}", flush=True)
    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    print("Embedding V2 passages...", flush=True)
    embeddings = embed_passages(model, prefixed_passages(chunks), batch_size)
    dimension = int(embeddings.shape[1])

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / DEFAULT_FAISS_FILENAME
    metadata_path = index_dir / DEFAULT_METADATA_FILENAME

    print("Writing V2 index files...", flush=True)
    faiss.write_index(index, str(index_path))
    write_metadata(metadata_path, chunks)

    written_index = faiss.read_index(str(index_path))
    metadata_rows = validate_metadata(metadata_path)
    if written_index.ntotal != metadata_rows:
        raise RuntimeError(f"FAISS rows ({written_index.ntotal}) != metadata rows ({metadata_rows})")
    if written_index.ntotal != len(chunks):
        raise RuntimeError(f"FAISS rows ({written_index.ntotal}) != V2 chunk count ({len(chunks)})")

    return {
        "v2_chunk_count": len(chunks),
        "faiss_rows": int(written_index.ntotal),
        "metadata_rows": metadata_rows,
        "embedding_dimension": int(written_index.d),
        "duplicate_chunk_ids": duplicate_ids,
        "index_path": index_path,
        "metadata_path": metadata_path,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"V2 chunk count: {summary['v2_chunk_count']}")
    print(f"FAISS rows: {summary['faiss_rows']}")
    print(f"metadata rows: {summary['metadata_rows']}")
    print(f"embedding dimension: {summary['embedding_dimension']}")
    print("")
    print(f"FAISS rows == metadata rows: {str(summary['faiss_rows'] == summary['metadata_rows']).lower()}")
    print(f"FAISS rows == V2 chunk count: {str(summary['faiss_rows'] == summary['v2_chunk_count']).lower()}")
    print(f"duplicate chunk IDs == 0: {str(summary['duplicate_chunk_ids'] == 0).lower()}")
    print("")
    print(f"Index file: {summary['index_path']}")
    print(f"Metadata file: {summary['metadata_path']}")
    print("")
    print("Existing Test 1/Test 2/Test 3 files and the original FAISS index were not modified.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    local_files_only = not args.allow_model_download if args.allow_model_download else args.local_files_only
    summary = build_index_v2(
        args.chunks_file,
        args.index_dir,
        args.model_name,
        args.batch_size,
        local_files_only,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
