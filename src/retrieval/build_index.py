"""Build a local FAISS index for AFAD earthquake chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_INDEX_DIR = Path("data/indexes/mobile")
DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS IndexFlatIP over AFAD earthquake chunks.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--local-files-only", action="store_true", help="Load the model only from the local HF cache.")
    return parser.parse_args()


def read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            chunks.append(record)
    return chunks


def prefixed_passages(chunks: Iterable[dict[str, Any]]) -> list[str]:
    return [f"passage: {str(chunk['text']).strip()}" for chunk in chunks]


def embed_passages(
    model: SentenceTransformer,
    passages: list[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def metadata_for_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "source_file": chunk["source_file"],
        "relative_path": chunk["relative_path"],
        "organization": chunk.get("organization", "AFAD"),
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "chunk_index": chunk["chunk_index"],
        "source_relevances": chunk["source_relevances"],
    }


def write_metadata(metadata_path: Path, chunks: list[dict[str, Any]]) -> None:
    with metadata_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(metadata_for_chunk(chunk), ensure_ascii=False) + "\n")


def build_index(
    chunks_file: Path,
    index_dir: Path,
    model_name: str,
    batch_size: int,
    local_files_only: bool,
) -> None:
    chunks = read_chunks(chunks_file)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_file}")

    print(f"Chunks loaded: {len(chunks)}", flush=True)
    print(f"Loading model: {model_name}", flush=True)
    model = SentenceTransformer(model_name, local_files_only=local_files_only)
    print("Embedding passages...", flush=True)
    embeddings = embed_passages(model, prefixed_passages(chunks), batch_size)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print("Writing index files...", flush=True)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "afad_e5_small.faiss"))
    write_metadata(index_dir / "metadata.jsonl", chunks)

    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Index file: {index_dir / 'afad_e5_small.faiss'}")
    print(f"Metadata file: {index_dir / 'metadata.jsonl'}")


def main() -> None:
    args = parse_args()
    build_index(args.chunks_file, args.index_dir, args.model_name, args.batch_size, args.local_files_only)


if __name__ == "__main__":
    main()
