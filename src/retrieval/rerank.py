"""Experimental CrossEncoder reranking over existing FAISS retrieval results."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any

import faiss
from sentence_transformers import CrossEncoder

from .search import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, embed_query, read_metadata


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_FAISS_TOP_K = 20
DEFAULT_RERANK_TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FAISS Top-20 retrieval followed by CrossEncoder reranking.")
    parser.add_argument("query", help="User query to search and rerank.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def load_embedding_model(model_name: str, allow_model_download: bool) -> Any:
    from .search import load_model

    return load_model(model_name, allow_model_download)


def load_reranker(model_name: str, allow_model_download: bool) -> CrossEncoder:
    # Keep CLI/evaluation output focused; tracebacks still appear if loading fails.
    with contextlib.redirect_stderr(io.StringIO()):
        return CrossEncoder(model_name, local_files_only=not allow_model_download)


def load_index_and_metadata(index_dir: Path) -> tuple[faiss.Index, list[dict[str, Any]]]:
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
    return index, metadata


def retrieve_faiss_candidates(
    query: str,
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    faiss_top_k: int = DEFAULT_FAISS_TOP_K,
) -> list[dict[str, Any]]:
    k = min(max(faiss_top_k, 1), len(metadata))
    query_embedding = embed_query(embedding_model, query)
    scores, indices = index.search(query_embedding, k)

    candidates: list[dict[str, Any]] = []
    row_indices: list[int] = []
    for original_rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
        row_index = int(row_index)
        if row_index < 0:
            continue
        record = metadata[row_index]
        row_indices.append(row_index)
        candidates.append(
            {
                "original_rank": original_rank,
                "original_similarity_score": float(score),
                "faiss_row": row_index,
                "chunk_id": record["chunk_id"],
                "source_file": record["source_file"],
                "page_start": record["page_start"],
                "page_end": record["page_end"],
                "text": str(record.get("text", "")),
            }
        )

    if len(candidates) != k:
        raise RuntimeError(f"Expected {k} FAISS candidates, got {len(candidates)}")
    if len(set(row_indices)) != len(row_indices):
        raise RuntimeError(f"Expected distinct FAISS rows, got duplicates in Top-{k}")
    return candidates


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: CrossEncoder,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[dict[str, Any]]:
    pairs = [(query, str(candidate["text"])) for candidate in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored_candidates: list[dict[str, Any]] = []

    for candidate, score in zip(candidates, scores):
        scored_candidate = dict(candidate)
        scored_candidate["rerank_score"] = float(score)
        scored_candidates.append(scored_candidate)

    scored_candidates.sort(
        key=lambda candidate: (
            -float(candidate["rerank_score"]),
            int(candidate["original_rank"]),
            str(candidate["chunk_id"]),
        )
    )

    reranked = scored_candidates[: max(top_k, 1)]
    for reranked_rank, candidate in enumerate(reranked, start=1):
        candidate["reranked_rank"] = reranked_rank
    return reranked


def rerank_query(
    query: str,
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    reranker: CrossEncoder,
    faiss_top_k: int = DEFAULT_FAISS_TOP_K,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[dict[str, Any]]:
    candidates = retrieve_faiss_candidates(query, index, metadata, embedding_model, faiss_top_k)
    return rerank_candidates(query, candidates, reranker, top_k)


def print_result(result: dict[str, Any]) -> None:
    print(f"\nreranked_rank: {result['reranked_rank']}")
    print(f"original_rank: {result['original_rank']}")
    print(f"original_similarity_score: {result['original_similarity_score']:.4f}")
    print(f"rerank_score: {result['rerank_score']:.4f}")
    print(f"chunk_id: {result['chunk_id']}")
    print(f"source_file: {result['source_file']}")
    print(f"page_start-page_end: {result['page_start']}-{result['page_end']}")
    print("text:")
    print(str(result["text"]).strip())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    index, metadata = load_index_and_metadata(args.index_dir)
    embedding_model = load_embedding_model(args.embedding_model, args.allow_model_download)
    reranker = load_reranker(args.reranker_model, args.allow_model_download)
    results = rerank_query(args.query, index, metadata, embedding_model, reranker, args.faiss_top_k, args.top_k)
    for result in results:
        print_result(result)


if __name__ == "__main__":
    main()
