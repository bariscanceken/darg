"""Hybrid dense + BM25 retrieval over the v2 AFAD index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .rerank import load_embedding_model, load_reranker
from .search import DEFAULT_MODEL_NAME, embed_query, read_metadata


DEFAULT_INDEX_DIR = Path("data/indexes/mobile_v2")
DEFAULT_INDEX_FILE = "afad_e5_small_v2.faiss"
DEFAULT_METADATA_FILE = "metadata_v2.jsonl"
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_DENSE_TOP_K = 50
DEFAULT_BM25_TOP_K = 50
DEFAULT_FINAL_TOP_K = 5
TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class HybridResources:
    index: faiss.Index
    metadata: list[dict[str, Any]]
    bm25: BM25Okapi
    tokenized_corpus: list[list[str]]
    embedding_model: Any
    reranker: CrossEncoder
    index_path: Path
    metadata_path: Path


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_v2_index_and_metadata(
    index_dir: Path = DEFAULT_INDEX_DIR,
    index_file: str = DEFAULT_INDEX_FILE,
    metadata_file: str = DEFAULT_METADATA_FILE,
) -> tuple[faiss.Index, list[dict[str, Any]], Path, Path]:
    index_path = index_dir / index_file
    metadata_path = index_dir / metadata_file
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"V2 index files not found: {index_path}, {metadata_path}")

    index = faiss.read_index(str(index_path))
    metadata = read_metadata(metadata_path)
    if index.ntotal != len(metadata):
        raise ValueError(f"V2 index rows ({index.ntotal}) do not match metadata rows ({len(metadata)})")
    return index, metadata, index_path, metadata_path


def build_bm25(metadata: list[dict[str, Any]]) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized_corpus = [tokenize(str(record.get("text", ""))) for record in metadata]
    return BM25Okapi(tokenized_corpus), tokenized_corpus


def load_hybrid_resources(
    index_dir: Path = DEFAULT_INDEX_DIR,
    index_file: str = DEFAULT_INDEX_FILE,
    metadata_file: str = DEFAULT_METADATA_FILE,
    embedding_model_name: str = DEFAULT_MODEL_NAME,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    allow_model_download: bool = False,
) -> HybridResources:
    index, metadata, index_path, metadata_path = load_v2_index_and_metadata(index_dir, index_file, metadata_file)
    bm25, tokenized_corpus = build_bm25(metadata)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)
    return HybridResources(
        index=index,
        metadata=metadata,
        bm25=bm25,
        tokenized_corpus=tokenized_corpus,
        embedding_model=embedding_model,
        reranker=reranker,
        index_path=index_path,
        metadata_path=metadata_path,
    )


def metadata_candidate(record: dict[str, Any], row_index: int) -> dict[str, Any]:
    return {
        "faiss_row": row_index,
        "chunk_id": record["chunk_id"],
        "source_file": record["source_file"],
        "relative_path": record.get("relative_path", ""),
        "organization": record.get("organization", "AFAD"),
        "page_start": record["page_start"],
        "page_end": record["page_end"],
        "chunk_index": record.get("chunk_index"),
        "source_relevances": record.get("source_relevances", []),
        "text": str(record.get("text", "")),
        "faiss_rank": None,
        "faiss_score": None,
        "bm25_rank": None,
        "bm25_score": None,
        "retrieved_by": "",
    }


def dense_retrieve(
    query: str,
    resources: HybridResources,
    top_k: int = DEFAULT_DENSE_TOP_K,
) -> list[dict[str, Any]]:
    k = min(max(top_k, 1), len(resources.metadata))
    query_embedding = embed_query(resources.embedding_model, query)
    scores, indices = resources.index.search(query_embedding, k)

    candidates: list[dict[str, Any]] = []
    row_indices: list[int] = []
    for rank, (score, row_index) in enumerate(zip(scores[0], indices[0]), start=1):
        row_index = int(row_index)
        if row_index < 0:
            continue
        row_indices.append(row_index)
        candidate = metadata_candidate(resources.metadata[row_index], row_index)
        candidate["faiss_rank"] = rank
        candidate["faiss_score"] = float(score)
        candidate["retrieved_by"] = "dense"
        candidates.append(candidate)

    if len(candidates) != k:
        raise RuntimeError(f"Expected {k} dense candidates, got {len(candidates)}")
    if len(set(row_indices)) != len(row_indices):
        raise RuntimeError(f"Expected distinct dense FAISS rows, got duplicates in Top-{k}")
    return candidates


def bm25_retrieve(
    query: str,
    resources: HybridResources,
    top_k: int = DEFAULT_BM25_TOP_K,
) -> list[dict[str, Any]]:
    k = min(max(top_k, 1), len(resources.metadata))
    scores = resources.bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:k]

    candidates: list[dict[str, Any]] = []
    for rank, row_index in enumerate(top_indices, start=1):
        row_index = int(row_index)
        candidate = metadata_candidate(resources.metadata[row_index], row_index)
        candidate["bm25_rank"] = rank
        candidate["bm25_score"] = float(scores[row_index])
        candidate["retrieved_by"] = "bm25"
        candidates.append(candidate)
    return candidates


def union_candidates(
    dense_candidates: list[dict[str, Any]],
    bm25_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    union: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for candidate in dense_candidates:
        chunk_id = str(candidate["chunk_id"])
        union[chunk_id] = dict(candidate)
        order.append(chunk_id)

    for candidate in bm25_candidates:
        chunk_id = str(candidate["chunk_id"])
        if chunk_id not in union:
            union[chunk_id] = dict(candidate)
            order.append(chunk_id)
            continue
        existing = union[chunk_id]
        existing["bm25_rank"] = candidate["bm25_rank"]
        existing["bm25_score"] = candidate["bm25_score"]
        existing["retrieved_by"] = "dense+bm25"

    candidates = [union[chunk_id] for chunk_id in order]
    if len(candidates) != len({str(candidate["chunk_id"]) for candidate in candidates}):
        raise RuntimeError("Duplicate chunk_id found in hybrid candidate union")
    return candidates


def rerank_hybrid_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: CrossEncoder,
    top_k: int = DEFAULT_FINAL_TOP_K,
) -> list[dict[str, Any]]:
    pairs = [(query, str(candidate["text"])) for candidate in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        updated = dict(candidate)
        updated["rerank_score"] = float(score)
        scored.append(updated)

    scored.sort(
        key=lambda candidate: (
            -float(candidate["rerank_score"]),
            candidate["faiss_rank"] if candidate["faiss_rank"] is not None else 999999,
            candidate["bm25_rank"] if candidate["bm25_rank"] is not None else 999999,
            str(candidate["chunk_id"]),
        )
    )
    final = scored[: max(top_k, 1)]
    for rank, candidate in enumerate(final, start=1):
        candidate["final_rank"] = rank
    return final


def hybrid_retrieve(
    query: str,
    resources: HybridResources,
    dense_top_k: int = DEFAULT_DENSE_TOP_K,
    bm25_top_k: int = DEFAULT_BM25_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dense_candidates = dense_retrieve(query, resources, dense_top_k)
    bm25_candidates = bm25_retrieve(query, resources, bm25_top_k)
    union = union_candidates(dense_candidates, bm25_candidates)
    final_results = rerank_hybrid_candidates(query, union, resources.reranker, final_top_k)
    return dense_candidates, bm25_candidates, union, final_results
