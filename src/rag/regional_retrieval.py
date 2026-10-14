"""Region-aware retrieval for questions that name a province or district.

When a location is detected, a second candidate pool is built from the chunks of
that province's IRAP (and, for a district, chunks mentioning the district), scored
with the same E5 query embedding. The general FAISS Top-50 and the regional pool
are reranked together by the same cross-encoder, and IRAP chunks of *other*
provinces are dropped. Reserving Top-5 slots for regional chunks was evaluated on
regional_dev_v1 (0/1/2 slots -> Hit@5 16/15/15 of 20) and was removed.
Questions without a location use the unchanged frozen pipeline.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.evaluate_reranked_v2 import retrieve_v2_candidates
from src.retrieval.rerank import rerank_candidates
from src.retrieval.search import embed_query

from .hybrid_retrieval import BM25_TOP_K, BM25_TOP_K_WITH_LLM, diversify, lexical_rows
from .llm_rerank import llm_rerank
from .location import IRAP_MAP_FILE, Location
from .query_normalization import turkish_lower


GENERAL_TOP_K = 50
REGIONAL_TOP_K = 20


_VECTOR_CACHE: dict[int, np.ndarray] = {}


@lru_cache(maxsize=1)
def irap_province_map() -> dict[str, str]:
    return json.loads(Path(IRAP_MAP_FILE).read_text(encoding="utf-8"))


def index_vectors(index: Any) -> np.ndarray:
    """All passage vectors of a flat FAISS index, reconstructed once per loaded index."""
    if id(index) not in _VECTOR_CACHE:
        _VECTOR_CACHE[id(index)] = index.reconstruct_n(0, index.ntotal)
    return _VECTOR_CACHE[id(index)]


def candidate_from_row(record: dict[str, Any], row: int, rank: int, score: float) -> dict[str, Any]:
    """Same shape as retrieve_v2_candidates() output."""
    return {
        "original_rank": rank,
        "original_similarity_score": score,
        "faiss_row": row,
        "chunk_id": record["chunk_id"],
        "source_file": record["source_file"],
        "relative_path": record.get("relative_path", ""),
        "organization": record.get("organization", "AFAD"),
        "page_start": record["page_start"],
        "page_end": record["page_end"],
        "chunk_index": record.get("chunk_index"),
        "source_relevances": record.get("source_relevances", []),
        "text": str(record.get("text", "")),
    }


def mentions(name: str, text: str) -> bool:
    return re.search(r"(?<![\wçğıöşü])" + re.escape(turkish_lower(name)) + r"(?![\wçğıöşü])", turkish_lower(text)) is not None


def regional_rows(location: Location, metadata: list[dict[str, Any]], irap_map: dict[str, str]) -> list[int]:
    """Candidate rows for the regional pool.

    Another province's IRAP is never regional evidence, even if it mentions the
    place (e.g. Sivas' plan naming Malatya, or the Ceyhan river in Kahramanmaraş).
    With a district, chunks naming the district come first; otherwise the
    province's whole IRAP; without an IRAP, non-IRAP documents naming the place.
    """
    allowed = [row for row, record in enumerate(metadata) if irap_map.get(record["source_file"]) in (None, location.province)]
    if location.district:
        named = [row for row in allowed if mentions(location.district, metadata[row]["text"])]
        if named:
            return named
    own_irap = [row for row in allowed if irap_map.get(metadata[row]["source_file"]) == location.province]
    if own_irap:
        return own_irap
    return [row for row in allowed if location.province and mentions(location.province, metadata[row]["text"])]


def retrieve_enhanced(
    question: str,
    location: Location | None,
    index: Any,
    metadata: list[dict[str, Any]],
    embedding_model: Any,
    reranker: Any,
    top_k: int = 5,
    lexical: bool = True,
    max_per_document: int | None = None,
    llm_reranker: tuple[Any, str] | None = None,
    bm25_top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Dense Top-50 (+ BM25-F5 pool) (+ regional pool), one rerank, optional diversity cap.

    With a location, IRAP chunks of other provinces are removed from every pool.
    """
    irap_map = irap_province_map()

    def allowed(source_file: str) -> bool:
        return location is None or irap_map.get(source_file) in (None, location.province)

    pool: dict[str, dict[str, Any]] = {}
    for candidate in retrieve_v2_candidates(question, index, metadata, embedding_model, GENERAL_TOP_K):
        if allowed(candidate["source_file"]):
            pool[candidate["chunk_id"]] = candidate
    rank = GENERAL_TOP_K
    if lexical:
        lexical_k = bm25_top_k or (BM25_TOP_K_WITH_LLM if llm_reranker is not None else BM25_TOP_K)
        for row, score in lexical_rows(question, metadata, lexical_k):
            if allowed(metadata[row]["source_file"]):
                rank += 1
                pool.setdefault(metadata[row]["chunk_id"], candidate_from_row(metadata[row], row, rank, score))
    regional_ids: set[str] = set()
    if location is not None and not location.is_ambiguous:
        rows = regional_rows(location, metadata, irap_map)
        if rows:
            query_vector = embed_query(embedding_model, question)[0]
            scores = index_vectors(index)[rows] @ query_vector
            for i in np.argsort(-scores)[:REGIONAL_TOP_K]:
                rank += 1
                candidate = candidate_from_row(metadata[rows[i]], rows[i], rank, float(scores[i]))
                regional_ids.add(candidate["chunk_id"])
                pool.setdefault(candidate["chunk_id"], candidate)

    ranked = rerank_candidates(question, list(pool.values()), reranker, top_k=len(pool))
    if llm_reranker is not None:
        client, model = llm_reranker
        final = llm_rerank(question, ranked, client, model, top_k)
    else:
        final = diversify(ranked, top_k, max_per_document) if max_per_document else ranked[:top_k]
    for candidate in final:
        candidate["regional"] = candidate["chunk_id"] in regional_ids
    return final


def retrieve_regional(question: str, location: Location, index: Any, metadata: list[dict[str, Any]],
                      embedding_model: Any, reranker: Any, top_k: int = 5) -> list[dict[str, Any]]:
    """Faz 1 configuration (regional pool only), kept for reproducing regional_dev_v1."""
    return retrieve_enhanced(question, location, index, metadata, embedding_model, reranker, top_k,
                             lexical=False, max_per_document=None)
