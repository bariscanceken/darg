"""Lexical (BM25-F5) candidate pool used next to dense retrieval.

E5 retrieval summarises a chunk's overall meaning, so a decisive word inside a
list ("... merdivenden inmemeli, asansöre binmemeli ...") can be lost: for
"depremde asansör kullanılır mı?" the answer chunk sits at FAISS rank 87 and never
reaches the reranker. A BM25 index with F5 stemming (first five letters, a standard
light stemmer for Turkish IR) catches such words (rank 12). Both pools are reranked
together by the same cross-encoder.

Ablations on dev + holdout + regional_dev_v1 (74 questions, LLM judge), Hit@5:
baseline 63; BM25 top-30 62; BM25 top-10 64; BM25 top-30 + LLM rerank 68 (no
question lost). Hence BM25_TOP_K = 10 without the LLM reranker and 30 with it.
A per-document diversity cap (diversify) did not change Hit@5 and lowered regional
P@5 (0.39 -> 0.28); it is kept as an option but disabled.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .query_normalization import turkish_lower


BM25_TOP_K = 10
# With the LLM reranker a wider lexical pool pays off (run 2: Hit@5 68/74, no loss vs baseline).
BM25_TOP_K_WITH_LLM = 30
MAX_PER_DOCUMENT = 2  # optional cap; disabled by default (see docstring)
STEM_LENGTH = 5
BM25_TOKEN_RE = re.compile(r"[a-zçğıöşüâîû0-9]+")


def f5_tokens(text: str) -> list[str]:
    return [word[:STEM_LENGTH] for word in BM25_TOKEN_RE.findall(turkish_lower(text))]


_BM25_CACHE: dict[int, BM25Okapi] = {}


def bm25_index(metadata: list[dict[str, Any]]) -> BM25Okapi:
    """Built once per loaded metadata list (~2 s for 24k chunks)."""
    if id(metadata) not in _BM25_CACHE:
        _BM25_CACHE[id(metadata)] = BM25Okapi([f5_tokens(str(record["text"])) for record in metadata])
    return _BM25_CACHE[id(metadata)]


def lexical_rows(question: str, metadata: list[dict[str, Any]], top_k: int = BM25_TOP_K) -> list[tuple[int, float]]:
    scores = bm25_index(metadata).get_scores(f5_tokens(question))
    best = np.argsort(-scores)[:top_k]
    return [(int(row), float(scores[row])) for row in best if scores[row] > 0]


def diversify(ranked: list[dict[str, Any]], top_k: int, max_per_document: int = MAX_PER_DOCUMENT) -> list[dict[str, Any]]:
    """Take candidates in rerank order, at most max_per_document per source file."""
    chosen: list[dict[str, Any]] = []
    per_document: dict[str, int] = {}
    for candidate in ranked:
        source = candidate["source_file"]
        if per_document.get(source, 0) >= max_per_document:
            continue
        chosen.append(candidate)
        per_document[source] = per_document.get(source, 0) + 1
        if len(chosen) == top_k:
            break
    return chosen
