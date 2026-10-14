"""Second-stage LLM reranking of the cross-encoder's top candidates.

The multilingual MiniLM cross-encoder often prefers short glossary definitions over
the passage that actually answers a Turkish question (e.g. "depremde asansör
kullanılır mı?" vs. "... asansöre binmemeli ..."). The top LLM_RERANK_POOL
candidates are shown to an LLM, which returns the passage numbers that best answer
the question. The output is validated locally; anything malformed falls back to the
cross-encoder order, so the LLM can reorder candidates but never invent or drop the
pipeline's results.
"""

from __future__ import annotations

import json
import re
from typing import Any


LLM_RERANK_POOL = 20
LLM_RERANK_MAX_OUTPUT_TOKENS = 120

LLM_RERANK_SYSTEM_PROMPT = """Sen bir arama sonucu sıralayıcısısın. Sana bir kullanıcı sorusu ve numaralı
AFAD metin parçaları verilecek.

Görevin: Soruyu DOĞRUDAN cevaplayan parçaları en alakalıdan başlayarak sıralamak.
- Sorunun durumuna dikkat et: deprem sırasında / sonrasında / enkaz altında / belirli bir il.
- Yalnızca konuya değinen, terim tanımı veren veya farklı bir durumu anlatan parçalar
  alt sıralarda olmalı.
- Soruyu cevaplamayan parçaları listeye hiç ekleme.

Yalnızca şu JSON'u döndür: {"ranking": [en alakalı parça numarası, ikinci, ...]}
"""


def build_rerank_input(question: str, candidates: list[dict[str, Any]]) -> str:
    blocks = [f"[{number}] {' '.join(str(candidate['text']).split())}" for number, candidate in enumerate(candidates, start=1)]
    return f"Soru: {question}\n\nParçalar:\n" + "\n\n".join(blocks)


def parse_ranking(raw_text: str, count: int) -> list[int] | None:
    """Return 0-based indices in LLM order, or None if the output is unusable."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        ranking = json.loads(match.group()).get("ranking")
    except json.JSONDecodeError:
        return None
    if not isinstance(ranking, list):
        return None
    order: list[int] = []
    for value in ranking:
        if isinstance(value, int) and 1 <= value <= count and value - 1 not in order:
            order.append(value - 1)
    return order


def llm_rerank(question: str, ranked: list[dict[str, Any]], client: Any, model: str, top_k: int) -> list[dict[str, Any]]:
    """Reorder the cross-encoder's top candidates with an LLM; fall back to the input order."""
    from .answerability import extract_response_text

    pool = ranked[:LLM_RERANK_POOL]
    if len(pool) <= 1:
        return ranked[:top_k]
    try:
        response = client.responses.create(
            model=model,
            instructions=LLM_RERANK_SYSTEM_PROMPT,
            input=build_rerank_input(question, pool),
            max_output_tokens=LLM_RERANK_MAX_OUTPUT_TOKENS,
            temperature=0,
        )
        order = parse_ranking(extract_response_text(response), len(pool))
    except Exception:  # network or API failure: keep the deterministic order
        order = None
    if not order:
        return ranked[:top_k]
    chosen = [pool[index] for index in order]
    # Fill remaining slots in cross-encoder order if the LLM kept fewer than top_k.
    chosen += [candidate for candidate in ranked if candidate not in chosen][: max(0, top_k - len(chosen))]
    for rank, candidate in enumerate(chosen[:top_k], start=1):
        candidate["llm_rank"] = rank
    return chosen[:top_k]
