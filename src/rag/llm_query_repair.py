"""LLM fallback for heavy misspellings the local normalizer leaves unresolved.

The LLM may only fix spelling. Its output is accepted only if a local,
deterministic check confirms that every word still maps to the user's own word
(same word count, each word within a small edit distance; a fused question
particle such as "binilirmi" -> "binilir mi" may be split off).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .query_normalization import (
    PROTECTED_WORDS,
    TOKEN_RE,
    QueryVocab,
    fold_diacritics,
    has_known_stem,
    load_query_vocab,
    osa_distance,
    turkish_lower,
)


REPAIR_MAX_OUTPUT_TOKENS = 200
QUESTION_PARTICLES = frozenset(word for word in PROTECTED_WORDS if word[0] == "m" and len(word) <= 8)

REPAIR_SYSTEM_PROMPT = """Sen Türkçe bir yazım düzeltme aracısın. Kullanıcının deprem ve afetlerle ilgili
sorusundaki yazım hatalarını düzeltirsin.

Kurallar:
- Yalnızca yazım hatalarını düzelt: eksik, fazla veya yer değiştirmiş harfler, eksik
  Türkçe karakterler (ı, ğ, ü, ş, ö, ç), bitişik yazılmış "mı/mi/mu/mü" soru eki.
- Kelime ekleme, kelime silme, eşanlamlı kelimeyle değiştirme ve soruyu yeniden yazma YASAKTIR.
- Doğru yazılmış kelimelere dokunma. Özel isim, marka ve kısaltmaları aynen bırak.
- Bir kelimenin hangi kelime olduğundan emin değilsen o kelimeyi OLDUĞU GİBİ bırak.
- Noktalama ve büyük/küçük harf düzenini koru.

Yalnızca şu JSON'u döndür:
{"corrected": "<düzeltilmiş soru>", "confident": true}
Sorunun ne sorduğundan emin değilsen "confident": false döndür.
"""


def unexplained_tokens(query: str, vocab: QueryVocab) -> list[str]:
    """Words the local layer cannot account for (unknown, not an inflection of a known stem)."""
    tokens: list[str] = []
    for index, match in enumerate(TOKEN_RE.finditer(query)):
        token = match.group()
        if index > 0 and token[0].isupper():
            continue  # likely a proper noun or brand
        if any(char.isupper() for char in token[1:]):
            continue  # acronym or brand
        word = turkish_lower(token)
        if word in vocab.words or word in PROTECTED_WORDS or has_known_stem(word, vocab):
            continue
        tokens.append(token)
    return tokens


def words_close(original: str, corrected: str) -> bool:
    a, b = fold_diacritics(turkish_lower(original)), fold_diacritics(turkish_lower(corrected))
    limit = max(2, math.ceil(0.4 * max(len(a), len(b))))
    return osa_distance(a, b, limit) <= limit


def is_spelling_only(original: str, corrected: str) -> bool:
    """Deterministic check that corrected is word-by-word a spelling fix of original."""
    source = TOKEN_RE.findall(original)
    target = TOKEN_RE.findall(corrected)
    i = j = 0
    while i < len(source) and j < len(target):
        if (j + 1 < len(target) and turkish_lower(target[j + 1]) in QUESTION_PARTICLES
                and words_close(source[i], target[j] + target[j + 1])
                and not (i + 1 < len(source) and turkish_lower(source[i + 1]) in QUESTION_PARTICLES)):
            i, j = i + 1, j + 2  # fused particle split off: "binilirmi" -> "binilir mi"
        elif words_close(source[i], target[j]):
            i, j = i + 1, j + 1
        else:
            return False
    return i == len(source) and j == len(target)


def parse_repair_response(raw_text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload.get("corrected"), str) or not isinstance(payload.get("confident"), bool):
        return None
    return payload


def repair_query(query: str, client: Any, model: str, vocab: QueryVocab | None = None) -> dict[str, Any]:
    """Ask the LLM for a spelling-only fix when local normalization leaves unknown words.

    Returns {"repaired": str, "status": ..., "unexplained": [...]}; status is one of
    not_needed, accepted, unchanged, not_confident, rejected_rewrite, invalid_response.
    """
    vocab = vocab or load_query_vocab()
    unexplained = unexplained_tokens(query, vocab)
    if not unexplained:
        return {"repaired": query, "status": "not_needed", "unexplained": []}

    from .answerability import extract_response_text

    response = client.responses.create(
        model=model,
        instructions=REPAIR_SYSTEM_PROMPT,
        input=query,
        max_output_tokens=REPAIR_MAX_OUTPUT_TOKENS,
        temperature=0,
    )
    payload = parse_repair_response(extract_response_text(response))
    result = {"repaired": query, "unexplained": unexplained}
    if payload is None:
        return {**result, "status": "invalid_response"}
    corrected = payload["corrected"].strip()
    if not payload["confident"]:
        return {**result, "status": "not_confident", "proposed": corrected}
    if corrected == query:
        return {**result, "status": "unchanged"}
    if not is_spelling_only(query, corrected):
        return {**result, "status": "rejected_rewrite", "proposed": corrected}
    return {**result, "repaired": corrected, "status": "accepted"}
