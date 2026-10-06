"""'Did you mean…?' suggestions for questions the answerability gate refused.

The LLM proposes up to three explicit questions the user may have meant. It never
answers and never picks one: the user chooses, and only the chosen question is
sent through the normal pipeline. Output is validated locally; anything malformed
yields no suggestions, so the CLI falls back to the generic refusal hint.
"""

from __future__ import annotations

import json
import re
from typing import Any


MAX_OPTIONS = 3
MAX_OPTION_LENGTH = 150
CLARIFY_MAX_OUTPUT_TOKENS = 250

CLARIFY_SYSTEM_PROMPT = """Sen DARG deprem ve afet bilgi asistanının soru netleştirme yardımcısısın.
Kullanıcının sorusu mevcut kaynaklarla cevaplanamadı. Bunun nedeni sorunun belirsiz,
eksik ya da alışılmadık biçimde ifade edilmiş olması olabilir.

Görevin: Kullanıcının kastetmiş olabileceği en fazla 3 açık ve kısa soru önermek.

Kurallar:
- Önerilerin her biri kullanıcının sorusunun KENDİSİ olmalı, yalnızca daha açık ve standart
  bir Türkçeyle yazılmış hali. Alt soru, yeni ayrıntı, yeni durum veya yeni konu ekleme.
- İlk öneri, sorunun en olası anlamının açık ifadesi olsun.
- Bir kelime veya ifade gerçekten birden fazla anlama gelebiliyorsa, her anlam için ayrı bir
  öneri yaz (ör. belirsiz bir durum ifadesi "enkaz altında" ya da "deprem anında" anlamına
  gelebiliyorsa ikisini de öner). Anlam tek ise tek öneri yeterlidir.
- Her öneri tek bir Türkçe soru olmalı ve soru işaretiyle bitmeli.
- Soru deprem, afet, acil durum veya hazırlıkla ilgili değilse boş liste döndür.
- Makul bir yorum yoksa boş liste döndür.

Yalnızca şu JSON'u döndür:
{"options": ["<soru 1>", "<soru 2>"]}
"""


def parse_options(raw_text: str, question: str) -> list[str]:
    """Validate the LLM output; return [] for anything unexpected."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    options = payload.get("options")
    if not isinstance(options, list):
        return []
    seen = {question.strip().casefold()}
    valid: list[str] = []
    for option in options:
        if not isinstance(option, str):
            continue
        text = " ".join(option.split())
        if not text.endswith("?") or len(text) > MAX_OPTION_LENGTH or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        valid.append(text)
    return valid[:MAX_OPTIONS]


def suggest_clarifications(question: str, client: Any, model: str) -> list[str]:
    from .answerability import extract_response_text

    response = client.responses.create(
        model=model,
        instructions=CLARIFY_SYSTEM_PROMPT,
        input=question,
        max_output_tokens=CLARIFY_MAX_OUTPUT_TOKENS,
        temperature=0,
    )
    return parse_options(extract_response_text(response), question)
