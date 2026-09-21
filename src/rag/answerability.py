"""Answerability gate for DARG RAG generation."""

from __future__ import annotations

from typing import Any


ANSWERABLE = "ANSWERABLE"
NOT_ANSWERABLE = "NOT_ANSWERABLE"
REFUSAL_ANSWER = "Bu konuda mevcut kaynaklarda yeterli güvenilir bilgi bulamadım."

ANSWERABILITY_SYSTEM_PROMPT = """Sen DARG RAG sistemi için answerability denetleyicisisin.

Görevin kullanıcıya cevap yazmak değil, verilen AFAD context'inin soruyu
doğrudan cevaplamaya yeterli olup olmadığını belirlemektir.

Sadece şu iki etiketten birini döndür:
ANSWERABLE
NOT_ANSWERABLE

ANSWERABLE kriterleri sıkıdır:
- Context, sorunun ana iddiasını, kararını veya istenen bilgiyi açıkça desteklemelidir.
- Context'in soruyla birebir aynı cümleyi kullanması gerekmez.
- Aynı anlamı açıkça ifade eden veya sorunun ana kararını doğrudan çıkarmaya
  yeterli olan kaynak bilgisi ANSWERABLE kabul edilebilir.
- Context'in sadece aynı genel konudan bahsetmesi yeterli değildir.
- Farklı koşullardaki bilgileri birleştirmek gerekiyorsa NOT_ANSWERABLE döndür.
- Genel model bilgisiyle boşluk doldurmak gerekiyorsa NOT_ANSWERABLE döndür.
- Kaynakta belirli kişiye veya duruma verilen tavsiyeyi başka duruma uygulamak
  gerekiyorsa NOT_ANSWERABLE döndür.
- "Enkaz altında kalan kişi" için verilen tavsiye genel bina tahliye tavsiyesi değildir.
- Dış tehlike için verilen "yerinde sığınak" bilgisi genel deprem sonrası binada
  kalma tavsiyesi değildir.
"""


def build_answerability_prompt(question: str, context: str) -> str:
    """Build the answerability check input."""
    return f"""Soru:
{question.strip()}

AFAD context:
{context.strip()}

Karar:
"""


def extract_response_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API response."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def parse_answerability_label(raw_text: str) -> str:
    """Parse an answerability model response, failing closed when uncertain."""
    normalized = raw_text.strip().upper()
    first_line = normalized.splitlines()[0].strip() if normalized else ""
    first_token = first_line.strip("` .:;,-")

    if first_token == ANSWERABLE:
        return ANSWERABLE
    if first_token == NOT_ANSWERABLE:
        return NOT_ANSWERABLE
    return NOT_ANSWERABLE


def check_answerability(client: Any, model: str, question: str, context: str) -> dict[str, str]:
    """Call the answerability gate and return a parsed decision."""
    response = client.responses.create(
        model=model,
        instructions=ANSWERABILITY_SYSTEM_PROMPT,
        input=build_answerability_prompt(question, context),
        max_output_tokens=16,
        temperature=0,
    )
    raw_text = extract_response_text(response)
    label = parse_answerability_label(raw_text)
    return {
        "label": label,
        "raw": raw_text,
    }


def is_answerable(label: str) -> bool:
    """Return True only for the exact ANSWERABLE label."""
    return label == ANSWERABLE
