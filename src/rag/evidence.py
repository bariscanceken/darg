"""Evidence-first answering: select verbatim supporting sentences before writing the answer.

The user's review found answers padded with off-question material, advice for the
wrong situation, and past events or other places presented as current guidance. Here
the model first quotes, per source, only the sentences that directly answer the
question for its situation and place, labelling what each quote is about. Quotes are
validated locally (they must appear in the cited passage); quotes marked as past-event
records, other places or other situations are dropped unless the question asks for
them. Quotes about a wider area than the place asked (Türkiye, the world, the whole
earthquake region) are kept but marked, so the answer states their real scope instead
of attributing them to the place. The answer is then written from the surviving quotes
only; with no quote left the pipeline refuses.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .query_normalization import turkish_lower


EVIDENCE_MAX_OUTPUT_TOKENS = 900
MIN_QUOTE_CHARS = 20

EVIDENCE_SYSTEM_PROMPT = """Bir deprem bilgi asistanı için kanıt seçicisin. Sana bir soru ve numaralı AFAD kaynak
parçaları ([Kaynak N]) verilecek. Cevap YAZMA; yalnızca kanıt seç.

1. Sorunun durumunu belirle: hazırlık / deprem sırasında / deprem sonrasında / enkaz altında /
   genel bilgi / geçmiş olaylar. Soru belirli bir yer içeriyorsa yeri de not et.
2. Kaynaklardan soruyu DOĞRUDAN cevaplayan cümleleri KELİMESİ KELİMESİNE alıntıla.
   - Tablo satırları ve listeler de alıntılanabilir (ör. ilçe adları, alan adları, sayılar); metni birebir kopyala.
   - En somut bilgiyi tercih et: isim, yer, sayı, liste. Soruyu kısmen cevaplayan somut bilgi de kanıttır.
   - Yalnızca konuya değinen, başka bir soruyu cevaplayan veya tanım/arka plan veren cümleleri alma.
   - Her alıntı için "about" alanına alıntının neyi anlattığını yaz:
     "same_situation" (sorunun durumu için tavsiye/bilgi),
     "other_situation" (başka durum için: ör. soru deprem sırasındayken deprem sonrası tavsiyesi,
       ya da enkaz altındaki kişiye yönelik tavsiye),
     "past_event" (geçmişteki belirli bir depremden sonra yapılanlar/olanlar),
     "other_place" (soruda geçmeyen başka bir il/bölge hakkında),
     "wider_area" (soru belirli bir yer içeriyorken alıntı o yeri adıyla anmıyor; Türkiye, dünya veya
       birçok ili kapsayan bir bölge hakkında; ör. soru Hatay'ı sorarken "ülkemizde 50783 kişi"),
     "general_fact" (durumdan bağımsız genel bilgi).
3. Uygun cümle yoksa boş liste döndür.

Yalnızca şu JSON'u döndür:
{"question_situation": "<durum>", "evidence": [{"source": <N>, "quote": "<birebir alıntı>", "about": "<etiket>"}]}
"""

ANSWER_FROM_EVIDENCE_INSTRUCTIONS = """
Kanıt kuralları:
- Cevabı YALNIZCA aşağıdaki "Seçilmiş kanıtlar" listesindeki alıntılara dayanarak yaz; başka bilgi ekleme.
- Her alıntıyı [Kaynak N] numarasıyla göster.
- Soruyu doğrudan ve kısa cevapla; kanıtta olmayan madde, gerekçe veya sonuç cümlesi ekleme.
- Geçmiş olaylara ait alıntıları kullanırsan geçmiş zamanla ve hangi olay olduğunu belirterek anlat.
- Kanıt soruyu yalnızca kısmen cevaplıyorsa kanıtın söylediğini ver ve neyin kaynaklarda bulunmadığını tek
  cümleyle belirt; bu durumda "bulamadım" demekle yetinme.
- Bilgi bir il planından veya rapordan geliyorsa ve metinde tarih/yıl varsa verinin yılını belirt.
- "(kapsam: daha geniş alan)" işaretli alıntıları sorulan yere ATFETME; kapsamını alıntıdaki gibi belirt
  (ör. "ülkemizde ...", "dünyada ...") ve sorulan yere özel bilginin kaynaklarda bulunmadığını söyle.
  Sorulan yer hakkında alıntıda olmayan hiçbir cümle kurma (ör. "X'te de ciddi kayıplar yaşandı").
- Sayıların kapsamını ve dönemini alıntıdakiyle aynen koru (dünya / ülke / il; yıl).
- Soruyu tekrar etme; doğrudan cevapla başla.
- Kanıtlardaki parantezli işaretleri ("(kapsam: ...)", "(doğrulanmış yerel bilgi)") cevaba yazma.
- "(doğrulanmış yerel bilgi)" işaretli kanıtları yalnızca soruyla ilgiliyse kullan; il planından gelenlerde
  planın yılını belirt.
- Kanıtta bir koşul varsa ("hasar yoksa", "eğer ...") cevabı o koşulla başlat; koşulsuz izin veya
  güvence cümlesi kurma.
- Cevabın girişinde durumu adlandırırsan kanıtların anlattığı durumu kullan; "deprem oldu", "deprem olduktan
  sonra" gibi ifadeler deprem SONRASIdır, "deprem sırasında" deme.
"""


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", turkish_lower(text)).strip()


def parse_evidence(raw_text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload.get("evidence"), list) else None


def allowed_labels(question_situation: str) -> set[str]:
    """Past events are kept only when the question itself is about past events."""
    labels = {"same_situation", "general_fact", "wider_area"}
    if "geçmiş" in turkish_lower(question_situation or ""):
        labels.add("past_event")
    return labels


def validate_evidence(payload: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep quotes that really occur in the cited chunk and carry an allowed label."""
    keep = allowed_labels(str(payload.get("question_situation", "")))
    texts = [normalise(str(chunk.get("text", ""))) for chunk in chunks]
    valid: list[dict[str, Any]] = []
    for item in payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        source, quote, about = item.get("source"), str(item.get("quote", "")).strip(), item.get("about")
        if not isinstance(source, int) or not 1 <= source <= len(chunks) or about not in keep:
            continue
        if len(quote) < MIN_QUOTE_CHARS or normalise(quote) not in texts[source - 1]:
            continue
        valid.append({"source": source, "quote": quote, "about": about})
    return valid


def mentions_place(quote: str, names: frozenset[str]) -> bool:
    lowered = turkish_lower(quote)
    return any(re.search(rf"(?<!\w){re.escape(name)}", lowered) for name in names)


def mark_wider_area(evidence: list[dict[str, Any]], names: frozenset[str]) -> list[dict[str, Any]]:
    """For place-specific questions, quotes that never name the place (or one of its districts) are wider-area."""
    return [item if mentions_place(item["quote"], names) else {**item, "about": "wider_area"} for item in evidence]


def select_evidence(question: str, context: str, chunks: list[dict[str, Any]], client: Any, model: str) -> dict[str, Any]:
    """Returns {"question_situation", "evidence": [validated quotes], "raw_count", "status"}."""
    from .answerability import extract_response_text

    try:
        response = client.responses.create(model=model, instructions=EVIDENCE_SYSTEM_PROMPT,
                                           input=f"Soru:\n{question}\n\nKaynaklar:\n{context}",
                                           max_output_tokens=EVIDENCE_MAX_OUTPUT_TOKENS, temperature=0)
        payload = parse_evidence(extract_response_text(response))
    except Exception:
        payload = None
    if payload is None:
        return {"question_situation": "", "evidence": [], "raw_count": 0, "status": "invalid"}
    evidence = validate_evidence(payload, chunks)
    return {"question_situation": payload.get("question_situation", ""), "evidence": evidence,
            "raw_count": len(payload.get("evidence", [])), "status": "ok" if evidence else "no_evidence"}


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    return "\n".join(f"- [Kaynak {item['source']}] \"{item['quote']}\""
                     + (" (kapsam: daha geniş alan)" if item.get("about") == "wider_area" else "")
                     + (" (doğrulanmış yerel bilgi)" if item.get("about") == "card" else "")
                     for item in evidence)
