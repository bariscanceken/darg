"""DARG pipeline v3: answer like a helpful assistant, verify against AFAD sources.

Pipelines v1/v2 answer ONLY from verbatim AFAD quotes and refuse otherwise. That keeps answers
grounded, but every missed passage becomes "bulamadım", and the filters added to keep that rule
became the fragile chain users experienced. A general chat model answers the same questions
fluently from its own knowledge. v3 combines the two:

1. understand_question (pipeline v2): spelling fix, type, situation, setting, search rewrites,
   and whether the question is about earthquakes/disasters at all.
2. retrieve_wide (pipeline v2) over the section-aware v5 index + verified province-card lines.
3. One gpt-4o call writes a helpful answer. AFAD passages come first; general knowledge may fill
   gaps. Every sentence states its basis: AFAD source(s) with a verbatim quote, or "genel".
4. Code checks every AFAD quote in its passage (src/rag/evidence.py: locate_quote). Sentences
   whose quote cannot be found are shown as general knowledge, never as AFAD-backed.
Each sentence shows its basis inline ([Kaynak N] or "(genel bilgi)").

Hybrid decision (the project form's "hibrit karar ajanı": vector database or web search): when the
AFAD answer has no AFAD-backed sentence, leans on general knowledge, or the question asks for
current information (2023, "şu an", which hospital, address...), an official-source web search
(gov.tr, Kızılay) is run and the answer is rewritten with the web results as extra numbered sources;
web quotes are verified the same way and shown with their URL.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .answerability import extract_response_text
from .evidence import locate_quote
from .pipeline_v2 import (
    card_passages,
    format_passages,
    parse_json,
    retrieve_wide,
    understand_question,
)
from .query_normalization import turkish_lower
from .situation import AFTER, passage_situation
from .source_routing import policy_for

logger = logging.getLogger("darg.pipeline")

# Writes and checks the answer. gpt-4o until 26 Oct 2026 (Final Test v2, expert-graded 40/42, was run with gpt-4o);
# switched by the project owner for cost after model_check_v1: gpt-4.1-mini 29/32 correct, ~6x cheaper per answer
# (one first-aid answer, M13, awaits an expert's safety opinion).
ANSWER_MODEL = "gpt-4.1-mini"
PASSAGES = 8
# Sentences carry verbatim quotes, so long answers are long; 900 cut cheaper (more verbose) models mid-JSON (25 Oct).
ANSWER_MAX_OUTPUT_TOKENS = 1400
NO_ANSWER = ("Bu soruya şu anda güvenilir bir cevap üretemedim. Acil bir durumdaysanız hemen 112 Acil Çağrı "
             "Merkezi'ni arayın; sorunuzu biraz farklı yazarak tekrar deneyebilirsiniz.")
OFF_TOPIC_ANSWER = ("DARG deprem ve afetlerle ilgili soruları cevaplıyor; bu soru bu kapsamın dışında görünüyor. "
                    "Depremle ilgili bir sorunuz varsa yardımcı olabilirim.")

SCOPE_PROMPT = """Bir deprem bilgi asistanına gelen soru bu asistanın kapsamında mı?
KAPSAM İÇİ: deprem, afet ve acil durumlar; afete hazırlık; ilk yardım ve psikolojik destek; toplanma/barınma
alanları; binaların depreme dayanıklılığı veya güvenliği (kişinin kendi binası dahil); deprem tahmini soruları;
bir il/ilçenin deprem riski, fayları, geçmiş depremleri; afet sonrası yardım ve haklar.
KAPSAM DIŞI: depremle ilgisi olmayan sağlık, yemek, para, yazılım, ulaşım, genel kültür soruları.
Örnekler:
"Binam depremde yıkılır mı?" -> true
"Büyük deprem ne zaman olacak?" -> true
"Depremden sonra panik atak geçiren birine nasıl yardım ederim?" -> true
"Grip olunca hangi ilacı içmeliyim?" -> false
"Python'da liste nasıl sıralanır?" -> false
"İstanbul'dan Ankara'ya en hızlı nasıl giderim?" -> false
Yalnızca JSON döndür: {"in_scope": true} ya da {"in_scope": false}"""
# Words that always mean "about earthquakes / disasters"; the model is not asked for these questions.
IN_SCOPE_RE = re.compile(r"deprem|afet|enkaz|artçı|artci|sarsıntı|sarsinti|tsunami|toplanma alan|barınma|fay\b|"
                         r"fay hat|afad|tahliye|yıkıl|yikil|hasar|çök-kapan|cok-kapan|112|binam|binanın|binanin|bina güven|"
                         r"depreme dayan|yaralı|yarali|kanama|kırık|kirik|kırığ|kirig|bayıl|bayil|ilk ?yardım|ilk ?yardim|"
                         r"kurtarma|hastane|acil servis|sağlık ocağı|saglik ocagi|ambulans")

ANSWER_PROMPT = """Sen DARG'sın: Türkiye'deki vatandaşlara deprem konusunda yardım eden bir asistan.
Soruyu bilgili, sakin ve yardımcı bir uzman gibi, doğrudan cevapla.

Kaynak kuralları:
- Önce verilen AFAD kaynak parçalarını, sonra "Web (resmî kaynak)" parçalarını kullan. Her parçanın başındaki "belge › bölüm" başlığı o bilginin hangi
  durum (deprem sırasında / sonrasında / enkaz altında / hazırlık) ve yer için yazıldığını gösterir.
- Kaynaklar soruyu tam karşılamıyorsa, doğru olduğundan emin olduğun genel bilgiyle tamamlayabilirsin.
- Her cümlenin dayanağını belirt: bir kaynak parçasından (AFAD ya da resmî web) geliyorsa kaynak numarasını ve o kaynaktan KELİMESİ KELİMESİNE
  bir alıntıyı ver; kendi genel bilginse "basis": "genel" yaz. Kaynakta olmayan bir şeyi kaynağa dayandırma.

Güvenlik kuralları:
- Deprem sırasında ve sonrasında yapılacakları karıştırma; tavsiyenin hangi duruma ait olduğunu belirt.
- Bazı parçaların başında "(durum: …)" etiketi var (başlıktan çıkarıldı). Etiket sorunun durumundan farklıysa o
  tavsiyeyi sorunun durumuna aitmiş gibi sunma; kullanacaksan durumunu açıkça yaz ("deprem sonrasında …").
- Web parçalarında tarih varsa cümlede belirt ("… (Mart 2024 itibarıyla)"); aynı konuda farklı tarihli rakamlar
  varsa en yenisini ver. Farklı olayların (ör. 6 Şubat ve 20 Şubat 2023 depremleri) bilgilerini birbirine karıştırma.
- Ortamı koru: bir ortam için yazılmış tavsiyeyi (ör. bina içi Çök-Kapan-Tutun) başka ortama (araç, sahil) uygulama.
- Belirli bir binanın güvenli olduğuna veya yıkılmayacağına dair ASLA güvence verme; bunun ancak uzman
  incelemesiyle (ör. yetkili mühendis, belediye, hasar tespit ekipleri) anlaşılacağını söyle.
- Depremin zamanı/yeri hakkında tahmin yapma.
- Yaralı, enkaz, yangın gibi acil durumlarda İLK cümlede 112 Acil Çağrı Merkezi'ni aramayı söyle.
- Kaynakta somut bilgi (sayı, isim, yer) varsa onu aynen ver; bilginin kapsamını koru ("X ili geneli" diyorsa
  ilçeye atfetme).
- Başka bir yerin veya geçmiş bir olayın bilgisini sorulan yere/bugüne atfetme; il planı bilgisinde planın yılını
  belirt ve 2023 depremlerinden önceki verilerin güncel olmayabileceğini söyle.
- Kısa tut: en fazla 7 cümle. Soruyu tekrar etme.

Yalnızca şu JSON'u döndür:
{"sentences": [{"text": "<cümle>", "basis": "afad" | "genel", "quotes": [{"source": <N>, "quote": "<birebir alıntı>"}]}]}"""


def in_scope(question: str, client: Any, model: str = "gpt-4o-mini", previous: str | None = None) -> bool:
    """previous = the last on-topic question of the conversation; a follow-up to it stays in scope."""
    if IN_SCOPE_RE.search(turkish_lower(question)):
        return True
    text = question if not previous else (f"Önceki soru (deprem sohbeti): {previous}\nYeni soru: {question}\n"
                                          "Yeni soru önceki depremle ilgili sohbetin devamıysa kapsam içidir.")
    try:
        response = client.responses.create(model=model, instructions=SCOPE_PROMPT, input=text,
                                           max_output_tokens=20, temperature=0)
        payload = parse_json(extract_response_text(response)) or {}
    except Exception:
        return True  # never refuse because the scope check failed
    return payload.get("in_scope") is not False


def verify_sentences(payload: dict[str, Any], passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each sentence -> {"text", "sources"}; sources stay empty unless a quote is found in its cited passage."""
    sentences = []
    for sentence in payload.get("sentences") or []:
        if not isinstance(sentence, dict) or not str(sentence.get("text", "")).strip():
            continue
        sources = []
        for quote in sentence.get("quotes") or []:
            if not isinstance(quote, dict):
                continue
            number = quote.get("source")
            if isinstance(number, str) and number.isdigit():
                number = int(number)
            if isinstance(number, int) and 1 <= number <= len(passages) \
                    and locate_quote(str(quote.get("quote", "")), passages[number - 1]["text"]):
                sources.append(number)
        sentences.append({"text": sentence["text"].strip(), "sources": sorted(set(sources)),
                          "claimed_afad": sentence.get("basis") == "afad"})
    return sentences


# ---------------------------------------------------------------- web search (hybrid decision)

WEB_MODEL = "gpt-4o"  # domain filters are rejected for gpt-4o-mini / gpt-4.1-mini (400); ~$0.07 per search
WEB_DOMAINS = ["gov.tr", "kizilay.org.tr"]  # official sources only: AFAD, e-Devlet, ministries, governorships
CURRENT_INFO_RE = re.compile(r"2023|2024|2025|2026|6 şubat|güncel|şu an|şimdi|bugün|hangi hastane|adres|telefon|"
                             r"numara|açık mı|nerede kal|başvuru|yardım başvuru")
WEB_PROMPT = """Aşağıdaki soruyu yalnızca resmî Türk kaynaklarında (AFAD, e-Devlet, bakanlıklar, valilikler, Kızılay)
web araması yaparak cevapla. Kısa, Türkçe ve yalnızca bulduğun bilgiyle yaz; bulamadıysan bunu söyle.
- Her bilginin yanına kaynağın yayın/güncelleme tarihini yaz ("(15.03.2024 itibarıyla)"); tarih yoksa "(tarih yok)".
- Sayılar (can kaybı, yaralı, bina sayısı vb.) için en güncel resmî açıklamayı ara ve onu ver; eski bir rakamı
  güncelmiş gibi yazma.
- Farklı olayları ayrı tut: hangi bilginin hangi depreme/tarihe ait olduğunu belirt (ör. 6 Şubat 2023 ile
  20 Şubat 2023 depremleri ayrı olaylardır)."""


def needs_web(question: str, sentences: list[dict[str, Any]]) -> bool:
    """The agent's choice: stay with the AFAD vector store, or also search official web sources.

    Web search costs ~5x a plain answer, so it runs only for current information (dates, hospitals, addresses,
    applications) or when no sentence is backed by an AFAD document. The former third rule ("2+ general-knowledge
    sentences") sent a quarter of all questions to the web, e.g. "deprem anında balkona çıkılır mı"; it was dropped on
    26 Oct 2026 for cost (general sentences still pass check_general_sentences). Not yet re-measured.
    """
    if CURRENT_INFO_RE.search(turkish_lower(question)):
        return True
    return not any(sentence["sources"] for sentence in sentences)


def web_passages(question: str, client: Any, model: str = WEB_MODEL) -> list[dict[str, Any]]:
    """Official-source web search; one passage per cited URL holding the sentences that cite it."""
    try:
        response = client.responses.create(
            model=model, instructions=WEB_PROMPT, input=question,
            tools=[{"type": "web_search", "filters": {"allowed_domains": WEB_DOMAINS},
                    "user_location": {"type": "approximate", "country": "TR"}}],
            max_output_tokens=700,
        )
    except Exception:
        return []
    by_url: dict[str, list[str]] = {}
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "") or ""
            for note in getattr(content, "annotations", []) or []:
                url = getattr(note, "url", None)
                if not url:
                    continue
                # the cited line (search answers come as bullet lines; dots would split "15.03.2024")
                start = text.rfind("\n", 0, getattr(note, "start_index", 0)) + 1
                end = text.find("\n", getattr(note, "end_index", 0))
                snippet = text[start:end + 1 if end >= 0 else len(text)].strip()
                snippet = re.sub(r"\(\[[^\]]*\]\([^)]*\)\)|\[[^\]]*\]\([^)]*\)", "", snippet)
                snippet = snippet.replace("**", "").replace("__", "").strip(" -•*\t")  # markdown emphasis
                if snippet and snippet not in by_url.setdefault(url, []):
                    by_url[url].append(snippet)
    return [{"text": f"Web (resmî kaynak): {url}\n" + " ".join(snippets), "source_file": url, "page_start": None,
             "page_end": None, "chunk_id": f"web:{url}"} for url, snippets in by_url.items() if snippets]


WEB_NOTE = ("Not: Web bilgileri resmî sitelerden (gov.tr, Kızılay) alındı; yayın tarihleri farklı olabilir, "
            "güncelliğini ilgili kurumdan kontrol edin.")
GENERAL_MARK = "(genel bilgi)"
GENERAL_LEGEND = "(genel bilgi) = AFAD kaynaklarında doğrulanmamış, genel bilgiye dayanan cümle."


def render(sentences: list[dict[str, Any]]) -> str:
    """Sentences in the model's order (so "call 112" can come first); each shows its basis inline."""
    if not sentences:  # in-scope question but no usable answer (e.g. unparsable model output): never "off topic"
        return NO_ANSWER
    lines = [f"{s['text']} " + ("".join(f"[Kaynak {n}]" for n in s["sources"]) if s["sources"] else GENERAL_MARK)
             for s in sentences]
    if any(not s["sources"] for s in sentences):
        lines.append("\n" + GENERAL_LEGEND)
    return "\n".join(lines)


def run_pipeline_v3(question: str, client: Any, index: Any, metadata: list[dict[str, Any]], embedding_model: Any,
                    reranker: Any, location: Any = None, local_info_builder: Any = None,
                    answer_model: str = ANSWER_MODEL, web_search: bool = True) -> dict[str, Any]:
    from .location import detect_location

    understanding = understand_question(question, client)
    corrected = understanding["corrected"]
    base = {"question": question, "understanding": understanding, "passages": [], "local_info": None,
            "location": location, "sentences": []}
    if not in_scope(corrected, client):
        return {**base, "answer": OFF_TOPIC_ANSWER, "status": "off_topic"}
    if location is None:
        location = detect_location(corrected)
    search_location = location if location is not None and not location.is_ambiguous else None
    passages = retrieve_wide(corrected, understanding["search_queries"], search_location,
                             policy_for(understanding["type"]), index, metadata, embedding_model, reranker,
                             top_k=PASSAGES)
    local_info = local_info_builder(location, corrected, understanding["type"]) if local_info_builder else None
    passages = passages + card_passages(local_info)
    header = f"Soru: {corrected}\nSorunun durumu: {understanding['situation']}"
    if understanding.get("setting"):
        header += f"; ortam: {understanding['setting']}"
    sentences = write_answer(header, passages, client, answer_model)
    used_web = False
    if web_search and needs_web(corrected, sentences):
        found = web_passages(corrected, client)
        if found:
            passages = passages + found
            sentences = write_answer(header, passages, client, answer_model)
            used_web = True
    return {**base, "location": location, "passages": passages, "local_info": local_info, "used_web": used_web,
            "sentences": sentences, "status": "ok" if sentences else "empty",
            "answer": render(sentences) + (f"\n\n{WEB_NOTE}" if used_web and any(
                str(passages[n - 1].get("chunk_id", "")).startswith("web:") for s in sentences for n in s["sources"])
                else "")}


def format_labelled_passages(passages: list[dict[str, Any]]) -> str:
    """Numbered passages; guide chunks carry the situation their section header names."""
    blocks = []
    for number, passage in enumerate(passages, start=1):
        situation = passage_situation(passage["text"])
        if passage.get("chunk_id") == "il_karti:brosur":
            situation = AFTER  # the assembly-area brochure is about after the earthquake
        label = f" (durum: {situation})" if situation else ""
        blocks.append(f"[Kaynak {number}]{label}\n{passage['text']}")
    return "\n\n".join(blocks)


class ServiceUnavailable(RuntimeError):
    """The language-model service cannot be used (invalid key, no credit, rate limit, network, outage).

    Not the question's fault, so it must not end as NO_ANSWER ("try asking differently"): the API turns it into a
    503 with a clear message, and the operator sees the real reason in the server log.
    """


def is_service_failure(error: BaseException) -> bool:
    import openai

    return isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError, openai.RateLimitError,
                              openai.APIConnectionError, openai.InternalServerError))


def salvage_sentences(text: str) -> dict[str, Any]:
    """Complete sentence objects from an answer whose JSON was cut off (output limit): {"sentences": [...]}.

    Without this, one unfinished last sentence threw the whole answer away (NO_ANSWER).
    """
    start = text.find('"sentences"')
    start = text.find("[", start) if start >= 0 else -1
    if start < 0:
        return {}
    decoder, position, sentences = json.JSONDecoder(), start + 1, []
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        try:
            item, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            sentences.append(item)
    return {"sentences": sentences} if sentences else {}


def write_answer(header: str, passages: list[dict[str, Any]], client: Any, model: str) -> list[dict[str, Any]]:
    try:
        response = client.responses.create(model=model, instructions=ANSWER_PROMPT,
                                           input=f"{header}\n\nKaynaklar (AFAD belgeleri; varsa resmî web sonuçları):\n"
                                                 f"{format_labelled_passages(passages)}",
                                           max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS, temperature=0)
        text = extract_response_text(response)
        payload = parse_json(text)
        if payload is None:
            payload = salvage_sentences(text)
            logger.warning("Answer JSON incomplete (status=%s); kept %d complete sentence(s)",
                           getattr(response, "status", "?"), len(payload.get("sentences", [])))
    except Exception as error:
        if is_service_failure(error):
            raise ServiceUnavailable(f"{type(error).__name__}: {error}") from error
        payload = {}  # a malformed answer is handled like an empty one
    return emergency_first(header, check_general_sentences(header, verify_sentences(payload, passages), client, model))


EMERGENCY_RE = re.compile(r"yaralı|yarali|yaralan|yaraland|kanama|kanıyor|enkaz|yangın|yangin|bayıl|bayil|nefes almıyor|acil")


CALL_112 = "Acil bir durumdaysanız önce hemen 112 Acil Çağrı Merkezi'ni arayın."


def emergency_first(header: str, sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """In an emergency question the answer starts with "call 112", whatever the model wrote.

    The model's own 112 sentence is moved to the front; if it wrote none (seen with cheaper models, 25 Oct), a fixed
    one is added. An empty answer stays empty: NO_ANSWER already points to 112.
    """
    if not sentences or not EMERGENCY_RE.search(turkish_lower(header)):
        return sentences
    calls = [sentence for sentence in sentences if "112" in sentence["text"]]
    if not calls:
        return [{"text": CALL_112, "sources": [], "claimed_afad": False}] + sentences
    return calls[:1] + [sentence for sentence in sentences if sentence is not calls[0]]


MAX_HISTORY_TURNS = 3
CONTEXT_PROMPT = """Bir deprem bilgi asistanıyla yapılan sohbetin önceki turları ve kullanıcının yeni mesajı verilecek.
Yeni mesaj önceki konuşmaya dayanıyorsa (ör. "peki hastane?", "orada tsunami olur mu?", "ya araçtaysam?"), onu önceki
turlardaki yer, durum ve konuyla tamamlayıp TEK BAŞINA anlaşılır bir soruya çevir. Önceki turlarda bir yer (il/ilçe)
geçtiyse ve yeni mesaj başka bir yer söylemiyorsa o yeri yeni soruya ekle. Yeni mesaj zaten tek başına
anlaşılıyorsa veya yeni bir konuysa aynen bırak. Yeni bilgi ekleme, cevap verme.
Yalnızca JSON döndür: {"standalone": "<soru>"}"""


def contextualize(question: str, history: list[dict[str, str]] | None, client: Any,
                  model: str = "gpt-4o-mini") -> str:
    """Turn a follow-up ("peki hastane?") into a standalone question using the last turns; no history, no call."""
    if not history:
        return question
    turns = "\n".join(f"Kullanıcı: {turn['question']}\nAsistan: {turn['answer'][:300]}"
                      for turn in history[-MAX_HISTORY_TURNS:])
    try:
        response = client.responses.create(model=model, instructions=CONTEXT_PROMPT,
                                           input=f"Önceki turlar:\n{turns}\n\nYeni mesaj: {question}",
                                           max_output_tokens=120, temperature=0)
        payload = parse_json(extract_response_text(response)) or {}
    except Exception:
        return question
    standalone = str(payload.get("standalone") or "").strip()
    return standalone or question


GENERAL_CHECK_PROMPT = """Bir deprem bilgi asistanının cevabındaki bazı cümleler kaynakla doğrulanamadı ve "genel bilgi"
olarak işaretlendi. Her cümleyi Türkiye'deki resmî afet ve ilk yardım tavsiyelerine (AFAD, Kızılay, Sağlık Bakanlığı)
göre değerlendir:
- "dogru": yerleşik, güvenilir ve soruya uygun bilgi;
- "supheli": yanlış olabilir, tartışmalı, eskimiş ya da soruya/duruma uymuyor;
- "yanlis": yanlış veya tehlikeli.
Yalnızca JSON döndür: {"checks": [{"index": <cümle numarası>, "verdict": "dogru" | "supheli" | "yanlis"}]}"""


def check_general_sentences(header: str, sentences: list[dict[str, Any]], client: Any, model: str) -> list[dict[str, Any]]:
    """Drop general-knowledge sentences a second check finds doubtful or wrong; AFAD/web-backed ones stay."""
    general = [index for index, sentence in enumerate(sentences) if not sentence["sources"]]
    if not general:
        return sentences
    listing = "\n".join(f"{number}. {sentences[index]['text']}" for number, index in enumerate(general, start=1))
    try:
        response = client.responses.create(model=model, instructions=GENERAL_CHECK_PROMPT,
                                           input=f"{header}\n\nGenel bilgi cümleleri:\n{listing}",
                                           max_output_tokens=300, temperature=0)
        payload = parse_json(extract_response_text(response)) or {}
    except Exception:
        return sentences  # a failed check never removes the answer
    rejected = {general[int(check["index"]) - 1] for check in payload.get("checks") or []
                if isinstance(check, dict) and str(check.get("index", "")).isdigit()
                and 1 <= int(check["index"]) <= len(general) and check.get("verdict") in ("supheli", "yanlis")}
    return [sentence for index, sentence in enumerate(sentences) if index not in rejected]
