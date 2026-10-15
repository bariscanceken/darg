"""DARG pipeline v2: understand -> retrieve wide -> answer with code-verified quotes.

Pipeline v1 grew one LLM filter per problem (spelling repair, question type, LLM rerank, gate,
evidence selection, place check, answer writer). Each step could drop a correct answer and the
losses compounded: in the 14 Oct 2026 diagnosis, 15 of 21 refused questions had usable
evidence that a later step still refused. v2 keeps three steps:

1. understand_question  one gpt-4o-mini call: spelling fix (accepted only if it passes the local
                        spelling-only check), question type, situation, setting, and three search
                        rewrites ("dışarıdaysam" -> "açık alanda", "sokakta", ...).
2. retrieve_wide        every query x (dense + BM25) over the section-aware v5 index, plus the
                        asked province's plan and the preferred source types; other provinces'
                        plans are removed; one cross-encoder rerank on the question; top 12.
3. answer_with_quotes   one gpt-4o call over the 12 passages (each shows its "document > section"
                        header) and the verified province-card lines. Every answer sentence must
                        carry verbatim quotes; quotes are located in their passage in code
                        (tolerating punctuation / small slips, then replaced by the source text),
                        and sentences without a verified quote are removed. Nothing left -> refusal.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

import numpy as np

from .answerability import REFUSAL_ANSWER, extract_response_text
from .evidence import normalise
from .hybrid_retrieval import lexical_rows
from .llm_query_repair import is_spelling_only
from .location import Location
from .regional_retrieval import candidate_from_row, index_vectors, irap_province_map, regional_rows
from .source_routing import QUESTION_TYPES, RoutingPolicy, policy_for, source_type

UNDERSTAND_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o"
DENSE_TOP_K = 30
LEXICAL_TOP_K = 15
POOL_TOP_K = 20
PASSAGES = 12
MIN_QUOTE_CHARS = 15
QUOTE_MATCH_RATIO = 0.9
ANSWER_MAX_OUTPUT_TOKENS = 900
SITUATIONS = ("hazırlık", "sırasında", "sonrasında", "enkaz altında", "genel", "geçmiş")

UNDERSTAND_PROMPT = """Bir deprem bilgi asistanına gelen soruyu çözümle. Yalnızca JSON döndür:
{"corrected": "<yazım hataları düzeltilmiş soru; yalnızca yazım: kelime ekleme/silme yok, konuşma dili
                kısaltmaları açılabilir (napcam -> ne yapacağım)>",
 "type": "davranis | bolgesel | tanim | gecmis | genel",
 "situation": "hazırlık | sırasında | sonrasında | enkaz altında | genel | geçmiş",
 "setting": "<kişinin bulunduğu ortam, kısa: ev, bina içi, açık alan, araç, sahil, okul, iş yeri ... ya da boş>",
 "search_queries": ["<soruyu AFAD rehberlerinin kullanacağı kelimelerle 3 farklı biçimde yeniden yaz>", "...", "..."]}
Tipler: davranis = kişinin ne yapması gerektiği; bolgesel = belirli bir il/ilçe hakkında bilgi (risk, fay,
toplanma alanı...); tanim = bir kavram; gecmis = geçmiş depremler; genel = diğer.
"deprem oldu" / "depremden sonra" -> sonrasında; "deprem olursa" / "depremde" -> sırasında (belirsizse genel)."""

ANSWER_PROMPT = """Sen DARG deprem bilgi asistanısın. Soruyu YALNIZCA verilen kaynak parçalarına dayanarak cevapla.
Her parçanın başında "belge › bölüm › alt bölüm" başlığı var; tavsiyenin hangi durum (deprem sırasında /
sonrasında / enkaz altında / hazırlık) ve hangi ortam için yazıldığını bu başlıktan ve metinden anla.

Kurallar:
- Soruyu doğrudan cevapla. Kaynaklar soruyu kısmen cevaplıyorsa cevaplanan kısmı ver, eksik kısmı "not_covered"
  alanında söyle. Hiçbir kaynak soruyla ilgili değilse "sentences" boş olsun.
- Her cümle, onu destekleyen kaynak numarasını ve kaynaktan KELİMESİ KELİMESİNE alıntıyı içersin.
- Kaynakta olmayan bilgi, gerekçe veya tavsiye ekleme; kendi bilgini kullanma.
- Başka bir durum için yazılmış tavsiyeyi sorunun durumuna aitmiş gibi sunma; gerekiyorsa durumu belirt
  ("deprem sonrasında ..."). Enkaz altındaki kişiye yönelik tavsiyeyi genel tavsiye gibi verme.
- Başka bir yerin veya geçmiş bir olayın bilgisini sorulan yere/bugüne atfetme; sayıların kapsamını (dünya / ülke /
  il / ilçe; yıl) kaynaktaki gibi koru. İl planı bilgisinde planın yılını belirt.
- Kaynaktaki koşulları koru ("hasar yoksa ..." gibi); koşulu cümlenin başına koy.
- Kısa, açık, uygulanabilir Türkçe yaz; madde listesi gerekiyorsa her madde ayrı cümle olsun.

Yalnızca şu JSON'u döndür:
{"sentences": [{"text": "<cevap cümlesi>", "quotes": [{"source": <N>, "quote": "<birebir alıntı>"}]}],
 "not_covered": "<sorunun kaynaklarda bulunmayan kısmı, yoksa boş>"}"""


def parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------- 1. understanding

def understand_question(question: str, client: Any, model: str = UNDERSTAND_MODEL) -> dict[str, Any]:
    """One structured call; the spelling fix is kept only if it passes the deterministic check."""
    fallback = {"corrected": question, "type": "genel", "situation": "genel", "setting": "", "search_queries": [],
                "status": "fallback"}
    try:
        response = client.responses.create(model=model, instructions=UNDERSTAND_PROMPT, input=question,
                                           max_output_tokens=300, temperature=0)
        payload = parse_json(extract_response_text(response))
    except Exception:
        payload = None
    if payload is None:
        return fallback
    corrected = str(payload.get("corrected") or question).strip()
    accepted = corrected == question or is_spelling_only(question, corrected)
    queries = [str(query).strip() for query in payload.get("search_queries") or [] if str(query).strip()][:3]
    return {
        "corrected": corrected if accepted else question,
        "spelling": "accepted" if accepted and corrected != question else ("rejected" if not accepted else "unchanged"),
        "type": payload.get("type") if payload.get("type") in QUESTION_TYPES else "genel",
        "situation": payload.get("situation") if payload.get("situation") in SITUATIONS else "genel",
        "setting": str(payload.get("setting") or "").strip(),
        "search_queries": queries,
        "status": "ok",
    }


# ---------------------------------------------------------------- 2. retrieval

def retrieve_wide(question: str, queries: list[str], location: Location | None, policy: RoutingPolicy | None,
                  index: Any, metadata: list[dict[str, Any]], embedding_model: Any, reranker: Any,
                  top_k: int = PASSAGES) -> list[dict[str, Any]]:
    """Union of dense + BM25 pools over every query; the cross-encoder ranks the union against the question."""
    from src.retrieval.evaluate_reranked_v2 import retrieve_v2_candidates
    from src.retrieval.rerank import rerank_candidates
    from src.retrieval.search import embed_query

    irap_map = irap_province_map()
    allowed_types = set(policy.allowed) if policy is not None else None
    if allowed_types is not None and location is None and "il_plani" not in policy.preferred:
        allowed_types.discard("il_plani")

    def allowed(record: dict[str, Any]) -> bool:
        if location is not None and irap_map.get(record["source_file"]) not in (None, location.province):
            return False  # another province's plan never answers about this place
        return allowed_types is None or source_type(record) in allowed_types

    pool: dict[str, dict[str, Any]] = {}
    rank = 0
    vectors = index_vectors(index)
    for query in [question, *[query for query in queries if query != question]]:
        for candidate in retrieve_v2_candidates(query, index, metadata, embedding_model, DENSE_TOP_K):
            if allowed(candidate):
                pool.setdefault(candidate["chunk_id"], candidate)
        for row, score in lexical_rows(query, metadata, LEXICAL_TOP_K):
            if allowed(metadata[row]):
                rank += 1
                pool.setdefault(metadata[row]["chunk_id"], candidate_from_row(metadata[row], row, rank, score))
    focus_rows: list[int] = []
    if location is not None and not location.is_ambiguous:
        focus_rows += regional_rows(location, metadata, irap_map)
    if policy is not None and policy.preferred:
        focus_rows += [row for row in range(len(metadata))
                       if source_type(metadata[row]) in policy.preferred and allowed(metadata[row])]
    if focus_rows:
        query_vector = embed_query(embedding_model, question)[0]
        scores = vectors[focus_rows] @ query_vector
        for i in np.argsort(-scores)[:POOL_TOP_K]:
            rank += 1
            row = focus_rows[i]
            pool.setdefault(metadata[row]["chunk_id"], candidate_from_row(metadata[row], row, rank, float(scores[i])))
    if not pool:
        return []
    return rerank_candidates(question, list(pool.values()), reranker, top_k=top_k)


# ---------------------------------------------------------------- 3. answer

def locate_quote(quote: str, passage: str) -> str | None:
    """The passage text a quote stands for: exact (normalised) match, else a >= 90% close span."""
    q = normalise(quote).strip(" .,;:!?\"'“”…")
    text = normalise(passage)
    if len(q) < MIN_QUOTE_CHARS:
        return None
    start = text.find(q)
    if start >= 0:
        return text[start:start + len(q)]
    matcher = difflib.SequenceMatcher(None, q, text, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 4]
    if not blocks:
        return None
    matched = sum(block.size for block in blocks)
    span_start, span_end = blocks[0].b, blocks[-1].b + blocks[-1].size
    if matched / len(q) >= QUOTE_MATCH_RATIO and span_end - span_start <= 1.2 * len(q):
        return text[span_start:span_end]
    return None


def format_passages(passages: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[Kaynak {number}]\n{passage['text']}" for number, passage in enumerate(passages, start=1))


def verify_answer(payload: dict[str, Any], passages: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep sentences with at least one quote found in its cited passage; cite those passages."""
    kept, dropped = [], []
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
        (kept if sources else dropped).append({"text": sentence["text"].strip(), "sources": sorted(set(sources))})
    return {"kept": kept, "dropped": dropped, "not_covered": str(payload.get("not_covered") or "").strip()}


def render_answer(verified: dict[str, Any]) -> str:
    if not verified["kept"]:
        return REFUSAL_ANSWER
    lines = [f"{sentence['text']} " + "".join(f"[Kaynak {number}]" for number in sentence["sources"])
             for sentence in verified["kept"]]
    answer = "\n".join(lines)
    if verified["not_covered"]:
        answer += f"\n\nKaynaklarda bulunmayan: {verified['not_covered']}"
    return answer


def answer_with_quotes(question: str, understanding: dict[str, Any], passages: list[dict[str, Any]], client: Any,
                       model: str = ANSWER_MODEL) -> dict[str, Any]:
    if not passages:
        return {"answer": REFUSAL_ANSWER, "verified": {"kept": [], "dropped": [], "not_covered": ""}, "status": "no_passages"}
    situation = understanding.get("situation", "genel")
    setting = understanding.get("setting", "")
    header = f"Soru: {question}\nSorunun durumu: {situation}" + (f"; ortam: {setting}" if setting else "")
    try:
        response = client.responses.create(model=model, instructions=ANSWER_PROMPT,
                                           input=f"{header}\n\nKaynaklar:\n{format_passages(passages)}",
                                           max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS, temperature=0)
        payload = parse_json(extract_response_text(response)) or {}
        status = "ok" if payload else "invalid"
    except Exception:
        payload, status = {}, "error"
    verified = verify_answer(payload, passages)
    return {"answer": render_answer(verified), "verified": verified, "status": status}


# ---------------------------------------------------------------- whole pipeline

def card_passages(local_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Verified province-card / brochure lines as extra passages (see src/rag/local_info.py)."""
    if not local_info:
        return []
    passages = []
    for line in local_info.get("general_lines", []):
        passages.append({"text": f"AFAD toplanma alanı broşürü\n{line['text']}", "source_file": line["source_file"],
                         "page_start": line["page"], "page_end": line["page"], "chunk_id": "il_karti:brosur"})
    for line in local_info.get("lines", []):
        passages.append({"text": f"{local_info['plan_label']} (doğrulanmış il kartı)\n{line['text']}",
                         "source_file": local_info["source_file"], "page_start": line["page"], "page_end": line["page"],
                         "chunk_id": f"il_karti:{local_info['province']}"})
    return passages


def run_pipeline_v2(question: str, client: Any, index: Any, metadata: list[dict[str, Any]], embedding_model: Any,
                    reranker: Any, location: Location | None = None, local_info_builder: Any = None,
                    answer_model: str = ANSWER_MODEL) -> dict[str, Any]:
    from .location import detect_location

    understanding = understand_question(question, client)
    corrected = understanding["corrected"]
    if location is None:
        location = detect_location(corrected)
    search_location = location if location is not None and not location.is_ambiguous else None
    policy = policy_for(understanding["type"])
    passages = retrieve_wide(corrected, understanding["search_queries"], search_location, policy, index, metadata,
                             embedding_model, reranker)
    local_info = local_info_builder(location, corrected, understanding["type"]) if local_info_builder else None
    passages = passages + card_passages(local_info)
    result = answer_with_quotes(corrected, understanding, passages, client, answer_model)
    return {"question": question, "understanding": understanding, "location": location, "passages": passages,
            "local_info": local_info, **result}
