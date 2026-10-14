"""Build verified "province cards" from IRAP plans (one structured card per province).

Regional questions failed for every province in the same way: key facts sit in IRAP
tables that PDF extraction flattened ("İlçe Altınözü Antakya ... Sayı 28 133 ..."),
and per-question retrieval could not read them reliably. Each IRAP is therefore read
once, offline: topic pages are selected by keywords, gpt-4o fills a fixed card schema
with a verbatim quote and page per item, and every item is validated locally (the
quote must occur on that page; every number and capitalised name in the value must
occur in the quote). Items that fail are dropped and logged in the card.

  python -m src.preprocessing.build_province_cards --provinces Hatay Kahramanmaraş Adana
  python -m src.preprocessing.build_province_cards --all
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from src.rag.query_normalization import turkish_lower


TEXT_ROOT = Path("data/processed/text_v3/plans/irap")
IRAP_MAP_FILE = Path("data/geo/irap_province_map.json")
GAZETTEER_FILE = Path("data/geo/turkiye_il_ilce.json")
OUTPUT_DIR = Path("data/province_cards")
BROCHURE_FILE = Path("data/processed/text_v3/publications/toplanma_alanlari.json")
DEFAULT_MODEL = "gpt-4o"

TOPICS = {
    "faults": r"fay (hatt|zon)|diri fay|aktif fay|fay sistemi",
    "historical_earthquakes": r"tarihsel deprem|geçmiş deprem|deprem kataloğu|aletsel dönem|\bdepremi\b|\bio ?[:=]",
    "assembly_areas": r"toplanma alan",
    "temporary_shelters": r"barınma alan|çadır ?kent",
    "health_facilities": r"hastane|sağlık tesis",
    "other_hazards": r"tsunami|sıvılaşma",
}
FIELD_LABELS = {
    "faults": "Fay hatları ve deprem tehlikesi",
    "high_risk_districts": "Deprem açısından riskli ilçeler/bölgeler",
    "historical_earthquakes": "Geçmiş önemli depremler",
    "assembly_areas": "Acil toplanma alanları",
    "temporary_shelters": "Geçici barınma alanları",
    "health_facilities": "Sağlık tesisleri",
    "other_hazards": "Tsunami, sıvılaşma ve diğer deprem kaynaklı tehlikeler",
}
MAX_QUOTE_CHARS = 600

EXTRACT_SYSTEM_PROMPT = """Bir il afet risk azaltma planından (IRAP) "il bilgi kartı" çıkarıyorsun.
Sana "=== Sayfa N ===" ile ayrılmış sayfalar verilecek. Yalnızca metinde AÇIKÇA yazanları kullan.

Alanlar:
- "data_year": Planın/verilerin yılı (metinde geçiyorsa, ör. "2021"); yoksa "".
- "faults": ilden geçen fay hatları/zonları ve deprem tehlikesi.
- "high_risk_districts": deprem açısından riskli olduğu belirtilen ilçeler/bölgeler.
- "historical_earthquakes": ili etkilemiş en önemli depremler (yıl, büyüklük/şiddet, kayıp varsa); en büyük ve
  en çok can kaybına yol açanları seç, hem tarihsel hem aletsel dönemden.
- "assembly_areas": acil toplanma alanları (ilçelere göre sayı, yer veya genel bilgi).
- "temporary_shelters": geçici barınma alanları (ilçe, yer, kapasite).
- "health_facilities": hastane/sağlık tesisi sayıları veya bilgileri.
- "other_hazards": tsunami, sıvılaşma gibi deprem kaynaklı diğer tehlikeler.

Her alan bir liste; her öğe: {"value": "<kısa Türkçe bilgi>", "page": <N>, "quote": "<o sayfadan BİREBİR, kesintisiz alıntı, en fazla 600 karakter; araya \"...\" koyma>"}.
- "value" içindeki her sayı ve özel isim "quote" içinde geçmeli. Tarihleri kaynaktaki biçimle yaz
  (kaynakta "13.08.1822" yazıyorsa "13 Ağustos 1822" değil "13.08.1822").
- Tablolar düzleşmiş olabilir (ör. "İlçe A B C Sayı 1 2 3"). Sıralama açıkça eşleşiyorsa çiftleri çıkar
  (ör. "A: 1"); emin değilsen o tabloyu kullanma.
- Eylem planı/proje maddeleri (ör. "H10-1 ... haritalandırılması 2021-2026") tehlike kanıtı değildir; tehlikeyi
  tespit eden cümleyi alıntıla.
- İçindekiler sayfasındaki başlıkları ("Tablo 1.20 ... 43") bilgi olarak kullanma.
- Tablo bilgisinde "quote"u tablonun kendi metninden al, tabloyu tanıtan cümleden değil.
- İlçelere göre sayı tablolarını (ör. toplanma alanı sayıları) TÜM ilçeleriyle tek öğede ver.
- Bir tablonun satırlarını alırken tablodaki TÜM satırları al (en fazla 8); satır atlama.
- Her alanda en fazla 8 öğe. Bilgi yoksa boş liste.
Yalnızca JSON döndür: {"data_year": "...", "faults": [...], "high_risk_districts": [...], "historical_earthquakes": [...],
"assembly_areas": [...], "temporary_shelters": [...], "health_facilities": [...], "other_hazards": [...]}
"""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalise(text: str) -> str:
    text = nfc(text).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", turkish_lower(text)).strip()


def irap_pages(source_file: str) -> list[dict[str, Any]]:
    for path in glob.glob(str(TEXT_ROOT / "*.json")):
        pages = json.loads(Path(path).read_text(encoding="utf-8"))
        if pages and nfc(pages[0]["source_file"]) == nfc(source_file):
            return pages
    raise FileNotFoundError(source_file)


def select_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [page for page in pages if any(re.search(rx, turkish_lower(page["text"])) for rx in TOPICS.values())]
    # Risky-district statements usually sit next to fault text, so fault pages cover them.
    return [page for page in wanted if not re.search(r"\.{8,}", page["text"])]  # skip table-of-contents pages


def numbers_in(text: str) -> list[str]:
    return [re.sub(r"[.,]", "", n) for n in re.findall(r"\d[\d.,]*\d|\d", text)]


GENERIC_WORDS = re.compile(r"^(deprem|fay|zon|mahalle|hastane|devlet|kişi|ilçe|il\b|merkez|tablo|risk)")


def names_in(text: str) -> list[str]:
    """Capitalised words, except generic nouns the model capitalises in labels ("Ceyhan Depremi")."""
    return [w for w in re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}", text) if not GENERIC_WORDS.match(turkish_lower(w))]


def table_quote(value: str, page_text: str) -> str | None:
    """Flattened tables ("İlçe A B C Sayı (ad) 1 2 3"): if the value's names and then its numbers occur
    on the page in the same order within a short span, return that span as the verbatim quote."""
    names, numbers = names_in(value), re.findall(r"\d[\d.,]*\d|\d", value)
    if len(names) < 3 or len(names) != len(numbers):
        return None
    text = " ".join(page_text.split())
    lower = turkish_lower(text)
    if len(lower) != len(text):
        return None
    for start in [m.start() for m in re.finditer(re.escape(turkish_lower(names[0])), lower)]:
        position = start
        for token in [turkish_lower(n) for n in names] + numbers:
            match = re.compile(rf"(?<![\w.,]){re.escape(token)}(?![\w]|[.,]\d)").search(lower, position)
            if not match:
                break
            position = match.end()
        else:
            if position - start <= 2 * MAX_QUOTE_CHARS:
                return text[start:position]
    return None


def letters(text: str) -> str:
    """Letters and digits only: PDF extraction drops hyphens/spaces ("Gölbaşıtürkoğlu") and the
    model may change trailing punctuation; the letter sequence itself must still be verbatim."""
    return re.sub(r"[\W_]", "", normalise(text))


def quote_on_page(quote: str, page_text: str) -> bool:
    """The quote, or each of its "..."-separated parts in order, must occur on the page."""
    position = 0
    for part in map(letters, re.split(r"(?:\.\.\.|…)", quote)):
        if not part:
            continue
        found = page_text.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def validate_item(item: Any, raw_pages: dict[int, str]) -> str | None:
    """Return None if valid, else the reason it was dropped. May correct an off-by-one page and
    replace the quote of a flattened table with the verbatim table span."""
    if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("value", "quote")):
        return "malformed"
    page = item.get("page")
    if not isinstance(page, int):
        return "unknown page"
    candidates = [n for n in (page, page - 1, page + 1) if n in raw_pages]
    if not candidates:
        return "unknown page"
    for n in candidates:
        span = table_quote(item["value"], raw_pages[n])
        if span:
            item["page"], item["quote"] = n, span
            return None
    quote = item["quote"].strip()
    if not quote or len(quote) > MAX_QUOTE_CHARS + 50:
        return "quote length"
    found = next((n for n in candidates if quote_on_page(quote, letters(raw_pages[n]))), None)
    if found is None:
        return "quote not on page"
    item["page"] = found
    quote_numbers = numbers_in(quote) + re.findall(r"\d+", quote)  # dates: "13.08.1822" also gives "1822"
    # "590.250" in the value may be written "590.250,00" in the plan.
    if any(not any(q.startswith(number) for q in quote_numbers) for number in numbers_in(item["value"])):
        return "number not in quote"
    quote_norm = letters(quote)
    if any(letters(name) not in quote_norm for name in names_in(item["value"])):
        return "name not in quote"
    return None


def assembly_lookup_item() -> dict[str, Any] | None:
    """Province-independent pointer from the AFAD brochure (verified verbatim)."""
    if not BROCHURE_FILE.exists():
        return None
    for page in json.loads(BROCHURE_FILE.read_text(encoding="utf-8")):
        text = " ".join(page["text"].split())
        match = re.search(r"Size en yakın.{0,120}?öğrenebilirsiniz\.", text)
        if match:
            return {"value": "Size en yakın afet ve acil durum toplanma alanını e-Devlet (www.turkiye.gov.tr) üzerinden öğrenebilirsiniz.",
                    "page": page["page"], "quote": match.group(), "source_file": page["source_file"]}
    return None


def extract_card(province: str, source_file: str, client: Any, model: str) -> dict[str, Any]:
    from src.rag.answerability import extract_response_text

    pages = select_pages(irap_pages(source_file))
    raw_pages = {page["page"]: page["text"] for page in pages}
    prompt = "\n\n".join(f"=== Sayfa {page['page']} ===\n{' '.join(page['text'].split())}" for page in pages)
    response = client.responses.create(model=model, instructions=EXTRACT_SYSTEM_PROMPT,
                                       input=f"İl: {province}\n\n{prompt}", max_output_tokens=4000, temperature=0)
    raw = extract_response_text(response)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    payload = json.loads(match.group()) if match else {}

    fields: dict[str, list[dict[str, Any]]] = {}
    dropped: list[dict[str, Any]] = []
    for field in FIELD_LABELS:
        kept = []
        for item in payload.get(field, []) or []:
            reason = validate_item(item, raw_pages)
            if reason:
                dropped.append({"field": field, "reason": reason, "item": item})
            else:
                kept.append({"value": item["value"].strip(), "page": item["page"], "quote": item["quote"].strip()})
        fields[field] = kept
    return {
        "province": province,
        "source_file": source_file,
        "data_year": str(payload.get("data_year", "")),
        "pages_used": [page["page"] for page in pages],
        "fields": fields,
        "assembly_area_lookup": assembly_lookup_item(),
        "dropped": dropped,
        "model": model,
        "built_at": datetime.date.today().isoformat(),
    }


def no_plan_card(province: str) -> dict[str, Any]:
    return {"province": province, "source_file": None, "data_year": "", "pages_used": [], "fields": {},
            "assembly_area_lookup": assembly_lookup_item(), "dropped": [], "model": None,
            "built_at": datetime.date.today().isoformat(), "no_regional_plan": True}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build verified province cards from IRAP plans.")
    parser.add_argument("--provinces", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    from src.rag.generate_answer import load_environment, load_openai_client

    irap_map: dict[str, str] = json.loads(IRAP_MAP_FILE.read_text(encoding="utf-8"))
    province_to_file = {province: source for source, province in irap_map.items()}
    all_provinces = [p["name"] for p in json.loads(GAZETTEER_FILE.read_text(encoding="utf-8"))["provinces"]]
    targets = all_provinces if args.all else args.provinces
    load_environment()
    client = load_openai_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for province in targets:
        if province in province_to_file:
            card = extract_card(province, province_to_file[province], client, args.model)
        else:
            card = no_plan_card(province)
        (OUTPUT_DIR / f"{province}.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        kept = sum(len(items) for items in card["fields"].values())
        print(f"{province}: kept {kept} items, dropped {len(card['dropped'])}"
              + (" (no regional plan)" if card.get("no_regional_plan") else ""), flush=True)


if __name__ == "__main__":
    main()
