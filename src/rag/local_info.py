"""Province-specific "local info" appended to answers from verified province cards.

Behaviour questions ("deprem oldu, ne yapmalıyım") are answered from citizen guides, so a
question naming a place got the same generic answer as any other place. When the question
names a province or district, a short local section is added from the province card
(src/preprocessing/build_province_cards.py) and the AFAD assembly-area brochure. Which card
fields are shown depends on the question: behaviour questions get assembly areas and shelters
(plus the brochure: what an assembly area is and how to find the nearest one), history
questions get past earthquakes, and keywords pick the rest (fault -> faults and risky
districts, hospital -> health facilities, tsunami -> other hazards, ...). Card lines come from quotes verified on their source page; brochure
lines are shown only if their source sentence is found verbatim in the brochure. Nothing here
is written by an LLM. The plan's data year is always shown, since plans predate the
6 February 2023 earthquakes. "Where should I go" questions get the brochure lines even without
a place name, since retrieval rarely ranks the brochure into the top five.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .location import Location
from .query_normalization import turkish_lower


CARD_DIR = Path("data/province_cards")
BROCHURE_FILE = Path("data/processed/text_v3/publications/toplanma_alanlari.json")
# Fields every question of a type gets; keywords below add more.
TYPE_FIELDS = {
    "davranis": ("assembly_areas", "temporary_shelters"),
    "gecmis": ("historical_earthquakes",),
}
FIELD_PATTERNS = {
    "faults": re.compile(r"\bfay|neden deprem|deprem bölgesi|deprem riski|riskli|tehlike"),
    "high_risk_districts": re.compile(r"riskli|deprem riski|tehlikeli|hangi ilçe"),
    "historical_earthquakes": re.compile(r"geçmiş|tarihte|tarihsel|eskiden|büyük deprem|hangi deprem|(?<!\d)1\d{3}(?!\d)"),
    "assembly_areas": re.compile(r"toplanma|nereye|tahliye"),
    "temporary_shelters": re.compile(r"barınma|barinma|çadır|cadir|nereye|nerede kal"),
    "health_facilities": re.compile(r"hastane|sağlık|saglik|yaralı|yarali"),
    "other_hazards": re.compile(r"tsunami|sıvılaşma|sivilasma|heyelan|zemin"),
}
FIELD_LABELS = {
    "faults": "Fay hatları",
    "high_risk_districts": "Deprem açısından riskli ilçeler",
    "historical_earthquakes": "Geçmiş depremler",
    "health_facilities": "Sağlık tesisleri",
    "other_hazards": "Diğer tehlikeler",
}
# The brochure (what an assembly area is, how to find yours) goes with these fields.
BROCHURE_FIELDS = frozenset({"assembly_areas", "temporary_shelters"})
MAX_SHELTERS = 3
# Question word -> stem expected in the card value.
HAZARD_WORDS = {"tsunami": "tsunami", "sıvılaşma": "sıvılaşma", "sivilasma": "sıvılaşma", "heyelan": "heyelan",
                "zemin": "sıvılaşma"}
MAX_ITEMS = 8
# (shown text, brochure page, sentence that must occur verbatim on that page)
BROCHURE_LINES = (
    ("Deprem gibi afetlerden sonra, geçici barınma merkezleri hazır olana kadar tehlikeli bölgeden uzaklaşıp "
     "afet ve acil durum toplanma alanına gidin; toplanma alanları bu amaçla belirlenmiş güvenli alanlardır.", 2,
     "Afet ve acil durumlar sonrasında geçici barınma merkezleri hazır olana kadar geçecek süre içerisinde yaşanacak "
     "paniği önlemek ve sağlıklı bilgi alışverişini sağlamak amacıyla halkın tehlikeli bölgeden uzaklaşarak "
     "toplanabileceği güvenli alanlardır."),
    ("Size en yakın toplanma alanını e-Devlet'ten (www.turkiye.gov.tr) öğrenebilirsiniz: T.C. kimlik numaranızla "
     "girin, AFAD hizmetlerinden \"Afet ve Acil Durum Toplanma Alanları\"nı seçin, adresinize en yakın alanı ve "
     "haritadaki yerini görün.", 1,
     "Ardından AFAD’ın sunduğu hizmetlerden Afet ve Acil Durum Toplanma Alanlarını seçin, • İkamet ettiğiniz adrese "
     "en yakın toplanma alanını belirleyin, • Harita özelliğinden alanın yerini öğrenin,"),
    ("Alana giderken afet ve acil durum çantanızı alın; geçici barınma merkezleri oluşturulana kadar bu alanda bulunun.", 1,
     "Alana giderken Afet ve Acil Durum Çantanızı alın, • Çantanızda bulunan pilli radyoyu çalıştırın, • Geçici "
     "barınma merkezleri oluşturulana kadar bu alanda bulunun."),
)


@lru_cache(maxsize=None)
def load_card(province: str, card_dir: str = str(CARD_DIR)) -> dict[str, Any] | None:
    path = Path(card_dir) / f"{province}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def brochure_lines(brochure_file: str = str(BROCHURE_FILE)) -> tuple[dict[str, Any], ...]:
    path = Path(brochure_file)
    if not path.exists():
        return ()
    pages = json.loads(path.read_text(encoding="utf-8"))
    texts = {page["page"]: " ".join(page["text"].split()) for page in pages}
    source = pages[0]["source_file"] if pages else ""
    return tuple({"text": text, "source_file": source, "page": page}
                 for text, page, quote in BROCHURE_LINES if quote in texts.get(page, ""))


def wanted_fields(question: str, question_type: str | None) -> list[str]:
    lowered = turkish_lower(question)
    fields = list(TYPE_FIELDS.get(question_type or "", ()))
    fields += [field for field, pattern in FIELD_PATTERNS.items() if pattern.search(lowered) and field not in fields]
    return fields


def plan_year(card: dict[str, Any]) -> str:
    if card.get("data_year"):
        return card["data_year"]
    match = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", card.get("source_file") or "")
    return match.group() if match else ""


def district_count(value: str, district: str) -> str | None:
    """'Altınözü: 28, Antakya: 133, ...' -> '133' for Antakya."""
    match = re.search(rf"(?:^|,\s*){re.escape(turkish_lower(district))}\s*:\s*([\d.]+)", turkish_lower(value))
    return match.group(1) if match else None


def starts_with_place(value: str, place: str) -> bool:
    return turkish_lower(value).startswith(turkish_lower(place))


def build_local_info(location: Location | None, question: str, question_type: str | None,
                     card_dir: str = str(CARD_DIR), brochure_file: str = str(BROCHURE_FILE)) -> dict[str, Any] | None:
    has_place = location is not None and location.province is not None and not location.is_ambiguous
    fields = wanted_fields(question, question_type)
    if not has_place:
        # Without a place only the brochure applies, and only when the question itself asks about it.
        fields = [field for field in fields if FIELD_PATTERNS[field].search(turkish_lower(question))]
    if not fields:
        return None
    info: dict[str, Any] = {
        "province": location.province if has_place else None,
        "district": location.district if has_place else None,
        "place": (location.district or location.province) if has_place else None,
        "general_lines": list(brochure_lines(brochure_file)) if BROCHURE_FIELDS & set(fields) else [],
        "lines": [],
        "plan_label": "",
        "source_file": None,
        "note": "",
    }
    card = load_card(location.province, card_dir) if has_place else None
    if card is not None:
        add_plan_lines(info, card, location, fields, question)
    return info if info["general_lines"] or info["lines"] else None


def add_plan_lines(info: dict[str, Any], card: dict[str, Any], location: Location, wanted: list[str],
                   question: str = "") -> None:
    year = plan_year(card)
    lines: list[dict[str, Any]] = []
    fields = card.get("fields") or {}
    for field in wanted:
        if field == "assembly_areas":
            lines += assembly_lines(fields.get(field, []), location)
        elif field == "temporary_shelters":
            lines += shelter_lines(fields.get(field, []), location)
        elif field == "other_hazards":
            lines += hazard_lines(fields.get(field, []), question)
        else:
            lines += [{"text": f"{FIELD_LABELS[field]}: {item['value']}", "page": item["page"]}
                      for item in fields.get(field, [])[:MAX_ITEMS]]

    if not lines:
        return
    info["lines"] = lines
    info["plan_label"] = f"{location.province} il planı" + (f" ({year})" if year else "")
    info["source_file"] = card["source_file"]
    if year and int(year) < 2023:
        info["note"] = (f"Not: Plan bilgileri {year} yılına aittir (6 Şubat 2023 depremlerinden önce); "
                        "güncel durum farklı olabilir, güncel bilgiyi e-Devlet'ten ve AFAD'dan kontrol edin.")
    elif not year:
        info["note"] = "Not: Plan bilgilerinin yılı belirtilmemiş; güncel bilgiyi e-Devlet'ten ve AFAD'dan kontrol edin."


def assembly_lines(items: list[dict[str, Any]], location: Location) -> list[dict[str, Any]]:
    for item in items:
        count = district_count(item["value"], location.district) if location.district else None
        if count:
            return [{"text": f"{location.district} ilçesinde {count} toplanma alanı belirlenmiş.", "page": item["page"]}]
        if not location.district:
            return [{"text": f"Toplanma alanları: {item['value']}", "page": item["page"]}]
    return []  # no item mentions this district


def hazard_lines(items: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """Only the hazard the question names (tsunami, liquefaction, ...), shown with its verified source sentence."""
    lowered = turkish_lower(question)
    asked = [word for word in HAZARD_WORDS if word in lowered]
    lines = []
    for item in items:
        value = turkish_lower(item["value"])
        if asked and not any(HAZARD_WORDS[word] in value for word in asked):
            continue
        lines.append({"text": f"{item['value']}: \"{item['quote']}\"", "page": item["page"]})
    return lines[:MAX_ITEMS]


def shelter_lines(items: list[dict[str, Any]], location: Location) -> list[dict[str, Any]]:
    if location.district:
        items = [item for item in items if starts_with_place(item["value"], location.district)]
        limit = MAX_SHELTERS
    else:
        limit = MAX_ITEMS
    return [{"text": f"Geçici barınma alanı: {item['value']}", "page": item["page"]} for item in items[:limit]]


def format_local_info(info: dict[str, Any]) -> str:
    output: list[str] = []
    if info["general_lines"]:
        output.append("Toplanma alanı (AFAD):")
        output.extend(f"- {line['text']}" for line in info["general_lines"])
        pages = "–".join(str(page) for page in sorted({line["page"] for line in info["general_lines"]}))
        output.append(f"  (Kaynak: {info['general_lines'][0]['source_file']}, s. {pages})")
    if info["lines"]:
        if output:
            output.append("")
        output.append(f"{info['place']} için {info['plan_label']} bilgisi:")
        output.extend(f"- {line['text']} (s. {line['page']})" for line in info["lines"])
        output.append(f"  (Kaynak: {info['source_file']})")
    if info["note"]:
        output.append(info["note"])
    return "\n".join(output)


CARD_CHUNK_PREFIX = "il_karti:"


def scope_label(info: dict[str, Any], line: dict[str, Any]) -> str:
    """When a district was asked, mark lines that are about the whole province (not that district)."""
    district = info.get("district")
    if not district or turkish_lower(district) in turkish_lower(line["text"]):
        return ""
    return f", {info['province']} ili geneli ({district} için ayrı bilgi yok)"


def local_info_evidence(info: dict[str, Any] | None, first_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verified local lines as extra evidence quotes numbered after the retrieved sources.

    Flattened plan tables defeat per-question extraction ("İlçe A B Sayı 1 2"), so the card's
    verified form ("Antakya ilçesinde 133 toplanma alanı") is given to the answer step directly.
    """
    if not info:
        return [], []
    entries = [(line["text"], line["source_file"], line["page"], "brosur") for line in info["general_lines"]]
    entries += [(f"{info['plan_label']}{scope_label(info, line)}: {line['text']}", info["source_file"], line["page"],
                 info["province"]) for line in info["lines"]]
    evidence, sources = [], []
    for number, (text, source_file, page, key) in enumerate(entries, start=first_number):
        evidence.append({"source": number, "quote": text, "about": "card"})
        sources.append({"source_label": f"Kaynak {number}", "chunk_id": f"{CARD_CHUNK_PREFIX}{key}",
                        "source_file": source_file, "page_start": page, "page_end": page,
                        "page_label": f"Sayfa {page}"})
    return evidence, sources

