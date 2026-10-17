"""Document library for the mobile app: every AFAD source document, browsable online and downloadable for offline use.

The catalog is built from data/raw/documents (the corpus), with titles (data/geo/document_titles.json), source types
(data/geo/source_types.json) and IRAP provinces (data/geo/irap_province_map.json). Page texts for offline keyword
search come from the structure-aware extraction (data/processed/structured_v5, OCR results included).
Documents are addressed by a stable id, never by path, so a request cannot reach other files.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RAW_ROOT = Path("data/raw/documents")
STRUCTURED_ROOT = Path("data/processed/structured_v5")
TITLES_FILE = Path("data/geo/document_titles.json")
TYPES_FILE = Path("data/geo/source_types.json")
IRAP_FILE = Path("data/geo/irap_province_map.json")

MEDIA_TYPES = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

CATEGORIES = {
    "vatandas_rehberi": "Vatandaş rehberleri",
    "il_plani": "İl afet planları",
    "ulusal_plan": "Ulusal planlar",
    "teknik_mesleki": "Eğitim ve teknik yayınlar",
    "gecmis_rapor": "Raporlar",
    "sozluk": "Sözlükler",
    "gorsel": "Bilgi görselleri",
}

# The one-tap "Temel rehberler" package (~50 MB): what a citizen needs without internet.
CORE_PACKAGE = [
    "ilk72saat1.pdf", "ilk72saat.pdf", "ilk72saatyetiskin.pdf", "ilk72saatcocuk.pdf", "ilk72saatengelliler.pdf",
    "psikolojikilkyardim.pdf", "bilgilenplanlahazirlan.pdf", "deprem_nedir.pdf", "toplanma_alanlari.pdf",
    "13_afad_bilgilendirme_afisleri_tr_afet_cantasi_info_.pdf", "SoruCevap-Kitapcik-2.pdf",
    "yapisalolmayanriskler.pdf",
]

# Readable titles where document_titles.json fell back to the file name (image-only or short documents).
TITLE_OVERRIDES = {
    "ilk72saat.pdf": "İlk 72 Saat Afet Hazırlık Yayını",
    "ilk72saatcocuk.pdf": "İlk 72 Saat: Çocuklar İçin",
    "ilk72saatyetiskin.pdf": "İlk 72 Saat: Yetişkinler İçin",
    "psikolojikilkyardim.pdf": "Psikolojik İlk Yardım",
    "bilgilenplanlahazirlan.pdf": "Bilgilen, Planla, Hazırlan",
    "04_ikaz_alarm_isaretleri_tr-REVIZE.pdf": "İkaz ve Alarm İşaretleri",
    "13_afad_bilgilendirme_afisleri_tr_afet_cantasi_info_.pdf": "Afet Çantası Bilgilendirme Afişi",
    "Afet-barinma-faaliyetleri.pdf": "Afet Barınma Faaliyetleri",
    "AYDES.pdf": "AYDES: Afet Yönetimi ve Karar Destek Sistemi",
    "deprem_nedir.pdf": "Deprem Nedir?",
    "hasar_tespit_-_10x21.pdf": "Hasar Tespit Broşürü",
    "psikososyal_-_10x21.pdf": "Psikososyal Destek Broşürü",
    "SoruCevap-Kitapcik-2.pdf": "Soru-Cevap Kitapçığı",
    "toplanma_alanlari.pdf": "Toplanma Alanları",
    "yapilacak_yardimlar_-_10x21.pdf": "Yapılacak Yardımlar Broşürü",
    "yer_secimi.pdf": "Yer Seçimi",
    "23052025-TASIP.pdf": "Türkiye Afet Sonrası İyileştirme Planı (TASİP)",
    "28032022-TARAP-kitap_V6.pdf": "Türkiye Afet Risk Azaltma Planı (TARAP)",
    "TAMP.pdf": "Türkiye Afet Müdahale Planı (TAMP)",
    "udsep_1402013_kitap.pdf": "Ulusal Deprem Stratejisi ve Eylem Planı (UDSEP)",
    "afad_red.pdf": "AFAD-RED",
    "deprem_haritasi.pdf": "Deprem Haritası",
    "deprem_takibi.pdf": "Deprem Takibi",
    "ENKAZDA-ARAMA-VE-KURTARMA-TEMEL-EĞİTİMİ-2025-.pdf": "Enkazda Arama ve Kurtarma Temel Eğitimi (2025)",
    "fayin_merkezi.pdf": "Fayın Merkezi",
    "gozlem_aglari.pdf": "Gözlem Ağları",
    "mudahale_kapasitesi.pdf": "Müdahale Kapasitesi",
    "OLAY-YERİ-YÖNETİMİ-TEMEL-EĞİTİMİ-220925.pdf": "Olay Yeri Yönetimi Temel Eğitimi",
    "risk_haritalari.pdf": "Risk Haritaları",
    "udap.pdf": "UDAP",
    "yonetmelik_brosur.pdf": "Yönetmelik Broşürü",
    "2020yilidogakaynakliolayistatistikleri.pdf": "2020 Yılı Doğa Kaynaklı Olay İstatistikleri",
    "2022-Yili-Doga-Kaynakli-Olay-Istatistikleri.pdf": "2022 Yılı Doğa Kaynaklı Olay İstatistikleri",
    "2023yilidogakaynakliolayistatistikleri-_1_.pdf": "2023 Yılı Doğa Kaynaklı Olay İstatistikleri",
    "2024-Yili-Doga-Kaynakli-Olay-Istatistikleri.pdf": "2024 Yılı Doğa Kaynaklı Olay İstatistikleri",
    "mudahale_edilen_olaylar.pdf": "Müdahale Edilen Olaylar",
    "ARAMA-VE-KURTARMA-AÇIKLAMALI-TERİMLER-SÖZLÜĞÜ-2025.pdf": "Arama ve Kurtarma Açıklamalı Terimler Sözlüğü (2025)",
    "94-201406021514-afetler_tum_dagilim_renkli_2.pdf": "Afetlerin Türkiye'deki Dağılımı",
    "95-2014060215230-afetler_yillar.pdf": "Afetlerin Yıllara Göre Dağılımı",
    "_afet_egitim_yili_5_milyon-_2_.jpeg": "Afet Eğitim Yılı",
    "Afete Hazırlık için AFAD DASK el ele____.jpg": "Afete Hazırlık için AFAD ve DASK El Ele",
    "YALNIZ-EYLEMLER.pdf": "Karabük İl Afet Risk Azaltma Planı (Eylemler)",  # irap_province_map: Karabük
}
PROVINCES_FILE = Path("data/geo/turkiye_il_ilce.json")


def _ascii_key(text: str) -> str:
    table = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")
    return re.sub(r"[^a-z]", "", nfc(text).translate(table).lower())


@lru_cache(maxsize=1)
def _province_names() -> list[str]:
    names = [province["name"] for province in _read_json(PROVINCES_FILE).get("provinces", [])]
    return sorted(names, key=len, reverse=True)  # longest first, so a longer name wins over a shorter prefix


def _province_from_file_name(file_name: str) -> str | None:
    key = _ascii_key(Path(file_name).stem)
    return next((name for name in _province_names() if key.startswith(_ascii_key(name))), None)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class LibraryDocument:
    id: str
    title: str
    category: str
    province: str | None
    file_name: str
    relative_path: str  # under RAW_ROOT
    size_bytes: int
    pages: int | None
    has_text: bool
    core: bool

    @property
    def path(self) -> Path:
        return RAW_ROOT / self.relative_path

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES[self.path.suffix.lower()]

    def to_api(self) -> dict:
        return {"id": self.id, "baslik": self.title, "kategori": self.category,
                "kategori_adi": CATEGORIES[self.category], "il": self.province, "dosya_adi": self.file_name,
                "tur": "gorsel" if self.path.suffix.lower() != ".pdf" else "pdf", "boyut": self.size_bytes,
                "sayfa": self.pages, "aranabilir": self.has_text, "temel": self.core}


def document_id(relative_path: str) -> str:
    return hashlib.sha1(nfc(relative_path).encode("utf-8")).hexdigest()[:12]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _readable_title(file_name: str) -> str:
    stem = re.sub(r"[_\-]+", " ", Path(file_name).stem).strip()
    return stem[:1].upper() + stem[1:] if stem else file_name


def _structured_record(relative_path: str) -> dict | None:
    path = STRUCTURED_ROOT / Path(relative_path).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _category(relative_path: str, file_name: str, types: dict[str, str]) -> str:
    folder = Path(relative_path).parent.as_posix()
    if folder.endswith("plans/irap"):
        return "il_plani"
    if "/plans/" in f"/{folder}/":
        return "ulusal_plan"
    if Path(file_name).suffix.lower() != ".pdf" or folder.endswith(("infographics", "maps")):
        return "gorsel"
    return types.get(file_name) if types.get(file_name) in CATEGORIES else "teknik_mesleki"


@lru_cache(maxsize=1)
def catalog() -> tuple[LibraryDocument, ...]:
    titles = {nfc(k): v for k, v in _read_json(TITLES_FILE).items()}
    types = {nfc(k): v for k, v in _read_json(TYPES_FILE).get("files", {}).items()}
    provinces = {nfc(k): v for k, v in _read_json(IRAP_FILE).items()}
    core = {nfc(name) for name in CORE_PACKAGE}
    documents = []
    for path in sorted(RAW_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MEDIA_TYPES:
            continue
        relative = nfc(path.relative_to(RAW_ROOT).as_posix())
        name = nfc(path.name)
        category = _category(relative, name, types)
        province = provinces.get(name) or (_province_from_file_name(name) if category == "il_plani" else None)
        title = titles.get(name, "")
        if name in TITLE_OVERRIDES:
            title = TITLE_OVERRIDES[name]
        elif category == "il_plani" and province:
            title = f"{province} İl Afet Risk Azaltma Planı"
        elif not title or title == Path(name).stem.replace("_", " ").replace("-", " ").strip():
            title = _readable_title(name)
        record = _structured_record(relative)
        pages = len(record["pages"]) if record else None
        has_text = bool(record) and any(block.get("text", "").strip() for page in record["pages"]
                                        for block in page["blocks"])
        documents.append(LibraryDocument(
            id=document_id(relative), title=title, category=category, province=province, file_name=name, relative_path=relative, size_bytes=path.stat().st_size,
            pages=pages, has_text=has_text, core=name in core,
        ))
    order = list(CATEGORIES)
    return tuple(sorted(documents, key=lambda d: (order.index(d.category), d.province or "", d.title.casefold())))


def find(document_id_: str) -> LibraryDocument | None:
    return next((d for d in catalog() if d.id == document_id_), None)


def page_texts(document: LibraryDocument) -> list[dict]:
    """[{"sayfa": n, "metin": "..."}] for pages that have text (for keyword search on the phone)."""
    record = _structured_record(document.relative_path)
    if not record:
        return []
    pages = []
    for page in record["pages"]:
        text = "\n".join(block["text"].strip() for block in page["blocks"] if block.get("text", "").strip())
        if text:
            pages.append({"sayfa": page["page"], "metin": text})
    return pages
