"""Readable document titles for chunk headers (data/geo/document_titles.json).

PDF metadata titles are unusable ("odd1", "BIREY AILE GIRIS") and file names are cryptic
("olagandisi.pdf", "document-_11_.pdf"). Provincial plans get "<İl> İl Afet Risk Azaltma Planı"
from the IRAP-province map; every other document gets a short title read from its first pages
by gpt-4o-mini (one call per document, ~131 calls, about $0.02). Existing titles are kept, so
re-runs only title new documents.

  python -m src.preprocessing.build_document_titles
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path


STRUCTURED_ROOT = Path("data/processed/structured_v5")
IRAP_MAP_FILE = Path("data/geo/irap_province_map.json")
OUTPUT_FILE = Path("data/geo/document_titles.json")
MODEL = "gpt-4o-mini"
FIRST_PAGES_CHARS = 2500

TITLE_PROMPT = """Aşağıda bir AFAD belgesinin dosya adı ve ilk sayfalarının metni var. Belgenin Türkçe başlığını yaz:
kısa (en fazla 8 kelime), belgenin kendi kullandığı başlığa en yakın biçimde, kurum adı eklemeden.
Yalnızca başlığı yaz."""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def fallback_title(name: str) -> str:
    return Path(name).stem.replace("_", " ").replace("-", " ").strip()


def first_pages_text(record: dict) -> str:
    """Text of the first pages that have any (covers are often image-only)."""
    parts: list[str] = []
    for page in record["pages"]:
        parts.extend(block["text"] for block in page["blocks"])
        if sum(map(len, parts)) >= FIRST_PAGES_CHARS:
            break
    return " ".join(" ".join(parts).split())[:FIRST_PAGES_CHARS]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from src.rag.answerability import extract_response_text
    from src.rag.generate_answer import load_environment, load_openai_client

    titles: dict[str, str] = json.loads(OUTPUT_FILE.read_text(encoding="utf-8")) if OUTPUT_FILE.exists() else {}
    irap = {nfc(source): province for source, province in json.loads(IRAP_MAP_FILE.read_text(encoding="utf-8")).items()}
    client = None
    for path in sorted(STRUCTURED_ROOT.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        name = nfc(record["source_file"])
        if name in titles and titles[name] != fallback_title(name):
            continue  # file-name fallbacks are retried (e.g. once OCR has filled an image-only document)
        if name in irap:
            titles[name] = f"{irap[name]} İl Afet Risk Azaltma Planı"
        else:
            text = first_pages_text(record)
            if not text:
                titles[name] = fallback_title(name)
            else:
                if client is None:
                    load_environment()
                    client = load_openai_client()
                response = client.responses.create(model=MODEL, instructions=TITLE_PROMPT,
                                                   input=f"Dosya: {name}\n\n{text}", max_output_tokens=40, temperature=0)
                titles[name] = extract_response_text(response).strip().strip('"').strip()
        print(f"{name} -> {titles[name]}", flush=True)
    OUTPUT_FILE.write_text(json.dumps(dict(sorted(titles.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(titles)} titles -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
