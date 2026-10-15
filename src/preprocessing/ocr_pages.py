"""OCR for pages without a text layer (posters, scanned booklets), as structured blocks.

extract_structured.py marks pages with almost no extractable text as "needs_ocr". Each such
page is rendered to an image and read by a vision model that returns the page text in a
small markdown dialect ("# " / "## " / "### " headings, "- " bullets, "header: value; ..."
table rows). Results are cached per page (data/processed/ocr_cache_v5/), so re-runs cost
nothing; the structured JSON files are then updated in place with the OCR blocks.

  python -m src.preprocessing.ocr_pages --only ilk72saat.pdf psikolojikilkyardim.pdf
  python -m src.preprocessing.ocr_pages                       # every page that needs OCR
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any


STRUCTURED_ROOT = Path("data/processed/structured_v5")
RAW_ROOT = Path("data/raw/documents")
CACHE_DIR = Path("data/processed/ocr_cache_v5")
DEFAULT_MODEL = "gpt-4o"
PROMPT_VERSION = "ocr_v1"
MAX_IMAGE_SIDE = 1600
EMPTY_MARK = "BOŞ"
PRICES = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6)}  # USD per million input / output tokens

OCR_PROMPT = f"""Bu, bir AFAD yayınının sayfa görüntüsü. Sayfadaki Türkçe metni okuma sırasıyla, KELİMESİ KELİMESİNE yaz.
- Ana başlıkları "# ", alt başlıkları "## ", daha küçük başlıkları "### " ile başlat.
- Madde işaretli satırları "- " ile başlat. Paragrafları boş satırla ayır.
- Tabloları her satır için tek satırda "sütun başlığı: değer; sütun başlığı: değer" biçiminde yaz.
- Sayfa numarası, logo yazısı ve tekrarlanan üst/alt bilgiyi yazma.
- Görseli betimleme, yorum yapma, metinde olmayan hiçbir şey ekleme; okunamayan kelimeyi atla.
- Sayfada okunabilir metin yoksa yalnızca {EMPTY_MARK} yaz."""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def page_image(pdf_path: Path, page_number: int) -> bytes:
    import pymupdf

    with pymupdf.open(pdf_path) as document:
        page = document[page_number - 1]
        zoom = MAX_IMAGE_SIDE / max(page.rect.width, page.rect.height)
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def cache_path(relative_path: str, page_number: int, model: str) -> Path:
    key = hashlib.sha1(f"{relative_path}|{page_number}|{model}|{PROMPT_VERSION}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{Path(relative_path).stem}_p{page_number}_{key}.json"


def ocr_page(client: Any, model: str, pdf_path: Path, relative_path: str, page_number: int) -> dict[str, Any]:
    path = cache_path(relative_path, page_number, model)
    if path.exists():
        return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
    from src.rag.answerability import extract_response_text

    image = base64.b64encode(page_image(pdf_path, page_number)).decode("ascii")
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": OCR_PROMPT},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image}", "detail": "high"},
        ]}],
        max_output_tokens=3000,
        temperature=0,
    )
    usage = getattr(response, "usage", None)
    record = {"relative_path": relative_path, "page": page_number, "model": model, "prompt_version": PROMPT_VERSION,
              "text": extract_response_text(response).strip(),
              "input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**record, "cached": False}


def markdown_blocks(text: str) -> list[dict[str, Any]]:
    """The OCR markdown dialect back into extract_structured blocks."""
    if not text or text.strip() == EMPTY_MARK:
        return []
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            joined = " ".join(paragraph)
            kind = "table" if all(": " in line and ";" in line for line in paragraph) else "text"
            blocks.append({"kind": kind, "text": "\n".join(paragraph) if kind == "table" else joined, "ocr": True})
            paragraph.clear()

    for raw in text.replace("```", "").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        level = len(line) - len(line.lstrip("#"))
        if 1 <= level <= 3 and line[level:level + 1] == " ":
            flush()
            blocks.append({"kind": "heading", "level": level, "text": line[level + 1:].strip(), "ocr": True})
            continue
        paragraph.append(line[2:] if line.startswith("- ") else line)
    flush()
    return blocks


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="OCR pages that have no text layer.")
    parser.add_argument("--only", nargs="*", default=[], help="Only these PDF file names.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after this many uncached pages.")
    parser.add_argument("--cache-only", action="store_true", help="Apply cached OCR results only; no API calls.")
    args = parser.parse_args()

    from src.rag.generate_answer import load_environment, load_openai_client

    client = None
    if not args.cache_only:
        load_environment()
        client = load_openai_client()
    wanted = {nfc(name) for name in args.only}
    input_price, output_price = PRICES.get(args.model, (0.0, 0.0))
    spent_in = spent_out = new_pages = 0
    for json_path in sorted(STRUCTURED_ROOT.rglob("*.json")):
        record = json.loads(json_path.read_text(encoding="utf-8"))
        if wanted and nfc(record["source_file"]) not in wanted:
            continue
        pdf_path = RAW_ROOT / record["relative_path"]
        changed = False
        for page in record["pages"]:
            if not page["needs_ocr"] or page.get("ocr_done"):
                continue
            if args.max_pages is not None and new_pages >= args.max_pages:
                break
            if args.cache_only and not cache_path(record["relative_path"], page["page"], args.model).exists():
                continue
            result = ocr_page(client, args.model, pdf_path, record["relative_path"], page["page"])
            if not result["cached"]:
                new_pages += 1
                spent_in += result["input_tokens"]
                spent_out += result["output_tokens"]
            page["blocks"] = page["blocks"] + markdown_blocks(result["text"])
            page["ocr_done"] = True
            page["ocr_model"] = args.model
            changed = True
            print(f"{record['source_file']} p{page['page']}: {len(result['text'])} chars"
                  + (" (cache)" if result["cached"] else f" ({result['input_tokens']}+{result['output_tokens']} tok)"),
                  flush=True)
        if changed:
            json_path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    cost = spent_in / 1e6 * input_price + spent_out / 1e6 * output_price
    print(f"new pages: {new_pages}, tokens in {spent_in} out {spent_out}, cost ≈ ${cost:.3f}"
          + (f" (${cost / new_pages:.4f}/page)" if new_pages else ""))


if __name__ == "__main__":
    main()
