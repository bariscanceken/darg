"""Structure-preserving PDF extraction: headings, paragraphs and tables per page.

The page-text extraction behind indexes v2-v4 flattened every page into one string, so
(1) chunks lost their section headings ("Afet Sonrası Yaşamın Sürdürülmesi > Açık alanda"),
which tell both retrieval and the answer step which situation a piece of advice is for;
(2) tables became unreadable runs ("İlçe Altınözü Antakya ... Sayı 28 133 ..."); and
(3) image-only pages (posters, scanned booklets) produced no text at all, so 10 documents,
among them AFAD's main "ilk 72 saat" guides, never reached the index.

Here each page becomes a list of blocks:
  {"kind": "heading", "level": 1-3, "text": ...}   from font size / boldness relative to body text
  {"kind": "text", "text": ...}                     one paragraph (a PyMuPDF text block)
  {"kind": "table", "text": ...}                    a detected table rendered as "header: value" lines
Pages without a text layer are marked "needs_ocr"; src/preprocessing/ocr_pages.py fills them.
Broken legacy-font Turkish is repaired per document (src/preprocessing/repair_encoding.py).

  python -m src.preprocessing.extract_structured                      # whole corpus
  python -m src.preprocessing.extract_structured --only olagandisi.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from src.preprocessing.repair_encoding import detect_family, repair_text


DEFAULT_INPUT_ROOT = Path("data/raw/documents")
DEFAULT_OUTPUT_ROOT = Path("data/processed/structured_v5")
ORGANIZATION = "AFAD"

LEVEL1_RATIO = 1.6        # font size / body size
LEVEL2_RATIO = 1.2
MAX_HEADING_CHARS = 100
MAX_BOLD_HEADING_CHARS = 60
REPEATED_LINE_SHARE = 0.4  # header/footer lines seen on >= 40% of pages are dropped
MIN_PAGES_FOR_REPEATS = 5
OCR_MIN_CHARS = 50         # a page with less extracted text than this is sent to OCR
HORIZONTAL_TABLE_MAX_ROWS = 3
BULLETS = "•●▪■◦○-–"
DIGITS_RE = re.compile(r"\d+")
TEXT_ONLY_FLAGS = pymupdf.TEXTFLAGS_TEXT  # never decode embedded images: slow (10 s on some pages) and unused
HYPHEN_BREAK_RE = re.compile(r"(?<=\w)[-­]\s+(?=[a-zçğıöşü])")


def is_bold(span: dict[str, Any]) -> bool:
    font = span.get("font", "")
    return bool(span.get("flags", 0) & 16) or any(mark in font for mark in ("Bold", "Black", "Heavy", "Semibold"))


def line_record(line: dict[str, Any]) -> dict[str, Any] | None:
    spans = [span for span in line["spans"] if span["text"].strip()]
    if not spans:
        return None
    text = "".join(span["text"] for span in line["spans"]).strip()
    chars = sum(len(span["text"].strip()) for span in spans)
    bold_chars = sum(len(span["text"].strip()) for span in spans if is_bold(span))
    size = max(spans, key=lambda span: len(span["text"].strip()))["size"]
    return {"text": text, "size": round(size * 2) / 2, "bold": bold_chars >= 0.8 * chars, "bbox": line["bbox"]}


def inside(bbox: tuple[float, ...], boxes: list[tuple[float, ...]]) -> bool:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for box in boxes)


def normalise_repeat(text: str) -> str:
    return DIGITS_RE.sub("#", " ".join(text.lower().split()))


def cell(value: Any) -> str:
    return " ".join(str(value or "").split())


def render_table(table: Any) -> str:
    """Rows as "header: value; header: value"; short wide tables (İlçe/Sayı) are paired by column position."""
    rows = [[cell(value) for value in row] for row in table.extract()]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if len(rows) <= HORIZONTAL_TABLE_MAX_ROWS and width >= 3 * len(rows):
        return render_horizontal(table, rows)
    boxes = [list(row.cells) for row in table.rows]
    headers = [(boxes[0][column], value) for column, value in enumerate(rows[0])
               if value and column < len(boxes[0]) and boxes[0][column] is not None]
    if not headers or len(rows) < 2:
        return "\n".join(" | ".join(value for value in row if value) for row in rows if any(row))

    def header_for(box: tuple[float, ...]) -> int:
        """The header whose (possibly merged) cell spans this cell's x-centre, else the nearest one."""
        x = (box[0] + box[2]) / 2
        spanning = [index for index, (head, _) in enumerate(headers) if head[0] - 1 <= x <= head[2] + 1]
        return spanning[0] if spanning else min(range(len(headers)), key=lambda index: abs((headers[index][0][0] + headers[index][0][2]) / 2 - x))

    records: list[dict[int, list[str]]] = []
    for row_index in range(1, len(rows)):
        assigned: dict[int, list[str]] = {}
        for column, value in enumerate(rows[row_index]):
            if value and column < len(boxes[row_index]) and boxes[row_index][column] is not None:
                assigned.setdefault(header_for(boxes[row_index][column]), []).append(value)
        if not assigned:
            continue
        if records and len(assigned) * 2 < len(records[-1]) and 0 not in assigned:
            for index, values in assigned.items():  # a cell's text wrapped onto the next table row
                records[-1].setdefault(index, []).extend(values)
            continue
        if 0 not in assigned and records and 0 in records[-1]:
            assigned[0] = records[-1][0]  # merged first-column cell (e.g. one district, several parcels)
        records.append(assigned)
    return "\n".join("; ".join(f"{headers[index][1]}: {' '.join(values)}" for index, values in sorted(record.items()))
                     for record in records)


def render_horizontal(table: Any, rows: list[list[str]]) -> str:
    """Header row of names, value row(s) below: assign each value to the header cell nearest in x."""
    cell_boxes = [list(row.cells) for row in table.rows]

    def centre(row: int, column: int) -> float | None:
        if row >= len(cell_boxes) or column >= len(cell_boxes[row]) or cell_boxes[row][column] is None:
            return None
        box = cell_boxes[row][column]
        return (box[0] + box[2]) / 2

    headers = [(centre(0, column), value) for column, value in enumerate(rows[0]) if value and centre(0, column) is not None]
    if not headers:
        return " | ".join(" ".join(value for value in row if value) for row in rows)
    lines = []
    for row_index in range(1, len(rows)):
        assigned: dict[int, list[str]] = {}
        label = ""
        for column, value in enumerate(rows[row_index]):
            x = centre(row_index, column)
            if not value or x is None:
                continue
            nearest = min(range(len(headers)), key=lambda index: abs(headers[index][0] - x))
            if abs(headers[nearest][0] - x) > 60 and not any(char.isdigit() for char in value):
                label = value  # row label such as "Sayı (ad)"
                continue
            assigned.setdefault(nearest, []).append(value)
        pairs = [f"{headers[index][1]}: {''.join(values)}" for index, values in sorted(assigned.items())]
        lines.append((f"{label} — " if label else "") + "; ".join(pairs))
    return "\n".join(lines)


def heading_level(line: dict[str, Any], body_size: float, single_line_block: bool) -> int | None:
    text = line["text"]
    letters = sum(char.isalpha() for char in text)
    if len(text) > MAX_HEADING_CHARS or letters < 3 or text[0] in BULLETS or text.endswith((".", ",", ";")):
        return None
    ratio = line["size"] / body_size
    if ratio >= LEVEL1_RATIO:
        return 1
    if ratio >= LEVEL2_RATIO:
        return 2
    if line["bold"] and ratio >= 0.95 and single_line_block and len(text) <= MAX_BOLD_HEADING_CHARS:
        return 3
    return None


def page_blocks(page: Any, body_size: float, repeated: set[str], family: str | None) -> tuple[list[dict[str, Any]], int]:
    tables = page.find_tables().tables
    table_boxes = [tuple(table.bbox) for table in tables]
    blocks: list[dict[str, Any]] = []
    chars = 0
    items: list[tuple[float, dict[str, Any]]] = []
    for block in page.get_text("dict", flags=TEXT_ONLY_FLAGS)["blocks"]:
        if block.get("type") != 0:
            continue
        lines = [record for record in map(line_record, block["lines"]) if record]
        lines = [line for line in lines if not inside(line["bbox"], table_boxes)
                 and normalise_repeat(line["text"]) not in repeated and not line["text"].isdigit()]
        if not lines:
            continue
        paragraph: list[str] = []
        for line in lines:
            text = repair_text(line["text"], family)
            chars += len(text)
            level = heading_level({**line, "text": text}, body_size, len(lines) == 1 or line is lines[0] and line["bold"])
            if level and not paragraph:
                if blocks_tail_heading(items, level):
                    if text not in items[-1][1]["text"]:  # decorative headings repeat the same line
                        items[-1][1]["text"] += " " + text  # a heading broken over two lines
                else:
                    items.append((line["bbox"][1], {"kind": "heading", "level": level, "text": text}))
                continue
            paragraph.append(text)
        if paragraph:
            joined = HYPHEN_BREAK_RE.sub("", " ".join(paragraph))
            items.append((block["bbox"][1], {"kind": "text", "text": joined}))
    for table in tables:
        rendered = repair_text(render_table(table), family)
        if rendered.strip():
            chars += len(rendered)
            items.append((table.bbox[1], {"kind": "table", "text": rendered}))
    items.sort(key=lambda item: item[0])
    blocks = [block for _, block in items]
    return blocks, chars


def blocks_tail_heading(items: list[tuple[float, dict[str, Any]]], level: int) -> bool:
    return bool(items) and items[-1][1]["kind"] == "heading" and items[-1][1]["level"] == level \
        and len(items[-1][1]["text"]) < MAX_HEADING_CHARS // 2


def document_profile(document: Any) -> tuple[float, set[str], str | None]:
    """Body font size (most common by characters), repeated header/footer lines, broken-font family."""
    sizes: Counter[float] = Counter()
    line_pages: Counter[str] = Counter()
    raw_text: list[str] = []
    for page in document:
        seen: set[str] = set()
        for block in page.get_text("dict", flags=TEXT_ONLY_FLAGS)["blocks"]:
            for line in block.get("lines", []):
                record = line_record(line)
                if not record:
                    continue
                sizes[record["size"]] += len(record["text"])
                seen.add(normalise_repeat(record["text"]))
                raw_text.append(record["text"])
        line_pages.update(seen)
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0
    repeated: set[str] = set()
    if document.page_count >= MIN_PAGES_FOR_REPEATS:
        repeated = {line for line, count in line_pages.items() if count >= REPEATED_LINE_SHARE * document.page_count}
    return body_size, repeated, detect_family("\n".join(raw_text))


def extract_document(pdf_path: Path, input_root: Path) -> dict[str, Any]:
    relative_path = pdf_path.relative_to(input_root).as_posix()
    with pymupdf.open(pdf_path) as document:
        body_size, repeated, family = document_profile(document)
        pages = []
        for index, page in enumerate(document):
            blocks, chars = page_blocks(page, body_size, repeated, family)
            pages.append({"page": index + 1, "blocks": blocks, "needs_ocr": chars < OCR_MIN_CHARS,
                          "has_images": bool(page.get_images())})
    return {"source_file": pdf_path.name, "relative_path": relative_path, "organization": ORGANIZATION,
            "body_font_size": body_size, "encoding_family": family, "pages": pages}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Structure-preserving PDF extraction.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--only", nargs="*", default=[], help="Process only these PDF file names.")
    args = parser.parse_args()

    pdfs = sorted(path for path in args.input_root.rglob("*.pdf"))
    if args.only:
        wanted = {unicodedata.normalize("NFC", name) for name in args.only}
        pdfs = [path for path in pdfs if unicodedata.normalize("NFC", path.name) in wanted]
    totals = Counter()
    for pdf_path in pdfs:
        record = extract_document(pdf_path, args.input_root)
        output = args.output_root / Path(record["relative_path"]).with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        ocr = sum(page["needs_ocr"] for page in record["pages"])
        headings = sum(block["kind"] == "heading" for page in record["pages"] for block in page["blocks"])
        tables = sum(block["kind"] == "table" for page in record["pages"] for block in page["blocks"])
        totals.update(pages=len(record["pages"]), ocr=ocr, headings=headings, tables=tables, documents=1)
        print(f"{record['relative_path']}: {len(record['pages'])} pages, {headings} headings, {tables} tables, "
              f"{ocr} need OCR" + (f", encoding {record['encoding_family']}" if record["encoding_family"] else ""),
              flush=True)
    print(dict(totals))


if __name__ == "__main__":
    main()
