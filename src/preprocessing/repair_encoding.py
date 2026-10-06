"""Repair broken Turkish character encodings in extracted PDF text.

Some AFAD booklets were typeset with legacy Turkish fonts, so PyMuPDF extracts
"dÈëarÈ çÈkmak" instead of "dışarı çıkmak". Each document is classified from its
own character statistics, and only documents that carry a known broken-font
signature are repaired. Genuine characters elsewhere (Ottoman Turkish "mezkûr",
French "é", the "ﬁ" ligature in "coğraﬁ") are left intact or only de-ligatured.

Runs as its own stage between extraction and earthquake filtering, writing to a
separate output root so earlier pipeline versions stay reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("data/processed/text")
DEFAULT_OUTPUT_ROOT = Path("data/processed/text_v3")
DEFAULT_REPORT_FILE = Path("data/processed/encoding_repair_report.json")

FAMILY_LEGACY_LATIN = "legacy_latin"  # È=ı ë=ş è=ğ é=İ ê=Ş, second variant ×=ı ù=ğ û=Ş ü=ş
FAMILY_LEGACY_SYMBOL = "legacy_symbol"  # ›=ı ﬂ=ş ¤=ğ ‹=İ ﬁ=Ş ⁄=Ğ

# A document is classified only when the signature character is both frequent and dense.
MIN_SIGNATURE_COUNT = 20
MIN_SIGNATURE_PER_1000_LETTERS = 2.0

LEGACY_LATIN_MAP = str.maketrans({"È": "ı", "ë": "ş", "è": "ğ", "é": "İ", "ê": "Ş", "×": "ı", "ù": "ğ", "û": "Ş", "‹": "İ"})
LEGACY_SYMBOL_MAP = str.maketrans({"›": "ı", "ﬂ": "ş", "¤": "ğ", "‹": "İ", "ﬁ": "Ş", "⁄": "Ğ"})
LIGATURE_MAP = str.maketrans({"ﬁ": "fi", "ﬂ": "fl"})

WORD_RE = re.compile(r"\S+")
LEGACY_LATIN_SECOND_VARIANT = re.compile(r"[×ù]")
# "Farkl2 senaryolar2": the digit 2 stands for ı after at least three letters.
DIGIT_TWO_AS_DOTLESS_I = re.compile(r"(?<=[A-Za-zÇĞİÖŞÜçğıöşü]{3})2(?=[a-zçğıöşü]|\b)")
# Characters that may appear lowercase inside an otherwise upper-case legacy word.
LEGACY_LOWERCASE_MARKS = set("çÈëèéêù×")


def letter_count(text: str) -> int:
    return sum(char.isalpha() for char in text)


def detect_family(document_text: str) -> str | None:
    """Classify a whole document by its broken-font signature, or None if it looks clean."""
    letters = max(letter_count(document_text), 1)
    for family, signature in ((FAMILY_LEGACY_LATIN, "È"), (FAMILY_LEGACY_SYMBOL, "›")):
        count = document_text.count(signature)
        if count >= MIN_SIGNATURE_COUNT and 1000 * count / letters >= MIN_SIGNATURE_PER_1000_LETTERS:
            return family
    return None


def repair_legacy_latin_word(word: str) -> str:
    if LEGACY_LATIN_SECOND_VARIANT.search(word):
        word = word.replace("ü", "ş").replace("Ü", "Ş")
    is_upper_word = sum(char.isupper() for char in word) >= 3 and all(
        not char.islower() or char in LEGACY_LOWERCASE_MARKS for char in word
    )
    if is_upper_word and "ç" in word:
        word = word.replace("ç", "Ğ")  # e.g. GÜVENLéçéNéN -> GÜVENLİĞİNİN
    return word.translate(LEGACY_LATIN_MAP)


def repair_text(text: str, family: str | None) -> str:
    if family == FAMILY_LEGACY_LATIN:
        text = WORD_RE.sub(lambda match: repair_legacy_latin_word(match.group()), text)
        return DIGIT_TWO_AS_DOTLESS_I.sub("ı", text)
    if family == FAMILY_LEGACY_SYMBOL:
        return text.translate(LEGACY_SYMBOL_MAP)
    return text.translate(LIGATURE_MAP)


def repair_document(pages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = "\n".join(str(page.get("text", "")) for page in pages)
    family = detect_family(original)
    repaired_pages = [{**page, "text": repair_text(str(page.get("text", "")), family)} for page in pages]
    changed_pages = sum(1 for before, after in zip(pages, repaired_pages) if before.get("text") != after["text"])
    return repaired_pages, {"family": family, "changed_pages": changed_pages}


def run(input_root: Path, output_root: Path, report_file: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"documents": {}, "families": Counter()}
    for input_path in sorted(input_root.rglob("*.json")):
        pages = json.loads(input_path.read_text(encoding="utf-8"))
        repaired_pages, info = repair_document(pages)
        output_path = output_root / input_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(repaired_pages, ensure_ascii=False, indent=2), encoding="utf-8")
        report["families"][str(info["family"])] += 1
        if info["changed_pages"]:
            report["documents"][input_path.relative_to(input_root).as_posix()] = info
    report["families"] = dict(report["families"])
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair broken Turkish font encodings in extracted text.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-file", type=Path, default=DEFAULT_REPORT_FILE)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.input_root.resolve() == args.output_root.resolve():
        raise SystemExit("Input and output roots must differ so earlier pipeline versions stay reproducible.")
    report = run(args.input_root, args.output_root, args.report_file)
    print(f"Document families: {report['families']}")
    for path, info in report["documents"].items():
        print(f"  {info['family'] or 'ligature-only':15} {info['changed_pages']:5} pages  {path}")
    print(f"Output: {args.output_root}")
    print(f"Report: {args.report_file}")


if __name__ == "__main__":
    main()
