"""Rule-based earthquake relevance filtering for processed page JSON files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("data/processed/text")
DEFAULT_OUTPUT_ROOT = Path("data/processed/earthquake_relevance")

LEADING_PAGE_NUMBER_RE = re.compile(r"^\s*(\d{1,3})(?:[.)]\s+|\s+)")

STRONG_EARTHQUAKE_PATTERNS: dict[str, re.Pattern[str]] = {
    "deprem": re.compile(r"(?<!\w)deprem\w*", re.IGNORECASE),
    "sismik": re.compile(r"(?<!\w)sismik\w*", re.IGNORECASE),
    "aktif fay": re.compile(r"(?<!\w)aktif\s+fay\w*", re.IGNORECASE),
    "fay": re.compile(r"(?<!\w)fay(?:\s+hatt\w*|\s+zon\w*|lar\w*|ler\w*|[ıiuü]\w*)?(?!\w)", re.IGNORECASE),
    "yer ivmesi": re.compile(r"(?<!\w)yer\s+ivme\w*", re.IGNORECASE),
    "artçı": re.compile(r"(?<!\w)art[çc][ıi]\w*", re.IGNORECASE),
    "bina hasarı": re.compile(r"(?<!\w)bina\s+hasar\w*", re.IGNORECASE),
    "yapı stoku": re.compile(r"(?<!\w)yap[ıi]\s+stoku\w*", re.IGNORECASE),
    "deprem tehlikesi": re.compile(r"(?<!\w)deprem\s+tehlike\w*", re.IGNORECASE),
    "deprem riski": re.compile(r"(?<!\w)deprem\s+risk\w*", re.IGNORECASE),
    "mikrobölgeleme": re.compile(r"(?<!\w)mikro(?:b[oö]lge|bolge)\w*", re.IGNORECASE),
}

SUPPORTING_EARTHQUAKE_PATTERNS: dict[str, re.Pattern[str]] = {
    "zemin": re.compile(r"(?<!\w)zemin\w*", re.IGNORECASE),
    "toplanma alanı": re.compile(r"(?<!\w)toplanma\s+alan\w*", re.IGNORECASE),
    "tahliye": re.compile(r"(?<!\w)tahliye\w*", re.IGNORECASE),
    "hasar tespiti": re.compile(r"(?<!\w)hasar\s+tespit\w*", re.IGNORECASE),
    "riskli yapı": re.compile(r"(?<!\w)riskli\s+yap[ıi]\w*", re.IGNORECASE),
    "yapısal risk": re.compile(r"(?<!\w)yap[ıi]sal\s+risk\w*", re.IGNORECASE),
}

OTHER_DISASTER_PATTERNS: dict[str, re.Pattern[str]] = {
    "heyelan": re.compile(r"(?<!\w)heyelan\w*", re.IGNORECASE),
    "çığ": re.compile(r"(?<!\w)[çc][ıi]g\w*|(?<!\w)[çc][ıi][ğg]\w*", re.IGNORECASE),
    "sel": re.compile(r"(?<!\w)sel(?:\s+afet\w*|ler\w*|in|de|den|e|i)?(?!\w)", re.IGNORECASE),
    "taşkın": re.compile(r"(?<!\w)ta[şs]k[ıi]n\w*", re.IGNORECASE),
    "orman yangını": re.compile(r"(?<!\w)orman\s+yang[ıi]n\w*", re.IGNORECASE),
    "su baskını": re.compile(r"(?<!\w)su\s+bask[ıi]n\w*", re.IGNORECASE),
    "kaya düşmesi": re.compile(r"(?<!\w)kaya\s+d[üu]s\w*|(?<!\w)kaya\s+d[üu][şs]\w*", re.IGNORECASE),
    "kuraklık": re.compile(r"(?<!\w)kurakl[ıi]k\w*", re.IGNORECASE),
    "fırtına": re.compile(r"(?<!\w)f[ıi]rt[ıi]na\w*", re.IGNORECASE),
    "hortum": re.compile(r"(?<!\w)hortum\w*", re.IGNORECASE),
    "maden kazası": re.compile(r"(?<!\w)maden\s+kaza\w*", re.IGNORECASE),
}


@dataclass
class FilterStats:
    relevant: int = 0
    uncertain: int = 0
    excluded: int = 0

    def add(self, relevance: str) -> None:
        if relevance == "relevant":
            self.relevant += 1
        elif relevance == "excluded":
            self.excluded += 1
        else:
            self.uncertain += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify processed AFAD pages by earthquake relevance.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def strip_leading_page_number(text: str, page_number: int | None) -> str:
    """Remove a leading standalone number when it matches the extracted page number."""
    if not text or page_number is None:
        return text

    match = LEADING_PAGE_NUMBER_RE.match(text)
    if not match:
        return text

    try:
        leading_number = int(match.group(1))
    except ValueError:
        return text

    if leading_number != page_number:
        return text

    return text[match.end() :].lstrip()


def find_matches(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    return [name for name, pattern in patterns.items() if pattern.search(text)]


def classify_page(text: str, relative_path: str, is_low_text: bool) -> tuple[str, list[str], list[str], str]:
    haystack = f"{relative_path} {text}".casefold()
    strong_matches = find_matches(haystack, STRONG_EARTHQUAKE_PATTERNS)
    supporting_matches = find_matches(haystack, SUPPORTING_EARTHQUAKE_PATTERNS)
    other_disaster_matches = find_matches(haystack, OTHER_DISASTER_PATTERNS)
    earthquake_matches = sorted(set(strong_matches + supporting_matches))

    if strong_matches:
        return "relevant", earthquake_matches, other_disaster_matches, "strong_earthquake_term"

    if len(supporting_matches) >= 2 and not other_disaster_matches:
        return "relevant", earthquake_matches, other_disaster_matches, "multiple_supporting_earthquake_terms"

    if other_disaster_matches and not earthquake_matches:
        return "excluded", earthquake_matches, other_disaster_matches, "other_disaster_only"

    if is_low_text or not text.strip():
        return "uncertain", earthquake_matches, other_disaster_matches, "low_or_empty_text"

    if supporting_matches:
        return "uncertain", earthquake_matches, other_disaster_matches, "supporting_term_without_clear_context"

    return "uncertain", earthquake_matches, other_disaster_matches, "no_rule_match"


def process_record(record: dict[str, Any]) -> dict[str, Any]:
    text = str(record.get("text", ""))
    page_number = record.get("page")
    page_number = page_number if isinstance(page_number, int) else None
    relative_path = str(record.get("relative_path", ""))
    cleaned_text = strip_leading_page_number(text, page_number)
    is_low_text = bool(record.get("is_low_text", not cleaned_text.strip()))
    relevance, earthquake_terms, other_disaster_terms, reason = classify_page(
        cleaned_text,
        relative_path,
        is_low_text,
    )

    return {
        **record,
        "text": cleaned_text,
        "relevance": relevance,
        "earthquake_terms": earthquake_terms,
        "other_disaster_terms": other_disaster_terms,
        "relevance_reason": reason,
    }


def process_file(input_path: Path, input_root: Path, output_root: Path, stats: FilterStats) -> None:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of page records: {input_path}")

    processed_records = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Expected page record objects in: {input_path}")
        processed_record = process_record(record)
        stats.add(str(processed_record["relevance"]))
        processed_records.append(processed_record)

    output_path = output_root / input_path.relative_to(input_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(processed_records, ensure_ascii=False, indent=2), encoding="utf-8")


def run_filter(input_root: Path, output_root: Path) -> FilterStats:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    stats = FilterStats()

    for input_path in sorted(input_root.rglob("*.json")):
        process_file(input_path, input_root, output_root, stats)

    return stats


def main() -> None:
    args = parse_args()
    stats = run_filter(args.input_root, args.output_root)
    print(f"Relevant pages: {stats.relevant}")
    print(f"Uncertain pages: {stats.uncertain}")
    print(f"Excluded pages: {stats.excluded}")


if __name__ == "__main__":
    main()
