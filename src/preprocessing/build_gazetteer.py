"""Build the location gazetteer used for region-aware retrieval.

Outputs (small, versioned files under data/geo/):
- turkiye_il_ilce.json: 81 provinces, 973 districts (TurkiyeAPI, MIT licence),
  with aliases, districts whose names exist in several provinces, and names that
  are also common Turkish words (only matched when capitalised in a query).
- irap_province_map.json: IRAP source file -> province, inferred from content
  because several file names do not name the province (e.g. "Irap-.pdf").
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.rag.query_normalization import turkish_lower


SOURCE_URL = "https://api.turkiyeapi.dev/v1/provinces?fields=id,name,districts&limit=100"
SOURCE_NOTE = "TurkiyeAPI (https://github.com/ubeydeozdmr/turkiye-api), MIT licence"
DEFAULT_METADATA_FILE = Path("data/indexes/mobile_v4/metadata_v2.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/geo")
ALIASES = {
    "Maraş": "Kahramanmaraş",
    "Antep": "Gaziantep",
    "Urfa": "Şanlıurfa",
    "Afyon": "Afyonkarahisar",
    "İçel": "Mersin",
}
# A place name is treated as a common word if most of its corpus occurrences are lowercase.
COMMON_WORD_LOWERCASE_SHARE = 0.5
MIN_OCCURRENCES_FOR_COMMON_WORD = 5
# ...or written lowercase often enough in absolute terms (IRAPs capitalise real place names a lot).
MIN_LOWERCASE_OCCURRENCES_FOR_COMMON_WORD = 20
# An IRAP file is assigned only if its top province clearly dominates the runner-up.
DOMINANCE = 3


def load_source(raw_file: Path | None) -> list[dict[str, Any]]:
    if raw_file is not None:
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    provinces = payload["data"]
    if len(provinces) != 81 or sum(len(p["districts"]) for p in provinces) != 973:
        raise SystemExit("Unexpected province/district counts; refusing to build gazetteer.")
    return provinces


TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]+")


def token_counts(texts: list[str]) -> tuple[Counter[str], Counter[str]]:
    """One pass over the corpus: occurrences per lowercased token, and how many were written lowercase."""
    total: Counter[str] = Counter()
    lowercase: Counter[str] = Counter()
    for text in texts:
        for token in TOKEN_RE.findall(text):
            key = turkish_lower(token)
            total[key] += 1
            if token[0].islower():
                lowercase[key] += 1
    return total, lowercase


def common_word_names(names: set[str], texts: list[str]) -> list[str]:
    """Names mostly written lowercase in the corpus (e.g. 'kale', 'pazar') are common words."""
    total, lowercase = token_counts(texts)
    common = []
    for name in sorted(names):
        key = turkish_lower(name)
        mostly_lowercase = total[key] >= MIN_OCCURRENCES_FOR_COMMON_WORD and lowercase[key] / total[key] > COMMON_WORD_LOWERCASE_SHARE
        if mostly_lowercase or lowercase[key] >= MIN_LOWERCASE_OCCURRENCES_FOR_COMMON_WORD:
            common.append(name)
    return common


def irap_province_map(metadata: list[dict[str, Any]], provinces: list[str]) -> dict[str, str]:
    texts: dict[str, list[str]] = defaultdict(list)
    for record in metadata:
        if "/irap/" in record["relative_path"]:
            texts[record["source_file"]].append(record["text"])
    mapping: dict[str, str] = {}
    for source_file, chunks in sorted(texts.items()):
        total, _ = token_counts(chunks)
        counts = Counter({province: total[turkish_lower(province)] for province in provinces})
        (top, top_count), (_, second_count) = counts.most_common(2)
        if top_count >= DOMINANCE * max(second_count, 1):
            mapping[source_file] = top
        else:
            print(f"  skipped (no dominant province): {source_file} {counts.most_common(2)}")
    return mapping


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build the province/district gazetteer.")
    parser.add_argument("--raw-file", type=Path, default=None, help="Use a downloaded TurkiyeAPI response instead of fetching.")
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = load_source(args.raw_file)
    metadata = [json.loads(line) for line in args.metadata_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    province_names = [p["name"] for p in source]

    district_provinces: dict[str, list[str]] = defaultdict(list)
    for province in source:
        for district in province["districts"]:
            district_provinces[district["name"]].append(province["name"])
    ambiguous = {name: sorted(owners) for name, owners in district_provinces.items() if len(owners) > 1}
    # Province names are distinctive and must match even when typed lowercase ("istanbulda").
    district_names = set(district_provinces) - set(province_names)
    common = common_word_names(district_names, [record["text"] for record in metadata])

    gazetteer = {
        "source": SOURCE_NOTE,
        "provinces": [
            {"name": p["name"], "districts": [{"name": d["name"], "population": d.get("population")} for d in p["districts"]]}
            for p in source
        ],
        "aliases": ALIASES,
        "ambiguous_districts": ambiguous,
        "common_word_names": common,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "turkiye_il_ilce.json").write_text(json.dumps(gazetteer, ensure_ascii=False, indent=1), encoding="utf-8")

    mapping = irap_province_map(metadata, province_names)
    (args.output_dir / "irap_province_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")

    covered = set(mapping.values())
    print(f"Provinces: {len(source)}  districts: {len(district_provinces)} unique names / {sum(len(p['districts']) for p in source)} total")
    print(f"Ambiguous district names: {len(ambiguous)}  e.g. {list(ambiguous.items())[:4]}")
    print(f"Common-word place names ({len(common)}): {common}")
    print(f"IRAP files mapped: {len(mapping)}  provinces covered: {len(covered)}")
    print(f"Provinces without IRAP: {sorted(set(province_names) - covered)}")


if __name__ == "__main__":
    main()
