"""Create source-type enriched metadata for the V2 mobile index."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_FILE = Path("data/indexes/mobile_v2/metadata_v2.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/indexes/mobile_v2/metadata_v2_source_enriched.jsonl")
DEFAULT_REVIEW_FILE = Path("data/evaluation/source_metadata_review.txt")
DEFAULT_UNKNOWN_FILE = Path("data/evaluation/source_metadata_unknown.txt")
DEFAULT_INDEX_FILE = Path("data/indexes/mobile_v2/afad_e5_small_v2.faiss")

DOCUMENT_TYPE_ORDER = ["citizen_guidance", "training_guidance", "plan", "technical", "unknown"]
PRIORITY_ORDER = [3, 2, 1]
REQUIRED_ADDED_FIELDS = {
    "document_type",
    "audience",
    "guidance_level",
    "source_priority",
    "source_priority_reason",
}
OLD_TYPE_COUNTS = {
    "citizen_guidance": 106,
    "training_guidance": 1502,
    "plan": 6003,
    "technical": 680,
    "unknown": 943,
}

CITIZEN_GUIDANCE = {
    "document_type": "citizen_guidance",
    "audience": "citizen",
    "guidance_level": "direct",
    "source_priority": 3,
    "source_priority_reason": "Direct AFAD citizen guidance document",
}
PRACTICAL_GUIDANCE = {
    "document_type": "training_guidance",
    "audience": "mixed",
    "guidance_level": "practical",
    "source_priority": 2,
    "source_priority_reason": "Practical disaster safety/support guidance",
}
PLAN_DOCUMENT = {
    "document_type": "plan",
    "audience": "institutional",
    "guidance_level": "indirect",
    "source_priority": 1,
    "source_priority_reason": "Institutional disaster plan or strategy document",
}
TECHNICAL_DOCUMENT = {
    "document_type": "technical",
    "audience": "professional",
    "guidance_level": "indirect",
    "source_priority": 1,
    "source_priority_reason": "Technical or scientific source",
}
UNKNOWN_DOCUMENT = {
    "document_type": "unknown",
    "audience": "unknown",
    "guidance_level": "unknown",
    "source_priority": 1,
    "source_priority_reason": "Could not classify confidently",
}

EXPLICIT_SOURCE_PATTERNS = [
    ("ilk72saat1.pdf", CITIZEN_GUIDANCE),
    ("ilk72saatengelliler.pdf", CITIZEN_GUIDANCE),
    ("olagandisi.pdf", CITIZEN_GUIDANCE),
    ("sorucevap-kitapcik-2.pdf", CITIZEN_GUIDANCE),
    ("hasar_tespit_-*10x21.pdf", CITIZEN_GUIDANCE),
    ("13_afad_bilgilendirme_afisleri_tr_afet_cantasi_info*.pdf", CITIZEN_GUIDANCE),
    ("psikososyal_-*10x21.pdf", PRACTICAL_GUIDANCE),
    ("yapilacak_yardimlar*-_10x21.pdf", PRACTICAL_GUIDANCE),
    ("yapilacak_yardimlar*10x21.pdf", PRACTICAL_GUIDANCE),
    ("afet-barinma-faaliyetleri.pdf", PRACTICAL_GUIDANCE),
    ("04_ikaz_alarm_isaretleri_tr-revize.pdf", PRACTICAL_GUIDANCE),
    ("yapisalolmayanriskler.pdf", PRACTICAL_GUIDANCE),
    ("yapisalriskler.pdf", PRACTICAL_GUIDANCE),
    ("tamp.pdf", PLAN_DOCUMENT),
    ("28032022-tarap-kitap_v6.pdf", PLAN_DOCUMENT),
    ("udsep_1402013_kitap.pdf", PLAN_DOCUMENT),
    ("23052025-tasip.pdf", PLAN_DOCUMENT),
    ("samsun.pdf", PLAN_DOCUMENT),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source-priority enriched V2 metadata without modifying the index.")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--unknown-file", type=Path, default=DEFAULT_UNKNOWN_FILE)
    parser.add_argument("--index-file", type=Path, default=DEFAULT_INDEX_FILE)
    return parser.parse_args()


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in normalized if not unicodedata.combining(character))


def contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)


def explicit_classification(normalized_source_file: str) -> dict[str, Any] | None:
    for pattern, classification in EXPLICIT_SOURCE_PATTERNS:
        if fnmatch.fnmatchcase(normalized_source_file, pattern):
            return dict(classification)
    return None


def classify_source(source_file: str) -> dict[str, Any]:
    normalized = normalize_filename(source_file)

    explicit = explicit_classification(normalized)
    if explicit is not None:
        return explicit

    training_terms = (
        "yapisalolmayanrisk",
        "yapisalrisk",
        "egitim",
        "ilk-yardim",
        "ilkyardim",
        "arama-ve-kurtarma",
        "arama kurtarma",
        "enkazda-arama-ve-kurtarma",
        "olay-yeri-yonetimi",
        "afet-yonetim-terimleri-sozlugu",
        "arama-ve-kurtarma-aciklamali-terimler-sozlugu",
        "sozluk",
        "abep",
        "tasip",
    )
    if contains_any(normalized, training_terms):
        return {
            "document_type": "training_guidance",
            "audience": "mixed",
            "guidance_level": "practical",
            "source_priority": 2,
            "source_priority_reason": "Training/safety guidance document",
        }

    plan_terms = (
        "irap",
        "i̇rap",
        "il-afet-risk",
        "afet-risk-azaltma-plani",
        "afet-risk-azaltma-plani",
        "risk-azaltma-plani",
        "strateji",
        "eylem-plani",
        "eylem_plani",
        "plan",
    )
    if contains_any(normalized, plan_terms):
        updated = dict(PLAN_DOCUMENT)
        updated["source_priority_reason"] = "Provincial risk reduction plan"
        return updated

    technical_terms = (
        "sismik",
        "mikrobolgeleme",
        "yerbilimsel",
        "jeoloji",
        "jeolojik",
        "muhendislik",
        "bilimsel",
        "deprem-tarihi",
        "tarihsel",
        "istanbulda-afetler",
    )
    if contains_any(normalized, technical_terms):
        return dict(TECHNICAL_DOCUMENT)

    return dict(UNKNOWN_DOCUMENT)


def read_metadata(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "chunk_id" not in record or "source_file" not in record:
                raise ValueError(f"Missing required metadata field at line {line_number}")
            records.append(record)
    return records


def enrich_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        updated.update(classify_source(str(record["source_file"])))
        enriched.append(updated)
    return enriched


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def unique_sources_by_type(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    sources: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sources[str(record["document_type"])].add(str(record["source_file"]))
    return {document_type: sorted(values, key=normalize_filename) for document_type, values in sources.items()}


def explicit_mapping_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(record["source_file"]) for record in records)
    by_source: dict[str, dict[str, Any]] = {}
    for record in records:
        source_file = str(record["source_file"])
        if source_file in by_source:
            continue
        if explicit_classification(normalize_filename(source_file)) is None:
            continue
        by_source[source_file] = {
            "filename": source_file,
            "document_type": record["document_type"],
            "source_priority": record["source_priority"],
            "chunk_count": counts[source_file],
        }
    return sorted(by_source.values(), key=lambda row: normalize_filename(str(row["filename"])))


def write_review(path: Path, records: list[dict[str, Any]]) -> None:
    by_type = unique_sources_by_type(records)
    lines: list[str] = []
    for document_type in DOCUMENT_TYPE_ORDER:
        examples = by_type.get(document_type, [])
        priority = records_by_type_priority(records, document_type)
        lines.extend(
            [
                "=" * 40,
                f"DOCUMENT TYPE: {document_type}",
                f"source_priority: {priority}",
                "",
            ]
        )
        for source_file in examples[:20]:
            lines.append(f"- {source_file}")
        if len(examples) > 20:
            lines.append(f"... ({len(examples) - 20} more)")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def records_by_type_priority(records: list[dict[str, Any]], document_type: str) -> str:
    priorities = sorted({int(record["source_priority"]) for record in records if record["document_type"] == document_type})
    return ", ".join(str(priority) for priority in priorities) if priorities else "N/A"


def write_unknown(path: Path, records: list[dict[str, Any]]) -> None:
    unknown_sources = unique_sources_by_type(records).get("unknown", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(unknown_sources) + ("\n" if unknown_sources else ""), encoding="utf-8")


def validate_output(input_records: list[dict[str, Any]], enriched_records: list[dict[str, Any]]) -> None:
    if len(input_records) != len(enriched_records):
        raise RuntimeError(f"Chunk count changed: input={len(input_records)}, output={len(enriched_records)}")
    input_ids = [str(record["chunk_id"]) for record in input_records]
    output_ids = [str(record["chunk_id"]) for record in enriched_records]
    if input_ids != output_ids:
        raise RuntimeError("Chunk IDs or row order changed during enrichment")
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("Duplicate chunk_id found in enriched metadata")
    for index, record in enumerate(enriched_records, start=1):
        missing = REQUIRED_ADDED_FIELDS - set(record)
        if missing:
            raise RuntimeError(f"Missing enriched fields at output row {index}: {', '.join(sorted(missing))}")


def print_summary(records: list[dict[str, Any]], source_unchanged: bool, index_unchanged: bool, review_file: Path, unknown_file: Path) -> None:
    type_counts = Counter(str(record["document_type"]) for record in records)
    priority_counts = Counter(int(record["source_priority"]) for record in records)
    unique_sources = {str(record["source_file"]) for record in records}
    sources = unique_sources_by_type(records)
    explicit_rows = explicit_mapping_rows(records)

    print(f"Total chunks: {len(records)}")
    print("")
    print("OLD:")
    for document_type in DOCUMENT_TYPE_ORDER:
        print(f"{document_type}: {OLD_TYPE_COUNTS.get(document_type, 0)}")
    print("")
    print("NEW:")
    for document_type in DOCUMENT_TYPE_ORDER:
        print(f"{document_type}: {type_counts.get(document_type, 0)}")
    print("")
    print("document_type counts:")
    for document_type in DOCUMENT_TYPE_ORDER:
        print(f"{document_type}: {type_counts.get(document_type, 0)}")
    print("")
    print("source_priority counts:")
    for priority in PRIORITY_ORDER:
        print(f"{priority}: {priority_counts.get(priority, 0)}")
    print("")
    print(f"Unique source files: {len(unique_sources)}")
    print("")
    print("explicit mapping table:")
    print("filename | document_type | source_priority | chunk_count")
    for row in explicit_rows:
        print(f"{row['filename']} | {row['document_type']} | {row['source_priority']} | {row['chunk_count']}")
    print("")
    for document_type in DOCUMENT_TYPE_ORDER:
        print(f"DOCUMENT TYPE: {document_type}")
        for source_file in sources.get(document_type, [])[:20]:
            print(f"- {source_file}")
        if len(sources.get(document_type, [])) > 20:
            print(f"... ({len(sources[document_type]) - 20} more)")
        print("")
    print(f"metadata_v2.jsonl unchanged: {str(source_unchanged).lower()}")
    print(f"V2 FAISS index unchanged: {str(index_unchanged).lower()}")
    print("chunk count unchanged: true")
    print("chunk_id order unchanged: true")
    print("retrieval results generated: false")
    print("score boost applied: false")
    print("")
    print(f"Review file: {review_file}")
    print(f"Unknown sources file: {unknown_file}")
    print("")
    print("Source metadata classification updated.")
    print("Retrieval behavior remains unchanged.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    input_hash_before = file_hash(args.input_file)
    index_hash_before = file_hash(args.index_file)
    input_records = read_metadata(args.input_file)
    enriched_records = enrich_records(input_records)
    validate_output(input_records, enriched_records)
    write_jsonl(args.output_file, enriched_records)
    write_review(args.review_file, enriched_records)
    write_unknown(args.unknown_file, enriched_records)
    input_hash_after = file_hash(args.input_file)
    index_hash_after = file_hash(args.index_file)
    print_summary(
        enriched_records,
        source_unchanged=input_hash_before == input_hash_after,
        index_unchanged=index_hash_before == index_hash_after,
        review_file=args.review_file,
        unknown_file=args.unknown_file,
    )


if __name__ == "__main__":
    main()
