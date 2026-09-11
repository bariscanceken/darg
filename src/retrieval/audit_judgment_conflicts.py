"""Audit conflicting development relevance judgments without modifying labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_JUDGMENT_FILES = [
    Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_score_fusion_judgments.csv"),
    Path("data/evaluation/retrieval_hybrid_judgments.csv"),
    Path("data/evaluation/retrieval_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_judgments.csv"),
]
DEFAULT_METADATA_FILES = [
    Path("data/indexes/mobile_v2/metadata_v2_source_enriched.jsonl"),
    Path("data/indexes/mobile_v2/metadata_v2.jsonl"),
    Path("data/indexes/mobile/metadata.jsonl"),
]
DEFAULT_JSON_FILE = Path("data/evaluation/judgment_conflicts.json")
DEFAULT_REVIEW_FILE = Path("data/evaluation/judgment_conflicts_review.txt")
SEPARATOR = "=" * 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit conflicting Q01-Q20 relevance judgments.")
    parser.add_argument("--judgment-files", nargs="+", type=Path, default=DEFAULT_JUDGMENT_FILES)
    parser.add_argument("--metadata-files", nargs="+", type=Path, default=DEFAULT_METADATA_FILES)
    parser.add_argument("--json-file", type=Path, default=DEFAULT_JSON_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    return parser.parse_args()


def read_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_id = str(record.get("chunk_id", ""))
                if chunk_id and chunk_id not in lookup:
                    lookup[chunk_id] = record
    return lookup


def row_source_info(row: dict[str, str]) -> dict[str, Any]:
    fields = [
        "alpha",
        "system",
        "rank",
        "final_rank",
        "reranked_rank",
        "original_faiss_rank",
        "faiss_rank",
        "similarity_score",
        "faiss_score",
        "rerank_score",
        "reranker_score",
        "source_file",
        "page_start",
        "page_end",
    ]
    return {field: row[field] for field in fields if field in row and str(row[field]).strip()}


def read_judgments(paths: list[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                label = str(row.get("relevant", "")).strip()
                question_id = str(row.get("question_id", "")).strip()
                chunk_id = str(row.get("chunk_id", "")).strip()
                if label not in {"0", "1"}:
                    continue
                if not question_id.startswith("Q") or not chunk_id:
                    continue
                grouped[(question_id, chunk_id)].append(
                    {
                        "label": label,
                        "question_id": question_id,
                        "question": str(row.get("question", "")),
                        "chunk_id": chunk_id,
                        "source_judgment_file": str(path),
                        "csv_row_number": row_number,
                        "source_information": row_source_info(row),
                    }
                )
    return grouped


def find_conflicts(grouped: dict[tuple[str, str], list[dict[str, Any]]], metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for (question_id, chunk_id), judgments in grouped.items():
        labels = {judgment["label"] for judgment in judgments}
        if len(labels) <= 1:
            continue
        record = metadata.get(chunk_id, {})
        first = judgments[0]
        source_file = record.get("source_file") or first["source_information"].get("source_file", "")
        page_start = record.get("page_start") or first["source_information"].get("page_start", "")
        page_end = record.get("page_end") or first["source_information"].get("page_end", "")
        conflicts.append(
            {
                "question_id": question_id,
                "question": first["question"],
                "chunk_id": chunk_id,
                "source_file": source_file,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_text": str(record.get("text", "")),
                "labels_found": sorted(labels),
                "judgments": sorted(
                    judgments,
                    key=lambda item: (item["label"], item["source_judgment_file"], int(item["csv_row_number"])),
                ),
                "final_label": "",
            }
        )
    conflicts.sort(key=lambda item: (item["question_id"], item["chunk_id"]))
    return conflicts


def render_source_information(info: dict[str, Any]) -> str:
    if not info:
        return "EMPTY"
    return "; ".join(f"{key}: {value}" for key, value in info.items())


def render_review(conflicts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for conflict in conflicts:
        lines = [
            SEPARATOR,
            f"QUESTION: {conflict['question_id']}",
            f"QUESTION TEXT: {conflict['question']}",
            "",
            f"CHUNK ID: {conflict['chunk_id']}",
            f"SOURCE FILE: {conflict['source_file']}",
            f"PAGE/PAGES: {conflict['page_start']}-{conflict['page_end']}",
            "",
            "CHUNK TEXT:",
            conflict["chunk_text"],
            "",
            "FOUND LABELS:",
            "",
        ]
        judgments_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for judgment in conflict["judgments"]:
            judgments_by_label[judgment["label"]].append(judgment)

        for label in sorted(judgments_by_label):
            lines.append(f"Label: {label}")
            for judgment in judgments_by_label[label]:
                lines.extend(
                    [
                        f"Source judgment file: {judgment['source_judgment_file']}",
                        f"Original row / source information: CSV line {judgment['csv_row_number']}; "
                        f"{render_source_information(judgment['source_information'])}",
                        "",
                    ]
                )
        lines.extend(["FINAL LABEL:", "", SEPARATOR])
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_json(path: Path, conflicts: list[dict[str, Any]], judgment_files: list[Path]) -> None:
    payload = {
        "conflicting_question_chunk_pairs": len(conflicts),
        "questions_affected": sorted({conflict["question_id"] for conflict in conflicts}),
        "files_scanned": [str(path) for path in judgment_files if path.exists()],
        "files_containing_conflicts": sorted(
            {
                judgment["source_judgment_file"]
                for conflict in conflicts
                for judgment in conflict["judgments"]
            }
        ),
        "conflicts": conflicts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_review(path: Path, conflicts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(conflicts), encoding="utf-8")


def print_summary(conflicts: list[dict[str, Any]]) -> None:
    questions = sorted({conflict["question_id"] for conflict in conflicts})
    files = sorted(
        {
            judgment["source_judgment_file"]
            for conflict in conflicts
            for judgment in conflict["judgments"]
        }
    )
    print(f"Conflicting question/chunk pairs: {len(conflicts)}")
    print(f"Questions affected: {', '.join(questions) if questions else 'NONE'}")
    print("Files containing conflicts:")
    if files:
        for path in files:
            print(f"- {path}")
    else:
        print("NONE")
    print("")
    print("Hiçbir relevance label değiştirilmedi.")
    print("Judgment conflict audit completed.")
    print("No labels were automatically modified.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    metadata = read_metadata(args.metadata_files)
    grouped = read_judgments(args.judgment_files)
    conflicts = find_conflicts(grouped, metadata)
    write_json(args.json_file, conflicts, args.judgment_files)
    write_review(args.review_file, conflicts)
    print_summary(conflicts)


if __name__ == "__main__":
    main()
