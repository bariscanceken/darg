"""Run the AFAD PDF preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

try:
    from .extract import extract_pdf_pages, find_pdfs, output_path_for
except ImportError:  # Allows running this module from the filesystem directly.
    from extract import extract_pdf_pages, find_pdfs, output_path_for


DEFAULT_RAW_ROOT = Path("data/raw/documents/afad")
DEFAULT_TEXT_ROOT = Path("data/processed/text")
DEFAULT_REJECTED_FILE = Path("data/processed/rejected/rejected_files.jsonl")


@dataclass
class PipelineStats:
    pdf_count: int = 0
    page_count: int = 0
    failed_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract page-level text from AFAD PDF files.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--rejected-file", type=Path, default=DEFAULT_REJECTED_FILE)
    return parser.parse_args()


def write_json(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def log_rejected(rejected_file: Path, pdf_path: Path, raw_root: Path, error: Exception) -> None:
    rejected_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        relative_path = pdf_path.relative_to(raw_root).as_posix()
    except ValueError:
        relative_path = str(pdf_path)

    payload = {
        "source_file": pdf_path.name,
        "relative_path": relative_path,
        "error": str(error),
    }

    with rejected_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_pipeline(raw_root: Path, text_root: Path, rejected_file: Path) -> PipelineStats:
    raw_root = raw_root.resolve()
    text_root = text_root.resolve()
    rejected_file = rejected_file.resolve()

    stats = PipelineStats()
    pdf_paths = find_pdfs(raw_root)
    stats.pdf_count = len(pdf_paths)

    rejected_file.parent.mkdir(parents=True, exist_ok=True)
    rejected_file.write_text("", encoding="utf-8")

    for pdf_path in pdf_paths:
        try:
            page_records = extract_pdf_pages(pdf_path, raw_root)
            output_path = output_path_for(pdf_path, raw_root, text_root)
            write_json(output_path, [record.to_dict() for record in page_records])
            stats.page_count += len(page_records)
        except Exception as exc:  # Keep the batch moving when a file is unreadable.
            stats.failed_count += 1
            log_rejected(rejected_file, pdf_path, raw_root, exc)

    return stats


def main() -> None:
    args = parse_args()
    stats = run_pipeline(args.raw_root, args.text_root, args.rejected_file)
    print(f"Processed PDFs: {stats.pdf_count}")
    print(f"Processed pages: {stats.page_count}")
    print(f"Failed PDFs: {stats.failed_count}")


if __name__ == "__main__":
    main()
