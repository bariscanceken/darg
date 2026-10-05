"""Build the offline query-normalization vocabulary from frozen V2 chunk metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from src.retrieval.evaluate_reranked_v2 import DEFAULT_INDEX_DIR, DEFAULT_METADATA_FILE

from .query_normalization import DEFAULT_VOCAB_FILE, TOKEN_RE, VOCAB_VERSION, turkish_lower


DEFAULT_MIN_FREQUENCY = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DARG query-normalization vocabulary.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_INDEX_DIR / DEFAULT_METADATA_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_VOCAB_FILE)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def count_words(metadata_path: Path) -> tuple[Counter[str], int]:
    """Count Turkish-lowercased word tokens in chunk texts (read-only)."""
    counts: Counter[str] = Counter()
    chunk_count = 0
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk_count += 1
            text = str(json.loads(line).get("text", ""))
            counts.update(TOKEN_RE.findall(turkish_lower(text)))
    return counts, chunk_count


def build_vocab(metadata_path: Path, min_frequency: int) -> dict:
    counts, chunk_count = count_words(metadata_path)
    words = {word: count for word, count in sorted(counts.items()) if count >= min_frequency}
    return {
        "version": VOCAB_VERSION,
        "source_file": metadata_path.as_posix(),
        "source_sha256": file_sha256(metadata_path),
        "chunk_count": chunk_count,
        "min_frequency": min_frequency,
        "token_pattern": TOKEN_RE.pattern,
        "word_count": len(words),
        "words": words,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    vocab = build_vocab(args.metadata_path, args.min_frequency)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(vocab, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")

    print(f"Source: {vocab['source_file']}")
    print(f"Source sha256: {vocab['source_sha256']}")
    print(f"Chunks: {vocab['chunk_count']}")
    print(f"Words (freq >= {vocab['min_frequency']}): {vocab['word_count']}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
