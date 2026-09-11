"""Find Q09 direct-guidance chunks and trace their retrieval/rerank ranks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rerank import (
    DEFAULT_RERANKER_MODEL,
    load_embedding_model,
    load_index_and_metadata,
    load_reranker,
    rerank_candidates,
    retrieve_faiss_candidates,
)
from .search import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME


QUESTION_ID = "Q09"
QUERY = "Deprem sonrası artçı sarsıntılarda ne yapmalıyım?"
DIRECT_SENTENCE = "Artçı şoklar için hazırlıklı olmalıyız"
FAISS_TOP_K = 100

DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_RERANKED_RESULTS_FILE = Path("data/evaluation/retrieval_reranked_results.json")
DEFAULT_RERANKED_JUDGMENTS_FILE = Path("data/evaluation/retrieval_reranked_judgments.csv")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/q09_direct_guidance_ranks.txt")
SEPARATOR = "=" * 88

TURKISH_AND_EXTRACTION_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "I": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
        "È": "i",
        "è": "g",
        "Ë": "s",
        "ë": "s",
        "É": "i",
        "é": "i",
    }
)


@dataclass(frozen=True)
class DirectChunk:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Q09 direct guidance chunk ranking.")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--reranked-results-file", type=Path, default=DEFAULT_RERANKED_RESULTS_FILE)
    parser.add_argument("--reranked-judgments-file", type=Path, default=DEFAULT_RERANKED_JUDGMENTS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(TURKISH_AND_EXTRACTION_MAP).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def contains_direct_sentence(text: str) -> bool:
    normalized_text = normalize(text)
    normalized_sentence = normalize(DIRECT_SENTENCE)
    if normalized_sentence in normalized_text:
        return True
    # Extraction may split punctuation or lose Turkish letters. Keep this anchored to
    # the known direct guidance wording instead of broad Q09 topic terms.
    return bool(re.search(r"\bartci\s+soklar.{0,80}\bhazirlikli\s+olmaliyiz\b", normalized_text))


def read_direct_chunks(chunks_file: Path) -> list[DirectChunk]:
    chunks: list[DirectChunk] = []
    seen: set[str] = set()
    with chunks_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", ""))
            if not contains_direct_sentence(text):
                continue
            chunk_id = str(record.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError(f"Missing chunk_id at {chunks_file}:{line_number}")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append(
                DirectChunk(
                    chunk_id=chunk_id,
                    source_file=str(record.get("source_file", "")),
                    page_start=int(record.get("page_start", 0)),
                    page_end=int(record.get("page_end", 0)),
                    text=text,
                )
            )
    return chunks


def candidate_by_chunk_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(candidate["chunk_id"]): candidate for candidate in candidates}


def read_q09_test2_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for question in payload.get("questions", []):
        if question.get("id") == QUESTION_ID:
            return list(question.get("results", []))
    raise ValueError(f"{QUESTION_ID} not found in {path}")


def read_q09_judgments(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("question_id") == QUESTION_ID]


def classify(
    direct_chunks: list[DirectChunk],
    faiss_lookup: dict[str, dict[str, Any]],
    rerank_lookup: dict[str, dict[str, Any]],
    test2_results: list[dict[str, Any]],
    q09_judgments: list[dict[str, str]],
) -> str:
    if not direct_chunks:
        return "DIRECT_GUIDANCE_CHUNK_NOT_FOUND"

    direct_ids = {chunk.chunk_id for chunk in direct_chunks}
    labels = {row["chunk_id"]: row.get("relevant", "").strip() for row in q09_judgments}
    test2_direct_ids = [str(result.get("chunk_id")) for result in test2_results if str(result.get("chunk_id")) in direct_ids]
    if any(labels.get(chunk_id) == "0" for chunk_id in test2_direct_ids):
        return "EVALUATION_LABEL_ERROR"

    direct_faiss_ranks = [
        int(faiss_lookup[chunk.chunk_id]["original_rank"])
        for chunk in direct_chunks
        if chunk.chunk_id in faiss_lookup
    ]
    direct_rerank_ranks = [
        int(rerank_lookup[chunk.chunk_id]["reranked_rank"])
        for chunk in direct_chunks
        if chunk.chunk_id in rerank_lookup
    ]

    if not direct_faiss_ranks:
        return "EMBEDDING_RETRIEVAL_FAILURE"
    # Test 2 reranks only FAISS Top-20. A direct chunk at rank 21-100 can score
    # well in this diagnostic's Top-100 rerank, but it was never a Test 2 candidate.
    if any(rank <= 20 for rank in direct_faiss_ranks):
        return "RERANKING_FAILURE"
    if any(21 <= rank <= 100 for rank in direct_faiss_ranks):
        return "CANDIDATE_RETRIEVAL_WEAK"
    return "EMBEDDING_RETRIEVAL_FAILURE"


def format_score(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "NOT_IN_TOP_100"


def render_direct_chunk(
    chunk: DirectChunk,
    faiss_lookup: dict[str, dict[str, Any]],
    rerank_lookup: dict[str, dict[str, Any]],
) -> str:
    faiss_candidate = faiss_lookup.get(chunk.chunk_id)
    rerank_candidate = rerank_lookup.get(chunk.chunk_id)
    faiss_rank = int(faiss_candidate["original_rank"]) if faiss_candidate else None
    faiss_score = float(faiss_candidate["original_similarity_score"]) if faiss_candidate else None
    rerank_rank = int(rerank_candidate["reranked_rank"]) if rerank_candidate else None
    rerank_score = float(rerank_candidate["rerank_score"]) if rerank_candidate else None
    return "\n".join(
        [
            f"chunk_id: {chunk.chunk_id}",
            f"source_file: {chunk.source_file}",
            f"page_start: {chunk.page_start}",
            f"page_end: {chunk.page_end}",
            f"FAISS rank: {faiss_rank if faiss_rank is not None else 'NOT_IN_TOP_100'}",
            f"FAISS similarity score: {format_score(faiss_score)}",
            f"Rerank rank: {rerank_rank if rerank_rank is not None else 'NOT_IN_TOP_100'}",
            f"Rerank score: {format_score(rerank_score)}",
            "full text:",
            chunk.text.strip(),
        ]
    )


def render_test2_result(result: dict[str, Any], labels: dict[str, str]) -> str:
    chunk_id = str(result.get("chunk_id"))
    return "\n".join(
        [
            f"rerank rank: {result.get('reranked_rank')}",
            f"original FAISS rank: {result.get('original_rank')}",
            f"chunk_id: {chunk_id}",
            f"source_file: {result.get('source_file')}",
            f"current relevant label: {labels.get(chunk_id, 'BLANK') or 'BLANK'}",
        ]
    )


def render_report(
    direct_chunks: list[DirectChunk],
    faiss_candidates: list[dict[str, Any]],
    reranked_candidates: list[dict[str, Any]],
    test2_results: list[dict[str, Any]],
    q09_judgments: list[dict[str, str]],
    diagnosis: str,
) -> str:
    faiss_lookup = candidate_by_chunk_id(faiss_candidates)
    rerank_lookup = candidate_by_chunk_id(reranked_candidates)
    labels = {row["chunk_id"]: row.get("relevant", "").strip() for row in q09_judgments}

    lines = [
        SEPARATOR,
        "Q09 Direct Guidance Rank Diagnostic",
        f"Question: {QUERY}",
        f"Known direct guidance sentence: {DIRECT_SENTENCE}",
        f"FAISS Top-K: {FAISS_TOP_K}",
        "",
        f"Direct guidance chunks found: {len(direct_chunks)}",
        "",
        SEPARATOR,
        "Direct Guidance Chunks",
    ]

    if not direct_chunks:
        lines.append("No chunks containing the known direct guidance sentence were found.")
    else:
        for index, chunk in enumerate(direct_chunks, start=1):
            lines.append("")
            lines.append(f"Direct chunk {index}")
            lines.append(render_direct_chunk(chunk, faiss_lookup, rerank_lookup))

    lines.extend(["", SEPARATOR, "Test 2 Q09 Top-5 Results"])
    for result in test2_results:
        lines.append("")
        lines.append(render_test2_result(result, labels))

    lines.extend(["", SEPARATOR, "Summary", f"Diagnosis: {diagnosis}"])
    for chunk in direct_chunks:
        faiss_candidate = faiss_lookup.get(chunk.chunk_id)
        rerank_candidate = rerank_lookup.get(chunk.chunk_id)
        lines.extend(
            [
                "",
                f"Chunk {chunk.chunk_id}:",
                f"FAISS rank: {faiss_candidate['original_rank'] if faiss_candidate else 'NOT_IN_TOP_100'}",
                f"FAISS similarity score: {format_score(float(faiss_candidate['original_similarity_score']) if faiss_candidate else None)}",
                f"Rerank rank: {rerank_candidate['reranked_rank'] if rerank_candidate else 'NOT_IN_TOP_100'}",
                f"Rerank score: {format_score(float(rerank_candidate['rerank_score']) if rerank_candidate else None)}",
            ]
        )

    lines.extend(
        [
            "",
            "Classification options:",
            "1. RERANKING_FAILURE",
            "2. CANDIDATE_RETRIEVAL_WEAK",
            "3. EMBEDDING_RETRIEVAL_FAILURE",
            "4. EVALUATION_LABEL_ERROR",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run_diagnostic(
    chunks_file: Path,
    index_dir: Path,
    reranked_results_file: Path,
    reranked_judgments_file: Path,
    output_file: Path,
    embedding_model_name: str,
    reranker_model_name: str,
    allow_model_download: bool,
) -> tuple[list[DirectChunk], list[dict[str, Any]], list[dict[str, Any]], str]:
    direct_chunks = read_direct_chunks(chunks_file)
    index, metadata = load_index_and_metadata(index_dir)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)
    faiss_candidates = retrieve_faiss_candidates(QUERY, index, metadata, embedding_model, FAISS_TOP_K)
    reranked_candidates = rerank_candidates(QUERY, faiss_candidates, reranker, top_k=FAISS_TOP_K)
    test2_results = read_q09_test2_results(reranked_results_file)
    q09_judgments = read_q09_judgments(reranked_judgments_file)
    diagnosis = classify(
        direct_chunks,
        candidate_by_chunk_id(faiss_candidates),
        candidate_by_chunk_id(reranked_candidates),
        test2_results,
        q09_judgments,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        render_report(direct_chunks, faiss_candidates, reranked_candidates, test2_results, q09_judgments, diagnosis),
        encoding="utf-8",
    )
    return direct_chunks, faiss_candidates, reranked_candidates, diagnosis


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    direct_chunks, faiss_candidates, reranked_candidates, diagnosis = run_diagnostic(
        args.chunks_file,
        args.index_dir,
        args.reranked_results_file,
        args.reranked_judgments_file,
        args.output_file,
        args.embedding_model,
        args.reranker_model,
        args.allow_model_download,
    )
    faiss_lookup = candidate_by_chunk_id(faiss_candidates)
    rerank_lookup = candidate_by_chunk_id(reranked_candidates)

    print(f"Direct guidance chunks found: {len(direct_chunks)}")
    for chunk in direct_chunks:
        faiss_candidate = faiss_lookup.get(chunk.chunk_id)
        rerank_candidate = rerank_lookup.get(chunk.chunk_id)
        print(f"Chunk: {chunk.chunk_id}")
        print(f"FAISS rank: {faiss_candidate['original_rank'] if faiss_candidate else 'NOT_IN_TOP_100'}")
        print(
            "FAISS similarity score: "
            f"{format_score(float(faiss_candidate['original_similarity_score']) if faiss_candidate else None)}"
        )
        print(f"Rerank rank: {rerank_candidate['reranked_rank'] if rerank_candidate else 'NOT_IN_TOP_100'}")
        print(f"Rerank score: {format_score(float(rerank_candidate['rerank_score']) if rerank_candidate else None)}")
    print(f"Diagnosis: {diagnosis}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
