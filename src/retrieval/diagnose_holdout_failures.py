"""Diagnose Test 4 holdout retrieval failures without changing retrieval state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .evaluate_reranked_v2 import (
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_FILE,
    DEFAULT_METADATA_FILE,
    load_v2_index_and_metadata,
    retrieve_v2_candidates,
)
from .evaluate_score_fusion import preview_text
from .rerank import DEFAULT_RERANKER_MODEL, load_embedding_model, load_reranker, rerank_candidates
from .search import DEFAULT_MODEL_NAME


FAILED_QUESTIONS = [
    {"id": "H02", "question": "Deprem sırasında merdivenlere yönelmek doğru mu?"},
    {"id": "H06", "question": "Sarsıntı bittikten hemen sonra binadan çıkmalı mıyım?"},
    {"id": "H10", "question": "Deprem sonrası kibrit veya çakmak kullanmak güvenli mi?"},
    {"id": "H13", "question": "Deprem sonrası binaya eşya almak için geri girebilir miyim?"},
    {"id": "H23", "question": "Depremden sonra açık alanda hangi yapılardan uzak durmalıyım?"},
    {"id": "H24", "question": "Sokaktayken deprem olursa binaların yanında beklemeli miyim?"},
    {"id": "H25", "question": "Araçtayken deprem olursa köprü üzerinde durmalı mıyım?"},
    {"id": "H26", "question": "Araçtayken sarsıntı başladığında aracı nasıl durdurmalıyım?"},
    {"id": "H27", "question": "Sahildeyken güçlü bir deprem hissedersem ne yapmalıyım?"},
]

LEXICAL_TERMS = {
    "H02": ["merdiven", "merdivenlere", "asansör", "çıkış"],
    "H06": ["sarsıntı bittikten sonra", "binadan çık", "tahliye", "yapı hasarı"],
    "H10": ["kibrit", "çakmak", "açık ateş", "doğalgaz"],
    "H13": ["binaya tekrar gir", "geri girmeyin", "eşya almak", "tehlike geçti"],
    "H23": ["açık alan", "bina", "elektrik direği", "duvar", "ağaç", "tabela"],
    "H24": ["sokak", "bina", "dışarıda", "açık alan"],
    "H25": ["araç", "köprü", "viyadük", "tünel"],
    "H26": ["araçtayken", "aracı durdur", "yol kenarı", "deprem sırasında araç"],
    "H27": ["sahil", "deniz", "tsunami", "yüksek yerlere"],
}

DEFAULT_FAISS_TOP_K = 100
DEFAULT_CURRENT_TOP_K = 5
DEFAULT_JSON_FILE = Path("data/evaluation/holdout_failure_diagnostics.json")
DEFAULT_TXT_FILE = Path("data/evaluation/holdout_failure_diagnostics.txt")
DEFAULT_REVIEW_FILE = Path("data/evaluation/holdout_failure_review.txt")
SEPARATOR = "=" * 80

PROTECTED_FILES = [
    Path("data/processed/chunks/afad_earthquake_chunks.jsonl"),
    Path("data/processed/chunks/afad_earthquake_chunks_v2.jsonl"),
    Path("data/indexes/mobile/afad_e5_small.faiss"),
    Path("data/indexes/mobile/metadata.jsonl"),
    Path("data/indexes/mobile_v2/afad_e5_small_v2.faiss"),
    Path("data/indexes/mobile_v2/metadata_v2.jsonl"),
    Path("data/evaluation/holdout_retrieval_judgments.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose failed holdout questions for Test 4 baseline.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--json-file", type=Path, default=DEFAULT_JSON_FILE)
    parser.add_argument("--txt-file", type=Path, default=DEFAULT_TXT_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CURRENT_TOP_K)
    parser.add_argument("--lexical-limit", type=int, default=20)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str | None]:
    return {str(path): file_hash(path) for path in paths}


def candidate_for_json(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "faiss_rank": int(candidate["original_rank"]),
        "faiss_score": float(candidate["original_similarity_score"]),
        "chunk_id": candidate["chunk_id"],
        "source_file": candidate["source_file"],
        "page_start": candidate["page_start"],
        "page_end": candidate["page_end"],
        "text": str(candidate["text"]),
    }


def reranked_for_json(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "reranker_rank": int(candidate["reranked_rank"]),
        "reranker_score": float(candidate["rerank_score"]),
        "faiss_rank": int(candidate["original_rank"]),
        "faiss_score": float(candidate["original_similarity_score"]),
        "chunk_id": candidate["chunk_id"],
        "source_file": candidate["source_file"],
        "page_start": candidate["page_start"],
        "page_end": candidate["page_end"],
        "text": str(candidate["text"]),
    }


def top5_current(reranked_all: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return [reranked_for_json(candidate) for candidate in reranked_all[:top_k]]


def top50_rerank_misses(reranked_all: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    misses = [
        candidate
        for candidate in reranked_all
        if int(candidate["original_rank"]) <= 50 and int(candidate["reranked_rank"]) > 5
    ]
    misses.sort(key=lambda candidate: (int(candidate["original_rank"]), int(candidate["reranked_rank"])))
    return [reranked_for_json(candidate) for candidate in misses[:limit]]


def lexical_matches(question_id: str, metadata: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    terms = LEXICAL_TERMS[question_id]
    matches: list[dict[str, Any]] = []
    for record in metadata:
        text = str(record.get("text", ""))
        searchable = text.casefold()
        matched_terms = [term for term in terms if term.casefold() in searchable]
        if not matched_terms:
            continue
        matches.append(
            {
                "chunk_id": record["chunk_id"],
                "source_file": record["source_file"],
                "page_start": record["page_start"],
                "page_end": record["page_end"],
                "matched_terms": matched_terms,
                "text": text,
            }
        )
    matches.sort(key=lambda item: (-len(item["matched_terms"]), str(item["source_file"]), int(item["page_start"])))
    return matches[:limit]


def possible_relevant_candidates(
    reranked_all: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    lexical_ids = {str(match["chunk_id"]) for match in lexical}
    candidates = [candidate for candidate in reranked_all if str(candidate["chunk_id"]) in lexical_ids]
    candidates.sort(key=lambda candidate: (int(candidate["reranked_rank"]), int(candidate["original_rank"])))
    return [reranked_for_json(candidate) for candidate in candidates[:limit]]


def run_diagnostic(
    index_dir: Path,
    index_file: str,
    metadata_file: str,
    embedding_model_name: str,
    reranker_model_name: str,
    faiss_top_k: int,
    top_k: int,
    lexical_limit: int,
    allow_model_download: bool,
) -> dict[str, Any]:
    index, metadata, index_path, metadata_path = load_v2_index_and_metadata(index_dir, index_file, metadata_file)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    question_diagnostics: list[dict[str, Any]] = []
    for question in FAILED_QUESTIONS:
        faiss_candidates = retrieve_v2_candidates(question["question"], index, metadata, embedding_model, faiss_top_k)
        if len({str(candidate["chunk_id"]) for candidate in faiss_candidates}) != len(faiss_candidates):
            raise RuntimeError(f"{question['id']}: duplicate chunk_id in FAISS Top-{faiss_top_k}")
        reranked_all = rerank_candidates(question["question"], faiss_candidates, reranker, top_k=faiss_top_k)
        lexical = lexical_matches(question["id"], metadata, lexical_limit)

        question_diagnostics.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "top5_current": top5_current(reranked_all, top_k),
                "possible_relevant_candidates": possible_relevant_candidates(reranked_all, lexical),
                "lexical_matches": lexical,
                "dense_retrieval_top100": [candidate_for_json(candidate) for candidate in faiss_candidates],
                "rerank_top100": [reranked_for_json(candidate) for candidate in reranked_all],
                "top50_candidates_outside_rerank_top5": top50_rerank_misses(reranked_all),
                "diagnostic_category": "UNKNOWN",
                "diagnostic_notes": "EMPTY",
            }
        )

    return {
        "experiment": "Holdout failure diagnostic for Test 4 baseline",
        "note": "Diagnostic only. No retrieval parameters were modified.",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "questions_analyzed": len(question_diagnostics),
        "faiss_candidates_per_question": faiss_top_k,
        "questions": question_diagnostics,
    }


def compact_result_lines(result: dict[str, Any], rank_label: str) -> list[str]:
    return [
        f"{rank_label}: {result.get('reranker_rank', result.get('faiss_rank'))}",
        f"FAISS rank: {result['faiss_rank']}",
        f"FAISS score: {float(result['faiss_score']):.6f}",
        f"reranker rank: {result.get('reranker_rank', 'N/A')}",
        f"reranker score: {float(result['reranker_score']):.6f}" if "reranker_score" in result else "reranker score: N/A",
        f"chunk_id: {result['chunk_id']}",
        f"source_file: {result['source_file']}",
        f"pages: {result['page_start']}-{result['page_end']}",
        f"text: {preview_text(str(result['text']), 900)}",
        "",
    ]


def render_review(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for question in payload["questions"]:
        lines = [
            SEPARATOR,
            f"QUESTION: {question['question_id']}",
            question["question"],
            "",
            "CURRENT TEST4 TOP-5",
            "",
        ]
        for result in question["top5_current"]:
            lines.extend(compact_result_lines(result, "rank"))

        lines.extend(["FAISS TOP-100 içinde dikkat çeken adaylar", ""])
        for result in question["dense_retrieval_top100"][:20]:
            lines.extend(compact_result_lines(result, "FAISS rank"))

        lines.extend(["RERANK sonrası dikkat çeken adaylar", ""])
        for result in question["rerank_top100"][:20]:
            lines.extend(compact_result_lines(result, "reranker rank"))

        lines.extend(["FAISS Top-50 içinde olup reranker Top-5 dışında kalan adaylar", ""])
        for result in question["top50_candidates_outside_rerank_top5"][:20]:
            lines.extend(compact_result_lines(result, "reranker rank"))

        lines.extend(["LEXICAL MATCHES", ""])
        if not question["lexical_matches"]:
            lines.append("No lexical matches found.")
            lines.append("")
        for index, match in enumerate(question["lexical_matches"], start=1):
            lines.extend(
                [
                    f"lexical match {index}",
                    f"chunk_id: {match['chunk_id']}",
                    f"source_file: {match['source_file']}",
                    f"pages: {match['page_start']}-{match['page_end']}",
                    f"matched_terms: {', '.join(match['matched_terms'])}",
                    f"text: {preview_text(str(match['text']), 900)}",
                    "",
                ]
            )

        lines.extend(["DIAGNOSTIC CATEGORY:", question["diagnostic_category"], "", "NOTES:", question["diagnostic_notes"], SEPARATOR])
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def render_diagnostics_txt(payload: dict[str, Any]) -> str:
    lines = [
        "Holdout Failure Diagnostics",
        f"Questions analyzed: {payload['questions_analyzed']}",
        f"FAISS candidates per question: {payload['faiss_candidates_per_question']}",
        "",
    ]
    for question in payload["questions"]:
        lines.extend(
            [
                SEPARATOR,
                f"question_id: {question['question_id']}",
                f"question: {question['question']}",
                "",
                "top5_current:",
            ]
        )
        for result in question["top5_current"]:
            lines.extend(compact_result_lines(result, "rank"))
        lines.append("possible_relevant_candidates:")
        if not question["possible_relevant_candidates"]:
            lines.extend(["NONE", ""])
        for result in question["possible_relevant_candidates"]:
            lines.extend(compact_result_lines(result, "reranker rank"))
        lines.append("lexical_matches:")
        if not question["lexical_matches"]:
            lines.extend(["NONE", ""])
        for match in question["lexical_matches"]:
            lines.extend(
                [
                    f"chunk_id: {match['chunk_id']}",
                    f"source_file: {match['source_file']}",
                    f"pages: {match['page_start']}-{match['page_end']}",
                    f"matched_terms: {', '.join(match['matched_terms'])}",
                    f"text: {preview_text(str(match['text']), 900)}",
                    "",
                ]
            )
        lines.extend(
            [
                "diagnostic_category:",
                question["diagnostic_category"],
                "",
                "diagnostic_notes:",
                question["diagnostic_notes"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def print_summary(payload: dict[str, Any], json_file: Path, txt_file: Path, review_file: Path, protected_unchanged: bool) -> None:
    print(f"Questions analyzed: {payload['questions_analyzed']}")
    print(f"FAISS candidates per question: {payload['faiss_candidates_per_question']}")
    print(f"Total FAISS candidates: {payload['questions_analyzed'] * payload['faiss_candidates_per_question']}")
    print("Lexical matches found per question:")
    for question in payload["questions"]:
        print(f"{question['question_id']}: {len(question['lexical_matches'])}")
    print("")
    print(f"Existing index unchanged: {str(protected_unchanged).lower()}")
    print(f"Existing chunk files unchanged: {str(protected_unchanged).lower()}")
    print(f"Holdout judgments unchanged: {str(protected_unchanged).lower()}")
    print("Retrieval parameters optimized: false")
    print("")
    print("Output files:")
    print(json_file)
    print(txt_file)
    print(review_file)
    print("")
    print("Failure diagnostic completed.")
    print("No retrieval parameters were modified.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    before = snapshot(PROTECTED_FILES)
    payload = run_diagnostic(
        index_dir=args.index_dir,
        index_file=args.index_file,
        metadata_file=args.metadata_file,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        faiss_top_k=args.faiss_top_k,
        top_k=args.top_k,
        lexical_limit=args.lexical_limit,
        allow_model_download=args.allow_model_download,
    )
    write_json(args.json_file, payload)
    write_text(args.txt_file, render_diagnostics_txt(payload))
    write_text(args.review_file, render_review(payload))
    after = snapshot(PROTECTED_FILES)
    print_summary(payload, args.json_file, args.txt_file, args.review_file, before == after)


if __name__ == "__main__":
    main()
