"""Trace remaining failed Test 2 queries through corpus coverage and retrieval."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
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


DEFAULT_TEXT_ROOT = Path("data/processed/text")
DEFAULT_RELEVANCE_ROOT = Path("data/processed/earthquake_relevance")
DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_QUESTIONS_FILE = Path("data/evaluation/retrieval_questions.json")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/remaining_failed_queries_trace.txt")
FAILED_QUESTION_IDS = ("Q09", "Q13", "Q14", "Q20")
FAISS_TOP_K = 100
SEPARATOR = "=" * 88

TURKISH_ASCII_MAP = str.maketrans(
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
    }
)


@dataclass(frozen=True)
class PatternSpec:
    label: str
    regex: bool = False


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_file: str
    relative_path: str
    page_start: int
    page_end: int
    source_relevances: list[str]
    text: str


@dataclass
class TraceCandidate:
    question_id: str
    source_file: str
    relative_path: str
    page: int
    relevance_status: str
    chunk_id: str
    page_start: int | None
    page_end: int | None
    matched_patterns: list[str] = field(default_factory=list)
    category: str = "UNKNOWN"
    context_mismatch: bool = False
    matched_sentence: str = ""
    previous_sentences: list[str] = field(default_factory=list)
    next_sentences: list[str] = field(default_factory=list)
    full_chunk_text: str = ""
    extracted_page_text: str = ""
    faiss_rank: int | None = None
    faiss_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


QUESTION_PATTERNS: dict[str, list[PatternSpec]] = {
    "Q09": [
        PatternSpec("artçı sarsıntılara hazırlıklı"),
        PatternSpec("artçı sarsıntılarda"),
        PatternSpec("artçı deprem sırasında"),
        PatternSpec("artçı sarsıntı sırasında"),
        PatternSpec("hasarlı binalardan uzak"),
        PatternSpec("hasarlı yapılardan uzak"),
        PatternSpec("binaya girmeyin"),
        PatternSpec("yapılara girmeyin"),
        PatternSpec("artçı.*uzak", regex=True),
        PatternSpec("artçı.*girmeyin", regex=True),
    ],
    "Q13": [
        PatternSpec("dışarıdaysanız"),
        PatternSpec("dışarıda iseniz"),
        PatternSpec("açık alandaysanız"),
        PatternSpec("açık alanda"),
        PatternSpec("bina dışındaysanız"),
        PatternSpec("binalardan uzak durun"),
        PatternSpec("binalardan uzaklaşın"),
        PatternSpec("enerji hatlarından uzak"),
        PatternSpec("elektrik direklerinden uzak"),
        PatternSpec("ağaçlardan uzak"),
        PatternSpec("duvarlardan uzak"),
        PatternSpec("dışarıda.*uzak", regex=True),
        PatternSpec("açık alan.*uzak", regex=True),
    ],
    "Q14": [
        PatternSpec("araç kullanıyorsanız"),
        PatternSpec("araç kullanırken"),
        PatternSpec("araç içindeyseniz"),
        PatternSpec("araç içerisindeyseniz"),
        PatternSpec("sürücü"),
        PatternSpec("trafikte"),
        PatternSpec("güvenli bir yerde durun"),
        PatternSpec("güvenli yerde durun"),
        PatternSpec("yolun kenarına"),
        PatternSpec("sağa çek"),
        PatternSpec("aracı durdur"),
        PatternSpec("araçta kal"),
        PatternSpec("araç içinde kal"),
        PatternSpec("köprülerden uzak"),
        PatternSpec("viyadüklerden uzak"),
        PatternSpec("tünellerden uzak"),
    ],
    "Q20": [
        PatternSpec("yetkililerin talimatlarına uy"),
        PatternSpec("yetkililerin talimatlarına uymal"),
        PatternSpec("resmi makam"),
        PatternSpec("resmi makamların"),
        PatternSpec("tahliye talimatı"),
        PatternSpec("tahliye emri"),
        PatternSpec("tahliye uyarısı"),
        PatternSpec("uyarılara uy"),
        PatternSpec("verilen talimatlara uy"),
        PatternSpec("yetkili.*talimat", regex=True),
        PatternSpec("tahliye.*uy", regex=True),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace remaining failed reranked retrieval questions.")
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--relevance-root", type=Path, default=DEFAULT_RELEVANCE_ROOT)
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if models are not in the local cache.",
    )
    return parser.parse_args()


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(TURKISH_ASCII_MAP).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def phrase_to_pattern(phrase: str, is_regex: bool) -> re.Pattern[str]:
    normalized = normalize_for_match(phrase)
    if is_regex:
        pattern = normalized.replace(r"\ ", r"\s+")
    else:
        pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){pattern}(?!\w)")


COMPILED_PATTERNS = {
    question_id: [(spec, phrase_to_pattern(spec.label, spec.regex)) for spec in specs]
    for question_id, specs in QUESTION_PATTERNS.items()
}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list records in {path}")
    return records


def read_chunks(chunks_file: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with chunks_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(record["chunk_id"]),
                    source_file=str(record.get("source_file", "")),
                    relative_path=str(record.get("relative_path", "")),
                    page_start=int(record.get("page_start", 0)),
                    page_end=int(record.get("page_end", 0)),
                    source_relevances=[str(value) for value in record.get("source_relevances", [])],
                    text=str(record.get("text", "")),
                )
            )
    return chunks


def build_page_to_chunks(chunks: list[ChunkRecord]) -> dict[tuple[str, int], list[ChunkRecord]]:
    lookup: dict[tuple[str, int], list[ChunkRecord]] = {}
    for chunk in chunks:
        for page in range(chunk.page_start, chunk.page_end + 1):
            lookup.setdefault((chunk.relative_path, page), []).append(chunk)
    return lookup


def build_relevance_lookup(relevance_root: Path) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for path in sorted(relevance_root.rglob("*.json")):
        for record in read_json_list(path):
            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            relevance = str(record.get("relevance", "")).strip()
            if relative_path and isinstance(page, int):
                lookup[(relative_path, page)] = relevance or "not_found"
    return lookup


def load_questions(path: Path) -> dict[str, str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    questions: dict[str, str] = {}
    for record in records:
        question_id = str(record.get("id", ""))
        if question_id in FAILED_QUESTION_IDS:
            questions[question_id] = str(record.get("question", ""))
    missing = [question_id for question_id in FAILED_QUESTION_IDS if question_id not in questions]
    if missing:
        raise ValueError(f"Missing questions in {path}: {', '.join(missing)}")
    return questions


def split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9])", compact)
    return [part.strip() for part in parts if part.strip()]


def match_patterns(text: str, question_id: str) -> list[str]:
    normalized = normalize_for_match(text)
    return [spec.label for spec, pattern in COMPILED_PATTERNS[question_id] if pattern.search(normalized)]


def sentence_context(text: str, question_id: str) -> tuple[list[str], str, list[str], list[str]]:
    sentences = split_sentences(text)
    for index, sentence in enumerate(sentences):
        matched = match_patterns(sentence, question_id)
        if matched:
            return (
                sentences[max(0, index - 3) : index],
                sentence,
                sentences[index + 1 : index + 4],
                matched,
            )
    return [], "", [], []


def has_any(text: str, terms: list[str]) -> bool:
    normalized = normalize_for_match(text)
    return any(normalize_for_match(term) in normalized for term in terms)


def has_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, normalize_for_match(text)) is not None


def classify_candidate(question_id: str, context: str, full_text: str) -> tuple[str, bool]:
    scope = f"{context} {full_text[:1500]}"
    normalized = normalize_for_match(scope)
    quake_terms = ["deprem", "sarsinti", "artci"]
    other_disasters = ["sel", "taskin", "tsunami", "cig", "heyelan", "orman yangini", "yangin"]
    has_quake = any(term in normalized for term in quake_terms)

    if question_id == "Q09":
        artci_context = "artci" in normalized or "sarsinti" in normalized
        behavior = any(
            term in normalized
            for term in [
                "hazirlikli",
                "hasarli binalardan uzak",
                "hasarli yapilardan uzak",
                "binaya girmeyin",
                "yapilara girmeyin",
                "girmeyiniz",
                "girilmemeli",
                "uzak dur",
                "uzaklas",
            ]
        )
        if artci_context and behavior:
            return "DIRECT_GUIDANCE", False
        if artci_context or behavior:
            return "RELATED_GUIDANCE", False
        return "UNKNOWN", False

    if question_id == "Q13":
        outside_context = any(
            term in normalized
            for term in [
                "disaridaysaniz",
                "disarida iseniz",
                "acik alandaysaniz",
                "acik alanda",
                "bina disindaysaniz",
                "bina disinda",
            ]
        )
        behavior = any(
            term in normalized
            for term in [
                "uzak dur",
                "uzaklas",
                "enerji hatlarindan uzak",
                "elektrik direklerinden uzak",
                "agaclardan uzak",
                "duvarlardan uzak",
                "binalardan uzak",
            ]
        )
        post_quake_only = ("deprem sonrasi" in normalized or "depremden sonra" in normalized) and not (
            "deprem sirasinda" in normalized or "sarsinti sirasinda" in normalized
        )
        if has_quake and outside_context and behavior and not post_quake_only:
            return "DIRECT_GUIDANCE", False
        if outside_context or behavior:
            return "RELATED_GUIDANCE", False
        return "UNKNOWN", False

    if question_id == "Q14":
        vehicle_context = any(
            term in normalized
            for term in [
                "arac kullaniyorsaniz",
                "arac kullanirken",
                "arac icindeyseniz",
                "arac icerisindeyseniz",
                "surucu",
                "trafikte",
                "aracta",
                "arac icinde",
            ]
        )
        behavior = any(
            term in normalized
            for term in [
                "guvenli bir yerde durun",
                "guvenli yerde durun",
                "yolun kenarina",
                "saga cek",
                "araci durdur",
                "aracta kal",
                "arac icinde kal",
                "koprulerden uzak",
                "viyaduklerden uzak",
                "tunellerden uzak",
            ]
        )
        if has_quake and vehicle_context and behavior:
            return "DIRECT_GUIDANCE", False
        if vehicle_context or behavior or any(term in normalized for term in ["kopru", "viyaduk", "tunel"]):
            return "RELATED_GUIDANCE", False
        return "UNKNOWN", False

    if question_id == "Q20":
        authority_context = any(term in normalized for term in ["yetkili", "resmi makam", "talimat", "uyari"])
        evacuation_context = "tahliye" in normalized
        command = has_regex(scope, r"(talimat|uyari|tahliye).{0,80}\b(uy|uymal|uyun|uyunuz)") or has_regex(
            scope, r"\b(uy|uyun|uyunuz).{0,80}(talimat|uyari|tahliye)"
        )
        context_mismatch = not has_quake and any(term in normalized for term in other_disasters)
        if (authority_context or evacuation_context) and command:
            return "DIRECT_GUIDANCE", context_mismatch
        if authority_context or evacuation_context:
            return "RELATED_GUIDANCE", context_mismatch
        return "UNKNOWN", context_mismatch

    return "UNKNOWN", False


def category_priority(category: str) -> int:
    return {
        "DIRECT_GUIDANCE": 3,
        "RELATED_GUIDANCE": 2,
        "MENTION_ONLY": 1,
        "UNKNOWN": 0,
    }.get(category, 0)


def merge_candidate(existing: TraceCandidate, incoming: TraceCandidate) -> None:
    for pattern in incoming.matched_patterns:
        if pattern not in existing.matched_patterns:
            existing.matched_patterns.append(pattern)
    if category_priority(incoming.category) > category_priority(existing.category):
        existing.category = incoming.category
        existing.context_mismatch = incoming.context_mismatch
        existing.matched_sentence = incoming.matched_sentence
        existing.previous_sentences = incoming.previous_sentences
        existing.next_sentences = incoming.next_sentences
    elif incoming.category == existing.category:
        existing.context_mismatch = existing.context_mismatch or incoming.context_mismatch


def scan_extracted_pages(
    text_root: Path,
    relevance_lookup: dict[tuple[str, int], str],
    page_to_chunks: dict[tuple[str, int], list[ChunkRecord]],
) -> tuple[dict[str, list[TraceCandidate]], int]:
    candidates_by_key: dict[tuple[str, str], TraceCandidate] = {}
    scanned_pages = 0

    for path in sorted(text_root.rglob("*.json")):
        for record in read_json_list(path):
            scanned_pages += 1
            text = str(record.get("text", ""))
            relative_path = str(record.get("relative_path", "")).strip()
            page = record.get("page")
            if not relative_path or not isinstance(page, int):
                continue

            relevance_status = relevance_lookup.get((relative_path, page), "not_found")
            chunks = page_to_chunks.get((relative_path, page), [])

            for question_id in FAILED_QUESTION_IDS:
                page_patterns = match_patterns(text, question_id)
                if not page_patterns:
                    continue

                previous, matched_sentence, following, sentence_patterns = sentence_context(text, question_id)
                matched_patterns = sentence_patterns or page_patterns
                context = " ".join([*previous, matched_sentence, *following])

                if chunks:
                    for chunk in chunks:
                        category, context_mismatch = classify_candidate(question_id, context, chunk.text)
                        incoming = TraceCandidate(
                            question_id=question_id,
                            source_file=chunk.source_file,
                            relative_path=relative_path,
                            page=page,
                            relevance_status=relevance_status,
                            chunk_id=chunk.chunk_id,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            matched_patterns=list(matched_patterns),
                            category=category,
                            context_mismatch=context_mismatch,
                            matched_sentence=matched_sentence,
                            previous_sentences=previous,
                            next_sentences=following,
                            full_chunk_text=chunk.text,
                            extracted_page_text=text,
                        )
                        key = (question_id, chunk.chunk_id)
                        if key in candidates_by_key:
                            merge_candidate(candidates_by_key[key], incoming)
                        else:
                            candidates_by_key[key] = incoming
                else:
                    category, context_mismatch = classify_candidate(question_id, context, text)
                    key = (question_id, f"NO_CHUNK::{relative_path}::{page}")
                    candidates_by_key[key] = TraceCandidate(
                        question_id=question_id,
                        source_file=str(record.get("source_file", "")),
                        relative_path=relative_path,
                        page=page,
                        relevance_status=relevance_status,
                        chunk_id="NO_CHUNK",
                        page_start=None,
                        page_end=None,
                        matched_patterns=list(matched_patterns),
                        category=category,
                        context_mismatch=context_mismatch,
                        matched_sentence=matched_sentence,
                        previous_sentences=previous,
                        next_sentences=following,
                        full_chunk_text="",
                        extracted_page_text=text,
                    )

    grouped: dict[str, list[TraceCandidate]] = {question_id: [] for question_id in FAILED_QUESTION_IDS}
    for candidate in candidates_by_key.values():
        grouped[candidate.question_id].append(candidate)
    for question_id in grouped:
        grouped[question_id].sort(
            key=lambda candidate: (
                -category_priority(candidate.category),
                candidate.context_mismatch,
                candidate.source_file,
                candidate.page_start if candidate.page_start is not None else 999999,
                candidate.page,
                candidate.chunk_id,
            )
        )
    return grouped, scanned_pages


def run_retrieval_trace(
    questions: dict[str, str],
    candidates: dict[str, list[TraceCandidate]],
    index_dir: Path,
    embedding_model_name: str,
    reranker_model_name: str,
    allow_model_download: bool,
) -> None:
    index, metadata = load_index_and_metadata(index_dir)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    for question_id in FAILED_QUESTION_IDS:
        query = questions[question_id]
        faiss_candidates = retrieve_faiss_candidates(query, index, metadata, embedding_model, FAISS_TOP_K)
        reranked_candidates = rerank_candidates(query, faiss_candidates, reranker, top_k=FAISS_TOP_K)
        faiss_lookup = {str(candidate["chunk_id"]): candidate for candidate in faiss_candidates}
        rerank_lookup = {str(candidate["chunk_id"]): candidate for candidate in reranked_candidates}

        for candidate in candidates[question_id]:
            if candidate.category != "DIRECT_GUIDANCE" or candidate.chunk_id == "NO_CHUNK":
                continue
            faiss_candidate = faiss_lookup.get(candidate.chunk_id)
            if faiss_candidate is not None:
                candidate.faiss_rank = int(faiss_candidate["original_rank"])
                candidate.faiss_score = float(faiss_candidate["original_similarity_score"])
            rerank_candidate = rerank_lookup.get(candidate.chunk_id)
            if rerank_candidate is not None:
                candidate.rerank_rank = int(rerank_candidate["reranked_rank"])
                candidate.rerank_score = float(rerank_candidate["rerank_score"])


def direct_candidates(candidates: list[TraceCandidate]) -> list[TraceCandidate]:
    return [candidate for candidate in candidates if candidate.category == "DIRECT_GUIDANCE"]


def direct_chunk_candidates(candidates: list[TraceCandidate]) -> list[TraceCandidate]:
    return [candidate for candidate in direct_candidates(candidates) if candidate.chunk_id != "NO_CHUNK"]


def diagnose(candidates: list[TraceCandidate]) -> tuple[str, str]:
    direct = direct_candidates(candidates)
    direct_chunks = direct_chunk_candidates(candidates)
    non_mismatch_direct = [candidate for candidate in direct if not candidate.context_mismatch]
    non_mismatch_direct_chunks = [candidate for candidate in direct_chunks if not candidate.context_mismatch]

    if not direct:
        return "CORPUS_COVERAGE_GAP", "No rule-based DIRECT_GUIDANCE candidate was found in extracted pages."
    if direct and not direct_chunks:
        return "PREPROCESSING_OR_CHUNKING_LOSS", "DIRECT_GUIDANCE exists pre-filter, but no direct candidate is represented in chunks."
    if direct and not non_mismatch_direct:
        return "CONTEXT_COVERAGE_PARTIAL", "Only context-mismatched direct guidance was found."

    ranked_direct_chunks = non_mismatch_direct_chunks or direct_chunks
    if all(candidate.faiss_rank is None for candidate in ranked_direct_chunks):
        return "EMBEDDING_RETRIEVAL_FAILURE", "Direct chunks exist, but none are in FAISS Top-100."
    if any(candidate.rerank_rank is not None and candidate.rerank_rank <= 5 for candidate in ranked_direct_chunks):
        return "EVALUATION_LABEL_OR_REVIEW_ISSUE", "At least one direct chunk is in reranker Top-5."
    if any(candidate.faiss_rank is not None and candidate.faiss_rank <= 20 for candidate in ranked_direct_chunks):
        return "RERANKING_FAILURE", "A direct chunk is in FAISS Top-20, but reranker does not place it in Top-5."
    return "EMBEDDING_RETRIEVAL_FAILURE", "Direct chunks are present but not surfaced strongly enough for Top-20/Top-5."


def best_rank(candidates: list[TraceCandidate], attr: str) -> str:
    ranks = [getattr(candidate, attr) for candidate in direct_chunk_candidates(candidates) if getattr(candidate, attr) is not None]
    return str(min(ranks)) if ranks else "NOT_IN_TOP_100"


def format_score(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "NOT_IN_TOP_100"


def render_candidate(candidate: TraceCandidate) -> str:
    pages = "NO_CHUNK" if candidate.page_start is None or candidate.page_end is None else f"{candidate.page_start}-{candidate.page_end}"
    chunk_text = candidate.full_chunk_text if candidate.full_chunk_text else "NO_CHUNK"
    return "\n".join(
        [
            f"question_id: {candidate.question_id}",
            f"source_file: {candidate.source_file}",
            f"page: {candidate.page}",
            f"page_start-page_end: {pages}",
            f"relevance_status: {candidate.relevance_status}",
            f"chunk_id: {candidate.chunk_id}",
            f"matched_pattern: {', '.join(candidate.matched_patterns)}",
            f"category: {candidate.category}",
            f"context_mismatch: {str(candidate.context_mismatch).lower()}",
            f"FAISS rank: {candidate.faiss_rank if candidate.faiss_rank is not None else 'NOT_IN_TOP_100'}",
            f"FAISS similarity score: {format_score(candidate.faiss_score)}",
            f"Rerank rank: {candidate.rerank_rank if candidate.rerank_rank is not None else 'NOT_IN_TOP_100'}",
            f"Rerank score: {format_score(candidate.rerank_score)}",
            f"matched sentence: {candidate.matched_sentence or 'NONE'}",
            f"previous 3 sentences: {' | '.join(candidate.previous_sentences) if candidate.previous_sentences else 'NONE'}",
            f"next 3 sentences: {' | '.join(candidate.next_sentences) if candidate.next_sentences else 'NONE'}",
            "full chunk text:",
            chunk_text,
        ]
    )


def render_report(
    questions: dict[str, str],
    candidates: dict[str, list[TraceCandidate]],
    scanned_pages: int,
) -> str:
    lines = [
        SEPARATOR,
        "Remaining Failed Query Trace",
        f"Extracted pages scanned: {scanned_pages}",
        f"FAISS Top-K: {FAISS_TOP_K}",
        "",
    ]
    summary_rows: list[tuple[str, int, str, str, str]] = []

    for question_id in FAILED_QUESTION_IDS:
        question_candidates = candidates[question_id]
        direct_chunks = direct_chunk_candidates(question_candidates)
        diagnosis, explanation = diagnose(question_candidates)
        summary_rows.append(
            (
                question_id,
                len(direct_chunks),
                best_rank(question_candidates, "faiss_rank"),
                best_rank(question_candidates, "rerank_rank"),
                diagnosis,
            )
        )

        lines.extend(
            [
                SEPARATOR,
                question_id,
                f"Question: {questions[question_id]}",
                f"Candidate chunks/pages: {len(question_candidates)}",
                f"Direct guidance chunks: {len(direct_chunks)}",
                f"Best FAISS rank: {best_rank(question_candidates, 'faiss_rank')}",
                f"Best rerank rank: {best_rank(question_candidates, 'rerank_rank')}",
                f"Primary diagnosis: {diagnosis}",
                f"Diagnosis note: {explanation}",
                "",
            ]
        )

        if not question_candidates:
            lines.append("No answer-like candidates found.")
            lines.append("")
            continue

        for index, candidate in enumerate(question_candidates, start=1):
            lines.append(f"Candidate {index}")
            lines.append(render_candidate(candidate))
            lines.append("")

    lines.extend([SEPARATOR, "Summary Table", "Question | Direct chunks | Best FAISS rank | Best rerank rank | Diagnosis"])
    for question_id, direct_count, faiss_rank, rerank_rank, diagnosis in summary_rows:
        lines.append(f"{question_id} | {direct_count} | {faiss_rank} | {rerank_rank} | {diagnosis}")

    return "\n".join(lines).rstrip() + "\n"


def run_trace(
    text_root: Path,
    relevance_root: Path,
    chunks_file: Path,
    questions_file: Path,
    index_dir: Path,
    output_file: Path,
    embedding_model_name: str,
    reranker_model_name: str,
    allow_model_download: bool,
) -> tuple[dict[str, list[TraceCandidate]], dict[str, str], int]:
    questions = load_questions(questions_file)
    chunks = read_chunks(chunks_file)
    page_to_chunks = build_page_to_chunks(chunks)
    relevance_lookup = build_relevance_lookup(relevance_root)
    candidates, scanned_pages = scan_extracted_pages(text_root, relevance_lookup, page_to_chunks)
    run_retrieval_trace(
        questions,
        candidates,
        index_dir,
        embedding_model_name,
        reranker_model_name,
        allow_model_download,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(questions, candidates, scanned_pages), encoding="utf-8")
    return candidates, questions, scanned_pages


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    candidates, _questions, _scanned_pages = run_trace(
        args.text_root,
        args.relevance_root,
        args.chunks_file,
        args.questions_file,
        args.index_dir,
        args.output_file,
        args.embedding_model,
        args.reranker_model,
        args.allow_model_download,
    )

    print(f"Questions analyzed: {len(FAILED_QUESTION_IDS)}")
    for question_id in FAILED_QUESTION_IDS:
        print(f"{question_id} direct guidance chunks: {len(direct_chunk_candidates(candidates[question_id]))}")
    print("")
    for question_id in FAILED_QUESTION_IDS:
        diagnosis, _explanation = diagnose(candidates[question_id])
        print(f"{question_id} diagnosis: {diagnosis}")
    print("")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
