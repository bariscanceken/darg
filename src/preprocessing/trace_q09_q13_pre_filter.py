"""Trace Q09/Q13 answer coverage before earthquake relevance filtering."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TEXT_ROOT = Path("data/processed/text")
DEFAULT_RELEVANCE_ROOT = Path("data/processed/earthquake_relevance")
DEFAULT_CHUNKS_FILE = Path("data/processed/chunks/afad_earthquake_chunks.jsonl")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/q09_q13_pre_filter_trace.txt")
MAX_REPORT_CANDIDATES = 50
SEPARATOR = "=" * 88
QUESTION_IDS = ("Q09", "Q13")

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
class PatternSpec:
    label: str
    regex: bool = False


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    relative_path: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class ContextMatch:
    matched_sentence: str
    previous_sentences: list[str]
    next_sentences: list[str]
    matched_terms: list[str]
    category: str


@dataclass(frozen=True)
class CandidatePage:
    question_id: str
    source_file: str
    relative_path: str
    page: int
    relevance_status: str
    matched_terms: list[str]
    matched_sentence: str
    previous_sentences: list[str]
    next_sentences: list[str]
    page_represented_in_chunks: bool
    covering_chunk_ids: list[str]
    answer_candidate: str
    matched_sentence_preserved_in_chunks: bool
    answer_signal_preserved_in_chunks: bool


QUESTIONS = {
    "Q09": "Deprem sonrası artçı sarsıntılarda ne yapmalıyım?",
    "Q13": "Deprem sırasında dışarıdaysam ne yapmalıyım?",
}

SEARCH_PATTERNS: dict[str, list[PatternSpec]] = {
    "Q09": [
        PatternSpec("artçı"),
        PatternSpec("artçı sarsıntı"),
        PatternSpec("artçı deprem"),
        PatternSpec("artçılara"),
        PatternSpec("sarsıntılar devam"),
        PatternSpec("tekrar deprem"),
        PatternSpec("hasarlı binaya girmeyin"),
        PatternSpec("hasarlı yapılara girmeyin"),
        PatternSpec("binalardan uzak durun"),
        PatternSpec("çökmeye karşı"),
        PatternSpec("yıkılma tehlikesi"),
        PatternSpec("ikinci sarsıntı"),
    ],
    "Q13": [
        PatternSpec("dışarıda"),
        PatternSpec("dışarıdaysanız"),
        PatternSpec("dışarıda iseniz"),
        PatternSpec("açık alan"),
        PatternSpec("açık alanda"),
        PatternSpec("bina dışında"),
        PatternSpec("binalardan uzak"),
        PatternSpec("duvarlardan uzak"),
        PatternSpec("elektrik direklerinden uzak"),
        PatternSpec("enerji hatlarından uzak"),
        PatternSpec("ağaçlardan uzak"),
        PatternSpec("balkonlardan uzak"),
        PatternSpec("camlardan uzak"),
        PatternSpec("düşebilecek cisim"),
        PatternSpec("sahil"),
        PatternSpec("deniz kıyısı"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Q09/Q13 coverage in extracted page-level text.")
    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--relevance-root", type=Path, default=DEFAULT_RELEVANCE_ROOT)
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-candidates", type=int, default=MAX_REPORT_CANDIDATES)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(TURKISH_AND_EXTRACTION_MAP).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def compile_pattern(spec: PatternSpec) -> re.Pattern[str]:
    normalized = normalize(spec.label)
    if spec.regex:
        pattern = normalized.replace(r"\ ", r"\s+")
    else:
        pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){pattern}(?!\w)")


COMPILED_PATTERNS = {
    question_id: [(spec, compile_pattern(spec)) for spec in patterns]
    for question_id, patterns in SEARCH_PATTERNS.items()
}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list of JSON objects in {path}")
    return records


def read_chunks(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(record.get("chunk_id", "")),
                    relative_path=str(record.get("relative_path", "")),
                    page_start=int(record.get("page_start", 0)),
                    page_end=int(record.get("page_end", 0)),
                    text=str(record.get("text", "")),
                )
            )
            if not chunks[-1].chunk_id:
                raise ValueError(f"Missing chunk_id at {path}:{line_number}")
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


def matched_terms(text: str, question_id: str) -> list[str]:
    normalized = normalize(text)
    return [spec.label for spec, pattern in COMPILED_PATTERNS[question_id] if pattern.search(normalized)]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s*[•·]\s*", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…:])\s+(?=[A-ZÇĞİÖŞÜ0-9])", text)
    return [part.strip(" .") for part in parts if part.strip(" .")]


def has_any(text: str, terms: list[str]) -> bool:
    normalized = normalize(text)
    return any(normalize(term) in normalized for term in terms)


def has_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, normalize(text)) is not None


def classify_q09(context: str) -> str:
    normalized = normalize(context)
    artci_context = any(term in normalized for term in ["artci", "sarsinti", "ikinci sarsinti", "tekrar deprem"])
    behavior = any(
        term in normalized
        for term in [
            "hazirlikli",
            "hasarli binaya girmeyin",
            "hasarli binalardan uzak",
            "hasarli yapilara girmeyin",
            "hasarli yapilardan uzak",
            "binaya girmeyin",
            "binalardan uzak durun",
            "guvenli alanda kal",
            "acik alanda kal",
            "dusebilecek cisimlerden uzak",
            "cok kapan tutun",
            "yetkili aciklamalarini takip",
            "uyarilari takip",
        ]
    ) or (
        ("hasarli bina" in normalized or "hasarli yapi" in normalized or "bina" in normalized)
        and any(term in normalized for term in ["girmeyin", "girmeyiniz", "girilmemeli", "uzak dur", "uzaklas"])
    )
    if artci_context and behavior:
        return "DIRECT_GUIDANCE"
    if artci_context or behavior or has_any(context, SEARCH_PATTERNS_AS_TERMS["Q09"]):
        return "RELATED_GUIDANCE"
    return "UNKNOWN"


def classify_q13(context: str) -> str:
    normalized = normalize(context)
    event_context = any(
        term in normalized
        for term in [
            "deprem sirasinda",
            "deprem aninda",
            "deprem esnasinda",
            "sarsinti sirasinda",
            "sarsinti aninda",
            "sarsinti basladiginda",
        ]
    )
    outdoor_context = any(
        term in normalized
        for term in [
            "disaridaysaniz",
            "disarida iseniz",
            "disarida",
            "acik alandaysaniz",
            "acik alanda",
            "bina disinda",
            "sokakta",
        ]
    )
    behavior = any(
        term in normalized
        for term in [
            "binalardan uzak",
            "duvarlardan uzak",
            "elektrik direklerinden uzak",
            "enerji hatlarindan uzak",
            "agaclardan uzak",
            "balkonlardan uzak",
            "camlardan uzak",
            "dusebilecek cisim",
            "uzak dur",
            "uzaklas",
            "guvenli acik alana",
            "acik alana",
        ]
    )
    post_quake_context = any(term in normalized for term in ["deprem sonrasi", "depremden sonra"])
    if event_context and outdoor_context and behavior and not post_quake_context:
        return "DIRECT_GUIDANCE"
    if outdoor_context or behavior or has_any(context, SEARCH_PATTERNS_AS_TERMS["Q13"]):
        return "RELATED_GUIDANCE"
    return "UNKNOWN"


def classify_context(question_id: str, context: str) -> str:
    if question_id == "Q09":
        return classify_q09(context)
    if question_id == "Q13":
        return classify_q13(context)
    return "UNKNOWN"


def category_priority(category: str) -> int:
    return {
        "DIRECT_GUIDANCE": 3,
        "RELATED_GUIDANCE": 2,
        "MENTION_ONLY": 1,
        "UNKNOWN": 0,
    }.get(category, 0)


SEARCH_PATTERNS_AS_TERMS = {
    question_id: [spec.label for spec in specs if not spec.regex]
    for question_id, specs in SEARCH_PATTERNS.items()
}


def best_context_match(text: str, question_id: str, window: int = 4) -> ContextMatch:
    sentences = split_sentences(text)
    best = ContextMatch("", [], [], [], "UNKNOWN")
    for index, sentence in enumerate(sentences):
        terms = matched_terms(sentence, question_id)
        if not terms:
            continue
        previous = sentences[max(0, index - window) : index]
        following = sentences[index + 1 : index + 1 + window]
        context = " ".join([*previous, sentence, *following])
        category = classify_context(question_id, context)
        current = ContextMatch(sentence, previous, following, terms, category)
        if category_priority(current.category) > category_priority(best.category):
            best = current
        elif category_priority(current.category) == category_priority(best.category) and len(terms) > len(best.matched_terms):
            best = current
    return best


def chunk_has_answer_signal(question_id: str, chunks: list[ChunkRecord]) -> bool:
    return any(best_context_match(chunk.text, question_id, window=4).category == "DIRECT_GUIDANCE" for chunk in chunks)


def sentence_preserved_in_chunks(sentence: str, chunks: list[ChunkRecord]) -> bool:
    normalized_sentence = normalize(sentence)
    if not normalized_sentence:
        return False
    # Very short keyword-only snippets are too weak for exact preservation.
    if len(normalized_sentence.split()) < 4:
        return False
    return any(normalized_sentence in normalize(chunk.text) for chunk in chunks)


def scan_pages(
    text_root: Path,
    relevance_lookup: dict[tuple[str, int], str],
    page_to_chunks: dict[tuple[str, int], list[ChunkRecord]],
) -> tuple[dict[str, list[CandidatePage]], int]:
    candidates: dict[str, list[CandidatePage]] = {question_id: [] for question_id in QUESTION_IDS}
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
            chunk_ids = [chunk.chunk_id for chunk in chunks]

            for question_id in QUESTION_IDS:
                page_terms = matched_terms(text, question_id)
                if not page_terms:
                    continue
                context = best_context_match(text, question_id, window=4)
                terms = context.matched_terms or page_terms
                candidates[question_id].append(
                    CandidatePage(
                        question_id=question_id,
                        source_file=str(record.get("source_file", "")),
                        relative_path=relative_path,
                        page=page,
                        relevance_status=relevance_status,
                        matched_terms=terms,
                        matched_sentence=context.matched_sentence,
                        previous_sentences=context.previous_sentences,
                        next_sentences=context.next_sentences,
                        page_represented_in_chunks=bool(chunks),
                        covering_chunk_ids=chunk_ids,
                        answer_candidate=context.category,
                        matched_sentence_preserved_in_chunks=sentence_preserved_in_chunks(context.matched_sentence, chunks),
                        answer_signal_preserved_in_chunks=chunk_has_answer_signal(question_id, chunks),
                    )
                )

    for question_id in candidates:
        candidates[question_id].sort(
            key=lambda candidate: (
                -category_priority(candidate.answer_candidate),
                {"relevant": 0, "uncertain": 1, "excluded": 2, "not_found": 3}.get(candidate.relevance_status, 4),
                not candidate.page_represented_in_chunks,
                candidate.source_file,
                candidate.page,
            )
        )
    return candidates, scanned_pages


def direct_pages(candidates: list[CandidatePage]) -> list[CandidatePage]:
    return [candidate for candidate in candidates if candidate.answer_candidate == "DIRECT_GUIDANCE"]


def diagnose(candidates: list[CandidatePage]) -> str:
    direct = direct_pages(candidates)
    if not direct:
        return "EXTRACTED_CORPUS_COVERAGE_UNRESOLVED"
    if direct and all(candidate.relevance_status == "excluded" for candidate in direct):
        return "RELEVANCE_FILTER_LOSS"
    if direct and not any(candidate.page_represented_in_chunks for candidate in direct):
        return "RELEVANCE_FILTER_LOSS"
    if any(candidate.page_represented_in_chunks for candidate in direct) and not any(
        candidate.answer_signal_preserved_in_chunks for candidate in direct
    ):
        return "CHUNKING_LOSS"
    if any(candidate.answer_signal_preserved_in_chunks for candidate in direct):
        return "RETRIEVAL_OR_PATTERN_DIAGNOSTIC_FAILURE"
    return "EXTRACTED_CORPUS_COVERAGE_UNRESOLVED"


def count_direct_by_status(candidates: list[CandidatePage]) -> Counter[str]:
    counts: Counter[str] = Counter(candidate.relevance_status for candidate in direct_pages(candidates))
    for status in ("relevant", "uncertain", "excluded", "not_found"):
        counts.setdefault(status, 0)
    return counts


def render_list(values: list[str]) -> str:
    if not values:
        return "NONE"
    return " | ".join(values)


def render_candidate(candidate: CandidatePage) -> str:
    return "\n".join(
        [
            f"question_id: {candidate.question_id}",
            f"source_file: {candidate.source_file}",
            f"relative_path: {candidate.relative_path}",
            f"page: {candidate.page}",
            f"relevance_status: {candidate.relevance_status}",
            f"matched_terms: {', '.join(candidate.matched_terms)}",
            f"matched sentence: {candidate.matched_sentence or 'NONE'}",
            f"previous 4 sentences: {render_list(candidate.previous_sentences)}",
            f"next 4 sentences: {render_list(candidate.next_sentences)}",
            f"page represented in chunks: {str(candidate.page_represented_in_chunks).lower()}",
            f"covering chunk_ids: {', '.join(candidate.covering_chunk_ids) if candidate.covering_chunk_ids else 'NONE'}",
            f"answer_candidate: {candidate.answer_candidate}",
            f"matched_sentence_preserved_in_chunks: {str(candidate.matched_sentence_preserved_in_chunks).lower()}",
            f"answer_signal_preserved_in_chunks: {str(candidate.answer_signal_preserved_in_chunks).lower()}",
        ]
    )


def summary_row(question_id: str, candidates: list[CandidatePage]) -> tuple[str, int, int, int, int, int, int, str]:
    direct = direct_pages(candidates)
    status_counts = count_direct_by_status(candidates)
    represented = sum(1 for candidate in direct if candidate.page_represented_in_chunks)
    return (
        question_id,
        len(candidates),
        len(direct),
        status_counts["relevant"],
        status_counts["uncertain"],
        status_counts["excluded"],
        represented,
        diagnose(candidates),
    )


def render_report(candidates: dict[str, list[CandidatePage]], scanned_pages: int, max_candidates: int) -> str:
    lines = [
        SEPARATOR,
        "Q09/Q13 Pre-Filter Coverage Trace",
        f"Extracted pages scanned: {scanned_pages}",
        "",
    ]

    for question_id in QUESTION_IDS:
        question_candidates = candidates[question_id]
        direct = direct_pages(question_candidates)
        lines.extend(
            [
                SEPARATOR,
                question_id,
                f"Question: {QUESTIONS[question_id]}",
                f"Extracted matches: {len(question_candidates)}",
                f"Direct pages: {len(direct)}",
                f"Direct pages in chunks: {sum(1 for candidate in direct if candidate.page_represented_in_chunks)}",
                f"Diagnosis: {diagnose(question_candidates)}",
                "",
            ]
        )
        if not question_candidates:
            lines.append("No candidate pages found.")
            lines.append("")
            continue
        for index, candidate in enumerate(question_candidates[:max_candidates], start=1):
            lines.append(f"Candidate {index}")
            lines.append(render_candidate(candidate))
            lines.append("")

    lines.extend(
        [
            SEPARATOR,
            "Summary Table",
            "Question | Extracted matches | Direct pages | Direct pages relevant | Direct pages uncertain | Direct pages excluded | Direct pages represented in chunks | Diagnosis",
        ]
    )
    for row in [summary_row(question_id, candidates[question_id]) for question_id in QUESTION_IDS]:
        lines.append(" | ".join(str(value) for value in row))

    return "\n".join(lines).rstrip() + "\n"


def run_trace(
    text_root: Path,
    relevance_root: Path,
    chunks_file: Path,
    output_file: Path,
    max_candidates: int,
) -> tuple[dict[str, list[CandidatePage]], int]:
    chunks = read_chunks(chunks_file)
    page_to_chunks = build_page_to_chunks(chunks)
    relevance_lookup = build_relevance_lookup(relevance_root)
    candidates, scanned_pages = scan_pages(text_root, relevance_lookup, page_to_chunks)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_report(candidates, scanned_pages, max_candidates), encoding="utf-8")
    return candidates, scanned_pages


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    candidates, scanned_pages = run_trace(
        args.text_root,
        args.relevance_root,
        args.chunks_file,
        args.output_file,
        args.max_candidates,
    )

    print(f"Extracted pages scanned: {scanned_pages}")
    print("")
    for question_id in QUESTION_IDS:
        direct = direct_pages(candidates[question_id])
        print(f"{question_id}:")
        print(f"matches: {len(candidates[question_id])}")
        print(f"direct pages: {len(direct)}")
        print(f"direct pages in chunks: {sum(1 for candidate in direct if candidate.page_represented_in_chunks)}")
        print(f"diagnosis: {diagnose(candidates[question_id])}")
        print("")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
