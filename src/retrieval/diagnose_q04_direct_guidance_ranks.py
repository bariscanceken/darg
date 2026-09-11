"""Rank diagnostics for Q04 DIRECT_GUIDANCE chunks."""

from __future__ import annotations

import argparse
import re
import sys
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


DEFAULT_TRACE_FILE = Path("data/evaluation/q04_asansor_trace.txt")
DEFAULT_OUTPUT_FILE = Path("data/evaluation/q04_direct_guidance_ranks.txt")
QUERY = "Deprem sırasında asansör kullanmalı mıyım?"
FAISS_TOP_K = 100
SEPARATOR = "=" * 80


@dataclass(frozen=True)
class DirectGuidanceTrace:
    chunk_id: str
    trace_source_file: str
    trace_page: str
    matched_sentence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find Q04 DIRECT_GUIDANCE chunk ranks in FAISS Top-100 and reranking.")
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
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


def field_value(block: str, field: str) -> str:
    match = re.search(rf"^{re.escape(field)}:\s*(.*)$", block, re.MULTILINE)
    return match.group(1).strip() if match else ""


def read_direct_guidance_traces(trace_file: Path) -> list[DirectGuidanceTrace]:
    text = trace_file.read_text(encoding="utf-8")
    traces: list[DirectGuidanceTrace] = []
    seen_chunk_ids: set[str] = set()

    for block in re.split(r"(?m)^Match\s+\d+\s*$", text):
        if "category: DIRECT_GUIDANCE" not in block:
            continue

        chunk_ids_value = field_value(block, "covering_chunk_ids")
        if not chunk_ids_value or chunk_ids_value == "NONE":
            continue

        source_file = field_value(block, "source_file")
        page = field_value(block, "page")
        matched_sentence = field_value(block, "matched_sentence")
        for chunk_id in [item.strip() for item in chunk_ids_value.split(",") if item.strip()]:
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            traces.append(
                DirectGuidanceTrace(
                    chunk_id=chunk_id,
                    trace_source_file=source_file,
                    trace_page=page,
                    matched_sentence=matched_sentence,
                )
            )

    return traces


def candidate_by_chunk_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(candidate["chunk_id"]): candidate for candidate in candidates}


def render_chunk_report(
    trace: DirectGuidanceTrace,
    faiss_lookup: dict[str, dict[str, Any]],
    rerank_lookup: dict[str, dict[str, Any]],
) -> str:
    faiss_candidate = faiss_lookup.get(trace.chunk_id)
    rerank_candidate = rerank_lookup.get(trace.chunk_id)

    lines = [
        f"Chunk: {trace.chunk_id}",
        f"Trace source_file: {trace.trace_source_file}",
        f"Trace page: {trace.trace_page}",
        f"Trace matched_sentence: {trace.matched_sentence}",
    ]

    if faiss_candidate is None:
        lines.extend(
            [
                "FAISS rank: NOT_IN_TOP_100",
                "FAISS score: NOT_IN_TOP_100",
                "source_file: NOT_IN_TOP_100",
                "pages: NOT_IN_TOP_100",
                "Rerank rank: NOT_IN_TOP_100",
                "Rerank score: NOT_IN_TOP_100",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"FAISS rank: {faiss_candidate['original_rank']}",
            f"FAISS score: {faiss_candidate['original_similarity_score']:.6f}",
            f"source_file: {faiss_candidate['source_file']}",
            f"pages: {faiss_candidate['page_start']}-{faiss_candidate['page_end']}",
        ]
    )

    if rerank_candidate is None:
        lines.extend(["Rerank rank: NOT_IN_TOP_100", "Rerank score: NOT_IN_TOP_100"])
    else:
        lines.extend(
            [
                f"Rerank rank: {rerank_candidate['reranked_rank']}",
                f"Rerank score: {rerank_candidate['rerank_score']:.6f}",
            ]
        )
    return "\n".join(lines)


def run_diagnostics(
    trace_file: Path,
    index_dir: Path,
    output_file: Path,
    embedding_model_name: str,
    reranker_model_name: str,
    allow_model_download: bool,
) -> list[DirectGuidanceTrace]:
    traces = read_direct_guidance_traces(trace_file)
    index, metadata = load_index_and_metadata(index_dir)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    faiss_candidates = retrieve_faiss_candidates(QUERY, index, metadata, embedding_model, FAISS_TOP_K)
    reranked_candidates = rerank_candidates(QUERY, faiss_candidates, reranker, top_k=FAISS_TOP_K)
    faiss_lookup = candidate_by_chunk_id(faiss_candidates)
    rerank_lookup = candidate_by_chunk_id(reranked_candidates)

    report_lines = [
        SEPARATOR,
        "Q04 DIRECT_GUIDANCE Rank Diagnostic",
        f"Query: {QUERY}",
        f"FAISS Top-K: {FAISS_TOP_K}",
        f"Reranker: {reranker_model_name}",
        "",
        f"Direct guidance chunks: {len(traces)}",
        "",
    ]

    for trace in traces:
        report_lines.append(render_chunk_report(trace, faiss_lookup, rerank_lookup))
        report_lines.append("")

    report_lines.append(SEPARATOR)
    report_lines.append("Summary")
    report_lines.append(f"Direct guidance chunks: {len(traces)}")
    for trace in traces:
        report_lines.append("")
        report_lines.append(f"Chunk {trace.chunk_id}:")
        faiss_candidate = faiss_lookup.get(trace.chunk_id)
        rerank_candidate = rerank_lookup.get(trace.chunk_id)
        if faiss_candidate is None:
            report_lines.append("FAISS rank: NOT_IN_TOP_100")
            report_lines.append("FAISS score: NOT_IN_TOP_100")
        else:
            report_lines.append(f"FAISS rank: {faiss_candidate['original_rank']}")
            report_lines.append(f"FAISS score: {faiss_candidate['original_similarity_score']:.6f}")
        if rerank_candidate is None:
            report_lines.append("Rerank rank: NOT_IN_TOP_100")
            report_lines.append("Rerank score: NOT_IN_TOP_100")
        else:
            report_lines.append(f"Rerank rank: {rerank_candidate['reranked_rank']}")
            report_lines.append(f"Rerank score: {rerank_candidate['rerank_score']:.6f}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    return traces


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    traces = run_diagnostics(
        args.trace_file,
        args.index_dir,
        args.output_file,
        args.embedding_model,
        args.reranker_model,
        args.allow_model_download,
    )
    print(f"Direct guidance chunks: {len(traces)}")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()
