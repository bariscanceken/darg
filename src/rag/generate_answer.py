"""Generate DARG answers from local retrieval context using the OpenAI Responses API."""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.retrieval.evaluate_reranked_v2 import (
    DEFAULT_FAISS_TOP_K,
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_FILE,
    DEFAULT_METADATA_FILE,
    load_v2_index_and_metadata,
    retrieve_v2_candidates,
)
from src.retrieval.rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_K, load_embedding_model, load_reranker, rerank_candidates
from src.retrieval.search import DEFAULT_MODEL_NAME

from .answerability import REFUSAL_ANSWER, check_answerability, is_answerable
from .context import filter_sources_by_answer_references, format_context, format_sources_for_cli
from .prompt import SYSTEM_PROMPT, build_user_prompt


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DARG answer using local retrieval and OpenAI generation.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if local retrieval models are not in cache.",
    )
    return parser.parse_args()


@lru_cache(maxsize=4)
def load_retrieval_components(
    index_dir: str,
    index_file: str,
    metadata_file: str,
    embedding_model_name: str,
    reranker_model_name: str,
    allow_model_download: bool,
) -> tuple[Any, list[dict[str, Any]], Any, Any]:
    """Load existing local retrieval components once per process."""
    index, metadata, _, _ = load_v2_index_and_metadata(Path(index_dir), index_file, metadata_file)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)
    return index, metadata, embedding_model, reranker


def retrieve_top_chunks(
    question: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    index_file: str = DEFAULT_INDEX_FILE,
    metadata_file: str = DEFAULT_METADATA_FILE,
    embedding_model_name: str = DEFAULT_MODEL_NAME,
    reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    faiss_top_k: int = DEFAULT_FAISS_TOP_K,
    top_k: int = DEFAULT_RERANK_TOP_K,
    allow_model_download: bool = False,
) -> list[dict[str, Any]]:
    """Run the frozen V2 retrieval pipeline and return final Top-K chunks."""
    index, metadata, embedding_model, reranker = load_retrieval_components(
        str(index_dir),
        index_file,
        metadata_file,
        embedding_model_name,
        reranker_model_name,
        allow_model_download,
    )
    candidates = retrieve_v2_candidates(question, index, metadata, embedding_model, faiss_top_k)
    reranked = rerank_candidates(question, candidates, reranker, top_k=faiss_top_k)
    return reranked[: max(top_k, 1)]


def load_environment() -> None:
    """Load local environment variables from .env."""
    from dotenv import load_dotenv

    load_dotenv()


def load_openai_client() -> Any:
    """Create an OpenAI client without logging secrets."""
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyasına ekleyin veya ortam değişkeni olarak tanımlayın.")
    return OpenAI()


def response_text(response: Any) -> str:
    """Extract text from an OpenAI Responses API response."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def generate_answer(
    question: str,
    *,
    openai_model: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    allow_model_download: bool = False,
) -> dict[str, Any]:
    """Generate an answer with Python-owned retrieval source metadata."""
    chunks = retrieve_top_chunks(question, allow_model_download=allow_model_download)
    formatted = format_context(chunks)
    load_environment()
    model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    client = load_openai_client()
    answerability = check_answerability(client, model, question, formatted["context"])
    if not is_answerable(answerability["label"]):
        return {
            "question": question,
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved_sources": formatted["sources"],
            "answerability": answerability,
        }

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=build_user_prompt(question, formatted["context"]),
        max_output_tokens=max_output_tokens,
        temperature=0.2,
    )

    answer = response_text(response)
    cited_sources = filter_sources_by_answer_references(answer, formatted["sources"])

    return {
        "question": question,
        "answer": answer,
        "sources": cited_sources,
        "retrieved_sources": formatted["sources"],
        "answerability": answerability,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    question = input("Soru: ").strip()
    if not question:
        raise SystemExit("Soru boş olamaz.")

    result = generate_answer(
        question,
        openai_model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
        allow_model_download=args.allow_model_download,
    )

    print("")
    print("DARG:")
    print(result["answer"])
    print("")
    print("Kaynaklar:")
    source_text = format_sources_for_cli(result["sources"])
    if source_text:
        print(source_text)
    else:
        print("Cevapta kaynak referansı bulunmadı.")


if __name__ == "__main__":
    main()
