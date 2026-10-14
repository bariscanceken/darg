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

from .answerability import DEFAULT_GATE_VERSION, GATE_PROMPTS, REFUSAL_ANSWER, check_answerability, is_answerable
from .clarification import suggest_clarifications
from .context import filter_sources_by_answer_references, format_context, format_sources_for_cli
from .llm_query_repair import repair_query
from .location import Location, detect_location, has_regional_plan
from .prompt import DEFAULT_GENERATION_VERSION, GENERATION_PROMPTS, build_user_prompt
from .query_normalization import normalize_query as normalize_question_text
from .regional_retrieval import retrieve_enhanced


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 700
CLI_GATE_VERSION = "v1"  # switch to "v2" only after it passes the held-out evaluation
CLI_LLM_REPAIR = True  # enabled after query_typo_eval_v2 (fix 10/15, keep 8/8)
# Encoding-repaired, E5-window-sized chunks; holdout Hit@5 20/30 -> 26/30 vs mobile_v2 (LLM judge).
CLI_INDEX_DIR = Path("data/indexes/mobile_v4")
# Faithfulness pilot (20 questions, gpt-4o judge): answers with a critical issue 5/16 (v1) -> 1/14 (v2).
CLI_GENERATION_VERSION = "v2"
NO_REGIONAL_PLAN_NOTE = ("Not: {province} için il afet risk azaltma planı (IRAP) kaynaklarımda bulunmuyor; "
                         "cevap genel AFAD bilgilerine dayanmaktadır.")
REFUSAL_HINT = (
    "İpucu: Sorunuzu daha açık yazmayı deneyin; örneğin durumu ('deprem sırasında', "
    "'deprem sonrasında', 'enkaz altındaysam') ve ne yapmak istediğinizi belirtin."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DARG answer using local retrieval and OpenAI generation.")
    parser.add_argument("--index-dir", type=Path, default=CLI_INDEX_DIR)
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
    parser.add_argument(
        "--no-query-normalization",
        action="store_true",
        help="Disable local typo normalization of the retrieval query.",
    )
    parser.add_argument(
        "--gate-version",
        choices=sorted(GATE_PROMPTS),
        default=CLI_GATE_VERSION,
        help="Answerability gate prompt version (library default stays v1).",
    )
    parser.add_argument(
        "--llm-repair",
        action=argparse.BooleanOptionalAction,
        default=CLI_LLM_REPAIR,
        help="Send words the local normalizer cannot explain to a spelling-only LLM fix.",
    )
    parser.add_argument(
        "--clarify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the gate refuses, offer 'did you mean…?' questions for the user to choose from.",
    )
    parser.add_argument(
        "--regional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Region-aware retrieval when the question names a province or district.",
    )
    parser.add_argument(
        "--lexical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a BM25 (F5-stemmed) candidate pool to the dense pool before reranking.",
    )
    parser.add_argument(
        "--llm-rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reorder the cross-encoder top-20 with the LLM before taking the Top-5.",
    )
    parser.add_argument(
        "--generation-version",
        choices=sorted(GENERATION_PROMPTS),
        default=CLI_GENERATION_VERSION,
        help="Answer-generation prompt version (library default stays v1).",
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
    normalize_query: bool = False,
    gate_version: str = DEFAULT_GATE_VERSION,
    llm_repair: bool = False,
    index_dir: Path = DEFAULT_INDEX_DIR,
    regional: bool = False,
    location: Location | None = None,
    lexical: bool = False,
    llm_rerank: bool = False,
    return_context: bool = False,
    generation_version: str = DEFAULT_GENERATION_VERSION,
    generation_model: str | None = None,
) -> dict[str, Any]:
    """Generate an answer with Python-owned retrieval source metadata.

    With normalize_query, retrieval, the answerability gate and generation all
    receive the typo-normalized question; the result keeps the original question.
    With llm_repair as well, words the local normalizer cannot explain are sent to
    a spelling-only LLM fix that is accepted only after a local word-level check.
    Without these flags (the default), every step receives the original question.
    """
    normalized_question = question
    normalization_fields: dict[str, Any] = {}
    client = None
    model = openai_model or DEFAULT_OPENAI_MODEL
    if normalize_query:
        normalization = normalize_question_text(question)
        normalized_question = normalization["normalized"]
        normalization_fields = {
            "normalized_question": normalized_question,
            "query_corrections": normalization["corrections"],
        }
        if llm_repair:
            load_environment()
            model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
            client = load_openai_client()
            repair = repair_query(normalized_question, client, model)
            normalized_question = repair["repaired"]
            normalization_fields["normalized_question"] = normalized_question
            normalization_fields["query_repair"] = repair

    if regional and location is None:
        location = detect_location(normalized_question)
    location_for_search = location if regional and location is not None and not location.is_ambiguous else None
    if llm_rerank and client is None:
        load_environment()
        model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        client = load_openai_client()
    if location_for_search is not None or lexical or llm_rerank:
        index, metadata, embedding_model, reranker = load_retrieval_components(
            str(index_dir), DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, DEFAULT_MODEL_NAME, DEFAULT_RERANKER_MODEL,
            allow_model_download,
        )
        chunks = retrieve_enhanced(normalized_question, location_for_search, index, metadata, embedding_model, reranker,
                                   top_k=DEFAULT_RERANK_TOP_K, lexical=lexical,
                                   llm_reranker=(client, model) if llm_rerank else None)
    else:
        chunks = retrieve_top_chunks(normalized_question, index_dir=index_dir, allow_model_download=allow_model_download)
    if regional:
        normalization_fields["location"] = None if location is None else {
            "province": location.province,
            "district": location.district,
            "ambiguous_provinces": list(location.ambiguous_provinces),
            "has_regional_plan": has_regional_plan(location.province),
        }
    formatted = format_context(chunks)
    if return_context:
        normalization_fields["context"] = formatted["context"]
    if client is None:
        load_environment()
        model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        client = load_openai_client()
    answerability = check_answerability(client, model, normalized_question, formatted["context"], gate_version)
    if not is_answerable(answerability["label"]):
        return {
            "question": question,
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved_sources": formatted["sources"],
            "answerability": answerability,
            **normalization_fields,
        }

    response = client.responses.create(
        model=generation_model or model,
        instructions=GENERATION_PROMPTS[generation_version],
        input=build_user_prompt(normalized_question, formatted["context"]),
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
        **normalization_fields,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    question = input("Soru: ").strip()
    if not question:
        raise SystemExit("Soru boş olamaz.")

    location = choose_location(question) if args.regional else None
    result = answer_for_cli(question, args, location)
    print_result(result)
    if is_answerable(result["answerability"]["label"]) or not args.clarify:
        if not is_answerable(result["answerability"]["label"]):
            print(REFUSAL_HINT)
        return

    load_environment()
    model = args.openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    options = suggest_clarifications(result.get("normalized_question", question), load_openai_client(), model)
    if options:
        print("")
        print("Olası anlamlar kontrol ediliyor…")
    # Only offer questions the pipeline can actually answer; their answers are ready on selection.
    answerable = [
        (option, option_result)
        for option, option_result in ((option, answer_for_cli(option, args)) for option in options)
        if is_answerable(option_result["answerability"]["label"])
    ]
    if not answerable:
        print(REFUSAL_HINT)
        return

    print("")
    print("Sorunuzu tam anlayamadım. Şunlardan birini mi sormak istediniz?")
    for number, (option, _) in enumerate(answerable, start=1):
        print(f"  {number}) {option}")
    choice = input(f"Seçiminiz (1-{len(answerable)}, geçmek için Enter): ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(answerable):
        print(REFUSAL_HINT)
        return

    print("")
    print_result(answerable[int(choice) - 1][1])


def choose_location(question: str) -> Location | None:
    """Ask which province is meant when a district name exists in several provinces."""
    location = detect_location(question)
    if location is None or not location.is_ambiguous:
        return location
    print(f"'{location.district}' adında birden fazla ilçe var. Hangisini kastediyorsunuz?")
    for number, province in enumerate(location.ambiguous_provinces, start=1):
        print(f"  {number}) {location.district} ({province})")
    choice = input(f"Seçiminiz (1-{len(location.ambiguous_provinces)}, geçmek için Enter): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(location.ambiguous_provinces):
        province = location.ambiguous_provinces[int(choice) - 1]
        return Location(province=province, district=location.district, matched_text=location.matched_text)
    return location


def answer_for_cli(question: str, args: argparse.Namespace, location: Location | None = None) -> dict[str, Any]:
    return generate_answer(
        question,
        openai_model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
        allow_model_download=args.allow_model_download,
        normalize_query=not args.no_query_normalization,
        gate_version=args.gate_version,
        llm_repair=args.llm_repair,
        index_dir=args.index_dir,
        regional=args.regional,
        location=location,
        lexical=args.lexical,
        llm_rerank=args.llm_rerank,
        generation_version=args.generation_version,
    )


def print_result(result: dict[str, Any]) -> None:
    repair = result.get("query_repair") or {}
    corrections = result.get("query_corrections") or []
    if repair.get("status") == "accepted":
        print(f"(Sorunuzu şöyle anladım: {result['normalized_question']})")
    elif corrections:
        pairs = ", ".join(f"{item['original']} → {item['corrected']}" for item in corrections)
        print(f"(Arama için yazım düzeltildi: {pairs})")

    print("")
    print("DARG:")
    print(result["answer"])
    location = result.get("location") or {}
    if location.get("province") and not location.get("has_regional_plan"):
        print("")
        print(NO_REGIONAL_PLAN_NOTE.format(province=location["province"]))
    print("")
    print("Kaynaklar:")
    source_text = format_sources_for_cli(result["sources"])
    print(source_text if source_text else "Cevapta kaynak referansı bulunmadı.")


if __name__ == "__main__":
    main()
