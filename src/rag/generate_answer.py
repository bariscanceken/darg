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

from .answerability import (
    DEFAULT_GATE_VERSION,
    GATE_PROMPTS,
    REFUSAL_ANSWER,
    check_answerability,
    check_answerability_majority,
    is_answerable,
)
from .clarification import suggest_clarifications
from .context import filter_sources_by_answer_references, format_context, format_sources_for_cli, source_metadata
from .evidence import ANSWER_FROM_EVIDENCE_INSTRUCTIONS, format_evidence, mark_wider_area, select_evidence
from .llm_query_repair import repair_query
from .local_info import CARD_CHUNK_PREFIX, build_local_info, format_local_info, local_info_evidence
from .location import Location, detect_location, has_regional_plan, place_names
from .prompt import DEFAULT_GENERATION_VERSION, GENERATION_PROMPTS, build_user_prompt
from .query_normalization import normalize_query as normalize_question_text
from .regional_retrieval import retrieve_enhanced
from .source_routing import classify_question, policy_for


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 700
CLI_GATE_VERSION = "v1"  # switch to "v2" only after it passes the held-out evaluation
CLI_LLM_REPAIR = True  # enabled after query_typo_eval_v2 (fix 10/15, keep 8/8)
# mobile_v4: encoding-repaired, E5-window-sized chunks; holdout Hit@5 20/30 -> 26/30 vs mobile_v2 (LLM judge).
# mobile_v5: section-aware chunks with headers, structured tables, OCR'd pages. Full measurement 16 Oct 2026 with the
# same CLI pipeline (74 + 13, gpt-4o judge, not human-calibrated): answered 53 -> 59/74, false accepts 0 -> 0/13,
# fully faithful 85% -> 81%, answers with a critical issue 2% -> 0%.
CLI_INDEX_DIR = Path("data/indexes/mobile_v5")
# Faithfulness pilot (20 questions, gpt-4o judge): answers with a critical issue 5/16 (v1) -> 1/14 (v2).
CLI_GENERATION_VERSION = "v2"
# Source-type routing: retrieval Hit@5 67 vs 68/74 (losses were past-event reports for behaviour questions);
# blind side-by-side (12 user-flagged questions, assessed by Claude): better 5, worse 1, equal 6.
CLI_SOURCE_ROUTING = True
# Evidence-first answers on top of routing (same 12 questions): better 7, worse 3, equal 2; answers can be terse.
CLI_EVIDENCE_FIRST = True
CLI_GATE_VOTES = 1  # set after the full faithfulness / false-accept measurement
# Verified province-card lines (assembly-area lookup, plan counts, shelters) for questions naming a place.
CLI_LOCAL_INFO = True
# Question types whose answer must be about the named place; quotes not naming it are marked wider-area.
PLACE_SPECIFIC_TYPES = frozenset({"bolgesel", "gecmis"})
# Skip the gate when evidence-first is on (evidence selection refuses when no verbatim quote answers) and let
# gpt-4o write the answer from the selected evidence. Full measurement 14 Oct 2026 (74 answerable + 13 negatives,
# current CLI pipeline, gpt-4o judge, not human-calibrated): answered 56 -> 53/74, false accepts 0 -> 0/13,
# fully faithful 55% -> 85%, answers with a critical issue 16% -> 2%.
CLI_EVIDENCE_GATE = True
CLI_GENERATION_MODEL = "gpt-4o"
# Pipeline v2 (src/rag/pipeline_v2.py): understand -> wide retrieval on the section-aware v5 index -> answer with
# code-verified quotes. Opt-in with --pipeline v2 until measured against v1.
CLI_PIPELINE = "v1"
CLI_V2_INDEX_DIR = Path("data/indexes/mobile_v5")
NO_REGIONAL_PLAN_NOTE = ("Not: {province} için il afet risk azaltma planı (IRAP) kaynaklarımda bulunmuyor; "
                         "cevap genel AFAD bilgilerine dayanmaktadır.")
LOCAL_ONLY_LEAD = "Kaynaklarımdaki ilgili yerel bilgiler şunlar:"
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
        "--source-routing",
        action=argparse.BooleanOptionalAction,
        default=CLI_SOURCE_ROUTING,
        help="Classify the question type and restrict retrieval to suitable source types.",
    )
    parser.add_argument(
        "--evidence-first",
        action=argparse.BooleanOptionalAction,
        default=CLI_EVIDENCE_FIRST,
        help="Select verbatim evidence first and write the answer only from it.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("v1", "v2"),
        default=CLI_PIPELINE,
        help="v2: understand -> wide retrieval (mobile_v5) -> answer with code-verified quotes.",
    )
    parser.add_argument(
        "--generation-model",
        default=CLI_GENERATION_MODEL,
        help="Model that writes the final answer (default: same as --openai-model).",
    )
    parser.add_argument(
        "--evidence-gate",
        action=argparse.BooleanOptionalAction,
        default=CLI_EVIDENCE_GATE,
        help="With --evidence-first, skip the answerability gate and let evidence selection refuse.",
    )
    parser.add_argument(
        "--local-info",
        action=argparse.BooleanOptionalAction,
        default=CLI_LOCAL_INFO,
        help="Append verified province-card info (assembly areas, shelters) when the question names a place.",
    )
    parser.add_argument(
        "--gate-votes",
        type=int,
        default=CLI_GATE_VOTES,
        help="Majority vote over this many gate calls (odd number; library default 1).",
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
    gate_votes: int = 1,
    source_routing: bool = False,
    evidence_first: bool = False,
    local_info: bool = False,
    evidence_gate: bool = False,
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
    if (llm_rerank or source_routing) and client is None:
        load_environment()
        model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        client = load_openai_client()
    policy = None
    if source_routing:
        question_type = classify_question(normalized_question, client, model)
        policy = policy_for(question_type)
        normalization_fields["question_type"] = question_type
    if location_for_search is not None or lexical or llm_rerank or source_routing:
        index, metadata, embedding_model, reranker = load_retrieval_components(
            str(index_dir), DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, DEFAULT_MODEL_NAME, DEFAULT_RERANKER_MODEL,
            allow_model_download,
        )
        chunks = retrieve_enhanced(normalized_question, location_for_search, index, metadata, embedding_model, reranker,
                                   top_k=DEFAULT_RERANK_TOP_K, lexical=lexical,
                                   llm_reranker=(client, model) if llm_rerank else None, policy=policy)
    else:
        chunks = retrieve_top_chunks(normalized_question, index_dir=index_dir, allow_model_download=allow_model_download)
    if regional:
        normalization_fields["location"] = None if location is None else {
            "province": location.province,
            "district": location.district,
            "ambiguous_provinces": list(location.ambiguous_provinces),
            "has_regional_plan": has_regional_plan(location.province),
        }
    if local_info:
        normalization_fields["local_info"] = build_local_info(location, normalized_question,
                                                              normalization_fields.get("question_type"))
    formatted = format_context(chunks)
    if return_context:
        normalization_fields["context"] = formatted["context"]
    if client is None:
        load_environment()
        model = openai_model or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
        client = load_openai_client()
    card_evidence, card_sources = [], []
    if evidence_first:
        card_evidence, card_sources = local_info_evidence(normalization_fields.get("local_info"),
                                                          len(formatted["sources"]) + 1)
    all_sources = formatted["sources"] + card_sources
    if return_context and card_evidence:  # judges must see the local lines the answer may cite
        normalization_fields["context"] += "".join(f"\n\n[Kaynak {item['source']}]\nMetin: {item['quote']}"
                                                   for item in card_evidence)
    if evidence_first and evidence_gate:
        answerability = {"label": "ANSWERABLE", "raw": "skipped: evidence selection decides"}
    elif gate_votes > 1:
        answerability = check_answerability_majority(client, model, normalized_question, formatted["context"],
                                                     gate_version, gate_votes)
    else:
        answerability = check_answerability(client, model, normalized_question, formatted["context"], gate_version)
    gate_refused = not is_answerable(answerability["label"])
    if gate_refused and not card_evidence:
        return {
            "question": question,
            "answer": REFUSAL_ANSWER,
            "sources": [],
            "retrieved_sources": formatted["sources"],
            "answerability": answerability,
            **normalization_fields,
        }

    instructions = GENERATION_PROMPTS[generation_version]
    generation_context = formatted["context"]
    if evidence_first:
        if gate_refused:  # retrieved passages rejected; only the verified local lines remain
            selection = {"question_situation": "", "evidence": [], "raw_count": 0, "status": "gate_refused"}
        else:
            selection = select_evidence(normalized_question, formatted["context"], chunks, client, model)
        if location_for_search is not None and normalization_fields.get("question_type") in PLACE_SPECIFIC_TYPES:
            selection["evidence"] = mark_wider_area(selection["evidence"], place_names(location_for_search.province))
        selection["evidence"] = selection["evidence"] + card_evidence
        normalization_fields["evidence"] = selection
        if not selection["evidence"]:
            if evidence_gate:
                answerability = {"label": "NOT_ANSWERABLE", "raw": "no evidence selected (gate skipped)"}
            return {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "sources": [],
                "retrieved_sources": formatted["sources"],
                "answerability": answerability,
                **normalization_fields,
            }
        instructions = instructions + ANSWER_FROM_EVIDENCE_INSTRUCTIONS
        generation_context = "Seçilmiş kanıtlar:\n" + format_evidence(selection["evidence"])

    response = client.responses.create(
        model=generation_model or model,
        instructions=instructions,
        input=build_user_prompt(normalized_question, generation_context),
        max_output_tokens=max_output_tokens,
        temperature=0.2,
    )

    answer = response_text(response)
    if evidence_first and answer.startswith(REFUSAL_ANSWER):
        answer = REFUSAL_ANSWER  # drop "[Kaynak N]" leftovers or extra sentences after a refusal
        if evidence_gate or gate_refused:
            answerability = {"label": "NOT_ANSWERABLE", "raw": "answer refused (gate skipped or card only)"}
    elif gate_refused:
        answerability = {"label": "ANSWERABLE", "raw": "gate refused retrieved passages; answered from verified local lines"}
    cited_sources = filter_sources_by_answer_references(answer, all_sources)
    if normalization_fields.get("local_info") and not normalization_fields["local_info"]["place"] \
            and answer != REFUSAL_ANSWER:
        normalization_fields["local_info"] = None  # place-free brochure lines only stand in for a refusal

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
    if result["answer"] == REFUSAL_ANSWER and result.get("local_info"):
        return  # the verified local section answers it; no clarification round
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


def generate_answer_v2(question: str, *, index_dir: Path = CLI_V2_INDEX_DIR, location: Location | None = None,
                       local_info: bool = True, answer_model: str | None = None,
                       allow_model_download: bool = False) -> dict[str, Any]:
    """Pipeline v2 with the same result shape as generate_answer (for printing and evaluation)."""
    from .pipeline_v2 import ANSWER_MODEL, run_pipeline_v2

    load_environment()
    client = load_openai_client()
    index, metadata, embedding_model, reranker = load_retrieval_components(
        str(index_dir), DEFAULT_INDEX_FILE, DEFAULT_METADATA_FILE, DEFAULT_MODEL_NAME, DEFAULT_RERANKER_MODEL,
        allow_model_download,
    )
    result = run_pipeline_v2(question, client, index, metadata, embedding_model, reranker, location=location,
                             local_info_builder=build_local_info if local_info else None,
                             answer_model=answer_model or ANSWER_MODEL)
    sources = [source_metadata(number, passage) for number, passage in enumerate(result["passages"], start=1)]
    found = result["location"]
    understanding = result["understanding"]
    refused = result["answer"] == REFUSAL_ANSWER
    return {
        "question": question,
        "answer": result["answer"],
        "sources": filter_sources_by_answer_references(result["answer"], sources),
        "retrieved_sources": sources,
        "answerability": {"label": "NOT_ANSWERABLE" if refused else "ANSWERABLE", "raw": "pipeline v2"},
        "normalized_question": understanding["corrected"],
        "query_repair": {"status": "accepted" if understanding.get("spelling") == "accepted" else "not_needed"},
        "question_type": understanding["type"],
        "understanding": understanding,
        "verified": result["verified"],
        "context": "\n\n".join(f"[Kaynak {n}]\nMetin: {p['text']}" for n, p in enumerate(result["passages"], start=1)),
        "location": None if found is None else {
            "province": found.province, "district": found.district,
            "ambiguous_provinces": list(found.ambiguous_provinces), "has_regional_plan": has_regional_plan(found.province),
        },
        "local_info": result["local_info"],
    }


def answer_for_cli(question: str, args: argparse.Namespace, location: Location | None = None) -> dict[str, Any]:
    if args.pipeline == "v2":
        return generate_answer_v2(question, location=location, local_info=args.local_info,
                                  allow_model_download=args.allow_model_download)
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
        gate_votes=args.gate_votes,
        source_routing=args.source_routing,
        evidence_first=args.evidence_first,
        local_info=args.local_info,
        evidence_gate=args.evidence_gate,
        generation_model=args.generation_model,
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
    local_only = result["answer"] == REFUSAL_ANSWER and bool(result.get("local_info"))
    print(LOCAL_ONLY_LEAD if local_only else result["answer"])
    location = result.get("location") or {}
    if location.get("province") and not location.get("has_regional_plan"):
        print("")
        print(NO_REGIONAL_PLAN_NOTE.format(province=location["province"]))
    card_cited = any(str(source.get("chunk_id", "")).startswith(CARD_CHUNK_PREFIX) for source in result["sources"])
    if result.get("local_info") and card_cited:
        if result["local_info"]["note"]:
            print("")
            print(result["local_info"]["note"])  # the answer already uses the local lines
    elif result.get("local_info"):
        print("")
        print(format_local_info(result["local_info"]))
    if local_only:
        return  # the local section names its own sources
    print("")
    print("Kaynaklar:")
    source_text = format_sources_for_cli(result["sources"])
    print(source_text if source_text else "Cevapta kaynak referansı bulunmadı.")


if __name__ == "__main__":
    main()
