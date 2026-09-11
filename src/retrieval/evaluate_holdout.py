"""Run unseen holdout retrieval for Test 4 baseline vs Test 6 score fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .evaluate_reranked_v2 import (
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_FILE,
    DEFAULT_METADATA_FILE,
    load_v2_index_and_metadata,
    retrieve_v2_candidates,
)
from .evaluate_score_fusion import add_scores_and_norms, apply_fusion, preview_text
from .rerank import DEFAULT_RERANKER_MODEL, DEFAULT_RERANK_TOP_K, load_embedding_model, load_reranker, rerank_candidates
from .search import DEFAULT_MODEL_NAME


HOLDOUT_QUESTIONS = [
    {"id": "H01", "question": "Deprem anında yatağın yanındaysam nasıl korunmalıyım?"},
    {"id": "H02", "question": "Deprem sırasında merdivenlere yönelmek doğru mu?"},
    {"id": "H03", "question": "Sarsıntı sırasında balkona çıkmalı mıyım?"},
    {"id": "H04", "question": "Deprem anında camlardan uzak durmalı mıyım?"},
    {"id": "H05", "question": "Deprem sırasında mutfaktaysam ilk ne yapmalıyım?"},
    {"id": "H06", "question": "Sarsıntı bittikten hemen sonra binadan çıkmalı mıyım?"},
    {"id": "H07", "question": "Deprem sonrası elektrik şalterini kapatmalı mıyım?"},
    {"id": "H08", "question": "Depremden sonra gaz vanasını kapatmak gerekir mi?"},
    {"id": "H09", "question": "Evde gaz kaçağından şüpheleniyorsam ışıkları açabilir miyim?"},
    {"id": "H10", "question": "Deprem sonrası kibrit veya çakmak kullanmak güvenli mi?"},
    {"id": "H11", "question": "Binadan çıktıktan sonra nereye gitmeliyim?"},
    {"id": "H12", "question": "Toplanma alanına giderken hangi risklere dikkat etmeliyim?"},
    {"id": "H13", "question": "Deprem sonrası binaya eşya almak için geri girebilir miyim?"},
    {"id": "H14", "question": "Artçı sarsıntılar devam ederken hasarlı binada kalmak güvenli mi?"},
    {"id": "H15", "question": "Depremden sonra çevredeki yaralılara yardım etmeli miyim?"},
    {"id": "H16", "question": "İlk yardım bilmiyorsam ağır yaralı birini hareket ettirmeli miyim?"},
    {"id": "H17", "question": "Enkaz altındaysam sürekli bağırmalı mıyım?"},
    {"id": "H18", "question": "Enkaz altında sesimi duyuramıyorsam ne yapabilirim?"},
    {"id": "H19", "question": "Deprem sonrası telefonu hangi durumlarda kullanmalıyım?"},
    {"id": "H20", "question": "Yakınlarıma ulaşmak için arama yerine mesaj kullanmak daha uygun mu?"},
    {"id": "H21", "question": "Deprem çantasını binadan çıkarken yanıma almalı mıyım?"},
    {"id": "H22", "question": "Deprem sonrasında temiz suya ulaşamazsam ne yapmalıyım?"},
    {"id": "H23", "question": "Depremden sonra açık alanda hangi yapılardan uzak durmalıyım?"},
    {"id": "H24", "question": "Sokaktayken deprem olursa binaların yanında beklemeli miyim?"},
    {"id": "H25", "question": "Araçtayken deprem olursa köprü üzerinde durmalı mıyım?"},
    {"id": "H26", "question": "Araçtayken sarsıntı başladığında aracı nasıl durdurmalıyım?"},
    {"id": "H27", "question": "Sahildeyken güçlü bir deprem hissedersem ne yapmalıyım?"},
    {"id": "H28", "question": "Yetkililer tahliye çağrısı yaparsa ne yapmalıyım?"},
    {"id": "H29", "question": "Deprem sonrası resmi açıklamaları nereden takip etmeliyim?"},
    {"id": "H30", "question": "Binanın güvenli olup olmadığına kendim karar verebilir miyim?"},
]

SYSTEM_TEST4 = "test4_baseline"
SYSTEM_TEST6 = "test6_score_fusion_alpha_0.50"
DEFAULT_FAISS_TOP_K = 50
DEFAULT_ALPHA = 0.50
DEFAULT_RESULTS_FILE = Path("data/evaluation/holdout_retrieval_results.json")
DEFAULT_JUDGMENTS_FILE = Path("data/evaluation/holdout_retrieval_judgments.csv")
DEFAULT_REVIEW_FILE = Path("data/evaluation/holdout_retrieval_review.txt")
DEFAULT_MANUAL_REVIEW_FILE = Path("data/evaluation/holdout_retrieval_manual_review.txt")
DEFAULT_MANUAL_MAPPING_FILE = Path("data/evaluation/holdout_retrieval_manual_mapping.json")
SEPARATOR = "=" * 80

CSV_FIELDS = [
    "question_id",
    "question",
    "system",
    "final_rank",
    "chunk_id",
    "source_file",
    "page_start",
    "page_end",
    "faiss_rank",
    "faiss_score",
    "reranker_score",
    "final_fusion_score",
    "relevant",
]

PROTECTED_FILES = [
    Path("data/processed/chunks/afad_earthquake_chunks.jsonl"),
    Path("data/processed/chunks/afad_earthquake_chunks_v2.jsonl"),
    Path("data/indexes/mobile/afad_e5_small.faiss"),
    Path("data/indexes/mobile/metadata.jsonl"),
    Path("data/indexes/mobile_v2/afad_e5_small_v2.faiss"),
    Path("data/indexes/mobile_v2/metadata_v2.jsonl"),
    Path("data/evaluation/retrieval_results.json"),
    Path("data/evaluation/retrieval_judgments.csv"),
    Path("data/evaluation/retrieval_metrics.json"),
    Path("data/evaluation/retrieval_review.txt"),
    Path("data/evaluation/retrieval_reranked_results.json"),
    Path("data/evaluation/retrieval_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_reranked_metrics.json"),
    Path("data/evaluation/retrieval_reranked_review.txt"),
    Path("data/evaluation/retrieval_top50_reranked_results.json"),
    Path("data/evaluation/retrieval_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_top50_reranked_metrics.json"),
    Path("data/evaluation/retrieval_top50_reranked_review.txt"),
    Path("data/evaluation/retrieval_v2_top50_reranked_results.json"),
    Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_v2_top50_reranked_metrics.json"),
    Path("data/evaluation/retrieval_v2_top50_reranked_review.txt"),
    Path("data/evaluation/retrieval_hybrid_results.json"),
    Path("data/evaluation/retrieval_hybrid_judgments.csv"),
    Path("data/evaluation/retrieval_hybrid_metrics.json"),
    Path("data/evaluation/retrieval_hybrid_review.txt"),
    Path("data/evaluation/retrieval_score_fusion_results.json"),
    Path("data/evaluation/retrieval_score_fusion_judgments.csv"),
    Path("data/evaluation/retrieval_score_fusion_metrics.json"),
    Path("data/evaluation/retrieval_score_fusion_review.txt"),
    Path("data/evaluation/retrieval_score_fusion_manual_review.txt"),
    Path("data/evaluation/retrieval_score_fusion_manual_mapping.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unseen holdout evaluation inputs for Test 4 vs Test 6.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--judgments-file", type=Path, default=DEFAULT_JUDGMENTS_FILE)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--manual-review-file", type=Path, default=DEFAULT_MANUAL_REVIEW_FILE)
    parser.add_argument("--manual-mapping-file", type=Path, default=DEFAULT_MANUAL_MAPPING_FILE)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--faiss-top-k", type=int, default=DEFAULT_FAISS_TOP_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
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


def baseline_result(question_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "system": SYSTEM_TEST4,
        "final_rank": int(result["reranked_rank"]),
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "text": result["text"],
        "faiss_rank": int(result["original_rank"]),
        "faiss_score": float(result["original_similarity_score"]),
        "reranker_score": float(result["rerank_score"]),
        "final_fusion_score": None,
    }


def fusion_result(question_id: str, question: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "system": SYSTEM_TEST6,
        "final_rank": int(result["final_rank"]),
        "chunk_id": result["chunk_id"],
        "source_file": result["source_file"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "text": result["text"],
        "faiss_rank": int(result["faiss_rank"]),
        "faiss_score": float(result["faiss_score"]),
        "reranker_score": float(result["reranker_score"]),
        "final_fusion_score": float(result["final_score"]),
    }


def run_holdout(
    index_dir: Path,
    index_file: str,
    metadata_file: str,
    embedding_model_name: str,
    reranker_model_name: str,
    faiss_top_k: int,
    top_k: int,
    alpha: float,
    allow_model_download: bool,
) -> dict[str, Any]:
    index, metadata, index_path, metadata_path = load_v2_index_and_metadata(index_dir, index_file, metadata_file)
    embedding_model = load_embedding_model(embedding_model_name, allow_model_download)
    reranker = load_reranker(reranker_model_name, allow_model_download)

    question_blocks: list[dict[str, Any]] = []
    overlap_ratios: list[float] = []
    for question in HOLDOUT_QUESTIONS:
        candidates = retrieve_v2_candidates(question["question"], index, metadata, embedding_model, faiss_top_k)
        if len({str(candidate["chunk_id"]) for candidate in candidates}) != len(candidates):
            raise RuntimeError(f"{question['id']}: duplicate chunk_id in FAISS Top-{faiss_top_k}")

        reranked_all = rerank_candidates(question["question"], candidates, reranker, top_k=faiss_top_k)
        baseline_top5 = reranked_all[:top_k]
        scored_candidates = add_scores_and_norms(candidates, reranked_all)
        fusion_top5, _ = apply_fusion(scored_candidates, alpha, top_k)

        baseline_results = [baseline_result(question["id"], question["question"], result) for result in baseline_top5]
        fusion_results = [fusion_result(question["id"], question["question"], result) for result in fusion_top5]
        baseline_ids = {str(result["chunk_id"]) for result in baseline_results}
        fusion_ids = {str(result["chunk_id"]) for result in fusion_results}
        overlap_ratios.append(len(baseline_ids & fusion_ids) / float(top_k))

        question_blocks.append(
            {
                "id": question["id"],
                "question": question["question"],
                "faiss_candidate_count": len(candidates),
                "systems": [
                    {"system": SYSTEM_TEST4, "results": baseline_results},
                    {"system": SYSTEM_TEST6, "alpha": alpha, "results": fusion_results},
                ],
                "top5_overlap_count": len(baseline_ids & fusion_ids),
                "top5_overlap_ratio": len(baseline_ids & fusion_ids) / float(top_k),
            }
        )

    return {
        "experiment": "Unseen holdout retrieval evaluation setup",
        "note": "This holdout set was used only for final retrieval evaluation. No retrieval parameters were tuned using holdout results.",
        "systems": {
            SYSTEM_TEST4: "V2 chunks -> multilingual-e5-small -> V2 FAISS Top-50 -> CrossEncoder -> Top-5",
            SYSTEM_TEST6: "V2 chunks -> multilingual-e5-small -> V2 FAISS Top-50 -> CrossEncoder -> min-max score fusion alpha=0.50 -> Top-5",
        },
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "faiss_candidates_per_question": faiss_top_k,
        "final_results_per_system_per_question": top_k,
        "alpha": alpha,
        "average_top5_overlap_between_systems": statistics.mean(overlap_ratios) if overlap_ratios else 0.0,
        "questions": question_blocks,
    }


def validate_payload(payload: dict[str, Any], top_k: int) -> None:
    questions = payload.get("questions", [])
    if len(questions) != 30:
        raise RuntimeError(f"Expected 30 holdout questions, got {len(questions)}")
    for question in questions:
        for system_block in question["systems"]:
            results = system_block.get("results", [])
            if len(results) != top_k:
                raise RuntimeError(f"{question['id']} {system_block['system']}: expected {top_k} results")
            chunk_ids = [str(result["chunk_id"]) for result in results]
            if len(chunk_ids) != len(set(chunk_ids)):
                raise RuntimeError(f"{question['id']} {system_block['system']}: duplicate chunk_id in Top-{top_k}")


def all_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in payload["questions"]:
        for system_block in question["systems"]:
            rows.extend(system_block["results"])
    return rows


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_judgments(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in all_results(payload):
            writer.writerow(
                {
                    "question_id": result["question_id"],
                    "question": result["question"],
                    "system": result["system"],
                    "final_rank": result["final_rank"],
                    "chunk_id": result["chunk_id"],
                    "source_file": result["source_file"],
                    "page_start": result["page_start"],
                    "page_end": result["page_end"],
                    "faiss_rank": result["faiss_rank"],
                    "faiss_score": f"{float(result['faiss_score']):.6f}",
                    "reranker_score": f"{float(result['reranker_score']):.6f}",
                    "final_fusion_score": ""
                    if result["final_fusion_score"] is None
                    else f"{float(result['final_fusion_score']):.6f}",
                    "relevant": "",
                }
            )


def render_review(payload: dict[str, Any]) -> str:
    blocks: list[str] = []
    for question in payload["questions"]:
        lines = [
            SEPARATOR,
            f"question_id: {question['id']}",
            f"question: {question['question']}",
            f"top5_overlap_count: {question['top5_overlap_count']}",
            f"top5_overlap_ratio: {question['top5_overlap_ratio']:.4f}",
            "",
        ]
        for system_block in question["systems"]:
            lines.extend([f"system: {system_block['system']}", ""])
            for result in system_block["results"]:
                lines.extend(
                    [
                        f"final rank: {result['final_rank']}",
                        f"chunk_id: {result['chunk_id']}",
                        f"source_file: {result['source_file']}",
                        f"page_start/page_end: {result['page_start']}-{result['page_end']}",
                        f"FAISS rank: {result['faiss_rank']}",
                        f"FAISS score: {float(result['faiss_score']):.6f}",
                        f"reranker score: {float(result['reranker_score']):.6f}",
                        "final fusion score: "
                        + (
                            "N/A"
                            if result["final_fusion_score"] is None
                            else f"{float(result['final_fusion_score']):.6f}"
                        ),
                        "text:",
                        str(result["text"]).strip(),
                        "",
                    ]
                )
        blocks.append("\n".join(lines).rstrip())
    return "\n\n".join(blocks) + "\n"


def write_review(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_review(payload), encoding="utf-8")


def grouped_manual_records(payload: dict[str, Any]) -> OrderedDict[tuple[str, str], list[dict[str, Any]]]:
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for result in all_results(payload):
        key = (str(result["question_id"]), str(result["chunk_id"]))
        grouped.setdefault(key, []).append(result)
    return grouped


def render_manual_review(grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]]) -> str:
    blocks: list[str] = []
    for index, records in enumerate(grouped.values(), start=1):
        first = records[0]
        systems = ", ".join(record["system"] for record in records)
        rank_lines = "\n".join(f"{record['system']} -> rank {record['final_rank']}" for record in records)
        blocks.append(
            "\n".join(
                [
                    SEPARATOR,
                    f"manual_review_id: M{index:03d}",
                    f"question_id: {first['question_id']}",
                    f"question: {first['question']}",
                    f"chunk_id: {first['chunk_id']}",
                    f"appeared_in_systems: {systems}",
                    "ranks:",
                    rank_lines,
                    f"source_file: {first['source_file']}",
                    f"page_start/page_end: {first['page_start']}-{first['page_end']}",
                    "",
                    "text:",
                    str(first["text"]).strip(),
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_manual_review(path: Path, grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_manual_review(grouped), encoding="utf-8")


def write_manual_mapping(path: Path, grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]]) -> None:
    records: list[dict[str, Any]] = []
    for index, grouped_records in enumerate(grouped.values(), start=1):
        first = grouped_records[0]
        records.append(
            {
                "manual_review_id": f"M{index:03d}",
                "question_id": first["question_id"],
                "question": first["question"],
                "chunk_id": first["chunk_id"],
                "source_file": first["source_file"],
                "page_start": first["page_start"],
                "page_end": first["page_end"],
                "judgment_rows": [
                    {
                        "question_id": record["question_id"],
                        "system": record["system"],
                        "final_rank": record["final_rank"],
                        "chunk_id": record["chunk_id"],
                    }
                    for record in grouped_records
                ],
            }
        )
    payload = {
        "source_judgments_file": str(DEFAULT_JUDGMENTS_FILE),
        "unique_key": "question_id + chunk_id",
        "manual_review_count": len(records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(
    payload: dict[str, Any],
    grouped: OrderedDict[tuple[str, str], list[dict[str, Any]]],
    results_file: Path,
    judgments_file: Path,
    review_file: Path,
    manual_review_file: Path,
    manual_mapping_file: Path,
    protected_unchanged: bool,
) -> None:
    results = all_results(payload)
    test4_results = sum(1 for result in results if result["system"] == SYSTEM_TEST4)
    test6_results = sum(1 for result in results if result["system"] == SYSTEM_TEST6)
    duplicate_eliminated = len(results) - len(grouped)
    print(f"Holdout questions: {len(payload['questions'])}")
    print(f"Test 4 results: {test4_results}")
    print(f"Test 6 results: {test6_results}")
    print(f"Total judgment rows: {len(results)}")
    print(f"Unique manual judgments required: {len(grouped)}")
    print(f"Duplicate judgments eliminated: {duplicate_eliminated}")
    print(f"Average Top-5 overlap between Test 4 and Test 6: {payload['average_top5_overlap_between_systems']:.4f}")
    print("")
    print(f"All relevant fields initially blank: true")
    print(f"Protected Test 1-6/chunk/index files unchanged: {str(protected_unchanged).lower()}")
    print("")
    print("Output files:")
    print(results_file)
    print(judgments_file)
    print(review_file)
    print(manual_review_file)
    print(manual_mapping_file)
    print("")
    print("This holdout set was used only for final retrieval evaluation.")
    print("No retrieval parameters were tuned using holdout results.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    before = snapshot(PROTECTED_FILES)
    payload = run_holdout(
        index_dir=args.index_dir,
        index_file=args.index_file,
        metadata_file=args.metadata_file,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        faiss_top_k=args.faiss_top_k,
        top_k=args.top_k,
        alpha=args.alpha,
        allow_model_download=args.allow_model_download,
    )
    validate_payload(payload, args.top_k)
    grouped = grouped_manual_records(payload)

    write_results(args.results_file, payload)
    write_judgments(args.judgments_file, payload)
    write_review(args.review_file, payload)
    write_manual_review(args.manual_review_file, grouped)
    write_manual_mapping(args.manual_mapping_file, grouped)

    after = snapshot(PROTECTED_FILES)
    print_summary(
        payload,
        grouped,
        args.results_file,
        args.judgments_file,
        args.review_file,
        args.manual_review_file,
        args.manual_mapping_file,
        before == after,
    )


if __name__ == "__main__":
    main()
