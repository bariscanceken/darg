"""Claim-level faithfulness evaluation of generated answers (Faz 3).

Answers are produced with the CLI configuration (normalisation, LLM spelling repair,
region-aware + BM25 retrieval, LLM rerank, gate v1, mobile_v4). A gpt-4o judge
splits each answer into claims and checks, per claim:
  supported        yes / partial / no   (against the [Kaynak N] passages shown)
  situation_match  yes / no / na        (advice given for the same situation as the
                                         question: preparation, during, after,
                                         trapped under debris, a specific group/place)
  critical         a safety instruction that is unsupported or applied to the wrong
                   situation and could endanger the user
plus whether the answer presents general guidance as local information for a place
named in the question (location_misattribution).

The judge is NOT yet calibrated against human labels for this task: the per-answer
report is written for a researcher to review a sample. Final Test files are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from src.rag.answerability import REFUSAL_ANSWER, extract_response_text
from src.rag.generate_answer import CLI_INDEX_DIR, generate_answer, generate_answer_v2, load_environment, load_openai_client
from src.retrieval.evaluate_holdout import HOLDOUT_QUESTIONS


JUDGE_MODEL = "gpt-4o"
JUDGE_PROMPT_VERSION = "faithfulness_v2"
CACHE_FILE = Path("data/evaluation/faithfulness_judge_cache.jsonl")
RESULTS_FILE = Path("data/evaluation/faithfulness_v2_results.jsonl")
REVIEW_FILE = Path("data/evaluation/faithfulness_v2_review.txt")
SUMMARY_FILE = Path("data/evaluation/faithfulness_v2_summary.txt")

FAITHFULNESS_SYSTEM_PROMPT = """Sen bir RAG cevap sadakati denetçisisin. Sana bir kullanıcı sorusu, AFAD
kaynak parçaları ([Kaynak N]) ve sistemin cevabı verilecek. Titiz ol: amacın,
kullanıcıyı yanlış durumda yanlış davranmaya yöneltebilecek hataları yakalamak.

Adım 1 - Sorunun durumunu belirle ("question_situation"): hazırlık / deprem SIRASINDA /
deprem SONRASINDA / enkaz altında / genel bilgi / belirli kişi grubu / belirli il-ilçe.

Adım 2 - Cevabı tek tek iddialara ayır (her madde veya bağımsız tavsiye/bilgi cümlesi).
Her iddia için:
- "source_situation": Kaynak bu bilgiyi HANGİ durum için veriyor? Seçenekler: hazırlık,
  sırasında, sonrasında, enkaz altında, genel bilgi, belirli grup, GEÇMİŞ OLAY KAYDI
  (geçmişteki belirli bir depremden sonra yapılanları anlatan rapor/plan metni),
  kaynakta yok.
- "supported": "yes" = kaynakta açıkça var; "partial" = genelleştirilmiş veya kaynakta
  olmayan ayrıntı eklenmiş; "no" = kaynakta yok.
- "situation_match": "no" eğer source_situation ile question_situation farklıysa ve
  cevap iddiayı sorunun durumu için geçerliymiş gibi sunuyorsa. Örnekler:
  * Soru deprem SIRASINDA iken gaz kaçağı, vana kapatma, artçı sarsıntı, tahliye gibi
    deprem SONRASI tavsiyeleri sunmak -> "no".
  * Geçmiş bir depremden sonra kurulmuş çadırkent, aşevi vb. bilgisini bugün gidilecek
    yer talimatı gibi sunmak -> "no".
  * Enkaz altındaki kişiye yönelik tavsiyeyi genel tavsiye gibi sunmak -> "no".
  "yes" = aynı durum; "na" = durumdan bağımsız genel bilgi.
- "critical": true eğer iddia bir güvenlik/davranış talimatıysa VE (supported "no" ise
  VEYA situation_match "no" ise).

Adım 3 - Soru belirli bir yer içeriyorsa ve cevap genel bilgiyi o yere özel bilgi gibi
sunuyorsa "location_misattribution": true.

Yalnızca şu JSON'u döndür:
{"question_situation": "<durum>",
 "claims": [{"claim": "<kısa özet>", "source_situation": "<durum>", "supported": "yes|partial|no",
             "situation_match": "yes|no|na", "critical": true|false}],
 "location_misattribution": true|false}
"""


def question_list() -> list[tuple[str, str]]:
    dev = [("dev", row["question"]) for row in json.loads(Path("data/evaluation/retrieval_questions.json").read_text(encoding="utf-8"))]
    holdout = [("holdout", row["question"]) for row in HOLDOUT_QUESTIONS]
    regional = [("regional", json.loads(line)["question"]) for line in Path("data/evaluation/regional_dev_v1.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["type"] != "ambiguous"]
    questions = dev + holdout + regional
    random.Random(0).shuffle(questions)  # fixed order so a pilot (--limit) is a stable mixed sample
    return questions


# Clearly unanswerable controls (personal guarantees, prediction, out of scope). rag_eval questions labelled
# NOT_ANSWERABLE only because the old retrieval missed them are excluded; gate_heldout_v2 is never used here.
NEGATIVE_RAG_EVAL_IDS = {"Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28", "Q29", "Q30"}
NEGATIVE_ROBUSTNESS_IDS = {"Q24-V1", "Q24-V2", "Q26-V1", "Q26-V2"}


def negative_questions() -> list[tuple[str, str]]:
    rag_eval = [json.loads(line) for line in Path("data/evaluation/rag_eval_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    robustness = [json.loads(line) for line in Path("data/evaluation/gate_robustness_v1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return ([("negative", row["question"]) for row in rag_eval if row["id"] in NEGATIVE_RAG_EVAL_IDS]
            + [("negative", row["question"]) for row in robustness if row["id"] in NEGATIVE_ROBUSTNESS_IDS])


class FaithfulnessJudge:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.cache: dict[str, dict[str, Any]] = {}
        if CACHE_FILE.exists():
            for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["verdict"]

    def judge(self, question: str, context: str, answer: str) -> dict[str, Any]:
        key = hashlib.sha256(f"{JUDGE_MODEL}|{JUDGE_PROMPT_VERSION}|{question}|{context}|{answer}".encode("utf-8")).hexdigest()
        if key not in self.cache:
            response = self.client.responses.create(
                model=JUDGE_MODEL,
                instructions=FAITHFULNESS_SYSTEM_PROMPT,
                input=f"Soru:\n{question}\n\nKaynaklar:\n{context}\n\nSistemin cevabı:\n{answer}",
                max_output_tokens=900,
                temperature=0,
            )
            raw = extract_response_text(response)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                verdict = json.loads(match.group()) if match else {"claims": [], "parse_error": raw}
            except json.JSONDecodeError:
                verdict = {"claims": [], "parse_error": raw}
            self.cache[key] = verdict
            with CACHE_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "verdict": verdict}, ensure_ascii=False) + "\n")
        return self.cache[key]


def answer_flags(verdict: dict[str, Any]) -> dict[str, Any]:
    claims = verdict.get("claims", [])
    return {
        "claims": len(claims),
        "unsupported": sum(c.get("supported") == "no" for c in claims),
        "partial": sum(c.get("supported") == "partial" for c in claims),
        "situation_mismatch": sum(c.get("situation_match") == "no" for c in claims),
        # The judge marks "critical"; it only counts when the claim is actually unsupported or
        # misapplied (the judge sometimes flags correct claims as critical).
        "critical": sum(bool(c.get("critical")) and (c.get("supported") == "no" or c.get("situation_match") == "no")
                        for c in claims),
        "location_misattribution": bool(verdict.get("location_misattribution")),
        "faithful": bool(claims) and all(c.get("supported") == "yes" and c.get("situation_match") != "no" for c in claims)
                    and not verdict.get("location_misattribution"),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Claim-level faithfulness evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions (pilot).")
    parser.add_argument("--generation-version", default="v1")
    parser.add_argument("--generation-model", default=None)
    parser.add_argument("--gate-votes", type=int, default=1)
    parser.add_argument("--with-negatives", action="store_true", help="Also run unanswerable control questions.")
    parser.add_argument("--tag", default=None, help="Suffix for output files.")
    parser.add_argument("--current-cli", action="store_true",
                        help="Also use the current CLI pipeline: source routing, evidence-first, local info.")
    parser.add_argument("--evidence-gate", action="store_true", help="Skip the gate; evidence selection refuses.")
    parser.add_argument("--index-dir", type=Path, default=CLI_INDEX_DIR, help="Index for pipeline v1 (e.g. data/indexes/mobile_v5).")
    parser.add_argument("--pipeline", choices=("v1", "v2"), default="v1",
                        help="v2: src/rag/pipeline_v2.py on the mobile_v5 index (other pipeline flags are ignored).")
    args = parser.parse_args()
    results_file, review_file, summary_file = (
        path.with_name(path.stem + f"_{args.tag}" + path.suffix) if args.tag else path
        for path in (RESULTS_FILE, REVIEW_FILE, SUMMARY_FILE)
    )

    load_environment()
    judge = FaithfulnessJudge(load_openai_client())
    questions = question_list()[: args.limit] if args.limit else question_list()
    if args.with_negatives:
        questions += negative_questions()

    results = []
    for index, (set_name, question) in enumerate(questions, start=1):
        if args.pipeline == "v2":
            result = generate_answer_v2(question)
        else:
            result = generate_answer(question, normalize_query=True, llm_repair=True, index_dir=args.index_dir,
                                     regional=True, lexical=True, llm_rerank=True, return_context=True,
                                     generation_version=args.generation_version, generation_model=args.generation_model,
                                     gate_votes=args.gate_votes, source_routing=args.current_cli,
                                     evidence_first=args.current_cli, local_info=args.current_cli,
                                     evidence_gate=args.evidence_gate)
        row: dict[str, Any] = {"set": set_name, "question": question, "answer": result["answer"],
                               "refused": result["answer"].strip() == REFUSAL_ANSWER}
        if not row["refused"] and set_name != "negative":
            verdict = judge.judge(question, result["context"], result["answer"])
            row.update(verdict=verdict, **answer_flags(verdict))
        results.append(row)
        status = ("NEGATIVE " + ("refused (ok)" if row["refused"] else "ANSWERED (false accept)")) if set_name == "negative" else             "refused" if row["refused"] else ("faithful" if row["faithful"] else
                 f"unsup={row['unsupported']} partial={row['partial']} mismatch={row['situation_mismatch']} critical={row['critical']}")
        print(f"[{index}/{len(questions)}] {status} | {question}", flush=True)

    with results_file.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    negatives = [r for r in results if r["set"] == "negative"]
    results_pos = [r for r in results if r["set"] != "negative"]
    answered = [r for r in results_pos if not r["refused"]]
    claims = sum(r["claims"] for r in answered)

    def pct(part: int, whole: int) -> str:
        return f"{part}/{whole} ({part / whole:.0%})" if whole else "0/0"

    lines = [
        f"DARG faithfulness [generation {args.generation_version}, model {args.generation_model or 'default'}, gate votes {args.gate_votes}"
        f"{', current CLI pipeline' if args.current_cli else ''}{', evidence gate' if args.evidence_gate else ''}"
        f"{', pipeline v2' if args.pipeline == 'v2' else ', index ' + args.index_dir.name}] "
        f"({JUDGE_PROMPT_VERSION}, judge {JUDGE_MODEL}, NOT human-calibrated)",
        "=" * 66,
        f"Answerable-set questions: {len(results_pos)} | answered: {pct(len(answered), len(results_pos))}",
        f"Unanswerable controls: {len(negatives)} | false accepts: {sum(not r['refused'] for r in negatives)}",
        f"Fully faithful answers: {pct(sum(r['faithful'] for r in answered), len(answered))}",
        f"Answers with >=1 critical issue: {pct(sum(r['critical'] > 0 for r in answered), len(answered))}",
        f"Answers with a situation mismatch: {pct(sum(r['situation_mismatch'] > 0 for r in answered), len(answered))}",
        f"Location misattribution: {pct(sum(r['location_misattribution'] for r in answered), len(answered))}",
        f"Claims: {claims} | unsupported {pct(sum(r['unsupported'] for r in answered), claims)}"
        f" | partial {pct(sum(r['partial'] for r in answered), claims)}"
        f" | situation mismatch {pct(sum(r['situation_mismatch'] for r in answered), claims)}"
        f" | critical {pct(sum(r['critical'] for r in answered), claims)}",
    ]
    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    review = []
    for number, row in enumerate(results, start=1):
        review.append(f"### {number}. [{row['set']}] {row['question']}  (soru durumu: {(row.get('verdict') or {}).get('question_situation', '-')})")
        review.append(row["answer"])
        for claim in (row.get("verdict") or {}).get("claims", []):
            flag = "  ⚠ KRİTİK" if claim.get("critical") else ""
            review.append(f"  - [{claim.get('supported')}/{claim.get('situation_match')}|kaynak: {claim.get('source_situation')}] {claim.get('claim')}{flag}")
        if row.get("location_misattribution"):
            review.append("  - ⚠ Yer bilgisi yanlış atfedilmiş")
        review.append("")
    review_file.write_text("\n".join(review), encoding="utf-8")
    print("\n".join(lines))
    print(f"Review file: {review_file}")


if __name__ == "__main__":
    main()
