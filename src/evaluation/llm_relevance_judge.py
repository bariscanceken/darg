"""LLM relevance judge for retrieval evaluation, calibrated against human labels.

The judge answers one question per (question, passage) pair: does the passage
contain information that directly helps answer the question? Results are cached
on disk keyed by model, prompt version, question and passage, so reruns are free
and reproducible.

  python -m src.evaluation.llm_relevance_judge --calibrate
compares the judge with the existing human labels (agreement, Cohen's kappa).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from src.rag.answerability import extract_response_text


JUDGE_MODEL = "gpt-4o"
JUDGE_PROMPT_VERSION = "judge_v1"
CACHE_FILE = Path("data/evaluation/llm_judge_cache.jsonl")
CALIBRATION_SUMMARY_FILE = Path("data/evaluation/llm_judge_calibration.txt")
HUMAN_JUDGMENT_FILES = [
    Path("data/evaluation/retrieval_v2_top50_reranked_judgments.csv"),
    Path("data/evaluation/retrieval_hybrid_judgments.csv"),
    Path("data/evaluation/retrieval_score_fusion_judgments.csv"),
    Path("data/evaluation/holdout_retrieval_judgments.csv"),
]
V2_METADATA_FILE = Path("data/indexes/mobile_v2/metadata_v2.jsonl")

JUDGE_SYSTEM_PROMPT = """Sen bir bilgi erişimi (retrieval) değerlendirme hakemisin.
Sana bir kullanıcı sorusu ve AFAD belgelerinden alınmış bir metin parçası verilecek.

Karar ver: Bu metin parçası, soruyu cevaplamak için DOĞRUDAN kullanılabilecek bilgi içeriyor mu?

1 (alakalı): Metin, sorunun istediği bilgiyi, tavsiyeyi veya kararı açıkça içeriyor;
  soruyu tamamen ya da önemli ölçüde cevaplamaya yetecek bir kısım var.
0 (alakasız): Metin yalnızca aynı genel konudan bahsediyor, farklı bir durumu anlatıyor
  (ör. soru deprem SIRASINDA iken metin deprem SONRASINI anlatıyor) veya soruyu
  cevaplamaya yetecek bilgi içermiyor.

Metinde bozuk Türkçe karakterler olabilir (ör. "dÈëarÈ" = "dışarı"); bunları doğru oku.
Yalnızca şu JSON'u döndür: {"relevant": 0 veya 1, "reason": "<en fazla 15 kelime>"}
"""


def cache_key(question: str, passage: str, model: str) -> str:
    payload = f"{model}|{JUDGE_PROMPT_VERSION}|{question}|{passage}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RelevanceJudge:
    def __init__(self, client: Any, model: str = JUDGE_MODEL, cache_file: Path = CACHE_FILE) -> None:
        self.client = client
        self.model = model
        self.cache_file = cache_file
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row

    def _call_with_retry(self, question: str, passage: str, attempts: int = 3) -> Any:
        """Retry transient connection/rate-limit errors so long evaluations are not lost."""
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        for attempt in range(1, attempts + 1):
            try:
                return self.client.responses.create(
                    model=self.model,
                    instructions=JUDGE_SYSTEM_PROMPT,
                    input=f"Soru:\n{question}\n\nMetin parçası:\n{passage}",
                    max_output_tokens=80,
                    temperature=0,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt == attempts:
                    raise
                time.sleep(5 * attempt)

    def judge(self, question: str, passage: str) -> int:
        key = cache_key(question, passage, self.model)
        if key not in self.cache:
            raw = extract_response_text(self._call_with_retry(question, passage))
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                payload = json.loads(match.group()) if match else {}
            except json.JSONDecodeError:
                payload = {}
            relevant = 1 if payload.get("relevant") in (1, "1", True) else 0
            row = {"key": key, "model": self.model, "prompt_version": JUDGE_PROMPT_VERSION,
                   "question": question, "relevant": relevant, "reason": payload.get("reason", ""), "raw": raw}
            self.cache[key] = row
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return int(self.cache[key]["relevant"])


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_a1 = sum(a for a, _ in pairs) / n
    p_b1 = sum(b for _, b in pairs) / n
    expected = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def human_labeled_pairs() -> dict[tuple[str, str], int]:
    pairs: dict[tuple[str, str], int] = {}
    for path in HUMAN_JUDGMENT_FILES:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["chunk_id"].startswith("afad_earthquake_v2_") and row.get("relevant", "").strip() in {"0", "1"}:
                    pairs[(row["question"], row["chunk_id"])] = int(row["relevant"])
    return pairs


def calibrate(judge: RelevanceJudge) -> str:
    texts = {}
    with V2_METADATA_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts[record["chunk_id"]] = record["text"]
    human = human_labeled_pairs()
    results = []
    for index, ((question, chunk_id), label) in enumerate(sorted(human.items()), start=1):
        results.append((label, judge.judge(question, texts[chunk_id]), question, chunk_id))
        if index % 50 == 0:
            print(f"  judged {index}/{len(human)}", flush=True)
    pairs = [(h, j) for h, j, _, _ in results]
    tp = sum(h == 1 and j == 1 for h, j in pairs)
    fp = sum(h == 0 and j == 1 for h, j in pairs)
    fn = sum(h == 1 and j == 0 for h, j in pairs)
    tn = sum(h == 0 and j == 0 for h, j in pairs)
    lines = [
        f"LLM judge calibration ({judge.model}, {JUDGE_PROMPT_VERSION}) vs human labels",
        "=" * 60,
        f"Pairs: {len(pairs)} (human relevant: {tp + fn}, not relevant: {fp + tn})",
        f"Agreement: {(tp + tn) / len(pairs):.1%}",
        f"Cohen's kappa: {cohen_kappa(pairs):.3f}",
        f"Judge precision (vs human): {tp / (tp + fp):.1%}" if tp + fp else "Judge precision: n/a",
        f"Judge recall (vs human): {tp / (tp + fn):.1%}" if tp + fn else "Judge recall: n/a",
        "",
        "Confusion (rows=human, cols=judge):",
        f"            judge=1  judge=0",
        f"  human=1   {tp:7}  {fn:7}",
        f"  human=0   {fp:7}  {tn:7}",
        "",
        "Disagreements:",
    ]
    for human_label, judge_label, question, chunk_id in results:
        if human_label != judge_label:
            lines.append(f"  human={human_label} judge={judge_label} | {question} | {chunk_id[-12:]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="LLM relevance judge utilities.")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--model", default=JUDGE_MODEL)
    args = parser.parse_args()

    from src.rag.generate_answer import load_environment, load_openai_client

    load_environment()
    judge = RelevanceJudge(load_openai_client(), args.model)
    if args.calibrate:
        summary = calibrate(judge)
        CALIBRATION_SUMMARY_FILE.write_text(summary, encoding="utf-8")
        print(summary.split("Disagreements:")[0])
        print(f"Full report: {CALIBRATION_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
