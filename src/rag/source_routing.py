"""Source-type aware routing: which kinds of documents may answer which kinds of questions.

Only ~3.6% of the corpus is citizen guidance ("what should I do"), while ~66% is
provincial plans (IRAP), ~13% technical material, ~9% glossaries and ~6% reports on
past events. Without routing, behaviour questions were answered from plans, past
recovery reports or glossaries. Each corpus file has a type (data/geo/source_types.json);
an LLM classifies the question type; the routing policy then restricts and focuses the
candidate pools. Classification failures fall back to "genel" (no restriction).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


SOURCE_TYPES_FILE = Path("data/geo/source_types.json")
QUESTION_TYPES = ("davranis", "bolgesel", "tanim", "gecmis", "genel")
ALL_SOURCE_TYPES = frozenset({"vatandas_rehberi", "il_plani", "ulusal_plan", "gecmis_rapor", "sozluk", "teknik_mesleki"})


@dataclass(frozen=True)
class RoutingPolicy:
    allowed: frozenset[str]     # source types that may appear in the Top-K at all
    preferred: frozenset[str]   # source types searched as an extra focused pool


POLICIES = {
    # "What should I do": citizen guidance only (own-province IRAP is still added by the regional pool).
    "davranis": RoutingPolicy(frozenset({"vatandas_rehberi", "il_plani"}), frozenset({"vatandas_rehberi"})),
    # Facts about a province: its plan first, plus citizen guides (e.g. how to look up one's assembly
    # area on e-Devlet, which applies to every province); glossaries never answer these.
    "bolgesel": RoutingPolicy(ALL_SOURCE_TYPES - {"sozluk"}, frozenset({"il_plani", "vatandas_rehberi"})),
    # "What is X": citizen guidance and glossaries; no province-specific plans or past reports.
    "tanim": RoutingPolicy(frozenset({"vatandas_rehberi", "sozluk", "ulusal_plan", "teknik_mesleki"}),
                           frozenset({"vatandas_rehberi", "sozluk"})),
    # Past events: reports and plans (which contain disaster histories).
    "gecmis": RoutingPolicy(ALL_SOURCE_TYPES - {"sozluk"}, frozenset({"gecmis_rapor", "il_plani"})),
    "genel": RoutingPolicy(ALL_SOURCE_TYPES, frozenset()),
}

CLASSIFY_SYSTEM_PROMPT = """Bir deprem bilgi asistanına gelen soruyu tek bir tipe ayır:
- "davranis": Kişinin ne yapması gerektiğini soruyor (deprem öncesi/sırası/sonrası, enkaz altında,
  dışarıda, araçta; "ne yapmalıyım", "...malı mıyım", "nasıl korunurum", "kullanılır mı").
- "bolgesel": Belirli bir il/ilçe hakkında bilgi soruyor (risk, fay, plan, barınma/toplanma alanları),
  davranış sormuyor.
- "tanim": Bir kavramın ne olduğunu soruyor ("nedir", "ne demek").
- "gecmis": Geçmişte olmuş depremleri veya o depremlerden sonra yapılanları soruyor.
- "genel": Diğer her şey veya emin değilsen.
Soru bir yer adı içerse bile kişinin şimdi ne yapacağını soruyorsa ("yaralı var ne yapalım", "nereye
gidelim", "hangi hastaneye gidelim", "evime girebilir miyim") tip "davranis"tır.
Yalnızca şu JSON'u döndür: {"type": "<tip>"}
"""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@lru_cache(maxsize=1)
def source_type_table() -> tuple[dict[str, str], dict[str, str]]:
    data = json.loads(SOURCE_TYPES_FILE.read_text(encoding="utf-8"))
    return data["folder_rules"], {nfc(name): kind for name, kind in data["files"].items()}


def source_type(record: dict[str, Any]) -> str | None:
    folder_rules, files = source_type_table()
    relative_path = str(record.get("relative_path", ""))
    for folder, kind in folder_rules.items():
        if relative_path.startswith(folder):
            return kind
    return files.get(nfc(str(record.get("source_file", ""))))


def classify_question(question: str, client: Any, model: str) -> str:
    """LLM question type; any failure or unexpected output becomes "genel" (no routing)."""
    from .answerability import extract_response_text

    try:
        response = client.responses.create(model=model, instructions=CLASSIFY_SYSTEM_PROMPT, input=question,
                                           max_output_tokens=20, temperature=0)
        match = re.search(r"\{.*\}", extract_response_text(response), re.DOTALL)
        kind = json.loads(match.group()).get("type") if match else None
    except Exception:
        kind = None
    return kind if kind in QUESTION_TYPES else "genel"


def policy_for(question_type: str) -> RoutingPolicy:
    return POLICIES.get(question_type, POLICIES["genel"])
