"""Which situation a passage's advice is for, read from its section headers (rule-based, no LLM).

v5 chunks start with "document › section › subsection". The headers of AFAD guides say when advice
applies ("Deprem Sırasında Yapılması Gerekenler", "Afet Sonrası Yaşamın Sürdürülmesi", "Enkaz
Altında..."), and the "İlk 72 Saat" guide uses a timeline: "0. saniye" = preparation, "3. saniye" =
during the shaking, "3. dakika / 30. dakika / 3. saat / 3. gün" = after. The deepest header that names
a situation wins. The answer step sees the label, so advice for one situation is not presented as
advice for another (e.g. "Çök-Kapan-Tutun" for someone driving).
"""

from __future__ import annotations

import re

from .query_normalization import turkish_lower

DURING, AFTER, RUBBLE, PREPARE = "deprem sırasında", "deprem sonrasında", "enkaz altında", "hazırlık"

RULES = (  # checked in this order within one header
    (RUBBLE, re.compile(r"enkaz alt")),
    (PREPARE, re.compile(r"\b0\. ?saniye")),
    (DURING, re.compile(r"\b3\. ?saniye")),
    (AFTER, re.compile(r"\b(3|30)\. ?dakika|\b3\. ?saat|\b3\. ?gün")),
    (DURING, re.compile(r"sırasında|sirasinda|anında|aninda|deprem anı|sarsıntı sırası|çök-kapan-tutun")),
    (AFTER, re.compile(r"sonrası|sonrasi|sonrasında|sonrasinda|sonra\b|tahliye|hayatın idamesi|barınma|iyileştirme")),
    (PREPARE, re.compile(r"öncesi|oncesi|öncesinde|hazırlık|hazirlik|aile afet planı|sabitle|yapısal olmayan risk|"
                         r"önlem")),
)


def header_situation(header: str) -> str | None:
    """Situation of a 'document › section › subsection' header line, deepest named section first.

    The document title is skipped ("İlk 72 Saat Hazırlık Rehberi" is not about preparation only).
    """
    sections = [turkish_lower(part) for part in header.split("›")][1:]  # the document title never decides
    for part in reversed(sections):
        for situation, pattern in RULES:
            if pattern.search(part):
                return situation
    return None


def passage_situation(text: str) -> str | None:
    """A v5 chunk's first line is its header; card/web passages have none."""
    first_line = text.split("\n", 1)[0]
    return header_situation(first_line) if "›" in first_line else None
