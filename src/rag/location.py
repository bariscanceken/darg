"""Detect Turkish province/district names in a question.

Uses the offline gazetteer in data/geo (81 provinces, 973 districts). Matching is
deliberately conservative:
- "Ceyhan'da", "Hatay'daki" (apostrophe) and exact tokens always match;
- glued suffixes ("Adanada") match only for names of 5+ letters followed by a
  known Turkish suffix chain, so "vanası" never becomes the province Van; a garbled
  chain ("antakyadaydayım") still matches if it starts with a locative/dative case
  ending (-da/-de/-ta/-te/-ya/-ye), the usual way people say where they are;
- a misspelt long name ("anatakyaEDayım") matches when one edit away from a 6+ letter
  name, followed by a short ending, and the word is not a known Turkish word;
- district names that are also common words (Araç, Bodrum, Maden, Kale, Pazar…)
  match only when capitalised, and "Merkez" never matches on its own;
- a district name shared by several provinces is resolved only if the question
  also names one of those provinces; otherwise the result is marked ambiguous.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .query_normalization import (
    fold_diacritics,
    has_known_stem,
    is_suffix_chain,
    load_query_vocab,
    osa_distance,
    turkish_lower,
)


GAZETTEER_FILE = Path("data/geo/turkiye_il_ilce.json")
IRAP_MAP_FILE = Path("data/geo/irap_province_map.json")
MIN_GLUED_NAME_LENGTH = 5
NEVER_ALONE = {"merkez"}
FUZZY_MIN_NAME_LENGTH = 6
FUZZY_MAX_TAIL = 7
CASE_START_RE = re.compile(r"^(?:[dt][ae]|y[ae])")  # Antakya-da..., Hatay-da..., Adana-ya...
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]+(?:['’][A-Za-zÇĞİÖŞÜçğıöşü]+)?")


@dataclass(frozen=True)
class Location:
    province: str | None = None
    district: str | None = None
    ambiguous_provinces: tuple[str, ...] = field(default_factory=tuple)
    matched_text: str = ""

    @property
    def is_ambiguous(self) -> bool:
        return self.province is None and bool(self.ambiguous_provinces)

    def label(self) -> str:
        if self.district and self.province:
            return f"{self.district} ({self.province})"
        return self.province or self.district or ""


@dataclass(frozen=True)
class Gazetteer:
    names: dict[str, tuple[str, str | None, tuple[str, ...]]]  # lowered name -> (kind, canonical, provinces)
    common_words: frozenset[str]
    irap_provinces: frozenset[str]


@lru_cache(maxsize=1)
def load_gazetteer(path: str = str(GAZETTEER_FILE), irap_path: str = str(IRAP_MAP_FILE)) -> Gazetteer:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names: dict[str, tuple[str, str | None, tuple[str, ...]]] = {}
    district_owners: dict[str, list[str]] = {}
    for province in data["provinces"]:
        for district in province["districts"]:
            district_owners.setdefault(district["name"], []).append(province["name"])
    for district, owners in district_owners.items():
        names[turkish_lower(district)] = ("district", district, tuple(sorted(owners)))
    for province in data["provinces"]:  # provinces win over same-named districts
        names[turkish_lower(province["name"])] = ("province", province["name"], (province["name"],))
    for alias, province in data["aliases"].items():
        names[turkish_lower(alias)] = ("province", province, (province,))
    irap = json.loads(Path(irap_path).read_text(encoding="utf-8"))
    return Gazetteer(
        names=names,
        common_words=frozenset(turkish_lower(name) for name in data["common_word_names"]),
        irap_provinces=frozenset(irap.values()),
    )


def match_token(token: str, gazetteer: Gazetteer) -> tuple[str, tuple[str, str | None, tuple[str, ...]]] | None:
    parts = re.split(r"['’]", token, maxsplit=1)
    stem, has_apostrophe = parts[0], len(parts) > 1
    lowered = turkish_lower(stem)
    if lowered in NEVER_ALONE:
        return None
    entry = gazetteer.names.get(lowered)
    if entry is None and not has_apostrophe and len(lowered) > MIN_GLUED_NAME_LENGTH:
        for cut in range(len(lowered) - 1, MIN_GLUED_NAME_LENGTH - 1, -1):
            candidate = gazetteer.names.get(lowered[:cut])
            if candidate is not None and is_suffix_chain(lowered[cut:]):
                lowered, entry = lowered[:cut], candidate
                break
        else:  # a typo in the suffix chain: accept the longest name followed by a case ending
            for cut in range(len(lowered) - 1, MIN_GLUED_NAME_LENGTH - 1, -1):
                candidate = gazetteer.names.get(lowered[:cut])
                if candidate is not None and CASE_START_RE.match(lowered[cut:]):
                    lowered, entry = lowered[:cut], candidate
                    break
    if entry is None and not has_apostrophe:
        fuzzy = fuzzy_name(lowered, gazetteer)
        if fuzzy is not None:
            lowered, entry = fuzzy, gazetteer.names[fuzzy]
    if entry is None:
        return None
    if lowered in gazetteer.common_words and not stem[:1].isupper():
        return None
    return lowered, entry


_FUZZY_CACHE: dict[int, tuple[tuple[str, str], ...]] = {}


def fuzzy_candidates(gazetteer: Gazetteer) -> tuple[tuple[str, str], ...]:
    """(name, diacritics-folded name) for long, non-common-word place names; built once per gazetteer."""
    if id(gazetteer) not in _FUZZY_CACHE:
        _FUZZY_CACHE[id(gazetteer)] = tuple(
            (name, fold_diacritics(name)) for name in gazetteer.names
            if len(name) >= FUZZY_MIN_NAME_LENGTH and name not in gazetteer.common_words and " " not in name)
    return _FUZZY_CACHE[id(gazetteer)]


def fuzzy_name(word: str, gazetteer: Gazetteer) -> str | None:
    """A misspelt place name: the one name one edit away from a prefix of an unknown word."""
    if len(word) < FUZZY_MIN_NAME_LENGTH:
        return None
    folded_word = fold_diacritics(word)
    for name, folded_name in fuzzy_candidates(gazetteer):  # typed without Turkish letters: "samandagda"
        tail = folded_word[len(folded_name):]
        if folded_word.startswith(folded_name) and (not tail or CASE_START_RE.match(tail) or is_suffix_chain(tail)):
            return name
    vocab = load_query_vocab()
    if word in vocab.words or has_known_stem(word, vocab):
        return None
    folded = fold_diacritics(word)
    found: set[str] = set()
    for cut in range(len(word), FUZZY_MIN_NAME_LENGTH - 2, -1):
        if len(word) - cut > FUZZY_MAX_TAIL:
            break
        prefix = folded[:cut]
        for name, folded_name in fuzzy_candidates(gazetteer):
            if abs(len(name) - cut) <= 1 and osa_distance(prefix, folded_name, 1) <= 1:
                found.add(name)
        if found:
            break
    return found.pop() if len(found) == 1 else None


def detect_location(question: str, gazetteer: Gazetteer | None = None) -> Location | None:
    """Return the location named in the question, or None if there is none."""
    gazetteer = gazetteer or load_gazetteer()
    provinces: list[str] = []
    districts: list[tuple[str, tuple[str, ...], str]] = []
    for match in WORD_RE.finditer(question):
        found = match_token(match.group(), gazetteer)
        if found is None:
            continue
        _, (kind, canonical, owners) = found
        if kind == "province":
            if canonical not in provinces:
                provinces.append(canonical)
        else:
            districts.append((canonical, owners, match.group()))

    for district, owners, text in districts:
        if len(owners) == 1:
            return Location(province=owners[0], district=district, matched_text=text)
        shared = [province for province in provinces if province in owners]
        if shared:
            return Location(province=shared[0], district=district, matched_text=text)
        return Location(district=district, ambiguous_provinces=owners, matched_text=text)
    if provinces:
        return Location(province=provinces[0], matched_text=provinces[0])
    return None


def place_names(province: str, gazetteer: Gazetteer | None = None) -> frozenset[str]:
    """Lowercased names that point to this province: its name, aliases and its districts (not common words)."""
    gazetteer = gazetteer or load_gazetteer()
    return frozenset(name for name, (_, _, owners) in gazetteer.names.items()
                     if province in owners and name not in gazetteer.common_words)


def has_regional_plan(province: str | None, gazetteer: Gazetteer | None = None) -> bool:
    gazetteer = gazetteer or load_gazetteer()
    return province is not None and province in gazetteer.irap_provinces
