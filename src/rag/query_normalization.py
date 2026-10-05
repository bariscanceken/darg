"""Conservative, deterministic typo normalization for DARG retrieval queries.

The normalized query is used only for retrieval. Unknown or ambiguous words are
left untouched, and a query without corrections is returned byte-for-byte.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


VOCAB_VERSION = "query_vocab_v1"
DEFAULT_VOCAB_FILE = Path("data/query_normalization/query_vocab_v1.json")

TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]+")
FOLD_TABLE = str.maketrans("çğıöşüâîû", "cgiosuaiu")

MIN_TOKEN_LENGTH = 5
MIN_STEM_LENGTH = 4
MIN_SUFFIX_LENGTH = 2
MIN_EDIT_CANDIDATE_FREQUENCY = 5
# Short words have denser d=1 neighbourhoods (78% of 5-letter vocab words), so demand stronger evidence.
SHORT_WORD_MAX_LENGTH = 6
MIN_SHORT_WORD_CANDIDATE_FREQUENCY = 20
DOMINANCE_RATIO = 10
IN_VOCAB_FOLD_RATIO = 100
MAX_CORRECTIONS = 2
MAX_OOV_RATIO = 0.5

METHOD_DIACRITICS = "diacritics"
METHOD_EDIT_DISTANCE = "edit_distance"


def _question_particles() -> frozenset[str]:
    """Generate mı/mi/mu/mü question particle forms for every vowel."""
    forms: set[str] = set()
    for v in "ıiuü":
        for suffix in ("", "y{v}m", "s{v}n", "y{v}z", "s{v}n{v}z", "d{v}r", "yd{v}", "yd{v}m", "yd{v}n",
                       "yd{v}k", "yd{v}n{v}z", "ym{v}ş"):
            forms.add(f"m{v}" + suffix.format(v=v))
    return frozenset(forms)


QUESTION_WORDS = frozenset({
    "acaba", "hangi", "hangisi", "hangileri", "nasıl", "neden", "niçin", "nerede", "nereye", "nereden",
    "nerelerde", "kimse", "kimler",
})
PROTECTED_WORDS = _question_particles() | QUESTION_WORDS

MIN_SUFFIX_GUARD_STEM_LENGTH = 3
# Closed-class Turkish inflectional suffixes (all vowel-harmony variants, 2+ letters).
INFLECTIONAL_SUFFIXES = frozenset({
    "lar", "ler",
    "da", "de", "ta", "te", "dan", "den", "tan", "ten", "ya", "ye", "na", "ne", "nda", "nde", "ndan", "nden",
    "la", "le", "yla", "yle",
    "ım", "im", "um", "üm", "yım", "yim", "yum", "yüm", "sın", "sin", "sun", "sün",
    "ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz", "nız", "niz", "nuz", "nüz",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün", "sı", "si", "su", "sü",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "ar", "er", "ır", "ir", "ur", "ür",
    "abil", "ebil", "yabil", "yebil",
    "malı", "meli", "mak", "mek", "ma", "me",
    "yor", "dı", "di", "du", "dü", "tı", "ti", "tu", "tü", "mış", "miş", "muş", "müş",
    "acak", "ecek", "yacak", "yecek", "sa", "se", "ysa", "yse", "ken", "yken",
})


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless i rules, preserving string length."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def turkish_capitalize(word: str) -> str:
    first = {"i": "İ", "ı": "I"}.get(word[0], word[0].upper())
    return first + word[1:]


def fold_diacritics(word: str) -> str:
    return word.translate(FOLD_TABLE)


def diacritics_consistent(token: str, candidate: str) -> bool:
    """True if candidate only adds Turkish characters the user typed as ASCII."""
    for typed, expected in zip(token, candidate):
        if typed == expected:
            continue
        if typed.isascii() and fold_diacritics(expected) == typed:
            continue
        return False
    return True


def osa_distance(a: str, b: str, max_distance: int) -> int:
    """Optimal string alignment distance; returns max_distance + 1 once exceeded."""
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    before_previous: list[int] = []
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                value = min(value, before_previous[j - 2] + 1)
            current[j] = value
        if min(current) > max_distance:
            return max_distance + 1
        before_previous, previous = previous, current
    return previous[len(b)]


class QueryVocab:
    """Corpus word frequencies with lookup structures for both correction stages."""

    def __init__(self, words: dict[str, int], source_sha256: str = "") -> None:
        self.words = words
        self.source_sha256 = source_sha256
        self.folded: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.edit_buckets: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
        for word, frequency in words.items():
            self.folded[fold_diacritics(word)].append((word, frequency))
            if frequency >= MIN_EDIT_CANDIDATE_FREQUENCY:
                self.edit_buckets[(word[0], len(word))].append((word, frequency))


@lru_cache(maxsize=2)
def load_query_vocab(path: str = str(DEFAULT_VOCAB_FILE)) -> QueryVocab:
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Query vocabulary bulunamadı: {vocab_path}. Önce `python -m src.rag.build_query_vocab` çalıştırın."
        )
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if payload.get("version") != VOCAB_VERSION:
        raise ValueError(f"Beklenmeyen vocabulary sürümü: {payload.get('version')}")
    return QueryVocab(payload["words"], payload.get("source_sha256", ""))


def pick_dominant(candidates: list[tuple[str, int]]) -> str | None:
    """Return the most frequent candidate only if it is unique or clearly dominant."""
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    if len(ranked) == 1 or ranked[0][1] >= DOMINANCE_RATIO * ranked[1][1]:
        return ranked[0][0]
    return None


def correct_by_diacritics(word: str, vocab: QueryVocab) -> str | None:
    candidates = [
        (candidate, frequency)
        for candidate, frequency in vocab.folded.get(fold_diacritics(word), [])
        if candidate != word and diacritics_consistent(word, candidate)
    ]
    best = pick_dominant(candidates)
    if best is None:
        return None
    if word in vocab.words:
        # Known ASCII spellings (often OCR noise such as "yangin") need overwhelming evidence.
        if not word.isascii() or vocab.words[best] < IN_VOCAB_FOLD_RATIO * vocab.words[word]:
            return None
    return best


def has_known_stem(word: str, vocab: QueryVocab) -> bool:
    """True if the word looks like a known stem plus a suffix (agglutinated valid form)."""
    for stem_length in range(MIN_STEM_LENGTH, len(word) - MIN_SUFFIX_LENGTH + 1):
        if word[:stem_length] in vocab.words:
            return True
    return False


@lru_cache(maxsize=4096)
def is_suffix_chain(tail: str) -> bool:
    """True if tail is a concatenation of known inflectional suffixes."""
    if not tail:
        return False
    return any(
        tail.startswith(suffix) and (len(tail) == len(suffix) or is_suffix_chain(tail[len(suffix):]))
        for suffix in INFLECTIONAL_SUFFIXES
    )


def changes_only_inflection(word: str, candidate: str) -> bool:
    """True if word = shared stem + suffix chain, i.e. the candidate keeps the stem and alters only endings."""
    for stem_length in range(MIN_SUFFIX_GUARD_STEM_LENGTH, len(word)):
        if candidate.startswith(word[:stem_length]) and is_suffix_chain(word[stem_length:]):
            return True
    return False


def correct_by_edit_distance(word: str, vocab: QueryVocab) -> str | None:
    is_short = len(word) <= SHORT_WORD_MAX_LENGTH
    max_distance = 1 if is_short else 2
    min_frequency = MIN_SHORT_WORD_CANDIDATE_FREQUENCY if is_short else MIN_EDIT_CANDIDATE_FREQUENCY
    best_distance = max_distance + 1
    candidates: list[tuple[str, int]] = []
    for length in range(len(word) - max_distance, len(word) + max_distance + 1):
        for candidate, frequency in vocab.edit_buckets.get((word[0], length), []):
            if frequency < min_frequency:
                continue
            distance = osa_distance(word, candidate, max_distance)
            if distance < best_distance:
                best_distance = distance
                candidates = [(candidate, frequency)]
            elif distance == best_distance and distance <= max_distance:
                candidates.append((candidate, frequency))
    best = pick_dominant(candidates)
    if best is None or changes_only_inflection(word, best):
        return None
    return best


def is_eligible(token: str, start: int, end: int, query: str) -> bool:
    if len(token) < MIN_TOKEN_LENGTH:
        return False
    if any(char.isupper() for char in token[1:]):
        return False  # acronyms (AFAD) and brand names (WhatsApp)
    if (start > 0 and query[start - 1].isdigit()) or (end < len(query) and query[end].isdigit()):
        return False
    return True


def correct_token(token: str, start: int, end: int, query: str, vocab: QueryVocab) -> tuple[str, str] | None:
    """Return (corrected lowercase word, method) or None to leave the token untouched."""
    if not is_eligible(token, start, end, query):
        return None
    word = turkish_lower(token)
    if word in PROTECTED_WORDS:
        return None

    corrected = correct_by_diacritics(word, vocab)
    if corrected is not None:
        return corrected, METHOD_DIACRITICS

    if word in vocab.words or has_known_stem(word, vocab):
        return None

    corrected = correct_by_edit_distance(word, vocab)
    if corrected is not None:
        return corrected, METHOD_EDIT_DISTANCE
    return None


def unchanged(query: str, reason: str | None = None) -> dict[str, Any]:
    return {"original": query, "normalized": query, "corrections": [], "skipped_reason": reason}


def normalize_query(query: str, vocab: QueryVocab | None = None) -> dict[str, Any]:
    """Correct only obvious misspellings; return the original string if unsure."""
    vocab = vocab or load_query_vocab()
    matches = list(TOKEN_RE.finditer(query))
    if not matches:
        return unchanged(query)

    unknown = [
        match for match in matches
        if turkish_lower(match.group()) not in vocab.words and turkish_lower(match.group()) not in PROTECTED_WORDS
    ]
    if len(unknown) / len(matches) > MAX_OOV_RATIO:
        return unchanged(query, "too_many_unknown_words")

    corrections: list[dict[str, Any]] = []
    for match in matches:
        result = correct_token(match.group(), match.start(), match.end(), query, vocab)
        if result is None:
            continue
        corrected, method = result
        if match.group()[0].isupper():
            corrected = turkish_capitalize(corrected)
        corrections.append({
            "original": match.group(),
            "corrected": corrected,
            "method": method,
            "start": match.start(),
            "end": match.end(),
        })

    if not corrections:
        return unchanged(query)
    if len(corrections) > MAX_CORRECTIONS:
        return unchanged(query, "too_many_corrections")

    parts: list[str] = []
    cursor = 0
    for correction in corrections:
        parts.append(query[cursor:correction["start"]])
        parts.append(correction["corrected"])
        cursor = correction["end"]
    parts.append(query[cursor:])
    return {
        "original": query,
        "normalized": "".join(parts),
        "corrections": corrections,
        "skipped_reason": None,
    }
