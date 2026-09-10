"""Text cleaning helpers for extracted PDF pages."""

from __future__ import annotations

import re


LOW_TEXT_MIN_CHARS = 30
LOW_TEXT_MIN_WORDS = 5


def clean_text(text: str) -> str:
    """Apply light normalization without changing the meaning of the page text."""
    if not text:
        return ""

    normalized = text.replace("\x00", "").replace("\ufeff", "").replace("\u00ad", "")

    # Join words split across line endings, e.g. "dep-\nrem" -> "deprem".
    normalized = re.sub(
        r"(?<=\w)[\-\u2010\u2011\u2012\u2013\u2014\u2212]\s*(?:\r\n|\r|\n)+\s*(?=\w)",
        "",
        normalized,
    )

    normalized = re.sub(r"(?:\r\n|\r|\n)+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def is_low_text(text: str) -> bool:
    """Mark pages that are empty or contain too little usable extracted text."""
    if not text:
        return True

    return len(text) < LOW_TEXT_MIN_CHARS or len(text.split()) < LOW_TEXT_MIN_WORDS
