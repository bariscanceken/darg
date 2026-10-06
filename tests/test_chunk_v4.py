"""Tests for E5-token-sized v4 chunking (uses the cached E5 tokenizer, no network)."""

from __future__ import annotations

import unittest

from src.preprocessing.chunk_earthquake_v4 import (
    MAX_TOKENS,
    MIN_EMIT_TOKENS,
    Unit,
    chunk_run,
    count_tokens,
    page_units,
    split_to_fit,
)


def units_of(token_sizes: list[int]) -> list[Unit]:
    return [Unit(f"s{index}", 1, "relevant", size) for index, size in enumerate(token_sizes)]


class PackingTests(unittest.TestCase):
    def test_no_chunk_exceeds_max(self) -> None:
        for chunk in chunk_run(units_of([90] * 30)):
            self.assertLessEqual(sum(unit.tokens for unit in chunk), MAX_TOKENS)

    def test_every_unit_is_covered(self) -> None:
        units = units_of([60, 120, 30, 200, 80, 40, 150, 20])
        covered = {unit.text for chunk in chunk_run(units) for unit in chunk}
        self.assertEqual(covered, {unit.text for unit in units})

    def test_consecutive_chunks_overlap(self) -> None:
        chunks = chunk_run(units_of([40] * 20))
        self.assertGreater(len(chunks), 1)
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left[-1].text, right[0].text)

    def test_short_tail_folded_into_previous(self) -> None:
        chunks = chunk_run(units_of([140, 140, 30]))
        self.assertTrue(all(sum(u.tokens for u in chunk) >= MIN_EMIT_TOKENS for chunk in chunks))


class TokenizationTests(unittest.TestCase):
    def test_long_sentence_split_to_fit(self) -> None:
        sentence = " ".join(["depremden sonra toplanma alanına gidilmelidir"] * 120)
        pieces = split_to_fit(sentence)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(count_tokens(piece) <= MAX_TOKENS for piece in pieces))
        self.assertEqual(" ".join(pieces).split(), sentence.split())

    def test_page_units_split_on_sentences(self) -> None:
        units = page_units("Deprem anında hemen dışarı çıkmak çok tehlikelidir. Asansöre binmeyin.", 25, "relevant")
        self.assertEqual([unit.text for unit in units],
                         ["Deprem anında hemen dışarı çıkmak çok tehlikelidir.", "Asansöre binmeyin."])
        self.assertEqual(units[0].page, 25)


if __name__ == "__main__":
    unittest.main()
