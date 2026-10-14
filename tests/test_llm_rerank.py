"""Offline tests for the LLM rerank stage (fake client, no network)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.rag.llm_rerank import llm_rerank, parse_ranking


def candidates(count: int) -> list[dict]:
    return [{"chunk_id": f"c{i}", "text": f"metin {i}", "source_file": "x.pdf"} for i in range(count)]


class FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.responses = SimpleNamespace(create=self._create)
        self._text, self._error = text, error

    def _create(self, **_: object) -> SimpleNamespace:
        if self._error:
            raise self._error
        return SimpleNamespace(output_text=self._text)


class ParseRankingTests(unittest.TestCase):
    def test_valid_ranking(self) -> None:
        self.assertEqual(parse_ranking('{"ranking": [3, 1, 2]}', 5), [2, 0, 1])

    def test_out_of_range_and_duplicates_are_dropped(self) -> None:
        self.assertEqual(parse_ranking('{"ranking": [2, 2, 9, 0, "1", 1]}', 3), [1, 0])

    def test_malformed_output(self) -> None:
        self.assertIsNone(parse_ranking("tamam", 3))
        self.assertIsNone(parse_ranking('{"ranking": "1,2"}', 3))


class RerankTests(unittest.TestCase):
    def test_llm_order_is_applied(self) -> None:
        ranked = candidates(6)
        top = llm_rerank("soru", ranked, FakeClient('{"ranking": [4, 1]}'), "m", top_k=3)
        self.assertEqual([c["chunk_id"] for c in top], ["c3", "c0", "c1"])  # remaining slot filled in original order

    def test_fallback_on_malformed_output(self) -> None:
        ranked = candidates(6)
        top = llm_rerank("soru", ranked, FakeClient("anlamadım"), "m", top_k=3)
        self.assertEqual([c["chunk_id"] for c in top], ["c0", "c1", "c2"])

    def test_fallback_on_api_error(self) -> None:
        ranked = candidates(6)
        top = llm_rerank("soru", ranked, FakeClient(error=RuntimeError("bağlantı yok")), "m", top_k=3)
        self.assertEqual([c["chunk_id"] for c in top], ["c0", "c1", "c2"])

    def test_never_returns_unknown_candidates(self) -> None:
        ranked = candidates(4)
        top = llm_rerank("soru", ranked, FakeClient('{"ranking": [7, 8, 2]}'), "m", top_k=4)
        self.assertTrue(all(c in ranked for c in top))
        self.assertEqual(top[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
