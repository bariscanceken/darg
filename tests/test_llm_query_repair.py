"""Offline tests for the LLM repair safety checks (no OpenAI calls)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.rag.llm_query_repair import is_spelling_only, parse_repair_response, repair_query, unexplained_tokens
from src.rag.query_normalization import load_query_vocab


class FakeClient:
    """Minimal stand-in for the OpenAI client returning a fixed text."""

    def __init__(self, text: str) -> None:
        self.calls = 0
        self.responses = SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(output_text=self._text)


class SpellingOnlyValidationTests(unittest.TestCase):
    def test_accepts_word_level_spelling_fixes(self) -> None:
        self.assertTrue(is_spelling_only("deprem çantsına neler kymalıyım", "deprem çantasına neler koymalıyım"))
        self.assertTrue(is_spelling_only("depremde binwadan cikmali miyim", "depremde binadan çıkmalı mıyım"))

    def test_accepts_split_of_fused_question_particle(self) -> None:
        self.assertTrue(is_spelling_only("hasarlı binya girilirmi", "hasarlı binaya girilir mi"))

    def test_rejects_rewrites_and_added_words(self) -> None:
        self.assertFalse(is_spelling_only("depremde binwadan çikimi", "Deprem sırasında binadan çıkmalı mıyım?"))
        self.assertFalse(is_spelling_only("gaz vanası", "doğalgaz vanasını kapatmak"))
        self.assertFalse(is_spelling_only("asansör kullanılır mı", "merdiven kullanılır mı"))

    def test_rejects_dropped_words(self) -> None:
        self.assertFalse(is_spelling_only("deprem anında ne yapmalıyım", "deprem ne yapmalıyım"))


class TriggerTests(unittest.TestCase):
    def test_clean_query_needs_no_repair(self) -> None:
        vocab = load_query_vocab()
        self.assertEqual(unexplained_tokens("Deprem sırasında asansör kullanmalı mıyım?", vocab), [])
        self.assertEqual(unexplained_tokens("Kahramanmaraş'taki AFAD ekipleri nerede?", vocab), [])

    def test_garbled_word_triggers_repair(self) -> None:
        self.assertIn("xqzt", unexplained_tokens("deprem xqzt ne yapmalıyım", load_query_vocab()))

    def test_no_llm_call_when_not_needed(self) -> None:
        client = FakeClient('{"corrected": "x", "confident": true}')
        result = repair_query("Deprem sırasında asansör kullanmalı mıyım?", client, "model")
        self.assertEqual(result["status"], "not_needed")
        self.assertEqual(client.calls, 0)


class ResponseHandlingTests(unittest.TestCase):
    def test_rewrite_is_rejected_and_original_kept(self) -> None:
        query = "depremde xqzt ne yapmalıyım"
        client = FakeClient('{"corrected": "Deprem sırasında evde ne yapmalıyım?", "confident": true}')
        result = repair_query(query, client, "model")
        self.assertEqual(result["status"], "rejected_rewrite")
        self.assertEqual(result["repaired"], query)

    def test_not_confident_keeps_original(self) -> None:
        query = "depremde xqzt ne yapmalıyım"
        result = repair_query(query, FakeClient('{"corrected": "depremde xqzt ne yapmalıyım", "confident": false}'), "m")
        self.assertEqual(result["status"], "not_confident")
        self.assertEqual(result["repaired"], query)

    def test_invalid_json_keeps_original(self) -> None:
        self.assertIsNone(parse_repair_response("tamam"))
        result = repair_query("depremde xqzt ne yapmalıyım", FakeClient("tamam"), "m")
        self.assertEqual(result["status"], "invalid_response")

    def test_valid_fix_is_accepted(self) -> None:
        query = "deprem çantsına neler kymalıyım"
        client = FakeClient('{"corrected": "deprem çantasına neler koymalıyım", "confident": true}')
        result = repair_query(query, client, "model")
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["repaired"], "deprem çantasına neler koymalıyım")


if __name__ == "__main__":
    unittest.main()
