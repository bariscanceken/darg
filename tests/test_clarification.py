"""Offline tests for 'did you mean…?' suggestion validation."""

from __future__ import annotations

import unittest

from src.rag.clarification import MAX_OPTIONS, parse_options


class ParseOptionsTests(unittest.TestCase):
    def test_valid_options(self) -> None:
        raw = '{"options": ["Enkaz altındayım, ne yapmalıyım?", "Deprem anındayım, ne yapmalıyım?"]}'
        self.assertEqual(parse_options(raw, "deprem altındayım ne yapmalıyım"),
                         ["Enkaz altındayım, ne yapmalıyım?", "Deprem anındayım, ne yapmalıyım?"])

    def test_empty_or_malformed_gives_nothing(self) -> None:
        for raw in ['{"options": []}', "tamam", '{"options": "x"}', "{bozuk json"]:
            self.assertEqual(parse_options(raw, "soru"), [])

    def test_drops_non_questions_duplicates_and_echo(self) -> None:
        raw = '{"options": ["Enkaz altındayım", "Ne yapmalıyım?", "ne yapmalıyım?", "soru?", 5]}'
        self.assertEqual(parse_options(raw, "soru?"), ["Ne yapmalıyım?"])

    def test_caps_option_count_and_length(self) -> None:
        raw = '{"options": ["A?", "B?", "C?", "D?", "' + "x" * 200 + '?"]}'
        self.assertEqual(len(parse_options(raw, "q")), MAX_OPTIONS)


if __name__ == "__main__":
    unittest.main()
