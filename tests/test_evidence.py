"""Offline tests for evidence-first validation (no network)."""

from __future__ import annotations

import unittest

from src.rag.evidence import allowed_labels, format_evidence, mark_wider_area, parse_evidence, validate_evidence


CHUNKS = [
    {"text": "Eğer bina içerisindeysek depremi hissettiğimizde ayakta durmamalı, asansöre binmemeli ve pencereden atlamamalıyız."},
    {"text": "1999 depreminden sonra Gölcük'te çadırkentler kurulmuş ve aşevleri açılmıştır."},
]


def payload(items: list[dict], situation: str = "deprem sırasında") -> dict:
    return {"question_situation": situation, "evidence": items}


class ValidateEvidenceTests(unittest.TestCase):
    def test_verbatim_quote_is_kept(self) -> None:
        kept = validate_evidence(payload([{"source": 1, "quote": "depremi hissettiğimizde ayakta durmamalı, asansöre binmemeli",
                                           "about": "same_situation"}]), CHUNKS)
        self.assertEqual(len(kept), 1)

    def test_invented_quote_is_dropped(self) -> None:
        kept = validate_evidence(payload([{"source": 1, "quote": "deprem sırasında asansörle hızla aşağı inin",
                                           "about": "same_situation"}]), CHUNKS)
        self.assertEqual(kept, [])

    def test_quote_from_wrong_source_is_dropped(self) -> None:
        kept = validate_evidence(payload([{"source": 2, "quote": "depremi hissettiğimizde ayakta durmamalı",
                                           "about": "same_situation"}]), CHUNKS)
        self.assertEqual(kept, [])

    def test_other_situation_and_place_are_dropped(self) -> None:
        for about in ("other_situation", "other_place"):
            kept = validate_evidence(payload([{"source": 1, "quote": "depremi hissettiğimizde ayakta durmamalı",
                                               "about": about}]), CHUNKS)
            self.assertEqual(kept, [], about)

    def test_wider_area_kept_and_marked(self) -> None:
        item = {"source": 1, "quote": "depremi hissettiğimizde ayakta durmamalı", "about": "wider_area"}
        kept = validate_evidence(payload([item]), CHUNKS)
        self.assertEqual(len(kept), 1)
        self.assertIn("(kapsam: daha geniş alan)", format_evidence(kept))
        self.assertNotIn("kapsam", format_evidence([{**item, "about": "same_situation"}]))

    def test_quotes_not_naming_the_place_become_wider_area(self) -> None:
        names = frozenset({"hatay", "antakya", "iskenderun"})
        items = [{"source": 1, "quote": "ülkemizde 6 Şubat depremlerinde 50783 kişi", "about": "past_event"},
                 {"source": 2, "quote": "22 Ocak 1997 Antakya Depremi: 1 kişi yaşamını yitirmiş", "about": "past_event"},
                 {"source": 3, "quote": "HATAY İLİ toplanma alanı sayısı", "about": "general_fact"}]
        self.assertEqual([item["about"] for item in mark_wider_area(items, names)],
                         ["wider_area", "past_event", "general_fact"])

    def test_past_event_only_for_history_questions(self) -> None:
        item = [{"source": 2, "quote": "1999 depreminden sonra Gölcük'te çadırkentler kurulmuş", "about": "past_event"}]
        self.assertEqual(validate_evidence(payload(item, "deprem sonrasında"), CHUNKS), [])
        self.assertEqual(len(validate_evidence(payload(item, "geçmiş olaylar"), CHUNKS)), 1)
        self.assertIn("past_event", allowed_labels("Geçmiş olaylar"))

    def test_bad_source_numbers_and_short_quotes(self) -> None:
        items = [{"source": 0, "quote": "x" * 30, "about": "same_situation"},
                 {"source": "1", "quote": "asansöre binmemeli ve pencereden", "about": "same_situation"},
                 {"source": 1, "quote": "asansöre", "about": "same_situation"}]
        self.assertEqual(validate_evidence(payload(items), CHUNKS), [])

    def test_parse_evidence(self) -> None:
        self.assertIsNone(parse_evidence("yok"))
        self.assertIsNone(parse_evidence('{"evidence": "x"}'))
        self.assertEqual(parse_evidence('{"question_situation": "x", "evidence": []}')["evidence"], [])


if __name__ == "__main__":
    unittest.main()
