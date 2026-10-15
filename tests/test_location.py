"""Tests for province/district detection (uses data/geo, no network)."""

from __future__ import annotations

import unittest

from src.rag.location import detect_location, has_regional_plan


def loc(question: str):
    return detect_location(question)


class DetectionTests(unittest.TestCase):
    def test_district_with_apostrophe_resolves_province(self) -> None:
        found = loc("Deprem sonrası Ceyhan'da güvenli bir yere nasıl geçerim?")
        self.assertEqual((found.province, found.district), ("Adana", "Ceyhan"))

    def test_province_forms(self) -> None:
        for question in ["Hatay'daki barınma alanları", "hatayda toplanma alanı", "İstanbul deprem riski", "istanbulda ne yapmalıyım"]:
            self.assertIsNotNone(loc(question), question)
        self.assertEqual(loc("hatayda toplanma alanı").province, "Hatay")

    def test_garbled_case_suffix_still_matches(self) -> None:
        found = loc("antakyadaydayım deprem oldu hangi hastaneye gidicem")
        self.assertEqual((found.province, found.district), ("Hatay", "Antakya"))
        self.assertEqual(loc("hataydaydım").province, "Hatay")
        self.assertIsNone(loc("vanası kapalı mı"))

    def test_misspelt_and_diacritic_free_names(self) -> None:
        self.assertEqual(loc("derpem oldu anatakyaEDayım ne yapcam").district, "Antakya")
        self.assertEqual(loc("iskendrunda tsunami var mı").district, "İskenderun")
        self.assertEqual(loc("samandagda evime girebilir miyim").district, "Samandağ")
        for question in ["deprem oldu ne yapmalıyım", "asansöre binmeli miyim", "komşularımla buluşma noktası",
                         "hasarlı binaya girilir mi", "enkaz altında kalırsam"]:
            self.assertIsNone(loc(question), question)

    def test_aliases(self) -> None:
        self.assertEqual(loc("Maraş'ta deprem sonrası ne yapmalı?").province, "Kahramanmaraş")
        self.assertEqual(loc("Antep'te toplanma alanı").province, "Gaziantep")

    def test_common_words_are_not_places(self) -> None:
        for question in [
            "Araçtayken deprem olursa ne yapmalıyım?",
            "araç kullanırken sarsıntı olursa",
            "bodrum katta kalırsam ne yapmalıyım",
            "maden kazasında ne yapılır",
            "Gaz vanasını kapatmalı mıyım?",
            "toplanma merkezi nerede",
            "Deprem sırasında asansör kullanmalı mıyım?",
        ]:
            self.assertIsNone(loc(question), question)

    def test_capitalised_common_word_district_matches(self) -> None:
        self.assertEqual(loc("Bodrum'da tsunami riski var mı?").province, "Muğla")

    def test_ambiguous_district_needs_province(self) -> None:
        ambiguous = loc("Gölbaşı'nda toplanma alanı nerede?")
        self.assertTrue(ambiguous.is_ambiguous)
        self.assertIn("Ankara", ambiguous.ambiguous_provinces)
        resolved = loc("Ankara Gölbaşı'nda toplanma alanı nerede?")
        self.assertEqual((resolved.province, resolved.district), ("Ankara", "Gölbaşı"))

    def test_regional_plan_coverage(self) -> None:
        self.assertTrue(has_regional_plan("Adana"))
        self.assertFalse(has_regional_plan("Malatya"))


if __name__ == "__main__":
    unittest.main()
