"""Tests for the broken Turkish font encoding repair stage."""

from __future__ import annotations

import unittest

from src.preprocessing.repair_encoding import (
    FAMILY_LEGACY_LATIN,
    FAMILY_LEGACY_SYMBOL,
    detect_family,
    repair_document,
    repair_text,
)


class RepairTextTests(unittest.TestCase):
    def test_legacy_latin(self) -> None:
        text = "Deprem anÈnda hemen dÈëarÈ çÈkmak çok tehlikelidir. Eèer bina içerisindeysek asansöre binmemeli."
        self.assertEqual(
            repair_text(text, FAMILY_LEGACY_LATIN),
            "Deprem anında hemen dışarı çıkmak çok tehlikelidir. Eğer bina içerisindeysek asansöre binmemeli.",
        )

    def test_legacy_latin_second_variant_and_uppercase(self) -> None:
        self.assertEqual(repair_text("taü×y×c× ûekil", FAMILY_LEGACY_LATIN), "taşıyıcı Şekil")
        self.assertEqual(repair_text("GÜVENLéçéNéN SANéYE éÇéN", FAMILY_LEGACY_LATIN), "GÜVENLİĞİNİN SANİYE İÇİN")
        self.assertEqual(repair_text("Farkl2 senaryolar2 m2 alan", FAMILY_LEGACY_LATIN), "Farklı senaryoları m2 alan")

    def test_genuine_umlaut_and_cedilla_kept_in_legacy_latin(self) -> None:
        self.assertEqual(repair_text("güvenli çocuk Örneèin", FAMILY_LEGACY_LATIN), "güvenli çocuk Örneğin")

    def test_legacy_symbol(self) -> None:
        self.assertEqual(repair_text("çantas›n› eﬂyalar›n Ola¤and›ﬂ› ﬁekil G‹R‹ﬁ", FAMILY_LEGACY_SYMBOL),
                         "çantasını eşyaların Olağandışı Şekil GİRİŞ")

    def test_clean_document_only_deligatured(self) -> None:
        self.assertEqual(repair_text("coğraﬁ mezkûr Société", None), "coğrafi mezkûr Société")


class DetectionTests(unittest.TestCase):
    def test_detects_legacy_latin(self) -> None:
        self.assertEqual(detect_family("yangÈn sonrasÈ yardÈm nasÈl " * 10), FAMILY_LEGACY_LATIN)

    def test_detects_legacy_symbol(self) -> None:
        self.assertEqual(detect_family("çantas›n› faydal› baz› " * 10), FAMILY_LEGACY_SYMBOL)

    def test_clean_and_foreign_text_not_flagged(self) -> None:
        self.assertIsNone(detect_family("Deprem sırasında Çök-Kapan-Tutun uygulanır. " * 50))
        self.assertIsNone(detect_family("Gutiérrez et Pérez, Société Géologique, mezkûr hükûmet " * 30))

    def test_sparse_signature_not_flagged(self) -> None:
        self.assertIsNone(detect_family("È " + "normal Türkçe metin " * 5000))


class DocumentTests(unittest.TestCase):
    def test_repair_document_keeps_page_fields(self) -> None:
        pages = [{"page": 1, "text": "yangÈn sonrasÈ " * 20, "source_file": "x.pdf"}]
        repaired, info = repair_document(pages)
        self.assertEqual(info["family"], FAMILY_LEGACY_LATIN)
        self.assertEqual(repaired[0]["page"], 1)
        self.assertEqual(repaired[0]["source_file"], "x.pdf")
        self.assertNotIn("È", repaired[0]["text"])


if __name__ == "__main__":
    unittest.main()
