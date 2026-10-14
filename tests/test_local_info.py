import json
import tempfile
import unittest
from pathlib import Path

from src.rag.local_info import (BROCHURE_FILE, brochure_lines, build_local_info, district_count, format_local_info,
                                load_card, local_info_evidence, wanted_fields)
from src.rag.location import Location


CARD = {
    "province": "Hatay",
    "source_file": "HATAY-İRAP-2022.pdf",
    "data_year": "2021",
    "fields": {
        "faults": [{"value": "Ölü Deniz Fay Zonu", "page": 21, "quote": "..."}],
        "other_hazards": [{"value": "Sıvılaşma riski", "page": 38, "quote": "alüvyon zeminde sıvılaşma"},
                          {"value": "Tsunami riski", "page": 145, "quote": "deniz kabarması ve tsunami riski"}],
        "assembly_areas": [{"value": "Altınözü: 28, Antakya: 133, Arsuz: 41", "page": 44, "quote": "..."}],
        "temporary_shelters": [
            {"value": "Antakya Saçaklı-Suvatlı, 327.651 Kişi", "page": 44, "quote": "..."},
            {"value": "Arsuz Alakop (Höyük), 44.206 Kişi", "page": 44, "quote": "..."},
        ],
    },
}
NO_BROCHURE = "missing_brochure.json"


class LocalInfoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        Path(self.tmp.name, "Hatay.json").write_text(json.dumps(CARD, ensure_ascii=False), encoding="utf-8")
        load_card.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()
        load_card.cache_clear()

    def build(self, location, question="deprem oldu ne yapmalıyım", question_type="davranis", brochure=NO_BROCHURE):
        return build_local_info(location, question, question_type, card_dir=self.tmp.name, brochure_file=brochure)

    def test_district_gets_own_count_and_shelter(self):
        info = self.build(Location(province="Hatay", district="Antakya"))
        texts = [line["text"] for line in info["lines"]]
        self.assertIn("Antakya ilçesinde 133 toplanma alanı belirlenmiş.", texts)
        self.assertTrue(any("Saçaklı-Suvatlı" in text for text in texts))
        self.assertFalse(any("Alakop" in text for text in texts))
        self.assertIn("2021", info["note"])
        self.assertIn("Antakya için Hatay il planı (2021) bilgisi:", format_local_info(info))

    def test_province_only_lists_all(self):
        info = self.build(Location(province="Hatay"))
        self.assertTrue(any("Arsuz: 41" in line["text"] for line in info["lines"]))

    def test_fields_follow_question(self):
        faults = self.build(Location(province="Hatay"), "hatayda hangi faylar var", "bolgesel")
        self.assertEqual([line["text"] for line in faults["lines"]], ["Fay hatları: Ölü Deniz Fay Zonu"])
        self.assertEqual(faults["general_lines"], [])
        self.assertIsNone(self.build(Location(province="Hatay"), "hatayın nüfusu kaç", "bolgesel"))
        self.assertEqual(wanted_fields("antakyada toplanma alanı nerede", "bolgesel"), ["assembly_areas"])
        self.assertEqual(wanted_fields("hatayda geçmişte olan depremler", "gecmis"), ["historical_earthquakes"])

    def test_hazard_lines_only_for_asked_hazard(self):
        info = self.build(Location(province="Hatay"), "hatayda tsunami riski var mı", "bolgesel")
        self.assertEqual(info["lines"], [{"text": 'Tsunami riski: "deniz kabarması ve tsunami riski"', "page": 145}])

    def test_missing_card_or_ambiguous_location(self):
        self.assertIsNone(self.build(Location(province="Bolu")))
        self.assertIsNone(self.build(Location(district="Merkez", ambiguous_provinces=("A", "B"))))
        self.assertIsNone(self.build(None))

    @unittest.skipUnless(BROCHURE_FILE.exists(), "brochure text not extracted")
    def test_brochure_lines_verified_and_used_without_place(self):
        self.assertEqual(len(brochure_lines(str(BROCHURE_FILE))), 3)
        info = self.build(None, "deprem oldu nereye gideceğim", "davranis", brochure=str(BROCHURE_FILE))
        self.assertEqual(len(info["general_lines"]), 3)
        self.assertEqual(info["lines"], [])
        self.assertIn("Toplanma alanı (AFAD):", format_local_info(info))
        self.assertIsNone(self.build(None, "deprem oldu ne yapmalıyım", "davranis", brochure=str(BROCHURE_FILE)))
        bolu = self.build(Location(province="Bolu"), brochure=str(BROCHURE_FILE))  # no card: brochure only
        self.assertEqual(len(bolu["general_lines"]), 3)

    def test_local_lines_become_numbered_evidence(self):
        info = self.build(Location(province="Hatay", district="Antakya"))
        evidence, sources = local_info_evidence(info, 6)
        self.assertEqual(evidence[0], {"source": 6, "quote": "Hatay il planı (2021): Antakya ilçesinde 133 toplanma "
                                                            "alanı belirlenmiş.", "about": "card"})
        self.assertEqual(sources[0]["source_label"], "Kaynak 6")
        self.assertEqual(sources[0]["chunk_id"], "il_karti:Hatay")
        self.assertEqual(local_info_evidence(None, 6), ([], []))

    def test_province_wide_lines_marked_for_district_question(self):
        info = self.build(Location(province="Hatay", district="Antakya"), "antakyada fay var mı", "bolgesel")
        evidence, _ = local_info_evidence(info, 6)
        self.assertEqual(evidence[0]["quote"], "Hatay il planı (2021), Hatay ili geneli (Antakya için ayrı bilgi yok): "
                                               "Fay hatları: Ölü Deniz Fay Zonu")

    def test_citation_lists_are_read(self):
        from src.rag.context import referenced_source_numbers
        self.assertEqual(referenced_source_numbers("x [Kaynak 6, 7] y [Kaynak 1, Kaynak 3] [Kaynak 2]"), [6, 7, 1, 3, 2])

    def test_district_count_exact_name(self):
        self.assertEqual(district_count("Defne: 55, Antakya: 133", "Antakya"), "133")
        self.assertIsNone(district_count("Defne: 55", "Antakya"))
        self.assertEqual(wanted_fields("x", "davranis"), ["assembly_areas", "temporary_shelters"])


if __name__ == "__main__":
    unittest.main()
