import unittest

from src.rag.situation import AFTER, DURING, PREPARE, RUBBLE, header_situation, passage_situation


class SituationFromHeadersTest(unittest.TestCase):
    def test_timeline_of_the_72_hour_guide(self):
        title = "Afetlerde İlk 72 Saat Hazırlık Rehberi"
        self.assertEqual(header_situation(f"{title} › V. 0. SANİYE İÇİN HAZIRLIK"), PREPARE)
        self.assertEqual(header_situation(f"{title} › VI. 3. SANİYE İÇİN HAZIRLIK › Çök-Kapan-Tutun"), DURING)
        self.assertEqual(header_situation(f"{title} › VII. 3. DAKİKA İÇİN HAZIRLIK › İlkyardım"), AFTER)
        self.assertEqual(header_situation(f"{title} › X. 3. GÜN İÇİN HAZIRLIK › Barınma"), AFTER)

    def test_named_sections_and_rubble(self):
        self.assertEqual(header_situation("Engelliler için Afet Hazırlık Rehberi › DEPREM SIRASINDA YAPILMASI GEREKENLER"), DURING)
        self.assertEqual(header_situation("Olağandışı Durumlar › HAYATIN İDAMESİ › Enkaz Altında Hayatın İdamesi"), RUBBLE)
        self.assertEqual(header_situation("Olağandışı Durumlar › HAYATIN İDAMESİ › Afet Sonrası Yaşamın Sürdürülmesi"), AFTER)

    def test_document_title_alone_decides_nothing(self):
        self.assertIsNone(header_situation("Afetlerde İlk 72 Saat Hazırlık Rehberi › ÖNSÖZ"))
        self.assertIsNone(passage_situation("Hatay il planı (2021) (doğrulanmış il kartı)\nAntakya: 133"))
        self.assertEqual(passage_situation("Rehber › Deprem Sırasında\nmetin"), DURING)


if __name__ == "__main__":
    unittest.main()
