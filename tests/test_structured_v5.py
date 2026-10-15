import unittest
from unittest import mock

from src.preprocessing import chunk_structured_v5 as chunker
from src.preprocessing.extract_structured import heading_level
from src.preprocessing.ocr_pages import markdown_blocks


def record(pages):
    return {"source_file": "rehber.pdf", "relative_path": "afad/books/rehber.pdf", "organization": "AFAD",
            "pages": [{"page": number, "blocks": blocks} for number, blocks in enumerate(pages, start=1)]}


class HeadingLevelTest(unittest.TestCase):
    def line(self, text, size, bold=False):
        return {"text": text, "size": size, "bold": bold}

    def test_levels_from_size_and_bold(self):
        self.assertEqual(heading_level(self.line("Afet Sonrası Yaşamın Sürdürülmesi", 15), 10.5, True), 2)
        self.assertEqual(heading_level(self.line("İLK 72 SAAT", 20), 10.5, True), 1)
        self.assertEqual(heading_level(self.line("Açık alanda", 10.5, bold=True), 10.5, True), 3)

    def test_body_text_is_not_a_heading(self):
        self.assertIsNone(heading_level(self.line("Açık alanlarda binaların uzağında durun.", 10.5, bold=True), 10.5, True))
        self.assertIsNone(heading_level(self.line("Açık alanda", 10.5), 10.5, True))
        self.assertIsNone(heading_level(self.line("• Madde", 15), 10.5, True))


class OcrMarkdownTest(unittest.TestCase):
    def test_markdown_dialect_to_blocks(self):
        blocks = markdown_blocks("# Aile Afet Planı\n- Plan yapın.\n- Buluşma yeri belirleyin.\n\n"
                                 "İlçe: Antakya; Sayı: 133\nİlçe: Defne; Sayı: 55")
        self.assertEqual(blocks[0], {"kind": "heading", "level": 1, "text": "Aile Afet Planı", "ocr": True})
        self.assertEqual(blocks[1]["text"], "Plan yapın. Buluşma yeri belirleyin.")
        self.assertEqual(blocks[2]["kind"], "table")
        self.assertEqual(markdown_blocks("BOŞ"), [])


class ChunkerTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(chunker, "known_titles", return_value={"rehber.pdf": "Olağandışı Durumlar Rehberi"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_chunks_carry_section_path(self):
        chunks = chunker.document_chunks(record([[
            {"kind": "heading", "level": 2, "text": "Afet Sonrası Yaşamın Sürdürülmesi"},
            {"kind": "heading", "level": 3, "text": "Açık alanda"},
            {"kind": "text", "text": "Deprem sonrası açık alanlarda binaların ve trafoların uzağında durun."},
        ]]))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["text"].startswith(
            "Olağandışı Durumlar Rehberi › Afet Sonrası Yaşamın Sürdürülmesi › Açık alanda\n"))
        self.assertEqual(chunks[0]["section_path"], ["Afet Sonrası Yaşamın Sürdürülmesi", "Açık alanda"])

    def test_table_rows_stay_whole_and_other_disaster_pages_drop(self):
        chunks = chunker.document_chunks(record([
            [{"kind": "table", "text": "İlçe: Antakya; Sayı: 133 toplanma alanı\nİlçe: Defne; Sayı: 55 toplanma alanı"}],
            [{"kind": "text", "text": "Sel ve taşkın sırasında yüksek yerlere çıkın."}],
        ]))
        self.assertEqual(len(chunks), 1)
        self.assertIn("İlçe: Antakya; Sayı: 133 toplanma alanı\nİlçe: Defne", chunks[0]["text"])
        self.assertTrue(chunks[0]["has_table"])
        self.assertNotIn("Sel", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
