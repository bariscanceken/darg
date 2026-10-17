import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from src.api import app as api
from src.api import library


class LibraryTest(unittest.TestCase):
    """A small fake corpus: one citizen guide (in the core package), one IRAP with only a file name, one image."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        raw, structured, geo = root / "raw", root / "structured", root / "geo"
        for path, content in [(raw / "afad/books/ilk72saat1.pdf", b"%PDF-1.4 guide"),
                              (raw / "afad/plans/irap/KAHRAMANMARAS-IRAP.pdf", b"%PDF-1.4 plan"),
                              (raw / "afad/maps/harita.png", b"\x89PNG"),
                              (raw / "afad/books/notlar.docx", b"skip me")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        guide = {"pages": [{"page": 1, "blocks": [{"text": "Çök, kapan, tutun."}]}, {"page": 2, "blocks": []},
                           {"page": 3, "blocks": [{"text": "Gaz vanasını"}, {"text": "kapatın."}]}]}
        (structured / "afad/books").mkdir(parents=True)
        (structured / "afad/books/ilk72saat1.json").write_text(json.dumps(guide), encoding="utf-8")
        geo.mkdir()
        (geo / "titles.json").write_text(json.dumps({"ilk72saat1.pdf": "Afetlerde İlk 72 Saat Hazırlık Rehberi"}),
                                         encoding="utf-8")
        (geo / "types.json").write_text(json.dumps({"files": {"ilk72saat1.pdf": "vatandas_rehberi"}}), encoding="utf-8")
        (geo / "irap.json").write_text("{}", encoding="utf-8")
        (geo / "provinces.json").write_text(json.dumps({"provinces": [{"name": "Kahramanmaraş"}, {"name": "Maraş"}]}),
                                            encoding="utf-8")
        patches = [mock.patch.object(library, "RAW_ROOT", raw), mock.patch.object(library, "STRUCTURED_ROOT", structured),
                   mock.patch.object(library, "TITLES_FILE", geo / "titles.json"),
                   mock.patch.object(library, "TYPES_FILE", geo / "types.json"),
                   mock.patch.object(library, "IRAP_FILE", geo / "irap.json"),
                   mock.patch.object(library, "PROVINCES_FILE", geo / "provinces.json")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        library.catalog.cache_clear()
        library._province_names.cache_clear()
        self.addCleanup(library.catalog.cache_clear)
        self.addCleanup(library._province_names.cache_clear)
        self.client = TestClient(api.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_catalog_titles_categories_provinces_and_core_package(self):
        body = self.client.get("/kutuphane").json()
        by_name = {d["dosya_adi"]: d for d in body["belgeler"]}
        self.assertEqual(set(by_name), {"ilk72saat1.pdf", "KAHRAMANMARAS-IRAP.pdf", "harita.png"})  # .docx skipped
        guide, plan, image = by_name["ilk72saat1.pdf"], by_name["KAHRAMANMARAS-IRAP.pdf"], by_name["harita.png"]
        self.assertEqual((guide["kategori"], guide["temel"], guide["sayfa"], guide["aranabilir"]),
                         ("vatandas_rehberi", True, 3, True))
        self.assertEqual((plan["il"], plan["baslik"]), ("Kahramanmaraş", "Kahramanmaraş İl Afet Risk Azaltma Planı"))
        self.assertEqual((image["kategori"], image["tur"]), ("gorsel", "gorsel"))
        self.assertEqual(body["temel_paket"], [guide["id"]])

    def test_file_and_page_texts_by_id_only(self):
        guide = next(d for d in self.client.get("/kutuphane").json()["belgeler"] if d["temel"])
        response = self.client.get(f"/kutuphane/{guide['id']}/dosya")
        self.assertEqual((response.status_code, response.content), (200, b"%PDF-1.4 guide"))
        self.assertEqual(response.headers["content-type"], "application/pdf")
        pages = self.client.get(f"/kutuphane/{guide['id']}/metin").json()["sayfalar"]
        self.assertEqual(pages, [{"sayfa": 1, "metin": "Çök, kapan, tutun."}, {"sayfa": 3, "metin": "Gaz vanasını\nkapatın."}])
        self.assertEqual(self.client.get("/kutuphane/..%2F..%2Fsecret/dosya").status_code, 404)
        self.assertEqual(self.client.get("/kutuphane/000000000000/metin").status_code, 404)


if __name__ == "__main__":
    unittest.main()
