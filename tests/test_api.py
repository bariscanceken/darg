import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from src.api import app as api


def fake_answer(question, history=None, **_):
    standalone = "Antakya'da hangi hastaneye gidebilirim?" if history else question
    return {"answer": "Öncelikle 112'yi arayın. [Kaynak 1]", "standalone_question": standalone, "used_web": True,
            "sentences": [{"text": "Öncelikle 112'yi arayın.", "sources": [1], "claimed_afad": True},
                          {"text": "Sakin olun.", "sources": [], "claimed_afad": False}],
            "sources": [{"source_label": "Kaynak 1", "chunk_id": "web:https://x.gov.tr/a?utm_source=openai",
                         "source_file": "https://x.gov.tr/a?utm_source=openai"},
                        {"source_label": "Kaynak 2", "chunk_id": "afad_earthquake_v5_1", "source_file": "ilk72saat1.pdf",
                         "page_label": "Sayfa 32", "page_start": 32}]}


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        logs = Path(self.tmp.name)
        patches = [mock.patch.object(api, "LOG_DIR", logs), mock.patch.object(api, "QA_LOG", logs / "qa.jsonl"),
                   mock.patch.object(api, "FEEDBACK_LOG", logs / "fb.jsonl"),
                   mock.patch.object(api, "generate_answer_v3", side_effect=fake_answer),
                   mock.patch.object(api, "_library_by_file", return_value={
                       "ilk72saat1.pdf": mock.Mock(title="İlk 72 Saat", id="abc123")})]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        api._sessions.clear()
        self.client = TestClient(api.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ask_feedback_and_statistics(self):
        first = self.client.post("/soru", json={"soru": "antakyadayım deprem oldu"}).json()
        self.assertEqual(first["kaynaklar"][0], {"etiket": "Kaynak 1", "tur": "web", "kaynak": "https://x.gov.tr/a",
                                                 "sayfa": None, "baslik": None, "belge_id": None, "ilk_sayfa": None})
        afad = first["kaynaklar"][1]
        self.assertEqual((afad["tur"], afad["baslik"], afad["belge_id"], afad["ilk_sayfa"]),
                         ("afad", "İlk 72 Saat", "abc123", 32))  # links to the library document and page
        self.assertEqual(first["cumleler"], [{"metin": "Öncelikle 112'yi arayın.", "kaynaklar": [1]},
                                             {"metin": "Sakin olun.", "kaynaklar": []}])
        second = self.client.post("/soru", json={"soru": "peki hastane?", "oturum_id": first["oturum_id"]}).json()
        self.assertEqual(second["anlasilan_soru"], "Antakya'da hangi hastaneye gidebilirim?")  # history was used
        self.assertEqual(self.client.post("/geri-bildirim", json={"kayit_id": first["kayit_id"], "puan": "iyi"}).status_code, 200)
        self.client.post("/geri-bildirim", json={"kayit_id": second["kayit_id"], "puan": "kotu", "not": "eksik"})
        stats = self.client.get("/istatistik").json()
        self.assertEqual((stats["soru_sayisi"], stats["geri_bildirim_sayisi"], stats["olumlu_orani"]), (2, 2, 0.5))

    def test_service_failure_is_a_503_with_a_clear_message_and_is_not_logged_as_an_answer(self):
        from src.rag.pipeline_v3 import ServiceUnavailable

        with mock.patch.object(api, "generate_answer_v3", side_effect=ServiceUnavailable("AuthenticationError: 401")):
            response = self.client.post("/soru", json={"soru": "artvinde hangi hastaneler var"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("yapay zekâ hizmetine bağlanılamadı", response.json()["detail"])
        self.assertEqual(self.client.get("/istatistik").json()["soru_sayisi"], 0)

    def test_unknown_record_and_empty_question_are_rejected(self):
        self.assertEqual(self.client.post("/geri-bildirim", json={"kayit_id": "yok", "puan": "iyi"}).status_code, 404)
        self.assertEqual(self.client.post("/soru", json={"soru": ""}).status_code, 422)
        self.assertEqual(self.client.get("/saglik").json(), {"durum": "ok"})


if __name__ == "__main__":
    unittest.main()
