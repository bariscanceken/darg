import unittest

from src.rag.pipeline_v3 import IN_SCOPE_RE, needs_web, render, verify_sentences
from src.rag.query_normalization import turkish_lower

PASSAGES = [{"text": "İlk 72 Saat › Deprem Sırasında\nDeprem sırasında asansör ve merdiven kullanmayın."}]


class ScopeAndWebDecisionTest(unittest.TestCase):
    def test_earthquake_words_are_always_in_scope(self):
        for question in ["Binam depremde yıkılır mı?", "Binanın güvenli olup olmadığına karar verebilir miyim?",
                         "Büyük deprem ne zaman olacak?", "enkaz altındayım"]:
            self.assertTrue(IN_SCOPE_RE.search(turkish_lower(question)), question)
        self.assertIsNone(IN_SCOPE_RE.search(turkish_lower("Python'da liste nasıl sıralanır?")))
        for question in ["kırığı olduğunu düşündüğüm birini taşımalı mıyım", "kanaması olan yaralıya ne yapmalıyım"]:
            self.assertTrue(IN_SCOPE_RE.search(turkish_lower(question)), question)

    def test_empty_answer_is_not_called_off_topic(self):
        from src.rag.pipeline_v3 import NO_ANSWER, OFF_TOPIC_ANSWER
        self.assertEqual(render([]), NO_ANSWER)
        self.assertNotEqual(render([]), OFF_TOPIC_ANSWER)
        self.assertIn("112", NO_ANSWER)

    def test_web_only_when_needed(self):
        afad = [{"text": "a", "sources": [1]}]
        self.assertFalse(needs_web("deprem sırasında asansör kullanılır mı", afad))
        self.assertTrue(needs_web("6 şubat 2023 depreminde hatayda ne oldu", afad))
        self.assertTrue(needs_web("deprem çantası", [{"text": "a", "sources": []}]))


class VerifyAndRenderTest(unittest.TestCase):
    def test_unverified_quote_becomes_general_knowledge(self):
        payload = {"sentences": [
            {"text": "Asansör kullanmayın.", "basis": "afad",
             "quotes": [{"source": 1, "quote": "Deprem sırasında asansör ve merdiven kullanmayın."}]},
            {"text": "Pencerelerden uzak durun.", "basis": "afad",
             "quotes": [{"source": 1, "quote": "Pencerelerden ve camlardan uzak durun."}]},
        ]}
        sentences = verify_sentences(payload, PASSAGES)
        self.assertEqual([s["sources"] for s in sentences], [[1], []])
        answer = render(sentences)
        self.assertIn("Asansör kullanmayın. [Kaynak 1]", answer)
        self.assertIn("Pencerelerden uzak durun. (genel bilgi)", answer)


if __name__ == "__main__":
    unittest.main()


class GeneralCheckTest(unittest.TestCase):
    def test_doubtful_general_sentences_are_dropped(self):
        import json
        from src.rag.pipeline_v3 import check_general_sentences

        class Client:
            def __init__(self):
                self.responses = self

            def create(self, **_):
                class Response:
                    output_text = json.dumps({"checks": [{"index": 1, "verdict": "dogru"},
                                                         {"index": 2, "verdict": "supheli"}]})
                return Response()

        sentences = [{"text": "112'yi arayın.", "sources": []}, {"text": "AFAD kaynaklı.", "sources": [1]},
                     {"text": "Kapı eşiğinde durun.", "sources": []}]
        kept = check_general_sentences("Soru: x", sentences, Client(), "gpt-4o")
        self.assertEqual([s["text"] for s in kept], ["112'yi arayın.", "AFAD kaynaklı."])
