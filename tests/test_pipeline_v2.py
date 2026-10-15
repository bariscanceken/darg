import json
import unittest

from src.rag.answerability import REFUSAL_ANSWER
from src.rag.pipeline_v2 import locate_quote, render_answer, understand_question, verify_answer

PASSAGE = ("Olağandışı Durumlar › Afet Sonrası › Açık alanda\n"
           "Açık alanlarda öncelikle binaların, kulelerin, su depolarının ve trafoların uzağında durmalısınız. "
           "Önceden belirlenmiş bir toplanma yeri varsa oraya gidebilirsiniz.")


class FakeResponse:
    def __init__(self, text):
        self.output_text = text


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.responses = self

    def create(self, **_):
        return FakeResponse(json.dumps(self.payload, ensure_ascii=False))


class LocateQuoteTest(unittest.TestCase):
    def test_exact_and_punctuation_tolerant(self):
        self.assertIsNotNone(locate_quote("Önceden belirlenmiş bir toplanma yeri varsa oraya gidebilirsiniz.", PASSAGE))
        self.assertIsNotNone(locate_quote("binaların, kulelerin, su depolarının ve trafoların uzağında durmalısınız,", PASSAGE))

    def test_small_slip_returns_source_text(self):
        found = locate_quote("binaların, kulelerin, su depoların ve trafoların uzağında durmalısınız", PASSAGE)
        self.assertIn("su depolarının", found)

    def test_invented_or_short_quotes_rejected(self):
        self.assertIsNone(locate_quote("Açık alanda köprülerin altında bekleyin ve aracınızda kalın.", PASSAGE))
        self.assertIsNone(locate_quote("Açık alan", PASSAGE))


class VerifyAnswerTest(unittest.TestCase):
    def test_unsupported_sentences_are_removed(self):
        payload = {"sentences": [
            {"text": "Binalardan ve trafolardan uzak durun.",
             "quotes": [{"source": 1, "quote": "binaların, kulelerin, su depolarının ve trafoların uzağında durmalısınız"}]},
            {"text": "Köprü altında bekleyin.", "quotes": [{"source": 1, "quote": "köprülerin altında güvenle bekleyin"}]},
            {"text": "Uydurma kaynak.", "quotes": [{"source": 7, "quote": "Önceden belirlenmiş bir toplanma yeri varsa"}]},
        ], "not_covered": "araçtaki durum"}
        verified = verify_answer(payload, [{"text": PASSAGE}])
        self.assertEqual([sentence["text"] for sentence in verified["kept"]], ["Binalardan ve trafolardan uzak durun."])
        self.assertEqual(len(verified["dropped"]), 2)
        answer = render_answer(verified)
        self.assertIn("Binalardan ve trafolardan uzak durun. [Kaynak 1]", answer)
        self.assertIn("Kaynaklarda bulunmayan: araçtaki durum", answer)

    def test_nothing_verified_is_a_refusal(self):
        self.assertEqual(render_answer(verify_answer({"sentences": []}, [{"text": PASSAGE}])), REFUSAL_ANSWER)


class UnderstandQuestionTest(unittest.TestCase):
    def test_spelling_fix_checked_locally(self):
        client = FakeClient({"corrected": "deprem oldu ne yapacağım asansöre bineyim mi", "type": "davranis",
                             "situation": "sonrasında", "setting": "", "search_queries": ["asansör kullanımı deprem"]})
        result = understand_question("dprem oldu napcam asaönsore bineyimmi", client)
        self.assertEqual(result["corrected"], "deprem oldu ne yapacağım asansöre bineyim mi")
        self.assertEqual(result["type"], "davranis")
        self.assertEqual(result["search_queries"], ["asansör kullanımı deprem"])

    def test_rewrite_disguised_as_spelling_is_rejected(self):
        client = FakeClient({"corrected": "deprem sırasında merdiven kullanılır mı", "type": "bilinmeyen",
                             "situation": "x", "search_queries": []})
        result = understand_question("asansör kullanılır mı", client)
        self.assertEqual(result["corrected"], "asansör kullanılır mı")
        self.assertEqual(result["spelling"], "rejected")
        self.assertEqual((result["type"], result["situation"]), ("genel", "genel"))


if __name__ == "__main__":
    unittest.main()
