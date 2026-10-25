import json
import unittest
from unittest import mock

from src.rag import graph_v3, pipeline_v3

PASSAGE = {"text": "İlk 72 Saat › Deprem Sırasında\nDeprem sırasında asansör ve merdiven kullanmayın.",
           "chunk_id": "c1", "source_file": "ilk72saat1.pdf"}
WEB = {"text": "Web (resmî kaynak): https://example.gov.tr\nDefne Devlet Hastanesi acil servisi açıktır.",
       "chunk_id": "web:https://example.gov.tr", "source_file": "https://example.gov.tr"}


class FakeResponse:
    def __init__(self, text):
        self.output_text = text


class FakeClient:
    """Answers each prompt type with a fixed JSON payload."""

    def __init__(self, scope=True, answer_sentences=None):
        self.responses = self
        self.scope = scope
        self.answer_sentences = answer_sentences or []

    def create(self, **kwargs):
        instructions = kwargs.get("instructions", "")
        if instructions == pipeline_v3.SCOPE_PROMPT:
            return FakeResponse(json.dumps({"in_scope": self.scope}))
        if instructions == pipeline_v3.ANSWER_PROMPT:
            return FakeResponse(json.dumps({"sentences": self.answer_sentences}, ensure_ascii=False))
        return FakeResponse(json.dumps({"corrected": kwargs["input"], "type": "davranis", "situation": "sırasında",
                                        "setting": "", "search_queries": []}))


def run_both(question, client, web=None):
    patches = [mock.patch.object(module, "retrieve_wide", return_value=[PASSAGE]) for module in (graph_v3, pipeline_v3)]
    patches += [mock.patch.object(module, "web_passages", return_value=web or []) for module in (graph_v3, pipeline_v3)]
    for patch in patches:
        patch.start()
    try:
        flat = pipeline_v3.run_pipeline_v3(question, client, None, [], None, None)
        graph = graph_v3.run_graph_v3(question, client, None, [], None, None)
    finally:
        for patch in patches:
            patch.stop()
    return flat, graph


class GraphMatchesFlatPipelineTest(unittest.TestCase):
    def test_off_topic(self):
        flat, graph = run_both("Python'da liste nasıl sıralanır?", FakeClient(scope=False))
        self.assertEqual(graph["answer"], flat["answer"])
        self.assertEqual(graph["status"], "off_topic")

    def test_afad_answer_without_web(self):
        sentences = [{"text": "Asansör kullanmayın.", "basis": "afad",
                      "quotes": [{"source": 1, "quote": "Deprem sırasında asansör ve merdiven kullanmayın."}]}]
        flat, graph = run_both("deprem sırasında asansör kullanılır mı", FakeClient(answer_sentences=sentences))
        self.assertEqual(graph["answer"], flat["answer"])
        self.assertFalse(graph["used_web"])
        self.assertIn("[Kaynak 1]", graph["answer"])

    def test_web_used_when_question_asks_current_info(self):
        sentences = [{"text": "Defne Devlet Hastanesi acil servisi açıktır.", "basis": "afad",
                      "quotes": [{"source": 2, "quote": "Defne Devlet Hastanesi acil servisi açıktır."}]}]
        flat, graph = run_both("antakyada hangi hastane açık", FakeClient(answer_sentences=sentences), web=[WEB])
        self.assertEqual(graph["answer"], flat["answer"])
        self.assertTrue(graph["used_web"])
        self.assertIn(pipeline_v3.WEB_NOTE, graph["answer"])



class ServiceFailureTest(unittest.TestCase):
    """An invalid key or an outage must surface as ServiceUnavailable, not as "try asking differently"."""

    def run_with(self, error):
        class Client(FakeClient):
            def create(self, **kwargs):
                if kwargs.get("instructions") == pipeline_v3.ANSWER_PROMPT:
                    raise error
                return super().create(**kwargs)

        with mock.patch.object(graph_v3, "retrieve_wide", return_value=[PASSAGE]):
            return graph_v3.run_graph_v3("artvinde hangi hastaneler var", Client(), None, [], None, None,
                                         web_search=False)

    def test_invalid_key_is_a_service_failure(self):
        import httpx
        import openai

        response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        with self.assertRaises(pipeline_v3.ServiceUnavailable):
            self.run_with(openai.AuthenticationError("Your API key has been invalidated.", response=response,
                                                     body=None))
        with self.assertRaises(pipeline_v3.ServiceUnavailable):
            self.run_with(openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")))

    def test_other_errors_still_end_in_the_safe_empty_answer(self):
        result = self.run_with(ValueError("malformed model output"))
        self.assertEqual((result["status"], result["answer"]), ("empty", pipeline_v3.NO_ANSWER))


class TruncatedAnswerTest(unittest.TestCase):
    """A long answer cut off by the output limit keeps its complete sentences instead of becoming NO_ANSWER."""

    CUT = ('{"sentences": [{"text": "Elektrik düğmelerini kullanmayın.", "basis": "afad", "quotes": [{"source": 1, '
           '"quote": "asansör ve merdiven kullanmayın"}]}, {"text": "Vanayı kapatın.", "basis": "genel", "quotes": []},'
           ' {"text": "Binayı hemen terk edin ve', )[0]

    def test_complete_sentences_are_salvaged(self):
        payload = pipeline_v3.salvage_sentences(self.CUT)
        self.assertEqual([s["text"] for s in payload["sentences"]],
                         ["Elektrik düğmelerini kullanmayın.", "Vanayı kapatın."])
        self.assertEqual(pipeline_v3.salvage_sentences("tamamen bozuk"), {})

    def test_graph_answers_from_a_cut_off_response(self):
        cut = self.CUT

        class Client(FakeClient):
            def create(self, **kwargs):
                if kwargs.get("instructions") == pipeline_v3.ANSWER_PROMPT:
                    return FakeResponse(cut)
                return super().create(**kwargs)

        with mock.patch.object(graph_v3, "retrieve_wide", return_value=[PASSAGE]):
            result = graph_v3.run_graph_v3("doğalgaz kokusu alırsam ne yapmalıyım", Client(), None, [], None, None,
                                           web_search=False)
        self.assertEqual(result["status"], "ok")
        self.assertIn("Elektrik düğmelerini kullanmayın.", result["answer"])


class ConversationTest(unittest.TestCase):
    def test_follow_up_is_made_standalone_and_first_turn_is_untouched(self):
        from src.rag.pipeline_v3 import CONTEXT_PROMPT, contextualize

        class Client:
            def __init__(self):
                self.responses = self
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                assert kwargs["instructions"] == CONTEXT_PROMPT
                return FakeResponse(json.dumps({"standalone": "Antakya'da deprem sonrası hangi hastaneye gidebilirim?"}))

        client = Client()
        self.assertEqual(contextualize("peki hastane?", [], client), "peki hastane?")
        self.assertEqual(client.calls, 0)  # no history, no model call
        history = [{"question": "Antakya'da deprem oldu nereye gideyim?", "answer": "Toplanma alanına gidin."}]
        self.assertEqual(contextualize("peki hastane?", history, client),
                         "Antakya'da deprem sonrası hangi hastaneye gidebilirim?")


class PlaceCarryOverTest(unittest.TestCase):
    def test_place_of_earlier_turn_is_used_when_the_new_question_names_none(self):
        history = [{"question": "antakyadayım deprem oldu", "answer": "Toplanma alanına gidin.", "place": "Antakya"}]
        with mock.patch.object(graph_v3, "retrieve_wide", return_value=[PASSAGE]):
            result = graph_v3.run_graph_v3("depremde yaralandım hangi hastaneye gidebilirim", FakeClient(), None, [],
                                           None, None, history=history)
        self.assertEqual((result["location"].province, result["location"].district), ("Hatay", "Antakya"))
        self.assertEqual(result["turn"]["place"], "Antakya")
        self.assertNotEqual(result["status"], "off_topic")


if __name__ == "__main__":
    unittest.main()
