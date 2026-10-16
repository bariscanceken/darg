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


if __name__ == "__main__":
    unittest.main()
