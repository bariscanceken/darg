"""Pipeline v3 as a LangGraph state graph (the project form's LangChain-based orchestration).

The steps are the measured v3 functions (src/rag/pipeline_v3.py, pipeline_v2.py); LangGraph only
orchestrates them. The hybrid decision agent of the form is the conditional edge after "answer":
stay with the AFAD vector store, or search official web sources and answer again.

    contextualize ─► understand ─► scope ──(off topic)──────────────────────────► finish
                    │
                    └──► retrieve ─► answer ──(enough AFAD evidence)──► finish
                                       ▲  │
                                       │  └──(needs web)──► web_search ──(nothing found)──► finish
                                       └──────────────(found)──────┘
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .pipeline_v2 import card_passages, retrieve_wide, understand_question
from .pipeline_v3 import (
    ANSWER_MODEL,
    contextualize,
    OFF_TOPIC_ANSWER,
    PASSAGES,
    WEB_NOTE,
    in_scope,
    needs_web,
    render,
    web_passages,
    write_answer,
)
from .source_routing import policy_for


class DargState(TypedDict, total=False):
    question: str
    original_question: str
    history: list[dict[str, str]]
    understanding: dict[str, Any]
    corrected: str
    off_topic: bool
    location: Any
    passages: list[dict[str, Any]]
    local_info: dict[str, Any] | None
    header: str
    sentences: list[dict[str, Any]]
    web_tried: bool
    used_web: bool
    answer: str
    status: str


def build_graph(client: Any, index: Any, metadata: list[dict[str, Any]], embedding_model: Any, reranker: Any,
                local_info_builder: Any = None, answer_model: str = ANSWER_MODEL, web_search: bool = True) -> Any:
    """Compile the v3 graph around already loaded retrieval components and an OpenAI client."""

    def contextualize_question(state: DargState) -> DargState:
        from .location import detect_location

        history = state.get("history") or []
        standalone = contextualize(state["question"], history, client)
        update: DargState = {"original_question": state["question"], "question": standalone}
        if state.get("location") is None and detect_location(standalone) is None:
            # The conversation's place carries over in code, not only through the rewrite.
            place = next((turn.get("place") for turn in reversed(history) if turn.get("place")), None)
            if place:
                update["location"] = detect_location(place)
        return update

    def understand(state: DargState) -> DargState:
        understanding = understand_question(state["question"], client)
        return {"understanding": understanding, "corrected": understanding["corrected"]}

    def scope(state: DargState) -> DargState:
        previous = next((turn["question"] for turn in reversed(state.get("history") or [])
                         if turn.get("on_topic", True)), None)
        return {"off_topic": not in_scope(state["corrected"], client, previous=previous)}

    def retrieve(state: DargState) -> DargState:
        from .location import detect_location

        understanding = state["understanding"]
        location = state.get("location") or detect_location(state["corrected"])
        search_location = location if location is not None and not location.is_ambiguous else None
        passages = retrieve_wide(state["corrected"], understanding["search_queries"], search_location,
                                 policy_for(understanding["type"]), index, metadata, embedding_model, reranker,
                                 top_k=PASSAGES)
        local_info = (local_info_builder(location, state["corrected"], understanding["type"])
                      if local_info_builder else None)
        header = f"Soru: {state['corrected']}\nSorunun durumu: {understanding['situation']}"
        if understanding.get("setting"):
            header += f"; ortam: {understanding['setting']}"
        return {"location": location, "passages": passages + card_passages(local_info), "local_info": local_info,
                "header": header}

    def answer(state: DargState) -> DargState:
        return {"sentences": write_answer(state["header"], state["passages"], client, answer_model)}

    def search_web(state: DargState) -> DargState:
        found = web_passages(state["corrected"], client) or web_passages(state["corrected"], client)  # one retry
        if not found:
            return {"web_tried": True}
        return {"web_tried": True, "used_web": True, "passages": state["passages"] + found}

    def finish(state: DargState) -> DargState:
        if state.get("off_topic"):
            return {"answer": OFF_TOPIC_ANSWER, "status": "off_topic", "sentences": [], "passages": []}
        sentences = state.get("sentences", [])
        passages = state.get("passages", [])
        cites_web = any(str(passages[n - 1].get("chunk_id", "")).startswith("web:")
                        for sentence in sentences for n in sentence["sources"])
        text = render(sentences) + (f"\n\n{WEB_NOTE}" if cites_web else "")
        return {"answer": text, "status": "ok" if sentences else "empty"}

    # Conditional edges: the routing decisions, including the hybrid decision agent.
    def after_scope(state: DargState) -> str:
        return "finish" if state["off_topic"] else "retrieve"

    def after_answer(state: DargState) -> str:
        if web_search and not state.get("web_tried") and needs_web(state["corrected"], state["sentences"]):
            return "web_search"
        return "finish"

    def after_web(state: DargState) -> str:
        return "answer" if state.get("used_web") else "finish"

    graph = StateGraph(DargState)
    for name, node in [("contextualize", contextualize_question), ("understand", understand), ("scope", scope), ("retrieve", retrieve), ("answer", answer),
                       ("web_search", search_web), ("finish", finish)]:
        graph.add_node(name, node)
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "understand")
    graph.add_edge("understand", "scope")
    graph.add_conditional_edges("scope", after_scope, {"retrieve": "retrieve", "finish": "finish"})
    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges("answer", after_answer, {"web_search": "web_search", "finish": "finish"})
    graph.add_conditional_edges("web_search", after_web, {"answer": "answer", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


def run_graph_v3(question: str, client: Any, index: Any, metadata: list[dict[str, Any]], embedding_model: Any,
                 reranker: Any, location: Any = None, local_info_builder: Any = None,
                 answer_model: str = ANSWER_MODEL, web_search: bool = True,
                 history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Same result shape as pipeline_v3.run_pipeline_v3; history = earlier turns of the conversation."""
    app = build_graph(client, index, metadata, embedding_model, reranker, local_info_builder, answer_model, web_search)
    state = app.invoke({"question": question, "location": location, "history": history or []})
    return {
        "question": question,
        "standalone_question": state["question"],
        "understanding": state["understanding"],
        "location": state.get("location"),
        "passages": state.get("passages", []),
        "local_info": state.get("local_info"),
        "sentences": state.get("sentences", []),
        "used_web": state.get("used_web", False),
        "answer": state["answer"],
        "status": state["status"],
        "turn": history_turn(state),
    }


def history_turn(state: DargState) -> dict[str, Any]:
    """What the next turn of the conversation needs to remember from this one."""
    location = state.get("location")
    place = "" if location is None or location.is_ambiguous else (location.district or location.province or "")
    return {"question": state["question"], "answer": state["answer"], "place": place,
            "on_topic": state.get("status") != "off_topic"}
