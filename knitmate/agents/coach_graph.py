"""뜨개 코치 대화 그래프 (Supervisor/Router 기반 멀티 에이전트 + 대화 메모리).

                    ┌─▶ 기법 코치 (RAG + 영상 Tool, ReAct)
사용자 ─▶ [라우터] ──┼─▶ 진행 코치 (프로젝트 메모리 Tool, ReAct)
 (Structured Output)├─▶ 도안 해설가 (도안/용어 Tool, ReAct)
                    ├─▶ 계산 도우미 (게이지·실 양 계산 Tool, ReAct)
                    └─▶ 안내 (일반 대화)

- 대화 메모리: LangGraph 체크포인터(thread_id별)로 이전 대화를 기억
- 장기 메모리: 프로젝트 진행 상황은 SQLite(storage)에 저장되고 진행 코치가 Tool로 읽고 씀
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..config import get_callbacks, get_llm
from ..schemas import Route
from . import prompts
from .react import build_react_agent, collect_tool_outputs, guard_urls
from .tools import COMMON_TOOLS, GAUGE_TOOLS, make_project_tools

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # 구버전 호환
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

HISTORY_WINDOW = 12  # 에이전트에 넘길 최근 대화 수 (토큰 절약)


class CoachState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    craft: str          # knit | crochet | any
    project_id: str | None
    videos: list[dict]
    sources: list[str]
    tools_used: list[str]


AGENT_LABELS = {
    "technique": "🧶 기법 코치",
    "progress": "📏 진행 코치",
    "pattern": "📜 도안 해설가",
    "gauge": "🧮 계산 도우미",
    "chitchat": "💬 안내",
}


def _context_line(state: CoachState) -> str:
    craft = {"knit": "대바늘", "crochet": "코바늘"}.get(state.get("craft") or "", "지정 안 함")
    pid = state.get("project_id")
    return f"\n[현재 상황] 사용자 선호 공예: {craft} / 진행 중 프로젝트: {'있음 (ID ' + pid + ')' if pid else '없음'}"


def router_node(state: CoachState) -> dict:
    llm = get_llm(0.0).with_structured_output(Route, method="function_calling")
    recent = state["messages"][-4:]
    route: Route = llm.invoke([SystemMessage(content=prompts.ROUTER_SYSTEM + _context_line(state)), *recent])
    intent = route.intent
    if intent in ("progress", "pattern") and not state.get("project_id"):
        intent = "technique" if intent == "pattern" else intent  # 진행 코치는 프로젝트가 없다고 안내해 준다
    return {"intent": intent}


def _run_agent(state: CoachState, system: str, tools: list) -> dict:
    agent = build_react_agent(system + _context_line(state), tools)
    history = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-HISTORY_WINDOW:]
    result = agent.invoke({"messages": history, "steps": 0}, config={"recursion_limit": 25})
    new_msgs = result["messages"][len(history):]
    extra = collect_tool_outputs(new_msgs)
    final = result["messages"][-1]
    answer = guard_urls(final.text,  # .text: 블록 리스트로 오는 모델(Gemini)도 본문 텍스트만 추출
                         [v["url"] for v in extra["videos"]])
    return {
        "messages": [AIMessage(content=answer, name=state.get("intent", "agent"))],
        "videos": extra["videos"],
        "sources": extra["sources"],
        "tools_used": extra["tools_used"],
    }


def technique_node(state: CoachState) -> dict:
    return _run_agent(state, prompts.TECHNIQUE_SYSTEM, COMMON_TOOLS)


def progress_node(state: CoachState) -> dict:
    return _run_agent(state, prompts.PROGRESS_SYSTEM, make_project_tools(state.get("project_id")) + COMMON_TOOLS[:2])


def pattern_node(state: CoachState) -> dict:
    return _run_agent(state, prompts.PATTERN_QA_SYSTEM, make_project_tools(state.get("project_id"))[:2] + COMMON_TOOLS)


def gauge_node(state: CoachState) -> dict:
    return _run_agent(state, prompts.GAUGE_SYSTEM, GAUGE_TOOLS)


def chitchat_node(state: CoachState) -> dict:
    history = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-6:]
    res = get_llm(0.5).invoke([SystemMessage(content=prompts.CHITCHAT_SYSTEM), *history])
    return {"messages": [AIMessage(content=res.content, name="chitchat")], "videos": [], "sources": [], "tools_used": []}


def pick_agent(state: CoachState) -> Literal["technique", "progress", "pattern", "gauge", "chitchat"]:
    return state.get("intent", "chitchat")  # type: ignore[return-value]


def build_coach_graph(checkpointer=None):
    g = StateGraph(CoachState)
    g.add_node("router", router_node)
    g.add_node("technique", technique_node)
    g.add_node("progress", progress_node)
    g.add_node("pattern", pattern_node)
    g.add_node("gauge", gauge_node)
    g.add_node("chitchat", chitchat_node)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", pick_agent, {k: k for k in AGENT_LABELS})
    for k in AGENT_LABELS:
        g.add_edge(k, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def ask_coach(graph, thread_id: str, question: str, craft: str = "any", project_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id}, "callbacks": get_callbacks(), "run_name": "coach_chat"}
    out = graph.invoke(
        {"messages": [HumanMessage(content=question)], "craft": craft, "project_id": project_id},
        config=config,
    )
    return {
        "answer": out["messages"][-1].text,
        "intent": out.get("intent", "chitchat"),
        "agent": AGENT_LABELS.get(out.get("intent", "chitchat"), ""),
        "videos": out.get("videos", []),
        "sources": out.get("sources", []),
        "tools_used": out.get("tools_used", []),
    }
