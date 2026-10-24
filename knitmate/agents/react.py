"""범용 ReAct 에이전트 (LangGraph 서브그래프).

[agent: LLM + bind_tools] ⇄ [tools: 도구 실행]  (도구 호출이 없으면 종료, 최대 스텝 제한)
"""
from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..config import get_llm

YOUTUBE_URL = re.compile(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s)\]>\"']+")


class ReactState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


def build_react_agent(system_prompt: str, tools: list, max_steps: int = 5, temperature: float = 0.3):
    tool_map = {t.name: t for t in tools}
    llm = get_llm(temperature).bind_tools(tools) if tools else get_llm(temperature)

    def agent(state: ReactState) -> dict:
        msgs = [SystemMessage(content=system_prompt), *state["messages"]]
        if state["steps"] >= max_steps:
            # 도구 호출 한도 도달: 지금까지 정보로 최종 답변
            res = get_llm(temperature).invoke(msgs + [SystemMessage(content="도구 호출 한도에 도달했다. 지금까지 얻은 정보로 최종 답변을 작성해라.")])
        else:
            res = llm.invoke(msgs)
        return {"messages": [res], "steps": state["steps"] + 1}

    def run_tools(state: ReactState) -> dict:
        last = state["messages"][-1]
        outs = []
        for call in last.tool_calls:
            t = tool_map.get(call["name"])
            try:
                result = t.invoke(call["args"]) if t else f"알 수 없는 도구: {call['name']}"
            except Exception as e:  # 도구 오류도 관찰 결과로 돌려줘 에이전트가 대처하게 한다
                result = f"도구 실행 오류: {e}"
            outs.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"]))
        return {"messages": outs}

    def should_continue(state: ReactState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    g = StateGraph(ReactState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def collect_tool_outputs(messages: list[BaseMessage]) -> dict:
    """도구 결과에서 영상 목록·근거 출처를 모은다 (UI 카드 표시 + URL 검증용)."""
    videos, sources, tools_used = [], [], []
    for m in messages:
        if isinstance(m, ToolMessage):
            tools_used.append(m.name)
            if m.name == "find_videos":
                try:
                    for v in json.loads(m.content).get("videos", []):
                        if v["url"] not in [x["url"] for x in videos]:
                            videos.append(v)
                except Exception:
                    pass
            if m.name == "search_knowledge":
                for s in re.findall(r"\[출처: ([^\]]+)\]", m.content):
                    if s not in sources:
                        sources.append(s)
    return {"videos": videos, "sources": sources, "tools_used": tools_used}


def guard_urls(answer: str, allowed: list[str]) -> str:
    """도구가 주지 않은 YouTube URL은 제거한다 (환각 링크 방지 가드레일)."""
    allowed_set = set(allowed)

    def repl(m: re.Match) -> str:
        url = m.group(0)
        return url if url in allowed_set else "(확인되지 않은 링크 제거됨)"

    return YOUTUBE_URL.sub(repl, answer)
