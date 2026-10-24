"""도안 변환 파이프라인 (LangGraph 멀티 에이전트).

            ┌─▶ [번역 에이전트] ─────────────────────────────▶ END
[추출] ─────┤
            └─▶ [파서 에이전트] ─▶ [코 수 검증 Tool] ─┬─▶ [차트 렌더 Tool] ─▶ END
                      ▲                              │
                      └──── [수정 루프: 오류 시 최대 2회] ┘

- 추출: PDF 텍스트 추출, 스캔본·이미지는 GPT-4o 비전으로 판독
- 번역(Structured Output: KoreanPattern) / 파서(Structured Output: PatternSpec)는 병렬 실행
- 검증 결과 오류가 있으면 오류 내용을 피드백으로 파서에게 돌려보내 자기 수정(self-correction)
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ..config import get_callbacks, get_llm
from ..core import glossary
from ..core.chart import render_chart
from ..core.stitch_counter import validate
from ..schemas import KoreanPattern, PatternSpec
from . import prompts

MAX_TEXT = 15000
MAX_REPAIR = 2


class PatternState(TypedDict, total=False):
    filename: str
    raw_text: str
    images: list[str]
    craft_hint: str  # auto | knit | crochet
    korean: dict
    spec: dict
    report: dict
    attempts: int
    feedback: str
    chart_png: bytes
    log: Annotated[list[str], operator.add]


def _glossary_context(text: str, craft: str | None) -> str:
    """도안에 실제로 등장하는 약어만 골라 용어집 발췌 (RAG: 키워드 매칭 검색)."""
    low = f" {text.lower()} "
    lines = []
    for e in glossary.load_glossary():
        if craft in ("knit", "crochet") and e["craft"] != craft:
            continue
        keys = [e["code"], *e.get("aliases", [])]
        if any(f" {k.lower()}" in low or f"{k.lower()}," in low or f"{k.lower()} " in low for k in keys if k.isascii() and len(k) >= 2) or e["ko"] in text:
            lines.append(f"- {e['code']} = {e['ko']} ({e['en']}): {e['desc']}")
    return "\n".join(lines) or "(도안에서 용어집 약어를 찾지 못함)"


def extract_node(state: PatternState) -> dict:
    text = (state.get("raw_text") or "").strip()
    images = state.get("images") or []
    if images and len(text) < 80:
        llm = get_llm(0.0)
        content: list[Any] = [{"type": "text", "text": "다음 도안 이미지를 텍스트 도안으로 옮겨 주세요."}]
        content += [{"type": "image_url", "image_url": {"url": u}} for u in images[:4]]
        res = llm.invoke([SystemMessage(content=prompts.EXTRACT_VISION_SYSTEM), HumanMessage(content=content)])
        text = res.text
        return {"raw_text": text[:MAX_TEXT], "log": [f"🔎 추출 에이전트: 이미지 {len(images)}장을 비전 모델로 판독했습니다."]}
    return {"raw_text": text[:MAX_TEXT], "log": [f"🔎 추출 에이전트: 텍스트 {len(text):,}자를 읽었습니다."]}


def translate_node(state: PatternState) -> dict:
    craft = state.get("craft_hint")
    craft = craft if craft in ("knit", "crochet") else None
    llm = get_llm(0.2).with_structured_output(KoreanPattern, method="function_calling")
    hint = f"\n[사용자 지정 공예] {'대바늘' if craft == 'knit' else '코바늘'}" if craft else ""
    try:
        res: KoreanPattern = llm.invoke([
            SystemMessage(content=prompts.TRANSLATOR_SYSTEM + "\n[용어집]\n" + _glossary_context(state["raw_text"], craft)),
            HumanMessage(content=f"[도안 원문]{hint}\n{state['raw_text']}"),
        ])
    except Exception as e:  # 번역 실패해도 파서 결과로 한국어 도안을 만들 수 있으므로 파이프라인은 계속
        return {"log": [f"🌏 번역 에이전트: 실패({type(e).__name__}) → 구조화 결과로 한국어 도안 대체"]}
    return {"korean": res.model_dump(), "log": [f"🌏 번역 에이전트: 한국어 도안 작성 완료 ({res.terminology} 용어 체계)"]}


def parse_node(state: PatternState) -> dict:
    craft = state.get("craft_hint")
    craft = craft if craft in ("knit", "crochet") else None
    attempts = state.get("attempts", 0)
    llm = get_llm(0.0).with_structured_output(PatternSpec, method="function_calling")
    hint = f"\n[사용자 지정 공예] {craft}" if craft else ""
    messages = [
        SystemMessage(content=prompts.PARSER_SYSTEM + "\n[도안 속 약어 용어집]\n" + _glossary_context(state["raw_text"], craft)),
        HumanMessage(content=f"[도안 원문]{hint}\n{state['raw_text']}"),
    ]
    if state.get("feedback"):
        messages.append(HumanMessage(content=state["feedback"]))
    spec: PatternSpec = llm.invoke(messages)
    label = "구조화 완료" if attempts == 0 else f"검증 피드백 반영 재구조화 ({attempts}회차)"
    return {"spec": spec.model_dump(), "log": [f"🧩 파서 에이전트: {label} - {len(spec.rows)}개 단 묶음"]}


def validate_node(state: PatternState) -> dict:
    spec = PatternSpec(**state["spec"])
    report = validate(spec)
    attempts = state.get("attempts", 0)
    if report.errors:
        msg = f"🧮 코 수 검증 Tool: 오류 {len(report.errors)}건 발견"
        feedback = prompts.REPAIR_TEMPLATE.format(errors="\n".join(report.errors[:15]), spec=json.dumps(state["spec"], ensure_ascii=False)[:8000])
    else:
        msg = f"🧮 코 수 검증 Tool: {len(report.rows)}단 모두 통과 (마지막 {report.final_count}코)"
        feedback = ""
    return {"report": report.to_dict(), "feedback": feedback, "attempts": attempts + 1, "log": [msg]}


def route_after_validate(state: PatternState) -> str:
    if state.get("feedback") and state.get("attempts", 0) <= MAX_REPAIR:
        return "parse"
    return "chart"


def chart_node(state: PatternState) -> dict:
    from ..core.stitch_counter import ValidationReport

    spec = PatternSpec(**state["spec"])
    report = ValidationReport.from_dict(state["report"])
    png = render_chart(report, spec.craft, spec.title)
    return {"chart_png": png, "log": ["🎨 차트 렌더 Tool: 기호 차트 생성 완료"]}


def build_pattern_graph():
    g = StateGraph(PatternState)
    g.add_node("extract", extract_node)
    g.add_node("translate", translate_node)
    g.add_node("parse", parse_node)
    g.add_node("validate", validate_node)
    g.add_node("chart", chart_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "translate")  # 병렬 분기 1
    g.add_edge("extract", "parse")      # 병렬 분기 2
    g.add_edge("translate", END)
    g.add_edge("parse", "validate")
    g.add_conditional_edges("validate", route_after_validate, {"parse": "parse", "chart": "chart"})
    g.add_edge("chart", END)
    return g.compile()


def run_pattern_pipeline(raw_text: str, images: list[str] | None = None, craft_hint: str = "auto", filename: str = "") -> PatternState:
    graph = build_pattern_graph()
    return graph.invoke(
        {"filename": filename, "raw_text": raw_text, "images": images or [], "craft_hint": craft_hint, "attempts": 0, "log": []},
        config={"callbacks": get_callbacks(), "run_name": "pattern_pipeline"},
    )
