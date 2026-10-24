"""에이전트 그래프 배선 테스트: 가짜 LLM으로 LangGraph 흐름(병렬·수정 루프·ReAct·라우팅)을 검증한다."""
import json

from langchain_core.messages import AIMessage, ToolMessage

from knitmate.config import SAMPLE_DIR
from knitmate.core.formatting import korean_from_spec
from knitmate.schemas import Diagnosis, KoreanPattern, PatternSpec, Route

GOOD = json.loads((SAMPLE_DIR / "round_coaster_crochet_en.spec.json").read_text(encoding="utf-8"))


class _Structured:
    def __init__(self, parent, schema):
        self.parent, self.schema = parent, schema

    def invoke(self, messages, config=None):
        p = self.parent
        if self.schema is PatternSpec:
            p.parse_calls += 1
            spec = json.loads(json.dumps(GOOD))
            if p.parse_calls == 1:  # 첫 시도는 일부러 틀린 구조 → 검증 Tool이 잡아야 함
                spec["rows"][2]["segments"][1]["ops"][0]["count"] = 2
            return PatternSpec(**spec)
        if self.schema is KoreanPattern:
            return KoreanPattern(**korean_from_spec(GOOD))
        if self.schema is Route:
            return Route(intent=p.route, reason="test")
        if self.schema is Diagnosis:
            return Diagnosis(summary="s", likely_causes=["a"], how_to_fix=["b"], check_rows=[1])
        raise AssertionError(self.schema)


class _Tooled:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, messages, config=None):
        if not any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="", tool_calls=[
                {"name": "search_knowledge", "args": {"query": "한길긴뜨기", "craft": "crochet"}, "id": "c1"},
                {"name": "find_videos", "args": {"technique": "한길긴뜨기", "craft": "crochet"}, "id": "c2"},
            ])
        url = json.loads([m for m in messages if isinstance(m, ToolMessage) and m.name == "find_videos"][0].content)["videos"][0]["url"]
        return AIMessage(content=f"### 한길긴뜨기 (dc)\n영상: {url}\n가짜: https://www.youtube.com/watch?v=FAKE123")


class FakeLLM:
    def __init__(self):
        self.parse_calls = 0
        self.route = "technique"

    def with_structured_output(self, schema, method=None):
        return _Structured(self, schema)

    def bind_tools(self, tools):
        return _Tooled(self)

    def invoke(self, messages, config=None):
        return AIMessage(content="안녕하세요!")


def _patch(monkeypatch):
    fake = FakeLLM()
    from knitmate.agents import coach_graph, diagnosis, pattern_graph, react

    for mod in (pattern_graph, coach_graph, react, diagnosis):
        monkeypatch.setattr(mod, "get_llm", lambda *a, **k: fake)
    return fake


def test_pattern_pipeline_repair_loop(monkeypatch):
    fake = _patch(monkeypatch)
    from knitmate.agents.pattern_graph import run_pattern_pipeline

    out = run_pattern_pipeline("Rnd 1: ...", craft_hint="crochet")
    assert fake.parse_calls == 2  # 오류 → 피드백 → 재파싱
    assert out["report"]["errors"] == []
    assert out["chart_png"][:4] == b"\x89PNG"
    assert out["korean"]["craft"] == "crochet"
    assert any("재구조화" in line for line in out["log"])


def test_coach_react_and_url_guard(monkeypatch):
    _patch(monkeypatch)
    from knitmate.agents.coach_graph import ask_coach, build_coach_graph

    g = build_coach_graph()
    res = ask_coach(g, "t1", "한길긴뜨기 어떻게 떠?", "crochet")
    assert res["intent"] == "technique"
    assert "FAKE123" not in res["answer"]          # 지어낸 링크 제거
    assert res["videos"][0]["url"] in res["answer"]  # 도구가 준 링크는 유지
    assert "search_knowledge" in res["tools_used"]
    # 메모리: 같은 thread에 두 번째 질문 → 이전 대화가 상태에 남아 있음
    ask_coach(g, "t1", "고마워", "crochet")
    state = g.get_state({"configurable": {"thread_id": "t1"}})
    assert len(state.values["messages"]) == 4
