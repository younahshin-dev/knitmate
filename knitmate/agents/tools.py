"""에이전트가 호출하는 LangChain Tool 모음 (Function Calling).

계산·검색·영상 링크처럼 '틀리면 안 되는 일'은 모두 Tool이 맡는다.
"""
from __future__ import annotations

import json
import math

from langchain_core.tools import tool

from ..core import glossary, storage
from ..core.rag import format_hits, search_knowledge as _search
from ..core.stitch_counter import ValidationReport, compare_progress, suspicious_rows
from ..core.videos import find_videos as _find_videos


# ---------- 공통 지식 Tool ----------

@tool
def search_knowledge(query: str, craft: str = "any") -> str:
    """뜨개 기법·도안 읽기·실/바늘 지식 문서를 검색한다(RAG). 뜨는 방법, 용어 의미, 자주 하는 실수를 답하기 전에 반드시 먼저 호출한다.
    craft: 'knit'(대바늘), 'crochet'(코바늘), 'any'"""
    hits = _search(query, k=4, craft=None if craft == "any" else craft)
    return format_hits(hits) if hits else "관련 문서를 찾지 못했습니다."


@tool
def lookup_abbreviation(term: str) -> str:
    """도안 약어(예: k2tog, ssk, sc, dc, yo)나 한국어 용어의 정확한 뜻과 코 수 규칙을 용어집에서 찾는다."""
    found = glossary.lookup(term)
    if not found:
        return f"용어집에 '{term}'이 없습니다."
    return json.dumps(
        [{"약어": e["code"], "한국어": e["ko"], "영문": e["en"], "공예": "대바늘" if e["craft"] == "knit" else "코바늘",
          "설명": e["desc"], "코수규칙": f"{e['consumes']}코 사용 → {e['produces']}코 생성"} for e in found],
        ensure_ascii=False,
    )


@tool
def find_videos(technique: str, craft: str = "any") -> str:
    """기법 이름으로 참고 영상을 찾는다. 큐레이션 목록 → YouTube API → YouTube 검색 링크 순서로 대체된다.
    답변에 영상 링크를 넣을 때는 이 Tool 결과의 url만 사용한다."""
    res = _find_videos(technique, None if craft == "any" else craft)
    return json.dumps(res, ensure_ascii=False)


# ---------- 계산 Tool ----------

@tool
def gauge_convert(pattern_sts_per_10cm: float, my_sts_per_10cm: float, pattern_stitch_count: int) -> str:
    """도안 게이지와 내 게이지가 다를 때, 도안 코 수를 내 게이지 기준 코 수로 환산한다(같은 폭 유지)."""
    width_cm = pattern_stitch_count / pattern_sts_per_10cm * 10
    mine = width_cm * my_sts_per_10cm / 10
    return json.dumps({"도안 기준 폭(cm)": round(width_cm, 1), "내 게이지 필요 코 수(소수)": round(mine, 1),
                       "권장 코 수(반올림)": round(mine)}, ensure_ascii=False)


@tool
def stitches_for_width(width_cm: float, sts_per_10cm: float, multiple: int = 1, plus: int = 0) -> str:
    """원하는 폭(cm)과 게이지로 필요한 코 수를 계산한다. 무늬 반복 단위(multiple)와 추가 코(plus)를 맞춰 가장 가까운 코 수를 제안한다."""
    raw = width_cm * sts_per_10cm / 10
    if multiple > 1:
        k = max(1, round((raw - plus) / multiple))
        adj = k * multiple + plus
    else:
        adj = round(raw)
    return json.dumps({"계산값": round(raw, 1), "추천 코 수": adj, "실제 폭(cm)": round(adj / sts_per_10cm * 10, 1)}, ensure_ascii=False)


@tool
def yarn_balls_needed(total_meters: float, meters_per_ball: float, spare_ratio: float = 0.1) -> str:
    """필요한 실 길이와 1볼 길이로 필요한 볼 수를 계산한다(여유분 비율 포함)."""
    need = total_meters * (1 + spare_ratio)
    return json.dumps({"여유 포함 길이(m)": round(need), "필요 볼 수": math.ceil(need / meters_per_ball)}, ensure_ascii=False)


COMMON_TOOLS = [search_knowledge, lookup_abbreviation, find_videos]
GAUGE_TOOLS = [gauge_convert, stitches_for_width, yarn_balls_needed, search_knowledge]


# ---------- 프로젝트(메모리) Tool: 프로젝트 ID를 클로저로 묶어서 생성 ----------

def make_project_tools(project_id: str | None):
    def _project():
        if not project_id:
            return None, None
        p = storage.get_project(project_id)
        if not p:
            return None, None
        return p, ValidationReport.from_dict(p["report"])

    @tool
    def get_project_status() -> str:
        """현재 뜨고 있는 프로젝트의 이름, 현재 단, 그 단의 지시와 기대 코 수, 최근 기록을 가져온다."""
        p, rep = _project()
        if not p:
            return "선택된 프로젝트가 없습니다. '도안 변환' 탭에서 도안을 변환한 뒤 프로젝트를 시작하라고 안내하세요."
        r = rep.get_row(p["current_row"])
        return json.dumps({
            "프로젝트": p["name"], "공예": p["craft"], "총 단 수": len(rep.rows), "현재 단": p["current_row"],
            "현재 단 지시": r.instruction_ko if r else "도안 범위를 벗어남",
            "현재 단 시작 코 수": r.start_count if r else None,
            "현재 단 완료 시 기대 코 수": r.produced if r else None,
            "최근 기록": storage.get_logs(p["id"], 5),
        }, ensure_ascii=False)

    @tool
    def get_pattern_rows(start_row: int, end_row: int) -> str:
        """프로젝트 도안의 특정 단 범위 지시(한국어)와 단별 코 수 계산 결과를 가져온다."""
        p, rep = _project()
        if not p:
            return "선택된 프로젝트가 없습니다."
        rows = [r for r in rep.rows if start_row <= r.row <= end_row][:20]
        return json.dumps([{"단": r.row, "면": r.side, "지시": r.instruction_ko, "시작 코": r.start_count,
                            "완료 코": r.produced, "검증": r.status} for r in rows], ensure_ascii=False)

    @tool
    def check_stitch_count(row: int, actual_count: int) -> str:
        """사용자가 row단을 끝내고 센 코 수(actual_count)를 도안 기대값과 비교하고 기록한다. 다르면 원인 후보 단을 함께 돌려준다."""
        p, rep = _project()
        if not p:
            return "선택된 프로젝트가 없습니다."
        res = compare_progress(rep, row, actual_count)
        storage.add_log(p["id"], row, actual_count, res.get("expected"), res["status"], "채팅에서 확인")
        if res["status"] == "mismatch":
            res["원인 후보 단"] = suspicious_rows(rep, row, res["diff"])
        elif res["status"] == "ok":
            storage.set_current_row(p["id"], row + 1)
            res["다음 단"] = row + 1
        return json.dumps(res, ensure_ascii=False)

    @tool
    def set_current_row(row: int) -> str:
        """사용자가 지금 뜨고 있는 단 번호를 저장한다 (예: '지금 7단 뜨는 중')."""
        p, _ = _project()
        if not p:
            return "선택된 프로젝트가 없습니다."
        storage.set_current_row(p["id"], row)
        return f"현재 단을 {row}단으로 저장했습니다."

    return [get_project_status, get_pattern_rows, check_stitch_count, set_current_row]
