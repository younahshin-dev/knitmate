"""진행 체크 탭의 코 수 불일치 진단 에이전트 (RAG + Structured Output)."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_callbacks, get_llm
from ..core.rag import format_hits, search_knowledge
from ..core.stitch_counter import ValidationReport, suspicious_rows
from ..schemas import Diagnosis
from . import prompts


def diagnose(project: dict, row: int, actual: int, expected: int) -> Diagnosis:
    report = ValidationReport.from_dict(project["report"])
    diff = actual - expected
    cands = suspicious_rows(report, row, diff)
    query = "코 수가 다를 때 확인 순서 " + ("바늘비우기 모아뜨기" if project["craft"] == "knit" else "기둥 사슬 늘려뜨기 줄여뜨기")
    knowledge = format_hits(search_knowledge(query, k=3, craft=project["craft"]))
    payload = {
        "프로젝트": project["name"],
        "공예": "대바늘" if project["craft"] == "knit" else "코바늘",
        "확인한 단": row,
        "사용자가 센 코 수": actual,
        "기대 코 수": expected,
        "차이": diff,
        "최근 단 정보(원인 후보)": cands,
    }
    llm = get_llm(0.2).with_structured_output(Diagnosis, method="function_calling")
    return llm.invoke(
        [SystemMessage(content=prompts.DIAGNOSIS_SYSTEM),
         HumanMessage(content=f"[상황]\n{json.dumps(payload, ensure_ascii=False)}\n\n[관련 지식]\n{knowledge}")],
        config={"callbacks": get_callbacks(), "run_name": "diagnosis"},
    )


def rule_based_diagnosis(project: dict, row: int, actual: int, expected: int) -> Diagnosis:
    """LLM을 쓸 수 없을 때의 규칙 기반 진단 (폴백)."""
    report = ValidationReport.from_dict(project["report"])
    diff = actual - expected
    cands = suspicious_rows(report, row, diff)
    knit = project["craft"] == "knit"
    if diff > 0:
        causes = (["바늘비우기(yo)를 의도치 않게 한 번 더 했거나, 겉뜨기↔안뜨기 전환 때 실이 바늘 위로 넘어감", "모아뜨기(k2tog/ssk)를 빠뜨림", "단 끝 코를 두 번 뜸"]
                  if knit else ["마지막 코를 기둥 사슬 꼭대기에 한 번 더 뜸", "늘려뜨기 위치를 한 번 더 함", "단을 닫는 빼뜨기 자리에 코를 뜸"])
    else:
        causes = (["바늘비우기(yo)를 빠뜨림", "단 끝 코를 뜨지 않고 넘어감", "모아뜨기를 한 번 더 함"]
                  if knit else ["마지막 코를 기둥 사슬에 뜨지 않음", "늘려뜨기를 빠뜨림", "코를 건너뜀"])
    return Diagnosis(
        summary=f"{row}단 기준 {abs(diff)}코 {'많아요' if diff > 0 else '적어요'}.",
        likely_causes=causes,
        how_to_fix=["표시한 원인 후보 단의 코를 무늬 반복 단위로 끊어서 세어 보세요.",
                    "한 반복 안에서 차이가 나는 위치를 찾으면 그 단만 풀어서 다시 뜹니다.",
                    "무늬에 영향이 적은 1코 차이라면 다음 단 가장자리에서 늘림/줄임 1코로 맞출 수 있어요."],
        check_rows=[c["row"] for c in cands][:4],
    )
