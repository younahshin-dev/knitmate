"""구조화 출력(Structured Output)에 쓰는 Pydantic 스키마.

도안은 LLM이 '구조'만 만들고, 코 수 계산은 코드(stitch_counter)가 한다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class StitchOp(BaseModel):
    """뜨기 동작 1종류. 예: 'k2' -> code='k', count=2"""

    code: str = Field(description="용어집 코드. 대바늘: co,k,p,ktbl,ptbl,sl,yo,m1,kfb,k2tog,ssk,p2tog,k3tog,sk2p,s2kp,bo,pm,sm / 코바늘: ch,slst,sc,hdc,dc,tr,inc,dec,mr,sk,flo,blo")
    count: int = Field(default=1, ge=1, description="같은 동작을 연속으로 반복하는 횟수 (k3 -> 3)")
    consumes: Optional[int] = Field(default=None, description="이 동작 1회가 이전 단의 코를 몇 개 쓰는지. 기본값과 다를 때만 지정 (예: 매직링 안에 뜨는 sc=0, 단을 닫는 sl st=0, 1코로 세는 기둥사슬 ch3=1)")
    produces: Optional[int] = Field(default=None, description="이 동작 1회가 새로 만드는 코 수. 기본값과 다를 때만 지정 (예: 코로 세지 않는 기둥사슬=0, 기둥사슬 ch3를 1코로 세면 1)")
    note: Optional[str] = Field(default=None, description="원문 표기 또는 보충 설명 (예: 'ch 3 counts as dc')")


class Segment(BaseModel):
    """반복 묶음. *...* 이나 [..] x3 을 표현한다."""

    ops: list[StitchOp]
    times: int = Field(default=1, ge=1, description="반복 횟수 (반복이 아니면 1)")
    repeat_to_end: bool = Field(default=False, description="'rep to end' 또는 'rep to last N sts'처럼 반복 횟수를 남은 코 수로 정하는 경우 true. 한 단에 하나만")


class PatternRow(BaseModel):
    row_start: int = Field(description="단 번호 시작")
    row_end: int = Field(description="단 번호 끝 (한 단이면 row_start와 같음). 같은 지시가 연속될 때만 범위로 묶는다")
    kind: Literal["row", "round"] = "row"
    side: Literal["RS", "WS", "any"] = "any"
    segments: list[Segment]
    stated_count: Optional[int] = Field(default=None, description="도안에 적힌 단 종료 후 코 수 (예: '(24 sts)'). 없으면 null")
    instruction_ko: str = Field(description="이 단의 한국어 설명 한 줄")


class PatternSpec(BaseModel):
    reasoning: str = Field(description="분석 과정: 공예 종류, 용어 체계, 시작 코 수, 반복 구조 해석을 단계적으로 적는다")
    title: str
    craft: Literal["knit", "crochet"]
    terminology: Literal["US", "UK", "KR", "unknown"] = "unknown"
    cast_on: int = Field(description="1단을 뜨기 전 바늘에 있는 코 수. 대바늘=코잡기 수, 코바늘 기초사슬=실제 코로 쓰는 사슬 수, 매직링 시작=0")
    start_method: str = Field(description="시작 방법 한국어 설명 (예: '손가락 코잡기 15코', '원형코(매직링)')")
    rows: list[PatternRow]
    finishing_ko: Optional[str] = Field(default=None, description="마무리 방법 한국어 설명")
    notes: list[str] = Field(default_factory=list, description="해석상 주의점, 모호한 부분")


class KoreanPattern(BaseModel):
    """번역 에이전트 출력"""

    title_ko: str
    craft: Literal["knit", "crochet"]
    terminology: Literal["US", "UK", "KR", "unknown"]
    materials_ko: list[str] = Field(default_factory=list, description="실, 바늘, 부자재")
    gauge_ko: Optional[str] = None
    abbreviations: list[dict] = Field(default_factory=list, description="[{'abbr':'k2tog','ko':'왼코 겹쳐 2코 모아뜨기'}] 도안에 나온 약어만")
    pattern_md: str = Field(description="한국어 도안 본문 (마크다운). 단마다 한 줄, 코 수 표기 유지")
    translator_notes: list[str] = Field(default_factory=list, description="번역/해석 시 주의할 점")


class Route(BaseModel):
    intent: Literal["technique", "progress", "pattern", "gauge", "chitchat"] = Field(
        description="technique: 기법·용어·뜨는 방법/영상 질문, progress: 현재 프로젝트 진행·코 수 확인·단 기록, pattern: 업로드한 도안 내용에 관한 질문, gauge: 게이지·코 수 환산·실 양 계산, chitchat: 그 외 인사/잡담"
    )
    reason: str = Field(description="분류 이유 한 줄")


class Diagnosis(BaseModel):
    """진행 코치의 코 수 불일치 진단"""

    summary: str = Field(description="한두 문장 요약")
    likely_causes: list[str] = Field(description="가능성 높은 원인 순서대로")
    how_to_fix: list[str] = Field(description="바로 해볼 수 있는 해결 방법")
    check_rows: list[int] = Field(default_factory=list, description="다시 확인해 볼 단 번호")
