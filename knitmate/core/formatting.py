"""결과 문서(한국어 도안 마크다운) 생성."""
from __future__ import annotations

from . import glossary


def korean_from_spec(spec: dict) -> dict:
    """LLM 없이 구조화 도안만으로 한국어 도안을 만든다 (데모/폴백)."""
    used = []
    for r in spec["rows"]:
        for seg in r["segments"]:
            for op in seg["ops"]:
                if op["code"] not in used:
                    used.append(op["code"])
    lines = [f"- **시작**: {spec.get('start_method', '')}"]
    for r in spec["rows"]:
        rng = f"{r['row_start']}단" if r["row_start"] == r["row_end"] else f"{r['row_start']}~{r['row_end']}단"
        side = {"RS": "(겉면)", "WS": "(안면)"}.get(r.get("side", "any"), "")
        cnt = f" ({r['stated_count']}코)" if r.get("stated_count") else ""
        instr = r["instruction_ko"].replace("(겉면) ", "").replace("(안면) ", "")
        lines.append(f"- **{rng}{side}**: {instr}{cnt}")
    if spec.get("finishing_ko"):
        lines.append(f"- **마무리**: {spec['finishing_ko']}")
    return {
        "title_ko": spec["title"],
        "craft": spec["craft"],
        "terminology": spec.get("terminology", "unknown"),
        "materials_ko": [],
        "gauge_ko": None,
        "abbreviations": [{"abbr": c, "ko": glossary.ko_name(c)} for c in used],
        "pattern_md": "\n".join(lines),
        "translator_notes": spec.get("notes", []),
    }


def korean_markdown(korean: dict, report: dict | None = None) -> str:
    craft = "대바늘" if korean.get("craft") == "knit" else "코바늘"
    out = [f"# {korean.get('title_ko', '도안')}", "", f"- 공예: {craft}", f"- 용어 체계: {korean.get('terminology', '-')}"]
    if korean.get("materials_ko"):
        out += ["", "## 준비물", *[f"- {m}" for m in korean["materials_ko"]]]
    if korean.get("gauge_ko"):
        out += ["", "## 게이지", korean["gauge_ko"]]
    if korean.get("abbreviations"):
        out += ["", "## 약어", "| 약어 | 한국어 |", "|---|---|"]
        out += [f"| {a.get('abbr', '')} | {a.get('ko', '')} |" for a in korean["abbreviations"]]
    out += ["", "## 도안", korean.get("pattern_md", "")]
    if korean.get("translator_notes"):
        out += ["", "## 참고", *[f"- {n}" for n in korean["translator_notes"]]]
    if report:
        out += ["", "## 단별 코 수 (자동 검증)", "| 단 | 시작 코 | 완료 코 | 검증 |", "|---|---|---|---|"]
        mark = {"ok": "✓", "warn": "△", "error": "✗"}
        out += [f"| {r['row']} | {r['start_count']} | {r['produced']} | {mark[r['status']]} |" for r in report["rows"]]
    out += ["", "_뜨개메이트(KnitMate)로 변환한 도안입니다. 원 도안의 저작권은 원작자에게 있습니다._"]
    return "\n".join(out)
