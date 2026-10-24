"""코 수 검증 엔진 (결정적 Tool).

LLM은 계산을 틀리기 쉬우므로, 구조화된 도안(PatternSpec)을 받아
단마다 사용 코 수 / 생성 코 수 / 반복 횟수를 코드로 계산하고 도안 표기와 대조한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..schemas import PatternSpec, Segment, StitchOp
from . import glossary


@dataclass
class RowResult:
    row: int
    side: str
    kind: str
    instruction_ko: str
    start_count: int
    consumed: int
    produced: int
    stated: int | None
    status: str  # ok | warn | error
    messages: list[str] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)  # 뜨는 순서대로 펼친 코드 (차트용)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    rows: list[RowResult]
    final_count: int
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def expected_count(self, row: int) -> int | None:
        for r in self.rows:
            if r.row == row:
                return r.produced
        return None

    def get_row(self, row: int) -> RowResult | None:
        for r in self.rows:
            if r.row == row:
                return r
        return None

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "final_count": self.final_count,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationReport":
        return cls(
            rows=[RowResult(**r) for r in d["rows"]],
            final_count=d["final_count"],
            errors=d["errors"],
            warnings=d["warnings"],
        )

    def summary_text(self, limit: int = 40) -> str:
        lines = [f"총 {len(self.rows)}단, 마지막 코 수 {self.final_count}코"]
        for r in self.rows[:limit]:
            mark = {"ok": "✓", "warn": "△", "error": "✗"}[r.status]
            stated = f" / 도안 표기 {r.stated}" if r.stated is not None else ""
            msg = f" ({'; '.join(r.messages)})" if r.messages else ""
            lines.append(f"{mark} {r.row}단: {r.start_count}→{r.produced}코{stated}{msg}")
        return "\n".join(lines)


def _op_rule(op: StitchOp) -> tuple[int, int, str | None]:
    rule = glossary.stitch_rule(op.code)
    unknown = None
    if rule is None:
        rule = (1, 1)
        unknown = op.code
    consumes = op.consumes if op.consumes is not None else rule[0]
    produces = op.produces if op.produces is not None else rule[1]
    return consumes, produces, unknown


def _segment_totals(seg: Segment) -> tuple[int, int, list[str]]:
    c = p = 0
    unknown: list[str] = []
    for op in seg.ops:
        oc, opr, unk = _op_rule(op)
        c += oc * op.count
        p += opr * op.count
        if unk:
            unknown.append(unk)
    return c, p, unknown


def _expand(seg: Segment, times: int) -> list[str]:
    seq: list[str] = []
    for _ in range(times):
        for op in seg.ops:
            g = glossary.get(op.code)
            if g and g.get("symbol") == "none":
                continue
            seq.extend([op.code.lower()] * op.count)
    return seq


def validate(spec: PatternSpec) -> ValidationReport:
    results: list[RowResult] = []
    errors: list[str] = []
    warnings: list[str] = []
    prev = spec.cast_on
    expected_next = None
    is_crochet = spec.craft == "crochet"

    for prow in sorted(spec.rows, key=lambda r: r.row_start):
        if prow.row_end < prow.row_start:
            prow.row_end = prow.row_start
        if expected_next is not None and prow.row_start != expected_next:
            warnings.append(f"{expected_next}단 다음에 {prow.row_start}단이 나옵니다. 빠진 단이 없는지 확인하세요.")
        for n in range(prow.row_start, prow.row_end + 1):
            msgs: list[str] = []
            status = "ok"
            fixed_c = fixed_p = 0
            to_end: list[Segment] = []
            seg_times: dict[int, int] = {}
            unknown_all: list[str] = []
            for i, seg in enumerate(prow.segments):
                c, p, unk = _segment_totals(seg)
                unknown_all += unk
                if seg.repeat_to_end:
                    to_end.append(seg)
                else:
                    fixed_c += c * seg.times
                    fixed_p += p * seg.times
                    seg_times[i] = seg.times

            if unknown_all:
                msgs.append(f"모르는 약어: {', '.join(sorted(set(unknown_all)))} (1코 사용·1코 생성으로 가정)")
                status = "warn"

            if len(to_end) > 1:
                msgs.append("'끝까지 반복'이 한 단에 2개 이상입니다.")
                status = "error"
            elif to_end:
                seg = to_end[0]
                idx = prow.segments.index(seg)
                per_c, per_p, _ = _segment_totals(seg)
                remain = prev - fixed_c
                if per_c <= 0:
                    msgs.append("끝까지 반복하는 묶음이 이전 단 코를 사용하지 않아 반복 횟수를 정할 수 없습니다.")
                    status = "error"
                    seg_times[idx] = 1
                elif remain < 0:
                    msgs.append(f"반복 전 고정 지시가 {fixed_c}코를 쓰는데 이전 단은 {prev}코뿐입니다.")
                    status = "error"
                    seg_times[idx] = 0
                elif remain % per_c != 0:
                    msgs.append(f"남은 {remain}코가 반복 단위 {per_c}코로 나누어떨어지지 않습니다.")
                    status = "error"
                    seg_times[idx] = remain // per_c
                else:
                    seg_times[idx] = remain // per_c
                t = seg_times[idx]
                fixed_c += per_c * t
                fixed_p += per_p * t

            consumed, produced = fixed_c, fixed_p
            if consumed > prev:
                msgs.append(f"이 단은 {consumed}코를 사용하는데 이전 단은 {prev}코입니다.")
                status = "error"
            elif consumed < prev:
                left = prev - consumed
                msgs.append(f"이전 단 {left}코가 남습니다(뜨지 않은 코).")
                if status == "ok":
                    status = "warn"

            stated = prow.stated_count if n == prow.row_end else None
            if stated is not None and stated != produced:
                msgs.append(f"계산 결과 {produced}코, 도안 표기 {stated}코로 다릅니다.")
                status = "error"

            seq: list[str] = []
            for i, seg in enumerate(prow.segments):
                seq += _expand(seg, seg_times.get(i, 1))

            results.append(
                RowResult(
                    row=n,
                    side=prow.side,
                    kind=prow.kind,
                    instruction_ko=prow.instruction_ko,
                    start_count=prev,
                    consumed=consumed,
                    produced=produced,
                    stated=stated,
                    status=status,
                    messages=msgs,
                    sequence=seq,
                )
            )
            if status == "error":
                errors.append(f"{n}단: " + " ".join(msgs))
            elif status == "warn" and not (is_crochet and all("남습니다" in m for m in msgs)):
                warnings.append(f"{n}단: " + " ".join(msgs))
            prev = produced
        expected_next = prow.row_end + 1

    if not results:
        errors.append("단 정보가 없습니다.")
    return ValidationReport(rows=results, final_count=prev, errors=errors, warnings=warnings)


def compare_progress(report: ValidationReport, row: int, actual: int) -> dict:
    """사용자가 센 코 수와 기대 코 수 비교."""
    r = report.get_row(row)
    if r is None:
        return {"status": "unknown", "message": f"{row}단은 도안에 없습니다. (도안은 {len(report.rows)}단까지)"}
    diff = actual - r.produced
    if diff == 0:
        return {"status": "ok", "expected": r.produced, "diff": 0, "message": f"{row}단 완료 기준 {r.produced}코, 정확합니다!"}
    word = "많아요" if diff > 0 else "적어요"
    return {
        "status": "mismatch",
        "expected": r.produced,
        "diff": diff,
        "message": f"{row}단 완료 기준 {r.produced}코여야 하는데 {abs(diff)}코 {word}.",
    }


def suspicious_rows(report: ValidationReport, row: int, diff: int, window: int = 4) -> list[dict]:
    """불일치 원인이 될 만한 최근 단(코 수가 변하는 단, 바늘비우기·모아뜨기가 있는 단)."""
    cands = []
    for r in report.rows:
        if r.row > row or r.row < max(1, row - window):
            continue
        shaping = [c for c in r.sequence if c in ("yo", "m1", "kfb", "k2tog", "ssk", "p2tog", "k3tog", "sk2p", "s2kp", "inc", "dec", "ch", "sk")]
        if shaping or r.produced != r.start_count:
            cands.append({"row": r.row, "instruction": r.instruction_ko, "change": r.produced - r.start_count, "shaping": sorted(set(shaping))})
    return cands
