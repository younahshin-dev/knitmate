"""오프라인 테스트: 코 수 검증, RAG, 영상 폴백, 차트 (LLM 불필요)."""
import json

from knitmate.config import SAMPLE_DIR
from knitmate.core.chart import render_chart
from knitmate.core.rag import build_index, search_knowledge
from knitmate.core.stitch_counter import compare_progress, validate
from knitmate.core.videos import find_videos
from knitmate.schemas import PatternRow, PatternSpec, Segment, StitchOp


def load(name):
    return PatternSpec(**json.loads((SAMPLE_DIR / f"{name}.spec.json").read_text(encoding="utf-8")))


def test_knit_sample_counts():
    rep = validate(load("lace_coaster_knit_en"))
    assert rep.ok, rep.errors
    assert rep.expected_count(10) == 13
    assert rep.final_count == 9


def test_crochet_sample_counts():
    rep = validate(load("round_coaster_crochet_en"))
    assert rep.ok, rep.errors
    assert [r.produced for r in rep.rows] == [6, 12, 18, 24, 30]


def test_detects_wrong_stated_count():
    spec = PatternSpec(
        reasoning="", title="t", craft="knit", cast_on=10, start_method="",
        rows=[PatternRow(row_start=1, row_end=1, segments=[Segment(ops=[StitchOp(code="k")], repeat_to_end=True)],
                         stated_count=12, instruction_ko="")],
    )
    rep = validate(spec)
    assert not rep.ok
    assert "12" in rep.errors[0]


def test_repeat_not_divisible():
    spec = PatternSpec(
        reasoning="", title="t", craft="knit", cast_on=11, start_method="",
        rows=[PatternRow(row_start=1, row_end=1, segments=[Segment(ops=[StitchOp(code="k", count=2), StitchOp(code="p", count=2)], repeat_to_end=True)],
                         instruction_ko="")],
    )
    assert not validate(spec).ok


def test_progress_compare():
    rep = validate(load("lace_coaster_knit_en"))
    assert compare_progress(rep, 10, 13)["status"] == "ok"
    res = compare_progress(rep, 10, 14)
    assert res["status"] == "mismatch" and res["diff"] == 1


def test_rag_finds_technique():
    build_index(verbose=False)
    hits = search_knowledge("한길긴뜨기 어떻게 떠?", k=3)
    assert any("한길긴뜨기" in h["section"] for h in hits)


def test_video_fallback_link():
    res = find_videos("한길긴뜨기", "crochet")
    assert res["videos"] and res["videos"][0]["url"].startswith("https://www.youtube.com/")


def test_chart_png():
    spec = load("round_coaster_crochet_en")
    png = render_chart(validate(spec), spec.craft, spec.title)
    assert png[:4] == b"\x89PNG"
