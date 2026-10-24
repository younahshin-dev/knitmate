"""뜨개메이트 백엔드 API (FastAPI, 선택 실행)

실행: uvicorn api:app --reload --port 8000
문서: http://localhost:8000/docs
Streamlit UI와 같은 에이전트·도구를 REST로 노출해 다른 프론트엔드(앱, 웹)에서도 쓸 수 있게 한다.
"""
from __future__ import annotations

import base64
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from knitmate import config
from knitmate.core import storage
from knitmate.core.pdf_reader import read_upload
from knitmate.core.rag import search_knowledge
from knitmate.core.stitch_counter import ValidationReport, compare_progress, validate
from knitmate.core.videos import find_videos
from knitmate.schemas import PatternSpec

app = FastAPI(title="KnitMate API", version="1.0.0", description="뜨개 도안 변환·코 수 검증·기법 코칭 API")
_coach = None


def coach():
    global _coach
    if _coach is None:
        from knitmate.agents.coach_graph import build_coach_graph

        _coach = build_coach_graph()
    return _coach


class ChatIn(BaseModel):
    question: str
    thread_id: str | None = None
    craft: str = "any"
    project_id: str | None = None


class CheckIn(BaseModel):
    row: int
    actual_count: int


@app.get("/health")
def health():
    return {"llm": config.llm_available(), "embedding": config.embedding_available(), "youtube": config.youtube_available()}


@app.post("/pattern/convert")
async def convert(file: UploadFile = File(...), craft: str = Form("auto")):
    if not config.llm_available():
        raise HTTPException(503, "LLM 설정(.env)이 필요합니다.")
    from knitmate.agents.pattern_graph import run_pattern_pipeline

    info = read_upload(file.filename or "pattern.txt", await file.read())
    out = run_pattern_pipeline(info["text"], info["images"], craft, file.filename or "")
    return {
        "korean": out.get("korean"),
        "spec": out["spec"],
        "report": out["report"],
        "chart_png_base64": base64.b64encode(out["chart_png"]).decode() if out.get("chart_png") else None,
        "log": out.get("log", []),
    }


@app.post("/pattern/validate")
def validate_spec(spec: PatternSpec):
    return validate(spec).to_dict()


@app.post("/chat")
def chat(body: ChatIn):
    if not config.llm_available():
        raise HTTPException(503, "LLM 설정(.env)이 필요합니다.")
    from knitmate.agents.coach_graph import ask_coach

    tid = body.thread_id or uuid.uuid4().hex
    return {"thread_id": tid, **ask_coach(coach(), tid, body.question, body.craft, body.project_id)}


@app.get("/knowledge/search")
def knowledge(q: str, craft: str | None = None, k: int = 4):
    return search_knowledge(q, k=k, craft=craft)


@app.get("/videos")
def videos(technique: str, craft: str | None = None):
    return find_videos(technique, craft)


@app.get("/projects")
def projects():
    return storage.list_projects()


@app.post("/projects/{pid}/check")
def check(pid: str, body: CheckIn):
    p = storage.get_project(pid)
    if not p:
        raise HTTPException(404, "프로젝트가 없습니다.")
    res = compare_progress(ValidationReport.from_dict(p["report"]), body.row, body.actual_count)
    storage.add_log(pid, body.row, body.actual_count, res.get("expected"), res["status"], "API")
    if res["status"] == "ok":
        storage.set_current_row(pid, body.row + 1)
    return res
