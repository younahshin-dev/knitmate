"""프로젝트 진행 상태 저장소 (SQLite, 장기 메모리).

대화 메모리는 LangGraph 체크포인터가, 뜨개 진행 상황(현재 단, 코 수 기록)은 여기서 관리한다.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

from ..config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    craft TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    korean_md TEXT,
    current_row INTEGER DEFAULT 1,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS progress_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    row INTEGER,
    actual_count INTEGER,
    expected_count INTEGER,
    status TEXT,
    note TEXT,
    created_at TEXT
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def create_project(name: str, craft: str, spec: dict, report: dict, korean_md: str = "") -> str:
    pid = uuid.uuid4().hex[:8]
    with _conn() as con:
        con.execute(
            "INSERT INTO projects VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, name, craft, json.dumps(spec, ensure_ascii=False), json.dumps(report, ensure_ascii=False), korean_md, 1, _now(), _now()),
        )
    return pid


def list_projects() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT id, name, craft, current_row, updated_at FROM projects ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_project(pid: str) -> dict | None:
    with _conn() as con:
        r = con.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["spec"] = json.loads(d.pop("spec_json"))
    d["report"] = json.loads(d.pop("report_json"))
    return d


def set_current_row(pid: str, row: int) -> None:
    with _conn() as con:
        con.execute("UPDATE projects SET current_row=?, updated_at=? WHERE id=?", (max(1, row), _now(), pid))


def delete_project(pid: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM projects WHERE id=?", (pid,))
        con.execute("DELETE FROM progress_logs WHERE project_id=?", (pid,))


def add_log(pid: str, row: int, actual: int | None, expected: int | None, status: str, note: str = "") -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO progress_logs (project_id,row,actual_count,expected_count,status,note,created_at) VALUES (?,?,?,?,?,?,?)",
            (pid, row, actual, expected, status, note, _now()),
        )
        con.execute("UPDATE projects SET updated_at=? WHERE id=?", (_now(), pid))


def get_logs(pid: str, limit: int = 50) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT row, actual_count, expected_count, status, note, created_at FROM progress_logs WHERE project_id=? ORDER BY id DESC LIMIT ?",
            (pid, limit),
        ).fetchall()
    return [dict(r) for r in rows]
