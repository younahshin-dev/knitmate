"""뜨개 용어집: 약어 ↔ 한국어 ↔ 코 수 규칙의 단일 출처(Single Source of Truth)."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from ..config import DATA_DIR


@lru_cache(maxsize=1)
def load_glossary() -> list[dict]:
    return json.loads((DATA_DIR / "glossary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _by_code() -> dict[str, dict]:
    return {e["code"]: e for e in load_glossary()}


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]", "", s.lower())


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for e in load_glossary():
        for key in [e["code"], e["ko"], e["en"], *e.get("aliases", [])]:
            idx.setdefault(_norm(key), e)
    return idx


def get(code: str) -> dict | None:
    return _by_code().get(code.lower()) or _alias_index().get(_norm(code))


def lookup(term: str, craft: str | None = None) -> list[dict]:
    """약어/한글/영문 이름으로 용어 찾기 (부분 일치 포함)."""
    t = _norm(term)
    exact = _alias_index().get(t)
    results = [exact] if exact else []
    for e in load_glossary():
        keys = [e["code"], e["ko"], e["en"], *e.get("aliases", [])]
        if e not in results and any(t and (t in _norm(k) or _norm(k) in t) for k in keys if len(_norm(k)) > 1):
            results.append(e)
    if craft in ("knit", "crochet"):
        results = [e for e in results if e["craft"] == craft] or results
    return results[:5]


def stitch_rule(code: str) -> tuple[int, int] | None:
    e = get(code)
    if not e:
        return None
    return e["consumes"], e["produces"]


def ko_name(code: str) -> str:
    e = get(code)
    return e["ko"] if e else code


def codes_for(craft: str) -> list[str]:
    return [e["code"] for e in load_glossary() if e["craft"] == craft]
