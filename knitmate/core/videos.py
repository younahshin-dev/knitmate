"""영상 추천 (3단 폴백).

1순위 큐레이션 목록(data/videos.json) → 2순위 YouTube Data API(키가 있을 때)
→ 3순위 YouTube 검색 링크. LLM이 URL을 지어내지 못하도록 URL은 여기서만 만든다.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from urllib.parse import quote_plus

from ..config import DATA_DIR, env


@lru_cache(maxsize=1)
def _curated() -> list[dict]:
    return json.loads((DATA_DIR / "videos.json").read_text(encoding="utf-8"))["techniques"]


def _norm(s: str) -> str:
    return re.sub(r"\s", "", s.lower())


def match_technique(query: str, craft: str | None = None) -> dict | None:
    q = _norm(query)
    best, best_len = None, 0
    for t in _curated():
        if craft in ("knit", "crochet") and t["craft"] != craft:
            continue
        for name in t["names"]:
            n = _norm(name)
            if len(n) >= 2 and n in q and len(n) > best_len:
                best, best_len = t, len(n)
    return best


def _youtube_api(query: str, max_results: int = 3) -> list[dict]:
    key = env("YOUTUBE_API_KEY")
    if not key:
        return []
    try:
        import requests

        r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "q": query, "type": "video", "maxResults": max_results,
                    "relevanceLanguage": "ko", "regionCode": "KR", "safeSearch": "strict", "key": key},
            timeout=6,
        )
        r.raise_for_status()
        out = []
        for item in r.json().get("items", []):
            vid = item["id"].get("videoId")
            if vid:
                out.append({
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "source": "youtube_api",
                })
        return out
    except Exception:
        return []


def search_link(query: str) -> dict:
    return {
        "title": f"YouTube에서 '{query}' 검색",
        "channel": "YouTube 검색",
        "url": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
        "source": "search_link",
    }


def find_videos(technique: str, craft: str | None = None, max_results: int = 3) -> dict:
    t = match_technique(technique, craft)
    query = t["query"] if t else f"{'코바늘 ' if craft == 'crochet' else '대바늘 ' if craft == 'knit' else '뜨개질 '}{technique}"
    videos: list[dict] = []
    if t and t.get("videos"):
        videos = [{**v, "source": "curated"} for v in t["videos"]][:max_results]
    if not videos:
        videos = _youtube_api(query, max_results)
    if not videos:
        videos = [search_link(query)]
    mode = videos[0]["source"]
    return {"technique": t["names"][0] if t else technique, "query": query, "mode": mode, "videos": videos}
